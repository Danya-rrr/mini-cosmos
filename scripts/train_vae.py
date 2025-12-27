"""
Mini-Cosmos: VAE Training Script
================================
Train the VAE on CARLA driving data.

Usage:
    python scripts/train_vae.py --epochs 50 --batch_size 8
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime

# Local imports
from src.data.dataset import CARLADataset, DatasetConfig, create_combined_dataloaders
from src.models.vae import VAE, VAEConfig, VAELoss


class VAETrainer:
    """
    Trainer for VAE model.
    
    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: VAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.loss_fn = VAELoss(
            beta=config.get('beta', 0.0001),
            recon_type=config.get('recon_type', 'mse')
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 50),
            eta_min=1e-6
        )
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'lr': []
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # Get frames [B, T, C, H, W]
            frames = batch['frames'].to(self.device)
            
            # Use only first frame for VAE training
            # Later we'll use sequences for world model
            x = frames[:, 0]  # [B, C, H, W]
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            
            # Compute loss
            losses = self.loss_fn(x, output['recon'], output['mu'], output['logvar'])
            
            # Backward pass
            losses['loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += losses['loss'].item()
            total_recon += losses['recon_loss'].item()
            total_kl += losses['kl_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'recon': f"{losses['recon_loss'].item():.4f}",
                'kl': f"{losses['kl_loss'].item():.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        for batch in self.val_loader:
            frames = batch['frames'].to(self.device)
            x = frames[:, 0]
            
            output = self.model(x)
            losses = self.loss_fn(x, output['recon'], output['mu'], output['logvar'])
            
            total_loss += losses['loss'].item()
            total_recon += losses['recon_loss'].item()
            total_kl += losses['kl_loss'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest
        checkpoint_path = self.output_dir / 'checkpoints' / 'vae_latest.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'vae_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  [*] Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        return checkpoint['epoch']
    
    def train(self, epochs: int, resume: str = None):
        """Full training loop"""
        start_epoch = 0
        
        # Resume from checkpoint
        if resume and Path(resume).exists():
            print(f"Resuming from {resume}")
            start_epoch = self.load_checkpoint(resume) + 1
        
        print(f"\nStarting training from epoch {start_epoch}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        for epoch in range(start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['recon'])
            self.history['train_kl'].append(train_metrics['kl'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['recon'])
            self.history['val_kl'].append(val_metrics['kl'])
            self.history['lr'].append(current_lr)
            
            # Print progress
            print(f"\nEpoch {epoch}/{epochs-1}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Recon: {val_metrics['recon']:.4f}, KL: {val_metrics['kl']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            self.save_checkpoint(epoch, is_best)
        
        # Save training history
        history_path = self.output_dir / 'vae_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 50)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints saved to: {self.output_dir / 'checkpoints'}")


def main():
    parser = argparse.ArgumentParser(description='Train VAE on CARLA data')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='./data/raw',
                        help='Path to CARLA data')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=4,
                        help='Latent space dimension')
    parser.add_argument('--beta', type=float, default=0.0001,
                        help='KL loss weight')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Dataset config
    dataset_config = DatasetConfig(
        sequence_length=8,
        frame_skip=1,
        image_size=(256, 256),
        augment=True,
    )
    
    # Create datasets
    print("Loading datasets...")
    train_loader, val_loader, _ = create_combined_dataloaders(
        carla_dir='./data/raw',
        nuscenes_dir='./data/processed/nuscenes',
        config=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # Check if we have enough data
    if train_loader is None or len(train_loader) == 0:
        print("[ERROR] No training data found!")
        print("Run: python scripts/collect_data.py --episodes 20 --frames 500")
        return
    
    # Create model
    vae_config = VAEConfig(
        in_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=(64, 128, 256, 256),
        beta=args.beta
    )
    
    model = VAE(vae_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    train_config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'beta': args.beta,
        'weight_decay': 1e-5,
        'recon_type': 'mse',
        'output_dir': args.output_dir,
    }
    
    # Create trainer
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device
    )
    
    # Train
    trainer.train(args.epochs, resume=args.resume)


if __name__ == '__main__':
    main()