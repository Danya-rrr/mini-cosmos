"""
Mini-Cosmos: VAE Training Script (Optimized)
=============================================
Train the VAE on CARLA + nuScenes driving data.

Optimizations:
    - Mixed Precision (AMP) for faster training & less memory
    - Gradient accumulation for larger effective batch size
    - Single frame loading (sequence_length=1) for VAE
    - Better logging and visualization

Usage:
    python scripts/train_vae.py --epochs 15 --batch_size 32

For Vast.ai (RTX 3090):
    python scripts/train_vae.py --epochs 15 --batch_size 64 --amp
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Local imports
from src.data.dataset import CARLADataset, DatasetConfig, create_combined_dataloaders
from src.models.vae import VAE, VAEConfig, VAELoss


class VAETrainer:
    """
    Trainer for VAE model with optimizations.
    
    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - KL annealing (optional)
    - Visualization of reconstructions
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
            weight_decay=config.get('weight_decay', 1e-5),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 15),
            eta_min=1e-6
        )
        
        # Mixed precision
        self.use_amp = config.get('amp', True) and device == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.grad_accum_steps = config.get('grad_accum_steps', 1)
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'samples').mkdir(exist_ok=True)
        
        # Logging
        self.history = {
            'train_loss': [], 'train_recon': [], 'train_kl': [],
            'val_loss': [], 'val_recon': [], 'val_kl': [],
            'lr': []
        }
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch with AMP and gradient accumulation."""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Get frames [B, T, C, H, W] or [B, C, H, W]
            frames = batch['frames'].to(self.device, non_blocking=True)
            
            # Handle both sequence and single frame formats
            if frames.dim() == 5:
                # Sequence format: use first frame only for VAE
                x = frames[:, 0]  # [B, C, H, W]
            else:
                x = frames  # Already [B, C, H, W]
            
            # Forward pass with AMP
            if self.use_amp:
                with autocast():
                    output = self.model(x)
                    losses = self.loss_fn(
                        x, output['recon'], 
                        output['mu'], output['logvar']
                    )
                    loss = losses['loss'] / self.grad_accum_steps
                
                # Backward with scaler
                self.scaler.scale(loss).backward()
            else:
                output = self.model(x)
                losses = self.loss_fn(
                    x, output['recon'], 
                    output['mu'], output['logvar']
                )
                loss = losses['loss'] / self.grad_accum_steps
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.use_amp:
                    # Unscale, clip, step, update
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Accumulate losses (use original loss, not divided)
            total_loss += losses['loss'].item()
            total_recon += losses['recon_loss'].item()
            total_kl += losses['kl_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'recon': f"{losses['recon_loss'].item():.4f}",
                'kl': f"{losses['kl_loss'].item():.2f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_kl = 0
        num_batches = 0
        
        for batch in self.val_loader:
            frames = batch['frames'].to(self.device, non_blocking=True)
            
            if frames.dim() == 5:
                x = frames[:, 0]
            else:
                x = frames
            
            if self.use_amp:
                with autocast():
                    output = self.model(x, deterministic=True)
                    losses = self.loss_fn(
                        x, output['recon'], 
                        output['mu'], output['logvar']
                    )
            else:
                output = self.model(x, deterministic=True)
                losses = self.loss_fn(
                    x, output['recon'], 
                    output['mu'], output['logvar']
                )
            
            total_loss += losses['loss'].item()
            total_recon += losses['recon_loss'].item()
            total_kl += losses['kl_loss'].item()
            num_batches += 1
        
        return {
            'loss': total_loss / num_batches,
            'recon': total_recon / num_batches,
            'kl': total_kl / num_batches
        }
    
    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save sample reconstructions."""
        self.model.eval()
        
        # Get a batch from validation
        batch = next(iter(self.val_loader))
        frames = batch['frames'].to(self.device)
        
        if frames.dim() == 5:
            x = frames[:num_samples, 0]
        else:
            x = frames[:num_samples]
        
        # Reconstruct
        if self.use_amp:
            with autocast():
                output = self.model(x, deterministic=True)
        else:
            output = self.model(x, deterministic=True)
        
        recon = output['recon']
        
        # Denormalize: [-1, 1] -> [0, 1]
        x = (x + 1) / 2
        recon = (recon + 1) / 2
        
        # Create figure
        fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
        
        for i in range(num_samples):
            # Original
            axes[0, i].imshow(x[i].permute(1, 2, 0).cpu().float().numpy())
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Reconstruction
            axes[1, i].imshow(recon[i].permute(1, 2, 0).cpu().clamp(0, 1).float().numpy())
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')
        
        plt.suptitle(f'Epoch {epoch}')
        plt.tight_layout()
        
        # Save
        sample_path = self.output_dir / 'samples' / f'epoch_{epoch:03d}.png'
        plt.savefig(sample_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest
        checkpoint_path = self.output_dir / 'checkpoints' / 'vae_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'checkpoints' / 'vae_best.pt'
            torch.save(checkpoint, best_path)
            print(f"  ★ New best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, path: str) -> int:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        return checkpoint['epoch']
    
    def save_history_plot(self):
        """Save training history plot."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        epochs = range(len(self.history['train_loss']))
        
        # Loss
        axes[0].plot(epochs, self.history['train_loss'], label='Train')
        axes[0].plot(epochs, self.history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].legend()
        axes[0].set_title('Total Loss')
        
        # Reconstruction
        axes[1].plot(epochs, self.history['train_recon'], label='Train')
        axes[1].plot(epochs, self.history['val_recon'], label='Val')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Reconstruction Loss')
        axes[1].legend()
        axes[1].set_title('Reconstruction Loss')
        
        # Learning rate
        axes[2].plot(epochs, self.history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.png', dpi=100)
        plt.close()
    
    def train(self, epochs: int, resume: str = None):
        """Full training loop."""
        start_epoch = 0
        
        # Resume from checkpoint
        if resume and Path(resume).exists():
            print(f"Resuming from {resume}")
            start_epoch = self.load_checkpoint(resume) + 1
        
        # Training info
        print(f"\n{'='*60}")
        print(f"VAE Training Configuration")
        print(f"{'='*60}")
        print(f"Device:              {self.device}")
        print(f"Mixed Precision:     {self.use_amp}")
        print(f"Training samples:    {len(self.train_loader.dataset):,}")
        print(f"Validation samples:  {len(self.val_loader.dataset):,}")
        print(f"Batch size:          {self.train_loader.batch_size}")
        print(f"Gradient accum:      {self.grad_accum_steps}")
        print(f"Effective batch:     {self.train_loader.batch_size * self.grad_accum_steps}")
        print(f"Learning rate:       {self.config.get('lr', 1e-4)}")
        print(f"Beta (KL weight):    {self.config.get('beta', 0.0001)}")
        print(f"Epochs:              {start_epoch} -> {epochs}")
        print(f"{'='*60}\n")
        
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
            print(f"  Train | Loss: {train_metrics['loss']:.4f} | "
                  f"Recon: {train_metrics['recon']:.4f} | KL: {train_metrics['kl']:.2f}")
            print(f"  Val   | Loss: {val_metrics['loss']:.4f} | "
                  f"Recon: {val_metrics['recon']:.4f} | KL: {val_metrics['kl']:.2f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Check best
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Save samples every 5 epochs
            if epoch % 5 == 0 or epoch == epochs - 1:
                self.save_samples(epoch)
        
        # Save final artifacts
        self.save_history_plot()
        
        history_path = self.output_dir / 'vae_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Checkpoints: {self.output_dir / 'checkpoints'}")
        print(f"Samples:     {self.output_dir / 'samples'}")


def main():
    parser = argparse.ArgumentParser(description='Train VAE on driving data')
    
    # Data
    parser.add_argument('--carla_dir', type=str, default='./data/raw',
                        help='Path to CARLA data')
    parser.add_argument('--nuscenes_dir', type=str, default='./data/processed/nuscenes',
                        help='Path to nuScenes data')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=4,
                        help='Latent space dimension')
    parser.add_argument('--beta', type=float, default=0.0001,
                        help='KL loss weight')
    
    # Training
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of epochs (default: 15)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--amp', action='store_true',
                        help='Use mixed precision training')
    parser.add_argument('--grad_accum', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Dataset config - single frames for VAE
    dataset_config = DatasetConfig(
        sequence_length=1,  # Single frames for VAE training
        frame_skip=1,
        image_size=(256, 256),
        augment=True,
        return_actions=False,  # Not needed for VAE
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_loader, val_loader, _ = create_combined_dataloaders(
        carla_dir=args.carla_dir,
        nuscenes_dir=args.nuscenes_dir,
        config=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    if train_loader is None or len(train_loader) == 0:
        print("\n[ERROR] No training data found!")
        print("Please ensure data exists in:")
        print(f"  - {args.carla_dir}")
        print(f"  - {args.nuscenes_dir}")
        return
    
    # Create model
    vae_config = VAEConfig(
        in_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=(64, 128, 256, 256),
        beta=args.beta
    )
    
    model = VAE(vae_config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,} ({trainable_params:,} trainable)")
    
    # Training config
    train_config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'beta': args.beta,
        'weight_decay': 1e-5,
        'recon_type': 'mse',
        'output_dir': args.output_dir,
        'amp': args.amp or (device == 'cuda'),  # Auto-enable AMP on CUDA
        'grad_accum_steps': args.grad_accum,
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