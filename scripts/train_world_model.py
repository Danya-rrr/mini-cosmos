"""
Mini-Cosmos: World Model Training Script
========================================
Train the World Model to predict future latent frames.

Usage:
    python scripts/train_world_model.py --epochs 30 --batch_size 8

Requirements:
    - Trained VAE checkpoint (outputs/checkpoints/vae_best.pt)
    - CARLA data in data/raw/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Local imports
from src.data.dataset import CARLADataset, DatasetConfig, create_combined_dataloaders
from src.models.vae import VAE, VAEConfig
from src.models.world_model import WorldModel, WorldModelConfig, WorldModelLoss


class WorldModelTrainer:
    """
    Trainer for World Model.
    
    Uses frozen VAE to encode/decode frames.
    Trains transformer to predict next latent frame.
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        vae: VAE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = 'cuda'
    ):
        self.world_model = world_model.to(device)
        self.vae = vae.to(device)
        self.vae.eval()  # Freeze VAE
        
        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.loss_fn = WorldModelLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            world_model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 0.01),
            betas=(0.9, 0.95)
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * config.get('epochs', 30)
        warmup_steps = int(total_steps * 0.1)
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.get('lr', 1e-4),
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )
        
        # Mixed precision
        self.use_amp = config.get('amp', True) and device == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.output_dir / 'world_model_samples').mkdir(exist_ok=True)
        
        # Context length
        self.context_length = config.get('context_length', 8)
        
        # Logging
        self.history = {
            'train_loss': [], 'val_loss': [], 'lr': []
        }
        
        self.best_val_loss = float('inf')
        self.global_step = 0
    
    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB frames to latent space using frozen VAE.
        
        Args:
            frames: [B, T, C, H, W] RGB frames
            
        Returns:
            [B, T, latent_dim, h, w] latent frames
        """
        B, T = frames.shape[:2]
        
        # Flatten batch and time
        frames_flat = frames.view(B * T, *frames.shape[2:])
        
        # Encode with VAE
        latents = self.vae.get_latent(frames_flat, deterministic=True)
        
        # Reshape back
        latents = latents.view(B, T, *latents.shape[1:])
        
        return latents
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.world_model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            # Get frames and actions
            frames = batch['frames'].to(self.device, non_blocking=True)
            actions = batch.get('actions')
            if actions is not None:
                actions = actions.to(self.device, non_blocking=True)
            
            # Need at least context_length + 1 frames
            if frames.shape[1] < self.context_length + 1:
                continue
            
            # Encode all frames to latent space
            with torch.no_grad():
                if self.use_amp:
                    with autocast('cuda'):
                        latents = self.encode_frames(frames)
                else:
                    latents = self.encode_frames(frames)
            
            # Split into context and target
            context_latents = latents[:, :self.context_length]  # First 8 frames
            target_latent = latents[:, self.context_length]     # 9th frame
            
            # Get context actions
            if actions is not None:
                context_actions = actions[:, :self.context_length]
            else:
                context_actions = None
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast('cuda'):
                    output = self.world_model(context_latents, context_actions)
                    losses = self.loss_fn(output['pred_latent'], target_latent)
                
                self.scaler.scale(losses['loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.world_model(context_latents, context_actions)
                losses = self.loss_fn(output['pred_latent'], target_latent)
                
                losses['loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            self.global_step += 1
            
            # Accumulate
            total_loss += losses['loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['loss'].item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return {'loss': total_loss / max(num_batches, 1)}
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate the model."""
        self.world_model.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            frames = batch['frames'].to(self.device, non_blocking=True)
            actions = batch.get('actions')
            if actions is not None:
                actions = actions.to(self.device, non_blocking=True)
            
            if frames.shape[1] < self.context_length + 1:
                continue
            
            # Encode frames
            if self.use_amp:
                with autocast('cuda'):
                    latents = self.encode_frames(frames)
            else:
                latents = self.encode_frames(frames)
            
            context_latents = latents[:, :self.context_length]
            target_latent = latents[:, self.context_length]
            
            if actions is not None:
                context_actions = actions[:, :self.context_length]
            else:
                context_actions = None
            
            # Forward
            if self.use_amp:
                with autocast('cuda'):
                    output = self.world_model(context_latents, context_actions)
                    losses = self.loss_fn(output['pred_latent'], target_latent)
            else:
                output = self.world_model(context_latents, context_actions)
                losses = self.loss_fn(output['pred_latent'], target_latent)
            
            total_loss += losses['loss'].item()
            num_batches += 1
        
        return {'loss': total_loss / max(num_batches, 1)}
    
    @torch.no_grad()
    def save_samples(self, epoch: int, num_samples: int = 4):
        """Save sample predictions."""
        self.world_model.eval()
        
        # Get a batch
        batch = next(iter(self.val_loader))
        frames = batch['frames'][:num_samples].to(self.device)
        actions = batch.get('actions')
        if actions is not None:
            actions = actions[:num_samples].to(self.device)
        
        if frames.shape[1] < self.context_length + 1:
            return
        
        # Encode frames
        latents = self.encode_frames(frames)
        
        context_latents = latents[:, :self.context_length]
        target_latent = latents[:, self.context_length]
        
        if actions is not None:
            context_actions = actions[:, :self.context_length]
        else:
            context_actions = None
        
        # Predict
        output = self.world_model(context_latents, context_actions)
        pred_latent = output['pred_latent']
        
        # Decode latents to images
        target_recon = self.vae.decode(target_latent)
        pred_recon = self.vae.decode(pred_latent)
        last_context_recon = self.vae.decode(context_latents[:, -1])
        
        # Denormalize
        def denorm(x):
            return (x + 1) / 2
        
        target_recon = denorm(target_recon)
        pred_recon = denorm(pred_recon)
        last_context_recon = denorm(last_context_recon)
        
        # Create figure
        fig, axes = plt.subplots(3, num_samples, figsize=(4 * num_samples, 12))
        
        for i in range(num_samples):
            # Last context frame
            axes[0, i].imshow(last_context_recon[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
            axes[0, i].set_title('Last Context')
            axes[0, i].axis('off')
            
            # Target (ground truth next frame)
            axes[1, i].imshow(target_recon[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
            axes[1, i].set_title('Target (GT)')
            axes[1, i].axis('off')
            
            # Prediction
            axes[2, i].imshow(pred_recon[i].permute(1, 2, 0).cpu().clamp(0, 1).numpy())
            axes[2, i].set_title('Prediction')
            axes[2, i].axis('off')
        
        plt.suptitle(f'World Model - Epoch {epoch}')
        plt.tight_layout()
        
        save_path = self.output_dir / 'world_model_samples' / f'epoch_{epoch:03d}.png'
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.world_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'config': self.config,
            'history': self.history,
            'best_val_loss': self.best_val_loss
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / 'checkpoints' / 'world_model_latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.output_dir / 'checkpoints' / 'world_model_best.pt')
            print(f"  ★ New best model (val_loss: {self.best_val_loss:.4f})")
    
    def load_checkpoint(self, path: str) -> int:
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.world_model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        return checkpoint['epoch']
    
    def train(self, epochs: int, resume: str = None):
        """Full training loop."""
        start_epoch = 0
        
        if resume and Path(resume).exists():
            print(f"Resuming from {resume}")
            start_epoch = self.load_checkpoint(resume) + 1
        
        print(f"\n{'='*60}")
        print(f"World Model Training")
        print(f"{'='*60}")
        print(f"Device:              {self.device}")
        print(f"Mixed Precision:     {self.use_amp}")
        print(f"Context length:      {self.context_length}")
        print(f"Training samples:    {len(self.train_loader.dataset):,}")
        print(f"Validation samples:  {len(self.val_loader.dataset):,}")
        print(f"Batch size:          {self.train_loader.batch_size}")
        print(f"Learning rate:       {self.config.get('lr', 1e-4)}")
        print(f"Epochs:              {start_epoch} -> {epochs}")
        print(f"{'='*60}\n")
        
        for epoch in range(start_epoch, epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Log
            current_lr = self.scheduler.get_last_lr()[0]
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['lr'].append(current_lr)
            
            # Print
            print(f"\nEpoch {epoch}/{epochs-1}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Check best
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Save
            self.save_checkpoint(epoch, is_best)
            
            # Samples every 5 epochs
            if epoch % 5 == 0 or epoch == epochs - 1:
                self.save_samples(epoch)
        
        # Save history
        with open(self.output_dir / 'world_model_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Train World Model')
    
    # Data
    parser.add_argument('--carla_dir', type=str, default='./data/raw')
    parser.add_argument('--nuscenes_dir', type=str, default='./data/processed/nuscenes')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--context_length', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # Training
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--amp', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default=None)
    
    # VAE checkpoint
    parser.add_argument('--vae_checkpoint', type=str, default='./outputs/checkpoints/vae_best.pt')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check VAE checkpoint
    vae_path = Path(args.vae_checkpoint)
    if not vae_path.exists():
        print(f"[ERROR] VAE checkpoint not found: {vae_path}")
        print("Train VAE first: python scripts/train_vae.py")
        return
    
    # Load VAE
    print("Loading VAE...")
    vae_checkpoint = torch.load(vae_path, map_location=device)
    
    vae_config = VAEConfig(
        in_channels=3,
        latent_dim=4,
        hidden_dims=(64, 128, 256, 256)
    )
    vae = VAE(vae_config)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    print(f"Loaded VAE from epoch {vae_checkpoint['epoch']}")
    
    # Dataset config - need sequences for world model
    dataset_config = DatasetConfig(
        sequence_length=args.context_length + 1,  # Context + target
        frame_skip=1,
        image_size=(256, 256),
        augment=True,
        return_actions=True  # Need actions for world model
    )
    
    # Create datasets
    print("\nLoading datasets...")
    train_loader, val_loader, _ = create_combined_dataloaders(
        carla_dir=args.carla_dir,
        nuscenes_dir=args.nuscenes_dir,
        config=dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if train_loader is None:
        print("[ERROR] No training data!")
        return
    
    # Create World Model
    wm_config = WorldModelConfig(
        latent_dim=4,
        latent_size=32,
        context_length=args.context_length,
        action_dim=3,
        use_actions=True,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1,
        latent_patch_size=4
    )
    
    world_model = WorldModel(wm_config)
    
    params = sum(p.numel() for p in world_model.parameters())
    print(f"\nWorld Model parameters: {params:,}")
    
    # Training config
    train_config = {
        'epochs': args.epochs,
        'lr': args.lr,
        'weight_decay': 0.01,
        'amp': args.amp,
        'output_dir': args.output_dir,
        'context_length': args.context_length
    }
    
    # Create trainer
    trainer = WorldModelTrainer(
        world_model=world_model,
        vae=vae,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device
    )
    
    # Train
    trainer.train(args.epochs, resume=args.resume)


if __name__ == '__main__':
    main()