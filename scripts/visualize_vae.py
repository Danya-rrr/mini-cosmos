"""
VAE Visualization Script

Visualizes VAE reconstruction quality and latent space statistics
for evaluating compression performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from src.data.dataset import CARLADataset, DatasetConfig
from src.models.vae import VAE, VAEConfig


def denormalize(tensor):
    """Convert from [-1, 1] to [0, 1] range."""
    return (tensor + 1) / 2


def visualize_reconstruction(
    model: VAE,
    dataset: CARLADataset,
    num_samples: int = 4,
    device: str = 'cuda',
    save_path: str = None
):
    """
    Visualize original images vs VAE reconstructions.
    
    Args:
        model: Trained VAE model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device for computation
        save_path: Path to save visualization
    """
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    with torch.no_grad():
        for i in range(num_samples):
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]
            
            x = sample['frames'][0].unsqueeze(0).to(device)
            
            output = model(x)
            recon = output['recon']
            
            x_np = denormalize(x[0]).cpu().permute(1, 2, 0).numpy()
            recon_np = denormalize(recon[0]).cpu().permute(1, 2, 0).numpy()
            
            x_np = np.clip(x_np, 0, 1)
            recon_np = np.clip(recon_np, 0, 1)
            
            axes[0, i].imshow(x_np)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon_np)
            axes[1, i].set_title('Reconstruction')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def visualize_latent_space(
    model: VAE,
    dataset: CARLADataset,
    num_samples: int = 16,
    device: str = 'cuda',
    save_path: str = None
):
    """
    Visualize latent space distribution statistics.
    
    Args:
        model: Trained VAE model
        dataset: Dataset to sample from
        num_samples: Number of samples for statistics
        device: Device for computation
        save_path: Path to save visualization
    """
    model.eval()
    
    mus = []
    logvars = []
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            x = sample['frames'][0].unsqueeze(0).to(device)
            
            mu, logvar = model.encode(x)
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
    
    mus = torch.cat(mus, dim=0)
    logvars = torch.cat(logvars, dim=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(mus.flatten().numpy(), bins=50, alpha=0.7)
    axes[0].set_title(f'Mu Distribution\nmean={mus.mean():.3f}, std={mus.std():.3f}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    
    axes[1].hist(logvars.flatten().numpy(), bins=50, alpha=0.7, color='orange')
    axes[1].set_title(f'Logvar Distribution\nmean={logvars.mean():.3f}, std={logvars.std():.3f}')
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


def main():
    """Main entry point for VAE visualization."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    checkpoint_path = Path('outputs/checkpoints/vae_best.pt')
    data_path = Path('data/raw')
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Train the model first: python scripts/train_vae.py")
        return
    
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vae_config = VAEConfig(
        in_channels=3,
        latent_dim=4,
        hidden_dims=(64, 128, 256, 256),
    )
    
    model = VAE(vae_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    print("Loading dataset...")
    dataset_config = DatasetConfig(
        sequence_length=8,
        frame_skip=1,
        image_size=(256, 256),
        augment=False,
    )
    
    dataset = CARLADataset(data_path, dataset_config, split='test')
    
    if len(dataset) == 0:
        print("No test data, using train split")
        dataset = CARLADataset(data_path, dataset_config, split='train')
    
    print(f"Dataset size: {len(dataset)}")
    
    print("\nVisualizing reconstruction...")
    visualize_reconstruction(
        model, dataset,
        num_samples=4,
        device=device,
        save_path=output_dir / 'vae_reconstruction.png'
    )
    
    print("\nVisualizing latent space...")
    visualize_latent_space(
        model, dataset,
        num_samples=50,
        device=device,
        save_path=output_dir / 'vae_latent_space.png'
    )
    
    print(f"\n[OK] Visualization complete!")
    print(f"Images saved to: {output_dir}")


if __name__ == '__main__':
    main()