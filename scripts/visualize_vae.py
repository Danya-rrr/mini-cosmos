"""
Mini-Cosmos: VAE Visualization
==============================
Visualize VAE reconstruction quality.

Usage:
    python scripts/visualize_vae.py
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
    """Convert from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def visualize_reconstruction(
    model: VAE,
    dataset: CARLADataset,
    num_samples: int = 4,
    device: str = 'cuda',
    save_path: str = None
):
    """
    Visualize original vs reconstructed images.
    """
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get sample
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]
            
            # Get first frame
            x = sample['frames'][0].unsqueeze(0).to(device)  # [1, 3, H, W]
            
            # Reconstruct
            output = model(x)
            recon = output['recon']
            
            # Convert to numpy
            x_np = denormalize(x[0]).cpu().permute(1, 2, 0).numpy()
            recon_np = denormalize(recon[0]).cpu().permute(1, 2, 0).numpy()
            
            # Clip values
            x_np = np.clip(x_np, 0, 1)
            recon_np = np.clip(recon_np, 0, 1)
            
            # Plot original
            axes[0, i].imshow(x_np)
            axes[0, i].set_title('Original')
            axes[0, i].axis('off')
            
            # Plot reconstruction
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
    Visualize latent space statistics.
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
    
    # Mu distribution
    axes[0].hist(mus.flatten().numpy(), bins=50, alpha=0.7)
    axes[0].set_title(f'Mu Distribution\nmean={mus.mean():.3f}, std={mus.std():.3f}')
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Count')
    
    # Logvar distribution
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
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    checkpoint_path = Path('outputs/checkpoints/vae_best.pt')
    data_path = Path('data/raw')
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check checkpoint exists
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("Train the model first: python scripts/train_vae.py")
        return
    
    # Load model
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
    
    # Load dataset
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
    
    # Visualize reconstruction
    print("\n" + "=" * 50)
    print("Visualizing reconstruction...")
    print("=" * 50)
    
    visualize_reconstruction(
        model, dataset,
        num_samples=4,
        device=device,
        save_path=output_dir / 'vae_reconstruction.png'
    )
    
    # Visualize latent space
    print("\n" + "=" * 50)
    print("Visualizing latent space...")
    print("=" * 50)
    
    visualize_latent_space(
        model, dataset,
        num_samples=50,
        device=device,
        save_path=output_dir / 'vae_latent_space.png'
    )
    
    print("\n[OK] Visualization complete!")
    print(f"Images saved to: {output_dir}")


if __name__ == '__main__':
    main()