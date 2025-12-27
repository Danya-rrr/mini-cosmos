"""
Mini-Cosmos: VAE (Variational Autoencoder)
==========================================
Complete VAE model combining encoder and decoder.

Architecture:
    Image (256x256) -> Encoder -> Latent (32x32x4) -> Decoder -> Reconstruction (256x256)

Loss:
    L = Reconstruction Loss + beta * KL Divergence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from src.models.encoder import Encoder
from src.models.decoder import Decoder


@dataclass
class VAEConfig:
    """VAE configuration"""
    in_channels: int = 3
    latent_dim: int = 4
    hidden_dims: tuple = (64, 128, 256, 256)
    beta: float = 0.0001  # KL weight (low for better reconstruction)
    recon_loss_type: str = 'mse'  # 'mse' or 'l1'


class VAE(nn.Module):
    """
    Variational Autoencoder for image compression.
    
    Compresses 256x256 images to 32x32x4 latent space (64x compression).
    Uses reparameterization trick for end-to-end training.
    """
    
    def __init__(self, config: Optional[VAEConfig] = None):
        super().__init__()
        
        self.config = config or VAEConfig()
        
        # Encoder
        self.encoder = Encoder(
            in_channels=self.config.in_channels,
            latent_dim=self.config.latent_dim,
            hidden_dims=list(self.config.hidden_dims)
        )
        
        # Decoder
        self.decoder = Decoder(
            latent_dim=self.config.latent_dim,
            out_channels=self.config.in_channels,
            hidden_dims=list(reversed(self.config.hidden_dims))
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution.
        
        Args:
            x: [B, 3, 256, 256] input images
            
        Returns:
            mu: [B, latent_dim, 32, 32] mean
            logvar: [B, latent_dim, 32, 32] log variance
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.
        
        Args:
            z: [B, latent_dim, 32, 32] latent
            
        Returns:
            x_recon: [B, 3, 256, 256] reconstructed image
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass: encode, sample, decode.
        
        Args:
            x: [B, 3, 256, 256] input images
            
        Returns:
            Dict with:
                - 'recon': reconstructed images
                - 'mu': latent mean
                - 'logvar': latent log variance
                - 'z': sampled latent
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def get_latent(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get latent representation of images.
        
        Args:
            x: input images
            deterministic: if True, return mu instead of sampling
            
        Returns:
            z: latent representation
        """
        mu, logvar = self.encode(x)
        
        if deterministic:
            return mu
        else:
            return self.reparameterize(mu, logvar)


class VAELoss(nn.Module):
    """
    VAE Loss = Reconstruction Loss + beta * KL Divergence
    
    Reconstruction: MSE or L1 between input and reconstruction
    KL: Regularizes latent space to be close to N(0, 1)
    """
    
    def __init__(self, beta: float = 0.0001, recon_type: str = 'mse'):
        super().__init__()
        self.beta = beta
        self.recon_type = recon_type
    
    def forward(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            x: original images
            recon: reconstructed images
            mu: latent mean
            logvar: latent log variance
            
        Returns:
            Dict with 'loss', 'recon_loss', 'kl_loss'
        """
        # Reconstruction loss
        if self.recon_type == 'mse':
            recon_loss = F.mse_loss(recon, x, reduction='mean')
        else:  # L1
            recon_loss = F.l1_loss(recon, x, reduction='mean')
        
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + self.beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


# Test
if __name__ == '__main__':
    print("Testing VAE...")
    print("=" * 50)
    
    # Create VAE
    config = VAEConfig(
        in_channels=3,
        latent_dim=4,
        hidden_dims=(64, 128, 256, 256),
        beta=0.0001
    )
    
    vae = VAE(config)
    loss_fn = VAELoss(beta=config.beta)
    
    # Test input
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output = vae(x)
    
    print(f"Input shape:      {x.shape}")
    print(f"Recon shape:      {output['recon'].shape}")
    print(f"Mu shape:         {output['mu'].shape}")
    print(f"Logvar shape:     {output['logvar'].shape}")
    print(f"Z shape:          {output['z'].shape}")
    
    # Compute loss
    losses = loss_fn(x, output['recon'], output['mu'], output['logvar'])
    
    print(f"\nLosses:")
    print(f"  Total:   {losses['loss'].item():.4f}")
    print(f"  Recon:   {losses['recon_loss'].item():.4f}")
    print(f"  KL:      {losses['kl_loss'].item():.4f}")
    
    # Count parameters
    params = sum(p.numel() for p in vae.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    # Compression ratio
    input_size = 3 * 256 * 256
    latent_size = 4 * 32 * 32
    print(f"Compression ratio: {input_size / latent_size:.1f}x")
    
    print("\n[OK] VAE works correctly!")