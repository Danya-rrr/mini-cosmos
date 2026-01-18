"""
VAE (Variational Autoencoder) Module

Complete VAE implementation combining encoder and decoder for image 
compression and reconstruction with learned latent representations.
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
    """Configuration for VAE architecture and training."""
    in_channels: int = 3
    latent_dim: int = 4
    hidden_dims: tuple = (64, 128, 256, 256)
    beta: float = 0.0001
    recon_loss_type: str = 'mse'
    l1_weight: float = 0.5


class VAE(nn.Module):
    """
    Variational Autoencoder for image compression and reconstruction.
    
    Compresses images to a compact latent representation using encoder-decoder
    architecture with reparameterization trick for end-to-end training.
    
    Args:
        config: VAE configuration dataclass
    """
    
    def __init__(self, config: Optional[VAEConfig] = None):
        super().__init__()
        
        self.config = config or VAEConfig()
        
        self.encoder = Encoder(
            in_channels=self.config.in_channels,
            latent_dim=self.config.latent_dim,
            hidden_dims=list(self.config.hidden_dims)
        )
        
        self.decoder = Decoder(
            latent_dim=self.config.latent_dim,
            out_channels=self.config.in_channels,
            hidden_dims=list(reversed(self.config.hidden_dims))
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to latent distribution parameters.
        
        Args:
            x: Input images of shape [B, C, H, W]
            
        Returns:
            Tuple of (mu, logvar) for latent distribution
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to images.
        
        Args:
            z: Latent representation of shape [B, latent_dim, h, w]
            
        Returns:
            Reconstructed images of shape [B, C, H, W]
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample from latent distribution using reparameterization trick.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(
        self, 
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Complete forward pass through VAE.
        
        Args:
            x: Input images of shape [B, C, H, W]
            deterministic: If True, use mean instead of sampling
            
        Returns:
            Dictionary containing reconstruction and latent parameters
        """
        mu, logvar = self.encode(x)
        
        z = mu if deterministic else self.reparameterize(mu, logvar)
        
        recon = self.decode(z)
        
        return {
            'recon': recon,
            'mu': mu,
            'logvar': logvar,
            'z': z
        }
    
    def get_latent(
        self, 
        x: torch.Tensor, 
        deterministic: bool = True
    ) -> torch.Tensor:
        """
        Extract latent representation from images.
        
        Args:
            x: Input images of shape [B, C, H, W]
            deterministic: If True, return mean (recommended for downstream tasks)
            
        Returns:
            Latent representation of shape [B, latent_dim, h, w]
        """
        mu, logvar = self.encode(x)
        return mu if deterministic else self.reparameterize(mu, logvar)
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images in inference mode.
        
        Args:
            x: Input images
            
        Returns:
            Reconstructed images
        """
        self.eval()
        output = self.forward(x, deterministic=True)
        return output['recon']


class VAELoss(nn.Module):
    """
    VAE training loss combining reconstruction and KL divergence terms.
    
    Loss = Reconstruction Loss + beta * KL Divergence
    
    Args:
        beta: Weight for KL divergence term
        recon_type: Type of reconstruction loss ('mse', 'l1', or 'mixed')
        l1_weight: Weight for L1 component in mixed mode
    """
    
    def __init__(
        self, 
        beta: float = 0.0001, 
        recon_type: str = 'mse',
        l1_weight: float = 0.5
    ):
        super().__init__()
        self.beta = beta
        self.recon_type = recon_type
        self.l1_weight = l1_weight
    
    def reconstruction_loss(
        self, 
        recon: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss based on configured type."""
        if self.recon_type == 'mse':
            return F.mse_loss(recon, target, reduction='mean')
        elif self.recon_type == 'l1':
            return F.l1_loss(recon, target, reduction='mean')
        else:
            mse = F.mse_loss(recon, target, reduction='mean')
            l1 = F.l1_loss(recon, target, reduction='mean')
            return (1 - self.l1_weight) * mse + self.l1_weight * l1
    
    def kl_divergence(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between learned distribution and standard normal.
        
        KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        """
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    def forward(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_override: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total VAE loss and its components.
        
        Args:
            x: Original images
            recon: Reconstructed images
            mu: Latent distribution mean
            logvar: Latent distribution log variance
            beta_override: Optional beta value for KL annealing
            
        Returns:
            Dictionary with total loss and individual components
        """
        recon_loss = self.reconstruction_loss(recon, x)
        kl_loss = self.kl_divergence(mu, logvar)
        
        beta = beta_override if beta_override is not None else self.beta
        loss = recon_loss + beta * kl_loss
        
        return {
            'loss': loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


def main():
    """Test VAE functionality."""
    config = VAEConfig(
        in_channels=3,
        latent_dim=4,
        hidden_dims=(64, 128, 256, 256),
        beta=0.0001
    )
    
    vae = VAE(config)
    loss_fn = VAELoss(beta=config.beta, recon_type='mse')
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    output = vae(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Recon shape:  {output['recon'].shape}")
    print(f"Mu shape:     {output['mu'].shape}")
    print(f"Logvar shape: {output['logvar'].shape}")
    print(f"Latent shape: {output['z'].shape}")
    
    assert output['recon'].shape == x.shape
    assert output['mu'].shape == (batch_size, 4, 32, 32)
    
    losses = loss_fn(x, output['recon'], output['mu'], output['logvar'])
    
    print(f"\nLosses:")
    print(f"  Total:        {losses['loss'].item():.4f}")
    print(f"  Recon:        {losses['recon_loss'].item():.4f}")
    print(f"  KL:           {losses['kl_loss'].item():.4f}")
    
    encoder_params = sum(p.numel() for p in vae.encoder.parameters())
    decoder_params = sum(p.numel() for p in vae.decoder.parameters())
    total_params = sum(p.numel() for p in vae.parameters())
    
    print(f"\nParameters:")
    print(f"  Encoder:      {encoder_params:,}")
    print(f"  Decoder:      {decoder_params:,}")
    print(f"  Total:        {total_params:,}")
    
    input_size = 3 * 256 * 256
    latent_size = 4 * 32 * 32
    print(f"\nCompression ratio: {input_size / latent_size:.1f}x")
    print("\nVAE test passed successfully!")


if __name__ == '__main__':
    main()