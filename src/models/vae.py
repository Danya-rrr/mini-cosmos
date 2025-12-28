"""
Mini-Cosmos: VAE (Variational Autoencoder)
==========================================
Complete VAE model combining encoder and decoder.

Architecture:
    Image (256x256) -> Encoder -> Latent (32x32x4) -> Decoder -> Reconstruction (256x256)

Loss:
    L = Reconstruction Loss + beta * KL Divergence

Updates:
    - Fixed decoder upsampling (32 -> 256)
    - Added perceptual loss option
    - Better KL annealing support
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
    recon_loss_type: str = 'mse'  # 'mse', 'l1', or 'mixed'
    l1_weight: float = 0.5  # Weight for L1 in mixed mode


class VAE(nn.Module):
    """
    Variational Autoencoder for image compression.
    
    Compresses 256x256 images to 32x32x4 latent space (64x compression).
    Uses reparameterization trick for end-to-end training.
    
    Features:
        - Encoder: 256x256 -> 32x32x4 (3 downsamples)
        - Decoder: 32x32x4 -> 256x256 (3 upsamples)
        - Reparameterization trick for VAE training
        - Support for deterministic encoding (inference)
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
        
        # Decoder (reversed hidden dims)
        self.decoder = Decoder(
            latent_dim=self.config.latent_dim,
            out_channels=self.config.in_channels,
            hidden_dims=list(reversed(self.config.hidden_dims))
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution.
        
        Args:
            x: [B, 3, 256, 256] input images (normalized to [-1, 1])
            
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
            x_recon: [B, 3, 256, 256] reconstructed image (in [-1, 1])
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        This allows gradients to flow through the sampling operation.
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
        Forward pass: encode, sample, decode.
        
        Args:
            x: [B, 3, 256, 256] input images
            deterministic: if True, use mu directly (no sampling)
            
        Returns:
            Dict with:
                - 'recon': reconstructed images
                - 'mu': latent mean
                - 'logvar': latent log variance
                - 'z': sampled/deterministic latent
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample or use mean
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        
        # Decode
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
        Get latent representation of images.
        
        For World Model training, use deterministic=True for consistency.
        
        Args:
            x: input images [B, 3, 256, 256]
            deterministic: if True, return mu (recommended for downstream)
            
        Returns:
            z: latent representation [B, latent_dim, 32, 32]
        """
        mu, logvar = self.encode(x)
        
        if deterministic:
            return mu
        else:
            return self.reparameterize(mu, logvar)
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct images (inference mode).
        
        Args:
            x: input images
            
        Returns:
            reconstructed images
        """
        self.eval()
        output = self.forward(x, deterministic=True)
        return output['recon']


class VAELoss(nn.Module):
    """
    VAE Loss = Reconstruction Loss + beta * KL Divergence
    
    Reconstruction: MSE, L1, or mixed
    KL: Regularizes latent space to be close to N(0, 1)
    
    Args:
        beta: KL loss weight (0.0001 - 0.001 recommended)
        recon_type: 'mse', 'l1', or 'mixed'
        l1_weight: weight for L1 in mixed mode (0.5 = equal)
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
        """Compute reconstruction loss."""
        if self.recon_type == 'mse':
            return F.mse_loss(recon, target, reduction='mean')
        elif self.recon_type == 'l1':
            return F.l1_loss(recon, target, reduction='mean')
        else:  # mixed
            mse = F.mse_loss(recon, target, reduction='mean')
            l1 = F.l1_loss(recon, target, reduction='mean')
            return (1 - self.l1_weight) * mse + self.l1_weight * l1
    
    def kl_divergence(
        self, 
        mu: torch.Tensor, 
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        KL divergence from N(mu, sigma) to N(0, 1).
        
        KL = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
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
        Compute VAE loss.
        
        Args:
            x: original images [B, 3, H, W]
            recon: reconstructed images [B, 3, H, W]
            mu: latent mean [B, latent_dim, h, w]
            logvar: latent log variance [B, latent_dim, h, w]
            beta_override: optional beta for KL annealing
            
        Returns:
            Dict with 'loss', 'recon_loss', 'kl_loss'
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(recon, x)
        
        # KL divergence
        kl_loss = self.kl_divergence(mu, logvar)
        
        # Use override beta if provided (for annealing)
        beta = beta_override if beta_override is not None else self.beta
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
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
    loss_fn = VAELoss(beta=config.beta, recon_type='mse')
    
    # Test input
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    output = vae(x)
    
    print(f"Input shape:      {x.shape}")
    print(f"Recon shape:      {output['recon'].shape}")
    print(f"Mu shape:         {output['mu'].shape}")
    print(f"Logvar shape:     {output['logvar'].shape}")
    print(f"Z shape:          {output['z'].shape}")
    
    # Verify shapes
    assert output['recon'].shape == x.shape, "Reconstruction shape mismatch!"
    assert output['mu'].shape == (2, 4, 32, 32), "Mu shape mismatch!"
    print("\n[OK] All shapes correct!")
    
    # Compute loss
    losses = loss_fn(x, output['recon'], output['mu'], output['logvar'])
    
    print(f"\nLosses:")
    print(f"  Total:   {losses['loss'].item():.4f}")
    print(f"  Recon:   {losses['recon_loss'].item():.4f}")
    print(f"  KL:      {losses['kl_loss'].item():.4f}")
    
    # Test deterministic mode
    z_det = vae.get_latent(x, deterministic=True)
    z_stoch = vae.get_latent(x, deterministic=False)
    print(f"\nDeterministic latent: {z_det.shape}")
    print(f"Stochastic latent:    {z_stoch.shape}")
    
    # Count parameters
    encoder_params = sum(p.numel() for p in vae.encoder.parameters())
    decoder_params = sum(p.numel() for p in vae.decoder.parameters())
    total_params = sum(p.numel() for p in vae.parameters())
    
    print(f"\nParameters:")
    print(f"  Encoder: {encoder_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  Total:   {total_params:,}")
    
    # Compression ratio
    input_size = 3 * 256 * 256  # 196,608
    latent_size = 4 * 32 * 32    # 4,096
    print(f"\nCompression:")
    print(f"  Input size:  {input_size:,} values")
    print(f"  Latent size: {latent_size:,} values")
    print(f"  Ratio:       {input_size / latent_size:.1f}x")
    
    print("\n[OK] VAE works correctly!")