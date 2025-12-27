"""
Mini-Cosmos: VAE Encoder
========================
Encodes images into compact latent representations.

Architecture: Convolutional encoder with residual blocks
Input: [B, 3, 256, 256] RGB images
Output: [B, latent_dim, 32, 32] latent representation (mu and logvar)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Skip connection if channels change
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        return F.silu(h + self.skip(x))


class DownsampleBlock(nn.Module):
    """Downsample with strided convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    """
    VAE Encoder
    
    Compresses 256x256 RGB images to 32x32 latent space.
    Downsampling: 256 -> 128 -> 64 -> 32 (3 downsample blocks)
    
    Args:
        in_channels: Input image channels (default: 3 for RGB)
        latent_dim: Latent space channels (default: 4)
        hidden_dims: Hidden layer dimensions
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        hidden_dims: List[int] = [64, 128, 256, 256],
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        
        in_ch = hidden_dims[0]
        for i, out_ch in enumerate(hidden_dims):
            # Residual block
            self.encoder_blocks.append(ResidualBlock(in_ch, out_ch))
            
            # Downsample (except last)
            if i < len(hidden_dims) - 1:
                self.encoder_blocks.append(DownsampleBlock(out_ch, out_ch))
            
            in_ch = out_ch
        
        # Final layers
        self.norm_out = nn.GroupNorm(8, hidden_dims[-1])
        
        # Output: mu and logvar for VAE
        self.conv_mu = nn.Conv2d(hidden_dims[-1], latent_dim, 1)
        self.conv_logvar = nn.Conv2d(hidden_dims[-1], latent_dim, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters.
        
        Args:
            x: [B, 3, 256, 256] input images
            
        Returns:
            mu: [B, latent_dim, 32, 32] mean
            logvar: [B, latent_dim, 32, 32] log variance
        """
        # Initial conv
        h = self.conv_in(x)
        
        # Encoder blocks
        for block in self.encoder_blocks:
            h = block(h)
        
        # Final norm + activation
        h = self.norm_out(h)
        h = F.silu(h)
        
        # Get mu and logvar
        mu = self.conv_mu(h)
        logvar = self.conv_logvar(h)
        
        return mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode and sample from latent distribution.
        
        Args:
            x: [B, 3, 256, 256] input images
            
        Returns:
            z: [B, latent_dim, 32, 32] sampled latent
        """
        mu, logvar = self.forward(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# Test
if __name__ == '__main__':
    print("Testing Encoder...")
    print("=" * 50)
    
    # Create encoder
    encoder = Encoder(
        in_channels=3,
        latent_dim=4,
        hidden_dims=[64, 128, 256, 256]
    )
    
    # Test input
    x = torch.randn(2, 3, 256, 256)
    
    # Forward pass
    mu, logvar = encoder(x)
    z = encoder.encode(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Mu shape:     {mu.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Z shape:      {z.shape}")
    
    # Count parameters
    params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    print("\n[OK] Encoder works correctly!")