"""
VAE Encoder Module

Compresses RGB images into compact latent representations using a 
convolutional architecture with residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


class ResidualBlock(nn.Module):
    """
    Residual block with group normalization and SiLU activation.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        
        self.skip = (nn.Conv2d(in_channels, out_channels, kernel_size=1) 
                     if in_channels != out_channels else nn.Identity())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.silu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        return F.silu(out + residual)


class DownsampleBlock(nn.Module):
    """
    Downsampling block using strided convolution.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Encoder(nn.Module):
    """
    VAE Encoder for image compression.
    
    Progressively downsamples images through residual blocks:
    256×256 → 128×128 → 64×64 → 32×32
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
        latent_dim: Dimension of latent space (default: 4)
        hidden_dims: Channel dimensions for each encoder stage
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        latent_dim: int = 4,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 256]
        
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        self.conv_in = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        
        self.encoder_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            out_ch = hidden_dims[i]
            
            self.encoder_blocks.append(ResidualBlock(in_ch, out_ch))
            
            if i < len(hidden_dims) - 1:
                self.encoder_blocks.append(DownsampleBlock(out_ch, out_ch))
        
        self.norm_out = nn.GroupNorm(num_groups=8, num_channels=hidden_dims[-1])
        
        self.conv_mu = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv2d(hidden_dims[-1], latent_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters.
        
        Args:
            x: Input images of shape [B, in_channels, H, W]
            
        Returns:
            Tuple of (mu, logvar), each of shape [B, latent_dim, H/8, W/8]
        """
        x = self.conv_in(x)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        
        mu = self.conv_mu(x)
        logvar = self.conv_logvar(x)
        
        return mu, logvar
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode image and sample from latent distribution.
        
        Args:
            x: Input images of shape [B, in_channels, H, W]
            
        Returns:
            Sampled latent representation of shape [B, latent_dim, H/8, W/8]
        """
        mu, logvar = self.forward(x)
        return self.reparameterize(mu, logvar)
    
    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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


def main():
    """Test encoder functionality."""
    encoder = Encoder(in_channels=3, latent_dim=4, hidden_dims=[64, 128, 256, 256])
    
    batch_size = 2
    x = torch.randn(batch_size, 3, 256, 256)
    
    mu, logvar = encoder(x)
    z = encoder.encode(x)
    
    print(f"Input shape:   {x.shape}")
    print(f"Mu shape:      {mu.shape}")
    print(f"Logvar shape:  {logvar.shape}")
    print(f"Latent shape:  {z.shape}")
    print(f"Parameters:    {sum(p.numel() for p in encoder.parameters()):,}")
    
    expected_shape = (batch_size, 4, 32, 32)
    assert mu.shape == expected_shape, f"Shape mismatch: expected {expected_shape}"
    print("\nEncoder test passed successfully!")


if __name__ == '__main__':
    main()