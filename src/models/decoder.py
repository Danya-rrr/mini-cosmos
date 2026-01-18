"""
VAE Decoder Module

Reconstructs RGB images from compressed latent representations using a 
convolutional architecture with residual connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


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


class UpsampleBlock(nn.Module):
    """
    Upsampling block using nearest neighbor interpolation followed by convolution.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Decoder(nn.Module):
    """
    VAE Decoder for image reconstruction.
    
    Progressively upsamples latent representations through residual blocks:
    32×32 → 64×64 → 128×128 → 256×256
    
    Args:
        latent_dim: Dimension of latent space (default: 4)
        out_channels: Number of output channels (default: 3 for RGB)
        hidden_dims: Channel dimensions for each decoder stage
    """
    
    def __init__(
        self,
        latent_dim: int = 4,
        out_channels: int = 3,
        hidden_dims: List[int] = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256, 128, 64]
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        self.conv_in = nn.Conv2d(latent_dim, hidden_dims[0], kernel_size=3, padding=1)
        
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_channels = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            out_ch = hidden_dims[i]
            
            self.res_blocks.append(ResidualBlock(in_channels, out_ch))
            
            if i < len(hidden_dims) - 1:
                self.upsample_blocks.append(UpsampleBlock(out_ch, out_ch))
        
        self.norm_out = nn.GroupNorm(num_groups=8, num_channels=hidden_dims[-1])
        self.conv_out = nn.Conv2d(hidden_dims[-1], out_channels, kernel_size=3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to image.
        
        Args:
            z: Latent tensor of shape [B, latent_dim, H, W]
            
        Returns:
            Reconstructed image of shape [B, out_channels, H*8, W*8]
        """
        x = self.conv_in(z)
        
        for i, res_block in enumerate(self.res_blocks):
            x = res_block(x)
            if i < len(self.upsample_blocks):
                x = self.upsample_blocks[i](x)
        
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        
        return torch.tanh(x)


def main():
    """Test decoder functionality."""
    decoder = Decoder(latent_dim=4, out_channels=3, hidden_dims=[256, 256, 128, 64])
    
    batch_size = 2
    z = torch.randn(batch_size, 4, 32, 32)
    x_recon = decoder(z)
    
    print(f"Input shape:  {z.shape}")
    print(f"Output shape: {x_recon.shape}")
    print(f"Output range: [{x_recon.min():.3f}, {x_recon.max():.3f}]")
    print(f"Parameters:   {sum(p.numel() for p in decoder.parameters()):,}")
    
    expected_shape = (batch_size, 3, 256, 256)
    assert x_recon.shape == expected_shape, f"Shape mismatch: expected {expected_shape}"
    print("\nDecoder test passed successfully!")


if __name__ == '__main__':
    main()