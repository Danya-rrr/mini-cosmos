"""
Mini-Cosmos: VAE Decoder
========================
Decodes latent representations back to images.

Architecture: Convolutional decoder with residual blocks
Input: [B, latent_dim, 32, 32] latent representation
Output: [B, 3, 256, 256] RGB images

Fixed: Correct upsampling 32 → 64 → 128 → 256 (3 upsample blocks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        h = self.conv2(h)
        h = self.norm2(h)
        
        return F.silu(h + self.skip(x))


class UpsampleBlock(nn.Module):
    """Upsample with nearest neighbor + convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class Decoder(nn.Module):
    """
    VAE Decoder
    
    Reconstructs 256x256 RGB images from 32x32 latent space.
    
    Architecture (fixed):
        32x32   -> ResBlock(256) -> Upsample
        64x64   -> ResBlock(256) -> Upsample  
        128x128 -> ResBlock(128) -> Upsample
        256x256 -> ResBlock(64)  -> Output Conv
    
    Args:
        latent_dim: Latent space channels (default: 4)
        out_channels: Output image channels (default: 3 for RGB)
        hidden_dims: Hidden layer dimensions [256, 256, 128, 64]
    """
    
    def __init__(
        self,
        latent_dim: int = 4,
        out_channels: int = 3,
        hidden_dims: List[int] = [256, 256, 128, 64],
    ):
        super().__init__()
        
        self.hidden_dims = hidden_dims
        
        # Initial convolution from latent
        self.conv_in = nn.Conv2d(latent_dim, hidden_dims[0], 3, padding=1)
        
        # Build decoder blocks
        # We need 3 upsamples: 32->64->128->256
        self.res_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()
        
        in_ch = hidden_dims[0]
        for i, out_ch in enumerate(hidden_dims):
            # Residual block
            self.res_blocks.append(ResidualBlock(in_ch, out_ch))
            
            # Upsample after first 3 blocks (not the last one)
            if i < len(hidden_dims) - 1:
                self.upsample_blocks.append(UpsampleBlock(out_ch, out_ch))
            
            in_ch = out_ch
        
        # Final layers
        self.norm_out = nn.GroupNorm(8, hidden_dims[-1])
        self.conv_out = nn.Conv2d(hidden_dims[-1], out_channels, 3, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.
        
        Args:
            z: [B, latent_dim, 32, 32] latent representation
            
        Returns:
            x_recon: [B, 3, 256, 256] reconstructed images
        """
        # Initial conv: [B, 4, 32, 32] -> [B, 256, 32, 32]
        h = self.conv_in(z)
        
        # Decoder blocks with upsampling
        for i, res_block in enumerate(self.res_blocks):
            h = res_block(h)
            if i < len(self.upsample_blocks):
                h = self.upsample_blocks[i](h)
        
        # Final norm + activation + conv
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        # Tanh to get [-1, 1] range (matching normalized images)
        return torch.tanh(h)


# Test
if __name__ == '__main__':
    print("Testing Decoder...")
    print("=" * 50)
    
    # Create decoder
    decoder = Decoder(
        latent_dim=4,
        out_channels=3,
        hidden_dims=[256, 256, 128, 64]
    )
    
    # Test input (latent)
    z = torch.randn(2, 4, 32, 32)
    
    # Forward pass
    x_recon = decoder(z)
    
    print(f"Latent shape:        {z.shape}")
    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Output range:        [{x_recon.min():.2f}, {x_recon.max():.2f}]")
    
    # Verify output size
    expected_shape = (2, 3, 256, 256)
    assert x_recon.shape == expected_shape, f"Expected {expected_shape}, got {x_recon.shape}"
    print(f"\n[OK] Output shape is correct: {expected_shape}")
    
    # Count parameters
    params = sum(p.numel() for p in decoder.parameters())
    print(f"Total parameters: {params:,}")
    
    # Show intermediate sizes
    print("\nArchitecture flow:")
    print("  Latent:  [B, 4, 32, 32]")
    print("  Conv_in: [B, 256, 32, 32]")
    print("  Block 0: [B, 256, 32, 32] -> Upsample -> [B, 256, 64, 64]")
    print("  Block 1: [B, 256, 64, 64] -> Upsample -> [B, 256, 128, 128]")
    print("  Block 2: [B, 128, 128, 128] -> Upsample -> [B, 128, 256, 256]")
    print("  Block 3: [B, 64, 256, 256]")
    print("  Output:  [B, 3, 256, 256]")
    
    print("\n[OK] Decoder works correctly!")