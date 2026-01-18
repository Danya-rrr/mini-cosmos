"""
World Model (Temporal Transformer)

Predicts future latent representations based on past frames and actions
using a transformer architecture for temporal dynamics modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """Configuration for World Model architecture and training."""
    latent_dim: int = 4
    latent_size: int = 32
    context_length: int = 8
    action_dim: int = 3
    use_actions: bool = True
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    latent_patch_size: int = 4


class LatentPatchEmbed(nn.Module):
    """
    Convert latent frames to patch embeddings.
    
    Args:
        latent_dim: Number of latent channels
        latent_size: Spatial size of latent feature maps
        patch_size: Size of each patch
        hidden_dim: Embedding dimension
    """
    
    def __init__(
        self,
        latent_dim: int = 4,
        latent_size: int = 32,
        patch_size: int = 4,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (latent_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            latent_dim, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Latent frame of shape [B, C, H, W]
        
        Returns:
            Patch embeddings of shape [B, num_patches, hidden_dim]
        """
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class LatentPatchDecode(nn.Module):
    """
    Convert patch embeddings back to latent frames.
    
    Args:
        latent_dim: Number of latent channels
        latent_size: Spatial size of latent feature maps
        patch_size: Size of each patch
        hidden_dim: Embedding dimension
    """
    
    def __init__(
        self,
        latent_dim: int = 4,
        latent_size: int = 32,
        patch_size: int = 4,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.num_patches_side = latent_size // patch_size
        
        self.proj = nn.Linear(hidden_dim, latent_dim * patch_size * patch_size)
        self.norm = nn.LayerNorm(latent_dim * patch_size * patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings of shape [B, num_patches, hidden_dim]
        
        Returns:
            Latent frame of shape [B, C, H, W]
        """
        B = x.shape[0]
        
        x = self.proj(x)
        x = self.norm(x)
        
        x = x.view(
            B,
            self.num_patches_side,
            self.num_patches_side,
            self.latent_dim,
            self.patch_size,
            self.patch_size
        )
        
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.latent_dim, self.latent_size, self.latent_size)
        
        return x


class ActionEncoder(nn.Module):
    """
    Encode action vectors to embeddings.
    
    Args:
        action_dim: Dimension of action vector
        hidden_dim: Embedding dimension
    """
    
    def __init__(self, action_dim: int = 3, hidden_dim: int = 512):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.net(actions)


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for temporal and spatial positions.
    
    Args:
        hidden_dim: Embedding dimension
        max_temporal: Maximum number of temporal positions
        max_spatial: Maximum number of spatial positions (patches)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_temporal: int = 16,
        max_spatial: int = 64
    ):
        super().__init__()
        
        self.temporal_embed = nn.Embedding(max_temporal, hidden_dim)
        self.spatial_embed = nn.Embedding(max_spatial, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        temporal_idx: int,
        num_patches: int
    ) -> torch.Tensor:
        """
        Add positional encodings to input.
        
        Args:
            x: Input tensor of shape [B, num_patches, hidden_dim]
            temporal_idx: Temporal position index
            num_patches: Number of spatial patches
        
        Returns:
            Tensor with added positional encodings
        """
        B = x.shape[0]
        device = x.device
        
        t_emb = self.temporal_embed(
            torch.tensor([temporal_idx], device=device)
        ).unsqueeze(0).expand(B, num_patches, -1)
        
        s_idx = torch.arange(num_patches, device=device)
        s_emb = self.spatial_embed(s_idx).unsqueeze(0).expand(B, -1, -1)
        
        return x + t_emb + s_emb


class TransformerBlock(nn.Module):
    """
    Transformer block with pre-normalization.
    
    Args:
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        mlp_ratio: Ratio of MLP hidden dim to embedding dim
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=mask, need_weights=False)
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        
        return x


class WorldModel(nn.Module):
    """
    World Model for predicting future latent representations.
    
    Uses transformer architecture to model temporal dynamics of latent
    representations conditioned on past frames and actions.
    
    Args:
        config: WorldModelConfig dataclass
    """
    
    def __init__(self, config: WorldModelConfig):
        super().__init__()
        
        self.config = config
        self.num_patches = (config.latent_size // config.latent_patch_size) ** 2
        
        self.patch_embed = LatentPatchEmbed(
            latent_dim=config.latent_dim,
            latent_size=config.latent_size,
            patch_size=config.latent_patch_size,
            hidden_dim=config.hidden_dim
        )
        
        self.patch_decode = LatentPatchDecode(
            latent_dim=config.latent_dim,
            latent_size=config.latent_size,
            patch_size=config.latent_patch_size,
            hidden_dim=config.hidden_dim
        )
        
        self.action_encoder = (
            ActionEncoder(config.action_dim, config.hidden_dim)
            if config.use_actions else None
        )
        
        self.pos_encoding = PositionalEncoding(
            hidden_dim=config.hidden_dim,
            max_temporal=config.context_length + 1,  # было + 8
            max_spatial=self.num_patches
        )
        
        self.pred_token = nn.Parameter(torch.randn(1, self.num_patches, config.hidden_dim))
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(config.hidden_dim)
    
    def encode_frames(
        self,
        latent_frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequence of latent frames and actions to transformer input.
        
        Args:
            latent_frames: Latent frames of shape [B, T, C, H, W]
            actions: Optional actions of shape [B, T, action_dim]
        
        Returns:
            Encoded sequence of shape [B, T * num_patches, hidden_dim]
        """
        B, T = latent_frames.shape[:2]
        
        all_patches = []
        
        for t in range(T):
            frame = latent_frames[:, t]
            patches = self.patch_embed(frame)
            patches = self.pos_encoding(patches, temporal_idx=t, num_patches=self.num_patches)
            
            if self.action_encoder is not None and actions is not None:
                action_emb = self.action_encoder(actions[:, t]).unsqueeze(1)
                patches = patches + action_emb
            
            all_patches.append(patches)
        
        sequence = torch.cat(all_patches, dim=1)
        
        return sequence
    
    def forward(
        self,
        latent_frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next latent frame from context.
        
        Args:
            latent_frames: Context frames of shape [B, T, C, H, W]
            actions: Optional actions of shape [B, T, action_dim]
        
        Returns:
            Dictionary containing predicted latent and features
        """
        B = latent_frames.shape[0]
        
        context = self.encode_frames(latent_frames, actions)
        
        pred_tokens = self.pred_token.expand(B, -1, -1)
        pred_tokens = self.pos_encoding(
            pred_tokens,
            temporal_idx=self.config.context_length,
            num_patches=self.num_patches
        )
        
        sequence = torch.cat([context, pred_tokens], dim=1)
        
        for block in self.transformer_blocks:
            sequence = block(sequence)
        
        pred_features = sequence[:, -self.num_patches:]
        pred_features = self.output_norm(pred_features)
        
        pred_latent = self.patch_decode(pred_features)
        
        return {
            'pred_latent': pred_latent,
            'features': pred_features
        }
    
    @torch.no_grad()
    def generate(
        self,
        initial_frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        num_frames: int = 8
    ) -> torch.Tensor:
        """
        Autoregressively generate future frames.
        
        Args:
            initial_frames: Initial context of shape [B, T, C, H, W]
            actions: Optional actions of shape [B, T + num_frames, action_dim]
            num_frames: Number of frames to generate
        
        Returns:
            Generated frames of shape [B, num_frames, C, H, W]
        """
        self.eval()
        
        B = initial_frames.shape[0]
        context_len = self.config.context_length
        
        context = initial_frames.clone()
        generated = []
        
        for i in range(num_frames):
            current_context = context[:, -context_len:]
            
            if actions is not None:
                start_idx = context.shape[1] - context_len
                end_idx = start_idx + context_len
                current_actions = actions[:, start_idx:end_idx]
            else:
                current_actions = None
            
            output = self.forward(current_context, current_actions)
            pred_frame = output['pred_latent']
            
            generated.append(pred_frame)
            context = torch.cat([context, pred_frame.unsqueeze(1)], dim=1)
        
        generated = torch.stack(generated, dim=1)
        
        return generated


class WorldModelLoss(nn.Module):
    """
    Loss function for World Model training.
    
    Args:
        feature_weight: Weight for optional feature loss
    """
    
    def __init__(self, feature_weight: float = 0.0):
        super().__init__()
        self.feature_weight = feature_weight
    
    def forward(
        self,
        pred_latent: torch.Tensor,
        target_latent: torch.Tensor,
        pred_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute prediction loss.
        
        Args:
            pred_latent: Predicted latent of shape [B, C, H, W]
            target_latent: Target latent of shape [B, C, H, W]
            pred_features: Optional predicted features
            target_features: Optional target features
        
        Returns:
            Dictionary with loss components
        """
        mse_loss = F.mse_loss(pred_latent, target_latent)
        
        return {
            'loss': mse_loss,
            'mse_loss': mse_loss
        }


def main():
    """Test World Model functionality."""
    config = WorldModelConfig(
        latent_dim=4,
        latent_size=32,
        context_length=8,
        action_dim=3,
        use_actions=True,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        latent_patch_size=4
    )
    
    model = WorldModel(config)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    B, T = 2, config.context_length
    latent_frames = torch.randn(B, T, 4, 32, 32)
    actions = torch.randn(B, T, 3)
    
    print(f"\nInput shapes:")
    print(f"  Latent frames: {latent_frames.shape}")
    print(f"  Actions:       {actions.shape}")
    
    output = model(latent_frames, actions)
    
    print(f"\nOutput shapes:")
    print(f"  Predicted:     {output['pred_latent'].shape}")
    print(f"  Features:      {output['features'].shape}")
    
    target_latent = torch.randn(B, 4, 32, 32)
    loss_fn = WorldModelLoss()
    losses = loss_fn(output['pred_latent'], target_latent)
    
    print(f"\nLoss: {losses['loss'].item():.4f}")
    
    all_actions = torch.randn(B, T + 4, 3)
    generated = model.generate(latent_frames, all_actions, num_frames=4)
    
    print(f"\nGenerated {generated.shape[1]} frames: {generated.shape}")
    print("\nWorld Model test passed successfully!")


if __name__ == '__main__':
    main()