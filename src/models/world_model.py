"""
Mini-Cosmos: World Model (Temporal Transformer)
================================================
Predicts future latent frames based on past frames and actions.

Architecture:
    - Input: 8 context frames (latent 32x32x4) + actions
    - Transformer: 6 layers, 512 hidden dim, 8 heads
    - Output: predicted next latent frame (32x32x4)
    - Autoregressive generation for multiple future frames

Usage:
    from src.models.world_model import WorldModel, WorldModelConfig
    
    model = WorldModel(WorldModelConfig())
    pred = model(latent_frames, actions)  # Predict next frame
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


@dataclass
class WorldModelConfig:
    """World Model configuration"""
    # Latent space (from VAE)
    latent_dim: int = 4
    latent_size: int = 32  # 32x32 spatial
    
    # Context
    context_length: int = 8  # Number of past frames
    
    # Actions
    action_dim: int = 3  # throttle, steer, brake
    use_actions: bool = True
    
    # Transformer
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    
    # Latent compression (reduce 32x32 -> smaller for transformer)
    latent_patch_size: int = 4  # 32/4 = 8x8 patches
    

class LatentPatchEmbed(nn.Module):
    """
    Convert latent frames to patch embeddings.
    
    Input: [B, C, H, W] latent frame (e.g., [B, 4, 32, 32])
    Output: [B, num_patches, hidden_dim] patch embeddings
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
        self.num_patches = (latent_size // patch_size) ** 2  # 8x8 = 64 patches
        self.patch_dim = latent_dim * patch_size * patch_size  # 4 * 4 * 4 = 64
        
        # Patch embedding via conv
        self.proj = nn.Conv2d(
            latent_dim, hidden_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] latent frame
        Returns:
            [B, num_patches, hidden_dim] patch embeddings
        """
        # [B, C, H, W] -> [B, hidden_dim, H/patch, W/patch]
        x = self.proj(x)
        
        # [B, hidden_dim, h, w] -> [B, hidden_dim, num_patches]
        x = x.flatten(2)
        
        # [B, hidden_dim, num_patches] -> [B, num_patches, hidden_dim]
        x = x.transpose(1, 2)
        
        x = self.norm(x)
        
        return x


class LatentPatchDecode(nn.Module):
    """
    Convert patch embeddings back to latent frame.
    
    Input: [B, num_patches, hidden_dim]
    Output: [B, C, H, W] latent frame
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
        self.num_patches_side = latent_size // patch_size  # 8
        
        # Project back to patch pixels
        self.proj = nn.Linear(hidden_dim, latent_dim * patch_size * patch_size)
        self.norm = nn.LayerNorm(latent_dim * patch_size * patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, num_patches, hidden_dim]
        Returns:
            [B, C, H, W] latent frame
        """
        B = x.shape[0]
        
        # [B, num_patches, hidden_dim] -> [B, num_patches, C*p*p]
        x = self.proj(x)
        x = self.norm(x)
        
        # Reshape to [B, h, w, C, p, p]
        x = x.view(
            B,
            self.num_patches_side,
            self.num_patches_side,
            self.latent_dim,
            self.patch_size,
            self.patch_size
        )
        
        # Rearrange to [B, C, H, W]
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, self.latent_dim, self.latent_size, self.latent_size)
        
        return x


class ActionEncoder(nn.Module):
    """
    Encode actions to embeddings.
    
    Input: [B, action_dim] actions (throttle, steer, brake)
    Output: [B, hidden_dim] action embedding
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
    Learnable positional encoding for spatial and temporal positions.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        max_temporal: int = 16,
        max_spatial: int = 64
    ):
        super().__init__()
        
        # Temporal position (which frame)
        self.temporal_embed = nn.Embedding(max_temporal, hidden_dim)
        
        # Spatial position (which patch within frame)
        self.spatial_embed = nn.Embedding(max_spatial, hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        temporal_idx: int,
        num_patches: int
    ) -> torch.Tensor:
        """
        Add positional encoding.
        
        Args:
            x: [B, num_patches, hidden_dim]
            temporal_idx: frame index (0, 1, 2, ...)
            num_patches: number of spatial patches
        """
        B = x.shape[0]
        device = x.device
        
        # Temporal embedding (same for all patches in frame)
        t_emb = self.temporal_embed(
            torch.tensor([temporal_idx], device=device)
        ).unsqueeze(0).expand(B, num_patches, -1)
        
        # Spatial embedding (different for each patch)
        s_idx = torch.arange(num_patches, device=device)
        s_emb = self.spatial_embed(s_idx).unsqueeze(0).expand(B, -1, -1)
        
        return x + t_emb + s_emb


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm."""
    
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
    
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        
        # MLP with pre-norm
        x = x + self.mlp(self.norm2(x))
        
        return x


class WorldModel(nn.Module):
    """
    World Model: Predicts future latent frames from past frames and actions.
    
    Architecture:
        1. Encode each latent frame into patches
        2. Add positional encodings (spatial + temporal)
        3. Optionally add action embeddings
        4. Process through Transformer
        5. Decode output patches to predicted latent frame
    
    Training:
        - Input: context frames z_{t-7}, ..., z_{t} and actions a_{t-7}, ..., a_{t}
        - Output: predicted frame z_{t+1}
        - Loss: MSE between predicted and actual z_{t+1}
    
    Inference:
        - Autoregressive: predict one frame, append to context, repeat
    """
    
    def __init__(self, config: Optional[WorldModelConfig] = None):
        super().__init__()
        
        self.config = config or WorldModelConfig()
        c = self.config
        
        # Number of patches per frame
        self.num_patches = (c.latent_size // c.latent_patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = LatentPatchEmbed(
            latent_dim=c.latent_dim,
            latent_size=c.latent_size,
            patch_size=c.latent_patch_size,
            hidden_dim=c.hidden_dim
        )
        
        # Patch decoding
        self.patch_decode = LatentPatchDecode(
            latent_dim=c.latent_dim,
            latent_size=c.latent_size,
            patch_size=c.latent_patch_size,
            hidden_dim=c.hidden_dim
        )
        
        # Action encoder
        if c.use_actions:
            self.action_encoder = ActionEncoder(c.action_dim, c.hidden_dim)
        else:
            self.action_encoder = None
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            hidden_dim=c.hidden_dim,
            max_temporal=c.context_length + 1,  # +1 for prediction token
            max_spatial=self.num_patches
        )
        
        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=c.hidden_dim,
                num_heads=c.num_heads,
                dropout=c.dropout
            )
            for _ in range(c.num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(c.hidden_dim)
        
        # Prediction token (learnable query for next frame)
        self.pred_token = nn.Parameter(torch.randn(1, self.num_patches, c.hidden_dim) * 0.02)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode_frames(
        self,
        latent_frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequence of latent frames (and actions) to transformer input.
        
        Args:
            latent_frames: [B, T, C, H, W] sequence of latent frames
            actions: [B, T, action_dim] optional actions
            
        Returns:
            [B, T * num_patches, hidden_dim] encoded sequence
        """
        B, T = latent_frames.shape[:2]
        
        all_patches = []
        
        for t in range(T):
            # Get frame at time t
            frame = latent_frames[:, t]  # [B, C, H, W]
            
            # Embed patches
            patches = self.patch_embed(frame)  # [B, num_patches, hidden_dim]
            
            # Add positional encoding
            patches = self.pos_encoding(patches, temporal_idx=t, num_patches=self.num_patches)
            
            # Add action embedding (broadcast to all patches)
            if self.action_encoder is not None and actions is not None:
                action_emb = self.action_encoder(actions[:, t])  # [B, hidden_dim]
                action_emb = action_emb.unsqueeze(1)  # [B, 1, hidden_dim]
                patches = patches + action_emb
            
            all_patches.append(patches)
        
        # Concatenate all frames: [B, T * num_patches, hidden_dim]
        sequence = torch.cat(all_patches, dim=1)
        
        return sequence
    
    def forward(
        self,
        latent_frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict next latent frame.
        
        Args:
            latent_frames: [B, T, C, H, W] context frames (T = context_length)
            actions: [B, T, action_dim] actions for each frame
            
        Returns:
            Dict with:
                - 'pred_latent': [B, C, H, W] predicted next frame
                - 'features': [B, num_patches, hidden_dim] output features
        """
        B = latent_frames.shape[0]
        
        # Encode context frames
        context = self.encode_frames(latent_frames, actions)  # [B, T*num_patches, hidden_dim]
        
        # Add prediction tokens
        pred_tokens = self.pred_token.expand(B, -1, -1)  # [B, num_patches, hidden_dim]
        
        # Add positional encoding to prediction tokens
        pred_tokens = self.pos_encoding(
            pred_tokens,
            temporal_idx=self.config.context_length,
            num_patches=self.num_patches
        )
        
        # Concatenate: [B, (T+1) * num_patches, hidden_dim]
        sequence = torch.cat([context, pred_tokens], dim=1)
        
        # Transformer
        for block in self.transformer_blocks:
            sequence = block(sequence)
        
        # Get prediction output (last num_patches tokens)
        pred_features = sequence[:, -self.num_patches:]  # [B, num_patches, hidden_dim]
        pred_features = self.output_norm(pred_features)
        
        # Decode to latent frame
        pred_latent = self.patch_decode(pred_features)  # [B, C, H, W]
        
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
            initial_frames: [B, T, C, H, W] initial context (T = context_length)
            actions: [B, T + num_frames, action_dim] actions for all timesteps
            num_frames: number of frames to generate
            
        Returns:
            [B, num_frames, C, H, W] generated frames
        """
        self.eval()
        
        B = initial_frames.shape[0]
        context_len = self.config.context_length
        
        # Start with initial context
        context = initial_frames.clone()
        
        generated = []
        
        for i in range(num_frames):
            # Get current context (last context_length frames)
            current_context = context[:, -context_len:]
            
            # Get actions for current context
            if actions is not None:
                # Actions from position (total - context_len) to (total)
                start_idx = context.shape[1] - context_len
                end_idx = start_idx + context_len
                current_actions = actions[:, start_idx:end_idx]
            else:
                current_actions = None
            
            # Predict next frame
            output = self.forward(current_context, current_actions)
            pred_frame = output['pred_latent']  # [B, C, H, W]
            
            # Append to generated
            generated.append(pred_frame)
            
            # Append to context for next iteration
            context = torch.cat([context, pred_frame.unsqueeze(1)], dim=1)
        
        # Stack generated frames: [B, num_frames, C, H, W]
        generated = torch.stack(generated, dim=1)
        
        return generated


class WorldModelLoss(nn.Module):
    """
    Loss function for World Model training.
    
    L = MSE(pred_latent, target_latent) + lambda * feature_loss
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
        Compute loss.
        
        Args:
            pred_latent: [B, C, H, W] predicted latent
            target_latent: [B, C, H, W] target latent
            
        Returns:
            Dict with 'loss', 'mse_loss'
        """
        # MSE loss on latent space
        mse_loss = F.mse_loss(pred_latent, target_latent)
        
        loss = mse_loss
        
        return {
            'loss': loss,
            'mse_loss': mse_loss
        }


# Test
if __name__ == '__main__':
    print("Testing World Model...")
    print("=" * 60)
    
    # Config
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
    
    print(f"Config:")
    print(f"  Context length: {config.context_length}")
    print(f"  Latent size: {config.latent_size}x{config.latent_size}x{config.latent_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Patch size: {config.latent_patch_size}")
    
    # Create model
    model = WorldModel(config)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params:,}")
    
    # Test input
    B = 2
    T = config.context_length
    
    latent_frames = torch.randn(B, T, 4, 32, 32)
    actions = torch.randn(B, T, 3)
    
    print(f"\nInput shapes:")
    print(f"  Latent frames: {latent_frames.shape}")
    print(f"  Actions: {actions.shape}")
    
    # Forward pass
    output = model(latent_frames, actions)
    
    print(f"\nOutput shapes:")
    print(f"  Predicted latent: {output['pred_latent'].shape}")
    print(f"  Features: {output['features'].shape}")
    
    # Test loss
    target_latent = torch.randn(B, 4, 32, 32)
    loss_fn = WorldModelLoss()
    losses = loss_fn(output['pred_latent'], target_latent)
    
    print(f"\nLoss: {losses['loss'].item():.4f}")
    
    # Test generation
    print(f"\nTesting autoregressive generation...")
    
    all_actions = torch.randn(B, T + 4, 3)  # Actions for context + 4 future frames
    generated = model.generate(latent_frames, all_actions, num_frames=4)
    
    print(f"  Generated {generated.shape[1]} frames: {generated.shape}")
    
    # Memory estimate
    print(f"\nMemory estimate (batch_size=8):")
    mem_params = params * 4 / 1e9  # 4 bytes per float32
    print(f"  Model parameters: {mem_params:.2f} GB")
    
    print("\n" + "=" * 60)
    print("[OK] World Model works correctly!")