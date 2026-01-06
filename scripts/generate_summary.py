"""
Summary image generator - shows predictions at different horizons.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data.dataset import CARLADataset, DatasetConfig
from src.models.vae import VAE, VAEConfig
from src.models.world_model import WorldModel, WorldModelConfig


def denormalize(tensor):
    return (tensor + 1) / 2


def tensor_to_numpy(tensor):
    img = denormalize(tensor).cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


class SummaryGenerator:
    
    def __init__(self, vae, world_model, device='cuda'):
        self.vae = vae.to(device)
        self.world_model = world_model.to(device)
        self.vae.eval()
        self.world_model.eval()
        self.device = device
        self.context_length = world_model.config.context_length
    
    @torch.no_grad()
    def encode_frames(self, frames):
        B, T = frames.shape[:2]
        frames_flat = frames.view(B * T, *frames.shape[2:])
        latents = self.vae.get_latent(frames_flat, deterministic=True)
        latents = latents.view(B, T, *latents.shape[1:])
        return latents
    
    @torch.no_grad()
    def decode_latent(self, latent):
        return self.vae.decode(latent)
    
    @torch.no_grad()
    def generate_at_horizons(self, context_frames, context_actions=None,
                              future_actions=None, horizons=[1, 10, 20, 40, 60]):
        context_latents = self.encode_frames(context_frames)
        
        # Get last context frame for display
        last_context = self.decode_latent(context_latents[:, -1])
        last_context_np = tensor_to_numpy(last_context[0])
        
        max_horizon = max(horizons)
        current_latents = context_latents.clone()
        
        predictions = {}
        
        for i in range(max_horizon):
            input_latents = current_latents[:, -self.context_length:]
            
            if context_actions is not None and future_actions is not None:
                all_actions = torch.cat([context_actions, future_actions], dim=1)
                start_idx = current_latents.shape[1] - self.context_length
                input_actions = all_actions[:, start_idx:start_idx + self.context_length]
            else:
                input_actions = None
            
            output = self.world_model(input_latents, input_actions)
            pred_latent = output['pred_latent']
            
            horizon = i + 1
            if horizon in horizons:
                pred_frame = self.decode_latent(pred_latent)
                predictions[horizon] = tensor_to_numpy(pred_frame[0])
            
            current_latents = torch.cat([
                current_latents,
                pred_latent.unsqueeze(1)
            ], dim=1)
        
        return last_context_np, predictions


def create_summary_grid(generator, dataset, args, device):
    horizons = [1, 10, 20, 40, 60]
    # Filter horizons based on available GT
    max_gt = args.num_future
    horizons = [h for h in horizons if h <= max_gt]
    
    num_samples = args.num_samples
    num_cols = 2 + len(horizons)  # Context + predictions + GT
    
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(3 * num_cols, 3 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    for row, idx in enumerate(tqdm(indices, desc="Generating")):
        sample = dataset[idx]
        frames = sample['frames'].unsqueeze(0).to(device)
        actions = sample.get('actions')
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)
        
        context_frames = frames[:, :args.context_length]
        gt_frames = frames[:, args.context_length:]
        
        if actions is not None:
            context_actions = actions[:, :args.context_length]
            future_actions = actions[:, args.context_length:]
        else:
            context_actions = None
            future_actions = None
        
        last_context, predictions = generator.generate_at_horizons(
            context_frames, context_actions, future_actions, horizons
        )
        
        # Column 0: Last context frame
        axes[row, 0].imshow(last_context)
        axes[row, 0].set_title('Context', fontsize=10)
        axes[row, 0].axis('off')
        
        # Columns 1 to N-1: Predictions at different horizons
        for col, h in enumerate(horizons):
            axes[row, col + 1].imshow(predictions[h])
            seconds = h / 20.0
            axes[row, col + 1].set_title(f't+{h} (~{seconds:.1f}s)', fontsize=10)
            axes[row, col + 1].axis('off')
        
        # Last column: Ground truth at max horizon
        gt_idx = horizons[-1] - 1
        if gt_idx < gt_frames.shape[1]:
            gt_np = tensor_to_numpy(gt_frames[0, gt_idx])
            axes[row, -1].imshow(gt_np)
            axes[row, -1].set_title(f'GT t+{horizons[-1]}', fontsize=10)
        axes[row, -1].axis('off')
    
    plt.suptitle('World Model Predictions at Different Horizons', fontsize=14, y=1.02)
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Generate prediction summary image')
    
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--context_length', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    
    parser.add_argument('--num_future', type=int, default=60)
    parser.add_argument('--num_samples', type=int, default=4)
    
    parser.add_argument('--vae_checkpoint', type=str, default='./outputs/checkpoints/vae_best.pt')
    parser.add_argument('--world_model_checkpoint', type=str, default='./outputs/checkpoints/world_model_best.pt')
    parser.add_argument('--data_dir', type=str, default='./data/raw')
    parser.add_argument('--output_dir', type=str, default='./outputs/summary')
    
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    vae_path = Path(args.vae_checkpoint)
    wm_path = Path(args.world_model_checkpoint)
    
    if not vae_path.exists():
        print(f"[ERROR] VAE checkpoint not found: {vae_path}")
        return
    
    if not wm_path.exists():
        print(f"[ERROR] World Model checkpoint not found: {wm_path}")
        return
    
    print("Loading VAE...")
    vae_checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
    vae_config = VAEConfig(
        in_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=(64, 128, 256, 256)
    )
    vae = VAE(vae_config)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {vae_checkpoint['epoch']}")
    
    print("Loading World Model...")
    wm_checkpoint = torch.load(wm_path, map_location=device, weights_only=False)
    wm_config = WorldModelConfig(
        latent_dim=args.latent_dim,
        latent_size=32,
        context_length=args.context_length,
        action_dim=3,
        use_actions=True,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1,
        latent_patch_size=4
    )
    world_model = WorldModel(wm_config)
    world_model.load_state_dict(wm_checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {wm_checkpoint['epoch']}")
    
    generator = SummaryGenerator(vae, world_model, device)
    
    print("Loading dataset...")
    total_frames_needed = args.context_length + args.num_future
    dataset_config = DatasetConfig(
        sequence_length=total_frames_needed,
        frame_skip=1,
        image_size=(256, 256),
        augment=False,
        return_actions=True
    )
    dataset = CARLADataset(args.data_dir, dataset_config, split='test')
    
    if len(dataset) == 0:
        print("No test data, using train split")
        dataset = CARLADataset(args.data_dir, dataset_config, split='train')
    
    print(f"  Dataset size: {len(dataset)}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating summary with {args.num_samples} samples...")
    
    fig = create_summary_grid(generator, dataset, args, device)
    
    output_path = output_dir / 'prediction_summary.png'
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n[OK] Saved: {output_path}")


if __name__ == '__main__':
    main()