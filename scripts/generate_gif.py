"""
Mini-Cosmos: GIF Generation Script
==================================
Generate GIF animations of predicted future frames.

Usage:
    python scripts/generate_gif.py --num_future 16 --output_dir outputs/gifs
    python scripts/generate_gif.py --num_future 16 --latent_dim 8

Requirements:
    - Trained VAE checkpoint
    - Trained World Model checkpoint
    - Test data
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from PIL import Image
import imageio

# Local imports
from src.data.dataset import CARLADataset, DatasetConfig
from src.models.vae import VAE, VAEConfig
from src.models.world_model import WorldModel, WorldModelConfig


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image [H, W, 3] uint8"""
    img = denormalize(tensor).cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return img


class GIFGenerator:
    """Generate GIF predictions from World Model."""
    
    def __init__(
        self,
        vae: VAE,
        world_model: WorldModel,
        device: str = 'cuda'
    ):
        self.vae = vae.to(device)
        self.world_model = world_model.to(device)
        self.vae.eval()
        self.world_model.eval()
        self.device = device
    
    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode RGB frames to latent."""
        B, T = frames.shape[:2]
        frames_flat = frames.view(B * T, *frames.shape[2:])
        latents = self.vae.get_latent(frames_flat, deterministic=True)
        latents = latents.view(B, T, *latents.shape[1:])
        return latents
    
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to RGB frame."""
        return self.vae.decode(latent)
    
    @torch.no_grad()
    def generate_future_frames(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor = None,
        future_actions: torch.Tensor = None,
        num_future: int = 8
    ) -> tuple:
        """
        Generate future frames autoregressively.
        
        Args:
            context_frames: [1, T, 3, H, W] context RGB frames
            context_actions: [1, T, 3] context actions (optional)
            future_actions: [1, num_future, 3] future actions (optional)
            num_future: number of future frames to generate
            
        Returns:
            predicted_frames: [num_future, 3, H, W] predicted RGB frames
            context_decoded: [T, 3, H, W] decoded context frames
        """
        context_length = self.world_model.config.context_length
        
        # Encode context to latent
        context_latents = self.encode_frames(context_frames)  # [1, T, C, h, w]
        
        # Decode context for visualization
        context_decoded = []
        for t in range(context_frames.shape[1]):
            decoded = self.decode_latent(context_latents[:, t])
            context_decoded.append(decoded[0])
        context_decoded = torch.stack(context_decoded)
        
        # Generate future frames autoregressively
        current_latents = context_latents.clone()
        predicted_frames = []
        
        for i in range(num_future):
            # Get last context_length latents
            input_latents = current_latents[:, -context_length:]
            
            # Get actions if available
            if context_actions is not None and future_actions is not None:
                # Combine context and future actions
                all_actions = torch.cat([context_actions, future_actions], dim=1)
                start_idx = current_latents.shape[1] - context_length
                input_actions = all_actions[:, start_idx:start_idx + context_length]
            else:
                input_actions = None
            
            # Predict next latent
            output = self.world_model(input_latents, input_actions)
            pred_latent = output['pred_latent']  # [1, C, h, w]
            
            # Decode to RGB
            pred_frame = self.decode_latent(pred_latent)
            predicted_frames.append(pred_frame[0])
            
            # Append to context for next iteration
            current_latents = torch.cat([
                current_latents,
                pred_latent.unsqueeze(1)
            ], dim=1)
        
        predicted_frames = torch.stack(predicted_frames)
        
        return predicted_frames, context_decoded
    
    def create_gif(
        self,
        context_frames: torch.Tensor,
        predicted_frames: torch.Tensor,
        ground_truth_frames: torch.Tensor = None,
        save_path: str = 'prediction.gif',
        fps: int = 5,
        include_comparison: bool = True
    ):
        """
        Create GIF from frames.
        
        Args:
            context_frames: [T, 3, H, W] context frames
            predicted_frames: [N, 3, H, W] predicted frames
            ground_truth_frames: [N, 3, H, W] ground truth (optional)
            save_path: path to save GIF
            fps: frames per second
            include_comparison: if True, show prediction vs GT side by side
        """
        frames_for_gif = []
        
        # Determine target size
        sample_img = tensor_to_numpy(context_frames[0])
        h, w = sample_img.shape[:2]
        
        if include_comparison and ground_truth_frames is not None:
            target_width = w * 2  # Side by side
        else:
            target_width = w
        target_height = h
        
        # Add context frames (with blue border)
        for i, frame in enumerate(context_frames):
            img = tensor_to_numpy(frame)
            img = self._add_border(img, color=(0, 100, 255), thickness=5)
            
            # Pad to match comparison width if needed
            if img.shape[1] < target_width:
                pad_width = target_width - img.shape[1]
                padding = np.zeros((img.shape[0], pad_width, 3), dtype=np.uint8)
                padding[:] = (40, 40, 40)  # Dark gray
                img = np.concatenate([img, padding], axis=1)
            
            frames_for_gif.append(img)
        
        # Add predicted frames (with green border)
        for i, pred_frame in enumerate(predicted_frames):
            pred_img = tensor_to_numpy(pred_frame)
            
            if include_comparison and ground_truth_frames is not None and i < len(ground_truth_frames):
                # Side by side: Prediction | Ground Truth
                gt_img = tensor_to_numpy(ground_truth_frames[i])
                
                pred_img = self._add_border(pred_img, color=(0, 255, 0), thickness=5)
                gt_img = self._add_border(gt_img, color=(255, 255, 0), thickness=5)
                
                # Concatenate horizontally
                combined = np.concatenate([pred_img, gt_img], axis=1)
                frames_for_gif.append(combined)
            else:
                pred_img = self._add_border(pred_img, color=(0, 255, 0), thickness=5)
                
                # Pad to match width if needed
                if pred_img.shape[1] < target_width:
                    pad_width = target_width - pred_img.shape[1]
                    padding = np.zeros((pred_img.shape[0], pad_width, 3), dtype=np.uint8)
                    padding[:] = (40, 40, 40)
                    pred_img = np.concatenate([pred_img, padding], axis=1)
                
                frames_for_gif.append(pred_img)
        
        # Save GIF
        imageio.mimsave(save_path, frames_for_gif, fps=fps, loop=0)
        print(f"Saved GIF: {save_path}")
    
    def _add_border(self, img: np.ndarray, color: tuple, thickness: int = 5) -> np.ndarray:
        """Add colored border to image."""
        img = img.copy()
        img[:thickness, :] = color
        img[-thickness:, :] = color
        img[:, :thickness] = color
        img[:, -thickness:] = color
        return img
    
    def _add_text(self, img: np.ndarray, text: str) -> np.ndarray:
        """Add text label to image (simple version without opencv)."""
        # Create a bar at the top for text
        h, w = img.shape[:2]
        bar_height = 25
        new_img = np.zeros((h + bar_height, w, 3), dtype=np.uint8)
        new_img[bar_height:] = img
        new_img[:bar_height] = (40, 40, 40)  # Dark gray bar
        
        # We'll skip actual text rendering to avoid opencv dependency
        # The border colors will indicate frame type
        return new_img


def main():
    parser = argparse.ArgumentParser(description='Generate GIF predictions')
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--context_length', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # Generation
    parser.add_argument('--num_future', type=int, default=16,
                        help='Number of future frames to generate')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of GIFs to generate')
    parser.add_argument('--fps', type=int, default=5,
                        help='Frames per second in GIF')
    
    # Paths
    parser.add_argument('--vae_checkpoint', type=str, 
                        default='./outputs/checkpoints/vae_best.pt')
    parser.add_argument('--world_model_checkpoint', type=str,
                        default='./outputs/checkpoints/world_model_best.pt')
    parser.add_argument('--data_dir', type=str, default='./data/raw')
    parser.add_argument('--output_dir', type=str, default='./outputs/gifs')
    
    # Options
    parser.add_argument('--with_gt', action='store_true', default=True,
                        help='Include ground truth comparison')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Check checkpoints
    vae_path = Path(args.vae_checkpoint)
    wm_path = Path(args.world_model_checkpoint)
    
    if not vae_path.exists():
        print(f"[ERROR] VAE checkpoint not found: {vae_path}")
        return
    
    if not wm_path.exists():
        print(f"[ERROR] World Model checkpoint not found: {wm_path}")
        return
    
    # Load VAE
    print("Loading VAE...")
    vae_checkpoint = torch.load(vae_path, map_location=device)
    vae_config = VAEConfig(
        in_channels=3,
        latent_dim=args.latent_dim,
        hidden_dims=(64, 128, 256, 256)
    )
    vae = VAE(vae_config)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    print(f"  Loaded from epoch {vae_checkpoint['epoch']}")
    
    # Load World Model
    print("Loading World Model...")
    wm_checkpoint = torch.load(wm_path, map_location=device)
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
    
    # Create generator
    generator = GIFGenerator(vae, world_model, device)
    
    # Load dataset
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
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate GIFs
    print(f"\nGenerating {args.num_samples} GIFs...")
    print("=" * 50)
    
    indices = np.random.choice(len(dataset), size=min(args.num_samples, len(dataset)), replace=False)
    
    for i, idx in enumerate(tqdm(indices, desc="Generating")):
        sample = dataset[idx]
        frames = sample['frames'].unsqueeze(0).to(device)  # [1, T, 3, H, W]
        actions = sample.get('actions')
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)
        
        # Split into context and ground truth future
        context_frames = frames[:, :args.context_length]
        gt_future_frames = frames[:, args.context_length:]
        
        if actions is not None:
            context_actions = actions[:, :args.context_length]
            future_actions = actions[:, args.context_length:]
        else:
            context_actions = None
            future_actions = None
        
        # Generate predictions
        predicted_frames, context_decoded = generator.generate_future_frames(
            context_frames,
            context_actions,
            future_actions,
            num_future=args.num_future
        )
        
        # Create GIF
        save_path = output_dir / f'prediction_{i:03d}.gif'
        
        gt_for_gif = gt_future_frames[0] if args.with_gt else None
        
        generator.create_gif(
            context_decoded,
            predicted_frames,
            ground_truth_frames=gt_for_gif,
            save_path=str(save_path),
            fps=args.fps,
            include_comparison=args.with_gt
        )
    
    print("=" * 50)
    print(f"[OK] Generated {args.num_samples} GIFs in {output_dir}")
    
    # Also create a summary image
    print("\nCreating summary image...")
    create_summary_image(generator, dataset, args, output_dir, device)


def create_summary_image(generator, dataset, args, output_dir, device):
    """Create a summary image showing multiple predictions."""
    import matplotlib.pyplot as plt
    
    num_examples = min(4, len(dataset))
    indices = np.random.choice(len(dataset), size=num_examples, replace=False)
    
    fig, axes = plt.subplots(num_examples, 6, figsize=(18, 4 * num_examples))
    
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        sample = dataset[idx]
        frames = sample['frames'].unsqueeze(0).to(device)
        actions = sample.get('actions')
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)
        
        context_frames = frames[:, :args.context_length]
        gt_future_frames = frames[:, args.context_length:]
        
        if actions is not None:
            context_actions = actions[:, :args.context_length]
            future_actions = actions[:, args.context_length:]
        else:
            context_actions = None
            future_actions = None
        
        # Generate
        predicted_frames, context_decoded = generator.generate_future_frames(
            context_frames,
            context_actions,
            future_actions,
            num_future=min(4, args.num_future)
        )
        
        # Plot: Last 2 context | Pred 1 | Pred 2 | GT 1 | GT 2
        # Last context frames
        for i in range(2):
            ctx_idx = -2 + i
            img = tensor_to_numpy(context_decoded[ctx_idx])
            axes[row, i].imshow(img)
            axes[row, i].set_title(f'Context {ctx_idx}')
            axes[row, i].axis('off')
        
        # Predictions
        for i in range(2):
            img = tensor_to_numpy(predicted_frames[i])
            axes[row, 2 + i].imshow(img)
            axes[row, 2 + i].set_title(f'Pred t+{i+1}')
            axes[row, 2 + i].axis('off')
        
        # Ground truth
        for i in range(2):
            if i < gt_future_frames.shape[1]:
                img = denormalize(gt_future_frames[0, i]).cpu().clamp(0, 1)
                img = img.permute(1, 2, 0).numpy()
                img = (img * 255).astype(np.uint8)
                axes[row, 4 + i].imshow(img)
            axes[row, 4 + i].set_title(f'GT t+{i+1}')
            axes[row, 4 + i].axis('off')
    
    plt.suptitle('World Model Predictions vs Ground Truth', fontsize=14)
    plt.tight_layout()
    
    summary_path = output_dir / 'prediction_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved summary: {summary_path}")


if __name__ == '__main__':
    main()