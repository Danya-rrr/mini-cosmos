"""
Mini-Cosmos: Evaluation Metrics Script
======================================
Compute metrics to evaluate World Model prediction quality.

Metrics:
    - SSIM: Structural Similarity Index
    - PSNR: Peak Signal-to-Noise Ratio
    - MSE: Mean Squared Error
    - LPIPS: Learned Perceptual Image Patch Similarity (optional)

Usage:
    python scripts/evaluate_metrics.py --num_samples 100 --latent_dim 8
    python scripts/evaluate_metrics.py --num_samples 100 --latent_dim 8 --use_lpips

Requirements:
    - Trained VAE and World Model checkpoints
    - pip install lpips (for LPIPS metric)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

# Local imports
from src.data.dataset import CARLADataset, DatasetConfig
from src.models.vae import VAE, VAEConfig
from src.models.world_model import WorldModel, WorldModelConfig

# Metrics
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Convert from [-1, 1] to [0, 1]"""
    return (tensor + 1) / 2


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy image [H, W, 3] float [0, 1]"""
    img = denormalize(tensor).cpu().clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    return img


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute SSIM between two images."""
    return ssim(pred, gt, channel_axis=2, data_range=1.0)


def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute PSNR between two images."""
    return psnr(gt, pred, data_range=1.0)


def compute_mse(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute MSE between two images."""
    return np.mean((pred - gt) ** 2)


class MetricsEvaluator:
    """Evaluate World Model predictions with various metrics."""
    
    def __init__(
        self,
        vae: VAE,
        world_model: WorldModel,
        device: str = 'cuda',
        use_lpips: bool = False
    ):
        self.vae = vae.to(device)
        self.world_model = world_model.to(device)
        self.vae.eval()
        self.world_model.eval()
        self.device = device
        
        # LPIPS (optional)
        self.lpips_fn = None
        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='alex').to(device)
                self.lpips_fn.eval()
                print("LPIPS metric enabled")
            except ImportError:
                print("LPIPS not installed. Run: pip install lpips")
    
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
    def compute_lpips(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute LPIPS between two images."""
        if self.lpips_fn is None:
            return 0.0
        
        # LPIPS expects [-1, 1] range
        pred_norm = pred * 2 - 1
        gt_norm = gt * 2 - 1
        
        # Add batch dimension if needed
        if pred_norm.dim() == 3:
            pred_norm = pred_norm.unsqueeze(0)
            gt_norm = gt_norm.unsqueeze(0)
        
        return self.lpips_fn(pred_norm, gt_norm).item()
    
    @torch.no_grad()
    def predict_future(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor = None,
        future_actions: torch.Tensor = None,
        num_future: int = 8
    ) -> torch.Tensor:
        """Generate future frame predictions."""
        context_length = self.world_model.config.context_length
        
        # Encode context
        context_latents = self.encode_frames(context_frames)
        
        # Generate autoregressively
        current_latents = context_latents.clone()
        predicted_frames = []
        
        for i in range(num_future):
            input_latents = current_latents[:, -context_length:]
            
            if context_actions is not None and future_actions is not None:
                all_actions = torch.cat([context_actions, future_actions], dim=1)
                start_idx = current_latents.shape[1] - context_length
                input_actions = all_actions[:, start_idx:start_idx + context_length]
            else:
                input_actions = None
            
            output = self.world_model(input_latents, input_actions)
            pred_latent = output['pred_latent']
            
            pred_frame = self.decode_latent(pred_latent)
            predicted_frames.append(pred_frame[0])
            
            current_latents = torch.cat([
                current_latents,
                pred_latent.unsqueeze(1)
            ], dim=1)
        
        return torch.stack(predicted_frames)
    
    def evaluate_sample(
        self,
        context_frames: torch.Tensor,
        gt_future_frames: torch.Tensor,
        context_actions: torch.Tensor = None,
        future_actions: torch.Tensor = None
    ) -> dict:
        """Evaluate metrics for a single sample."""
        num_future = gt_future_frames.shape[1]
        
        # Predict
        predicted_frames = self.predict_future(
            context_frames,
            context_actions,
            future_actions,
            num_future=num_future
        )
        
        # Compute metrics per timestep
        metrics_per_step = {
            'ssim': [],
            'psnr': [],
            'mse': [],
            'lpips': []
        }
        
        for t in range(num_future):
            pred = tensor_to_numpy(predicted_frames[t])
            gt = tensor_to_numpy(gt_future_frames[0, t])
            
            metrics_per_step['ssim'].append(compute_ssim(pred, gt))
            metrics_per_step['psnr'].append(compute_psnr(pred, gt))
            metrics_per_step['mse'].append(compute_mse(pred, gt))
            
            if self.lpips_fn is not None:
                pred_tensor = denormalize(predicted_frames[t]).to(self.device)
                gt_tensor = denormalize(gt_future_frames[0, t]).to(self.device)
                metrics_per_step['lpips'].append(self.compute_lpips(pred_tensor, gt_tensor))
        
        return metrics_per_step


def evaluate_model(
    evaluator: MetricsEvaluator,
    dataset: CARLADataset,
    num_samples: int,
    num_future: int,
    context_length: int,
    device: str
) -> dict:
    """Evaluate model on multiple samples."""
    
    all_metrics = {
        'ssim': [[] for _ in range(num_future)],
        'psnr': [[] for _ in range(num_future)],
        'mse': [[] for _ in range(num_future)],
        'lpips': [[] for _ in range(num_future)]
    }
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    for idx in tqdm(indices, desc="Evaluating"):
        sample = dataset[idx]
        frames = sample['frames'].unsqueeze(0).to(device)
        actions = sample.get('actions')
        if actions is not None:
            actions = actions.unsqueeze(0).to(device)
        
        context_frames = frames[:, :context_length]
        gt_future_frames = frames[:, context_length:context_length + num_future]
        
        if gt_future_frames.shape[1] < num_future:
            continue
        
        if actions is not None:
            context_actions = actions[:, :context_length]
            future_actions = actions[:, context_length:context_length + num_future]
        else:
            context_actions = None
            future_actions = None
        
        metrics = evaluator.evaluate_sample(
            context_frames,
            gt_future_frames,
            context_actions,
            future_actions
        )
        
        for t in range(num_future):
            all_metrics['ssim'][t].append(metrics['ssim'][t])
            all_metrics['psnr'][t].append(metrics['psnr'][t])
            all_metrics['mse'][t].append(metrics['mse'][t])
            if metrics['lpips']:
                all_metrics['lpips'][t].append(metrics['lpips'][t])
    
    # Compute statistics
    results = {
        'per_timestep': {},
        'average': {}
    }
    
    for metric_name in ['ssim', 'psnr', 'mse', 'lpips']:
        if not all_metrics[metric_name][0]:
            continue
            
        means = [np.mean(all_metrics[metric_name][t]) for t in range(num_future)]
        stds = [np.std(all_metrics[metric_name][t]) for t in range(num_future)]
        
        results['per_timestep'][metric_name] = {
            'mean': means,
            'std': stds
        }
        
        results['average'][metric_name] = {
            'mean': np.mean(means),
            'std': np.mean(stds)
        }
    
    return results


def plot_metrics(results: dict, output_dir: Path):
    """Plot metrics over prediction horizon."""
    
    metrics_to_plot = ['ssim', 'psnr', 'mse']
    if 'lpips' in results['per_timestep']:
        metrics_to_plot.append('lpips')
    
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, metric_name in zip(axes, metrics_to_plot):
        data = results['per_timestep'][metric_name]
        timesteps = range(1, len(data['mean']) + 1)
        
        ax.plot(timesteps, data['mean'], 'b-', linewidth=2, label='Mean')
        ax.fill_between(
            timesteps,
            np.array(data['mean']) - np.array(data['std']),
            np.array(data['mean']) + np.array(data['std']),
            alpha=0.3
        )
        
        ax.set_xlabel('Prediction Horizon (t+k)')
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f'{metric_name.upper()} vs Prediction Horizon')
        ax.grid(True, alpha=0.3)
        
        # Add average line
        avg = results['average'][metric_name]['mean']
        ax.axhline(y=avg, color='r', linestyle='--', alpha=0.5, label=f'Avg: {avg:.4f}')
        ax.legend()
    
    plt.tight_layout()
    
    save_path = output_dir / 'metrics_plot.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {save_path}")


def print_results(results: dict):
    """Print results in a nice format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print("\n📊 Average Metrics:")
    print("-" * 40)
    
    for metric_name, values in results['average'].items():
        print(f"  {metric_name.upper():8s}: {values['mean']:.4f} ± {values['std']:.4f}")
    
    print("\n📈 Metrics by Prediction Horizon:")
    print("-" * 40)
    
    for metric_name in results['per_timestep']:
        data = results['per_timestep'][metric_name]
        print(f"\n  {metric_name.upper()}:")
        print(f"    t+1:  {data['mean'][0]:.4f}")
        print(f"    t+5:  {data['mean'][min(4, len(data['mean'])-1)]:.4f}")
        print(f"    t+10: {data['mean'][min(9, len(data['mean'])-1)]:.4f}")
        if len(data['mean']) > 10:
            print(f"    t+{len(data['mean'])}: {data['mean'][-1]:.4f}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate World Model predictions')
    
    # Model
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--context_length', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # Evaluation
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')
    parser.add_argument('--num_future', type=int, default=16,
                        help='Number of future frames to predict')
    parser.add_argument('--use_lpips', action='store_true',
                        help='Use LPIPS metric (requires pip install lpips)')
    
    # Paths
    parser.add_argument('--vae_checkpoint', type=str,
                        default='./outputs/checkpoints/vae_best.pt')
    parser.add_argument('--world_model_checkpoint', type=str,
                        default='./outputs/checkpoints/world_model_best.pt')
    parser.add_argument('--data_dir', type=str, default='./data/raw')
    parser.add_argument('--output_dir', type=str, default='./outputs/metrics')
    
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
    vae_checkpoint = torch.load(vae_path, map_location=device, weights_only=False)
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
    
    # Create evaluator
    evaluator = MetricsEvaluator(vae, world_model, device, use_lpips=args.use_lpips)
    
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
    
    # Evaluate
    print(f"\nEvaluating on {args.num_samples} samples...")
    results = evaluate_model(
        evaluator,
        dataset,
        num_samples=args.num_samples,
        num_future=args.num_future,
        context_length=args.context_length,
        device=device
    )
    
    # Print results
    print_results(results)
    
    # Save results (convert numpy types to Python types for JSON)
    def convert_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(i) for i in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj
    
    results_path = output_dir / 'metrics_results.json'
    with open(results_path, 'w') as f:
        json.dump(convert_to_python(results), f, indent=2)
    print(f"\nSaved results: {results_path}")
    
    # Plot
    plot_metrics(results, output_dir)
    
    print("\n[OK] Evaluation complete!")


if __name__ == '__main__':
    main()