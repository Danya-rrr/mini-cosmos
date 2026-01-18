"""
Dataset Module

PyTorch datasets for loading CARLA and nuScenes driving sequences with support
for multi-source data loading, augmentation, and action conditioning.
"""

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T
from torchvision.transforms import functional as TF

import numpy as np
from pathlib import Path
from PIL import Image
import json
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and preprocessing."""
    sequence_length: int = 16
    frame_skip: int = 1
    image_size: Tuple[int, int] = (256, 256)
    normalize: bool = True
    augment: bool = True
    return_actions: bool = True


class VideoAugmentation:
    """
    Video-specific data augmentation with temporal consistency.
    
    Args:
        brightness: Brightness adjustment range
        contrast: Contrast adjustment range
        saturation: Saturation adjustment range
        hue: Hue adjustment range
        horizontal_flip_prob: Probability of horizontal flip
    """
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
        horizontal_flip_prob: float = 0.5,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.horizontal_flip_prob = horizontal_flip_prob
    
    def __call__(
        self,
        frames: torch.Tensor,
        actions: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply consistent augmentation across video sequence."""
        brightness_factor = 1.0 + random.uniform(-self.brightness, self.brightness)
        contrast_factor = 1.0 + random.uniform(-self.contrast, self.contrast)
        saturation_factor = 1.0 + random.uniform(-self.saturation, self.saturation)
        hue_factor = random.uniform(-self.hue, self.hue)
        do_flip = random.random() < self.horizontal_flip_prob
        
        augmented_frames = []
        for frame in frames:
            frame = TF.adjust_brightness(frame, brightness_factor)
            frame = TF.adjust_contrast(frame, contrast_factor)
            frame = TF.adjust_saturation(frame, saturation_factor)
            frame = TF.adjust_hue(frame, hue_factor)
            
            if do_flip:
                frame = TF.hflip(frame)
            
            augmented_frames.append(frame)
        
        frames = torch.stack(augmented_frames)
        
        if do_flip and actions is not None and actions.shape[-1] >= 2:
            actions = actions.clone()
            actions[:, 1] = -actions[:, 1]
        
        return frames, actions


class BaseVideoDataset(Dataset):
    """
    Base dataset class for video sequences.
    
    Args:
        data_dir: Root directory containing episode folders
        config: Dataset configuration
        split: Data split ('train', 'val', or 'test')
        split_ratio: Ratio for train/val/test split
        episode_prefix: Prefix for episode folder names
    """
    
    def __init__(
        self,
        data_dir: str,
        config: Optional[DatasetConfig] = None,
        split: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        episode_prefix: str = 'episode_'
    ):
        self.data_dir = Path(data_dir)
        self.config = config or DatasetConfig()
        self.split = split
        self.episode_prefix = episode_prefix
        
        self.episodes = self._load_episodes()
        self.episodes = self._apply_split(self.episodes, split, split_ratio)
        self.sequences = self._index_sequences()
        
        transforms_list = [
            T.Resize(self.config.image_size),
            T.ToTensor(),
        ]
        if self.config.normalize:
            transforms_list.append(T.Normalize(mean=[0.5]*3, std=[0.5]*3))
        
        self.base_transform = T.Compose(transforms_list)
        
        self.augmentation = (
            VideoAugmentation() if self.config.augment and split == 'train' else None
        )
    
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load episode metadata from filesystem."""
        episodes = []
        
        for episode_dir in sorted(self.data_dir.iterdir()):
            if not episode_dir.is_dir():
                continue
            if not episode_dir.name.startswith(self.episode_prefix):
                continue
            
            images_dir = episode_dir / 'images'
            if not images_dir.exists():
                images_dir = episode_dir
            
            frame_files = sorted(images_dir.glob('*.png'))
            if not frame_files:
                frame_files = sorted(images_dir.glob('*.jpg'))
            
            min_frames = self.config.sequence_length * self.config.frame_skip
            if len(frame_files) < min_frames:
                continue
            
            metadata_path = episode_dir / 'metadata.json'
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, encoding='utf-8') as f:
                    metadata = json.load(f)
            
            episodes.append({
                'dir': episode_dir,
                'images_dir': images_dir,
                'metadata': metadata,
                'frame_files': frame_files,
                'num_frames': len(frame_files)
            })
        
        return episodes
    
    def _apply_split(
        self,
        episodes: List[Dict],
        split: str,
        ratio: Tuple[float, float, float]
    ) -> List[Dict]:
        """Split episodes into train/val/test sets."""
        n = len(episodes)
        train_end = int(n * ratio[0])
        val_end = int(n * (ratio[0] + ratio[1]))
        
        if split == 'train':
            return episodes[:train_end]
        elif split == 'val':
            return episodes[train_end:val_end]
        else:
            return episodes[val_end:]
    
    def _index_sequences(self) -> List[Tuple[int, int]]:
        """Create index of all valid sequences."""
        sequences = []
        seq_len = self.config.sequence_length * self.config.frame_skip
        
        for ep_idx, episode in enumerate(self.episodes):
            num_frames = episode['num_frames']
            num_sequences = num_frames - seq_len + 1
            
            for start_idx in range(0, num_sequences, self.config.frame_skip):
                sequences.append((ep_idx, start_idx))
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load and return a video sequence."""
        ep_idx, start_idx = self.sequences[idx]
        episode = self.episodes[ep_idx]
        
        frame_indices = [
            start_idx + i * self.config.frame_skip
            for i in range(self.config.sequence_length)
        ]
        
        frames = []
        for frame_idx in frame_indices:
            img_path = episode['frame_files'][frame_idx]
            img = Image.open(img_path).convert('RGB')
            img = self.base_transform(img)
            frames.append(img)
        
        frames = torch.stack(frames)
        
        actions = None
        if self.config.return_actions and 'frames' in episode['metadata']:
            actions_list = []
            for frame_idx in frame_indices:
                if frame_idx < len(episode['metadata']['frames']):
                    frame_meta = episode['metadata']['frames'][frame_idx]
                    actions_list.append([
                        frame_meta.get('control_throttle', 0.0),
                        frame_meta.get('control_steer', 0.0),
                        frame_meta.get('control_brake', 0.0),
                    ])
                else:
                    actions_list.append([0.0, 0.0, 0.0])
            actions = torch.tensor(actions_list, dtype=torch.float32)
        
        if self.augmentation is not None:
            frames, actions = self.augmentation(frames, actions)
        
        result = {'frames': frames}
        if actions is not None:
            result['actions'] = actions
        
        return result


class CARLADataset(BaseVideoDataset):
    """Dataset for CARLA simulator data."""
    
    def __init__(self, data_dir: str, config=None, split='train', **kwargs):
        super().__init__(
            data_dir, config, split,
            episode_prefix='episode_',
            **kwargs
        )
        print(f"CARLADataset [{split}]: {len(self.episodes)} episodes, "
              f"{len(self.sequences)} sequences")


class NuScenesDataset(BaseVideoDataset):
    """Dataset for nuScenes data."""
    
    def __init__(self, data_dir: str, config=None, split='train', **kwargs):
        super().__init__(
            data_dir, config, split,
            episode_prefix='scene_',
            **kwargs
        )
        print(f"NuScenesDataset [{split}]: {len(self.episodes)} episodes, "
              f"{len(self.sequences)} sequences")


class CombinedDataset(Dataset):
    """Wrapper for combining multiple datasets."""
    
    def __init__(self, datasets: List[Dataset]):
        self.dataset = ConcatDataset(datasets)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.dataset[idx]


def create_combined_dataloaders(
    carla_dir: str = './data/raw',
    nuscenes_dir: str = './data/processed/nuscenes',
    config: Optional[DatasetConfig] = None,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders combining CARLA and nuScenes data.
    
    Args:
        carla_dir: Path to CARLA data directory
        nuscenes_dir: Path to nuScenes data directory
        config: Dataset configuration
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = config or DatasetConfig()
    
    datasets = {'train': [], 'val': [], 'test': []}
    
    carla_path = Path(carla_dir)
    if carla_path.exists() and any(carla_path.iterdir()):
        for split in ['train', 'val', 'test']:
            ds = CARLADataset(carla_dir, config, split=split)
            if len(ds) > 0:
                datasets[split].append(ds)
    
    nuscenes_path = Path(nuscenes_dir)
    if nuscenes_path.exists() and any(nuscenes_path.iterdir()):
        for split in ['train', 'val', 'test']:
            ds = NuScenesDataset(nuscenes_dir, config, split=split)
            if len(ds) > 0:
                datasets[split].append(ds)
    
    train_dataset = CombinedDataset(datasets['train']) if datasets['train'] else None
    val_dataset = CombinedDataset(datasets['val']) if datasets['val'] else None
    test_dataset = CombinedDataset(datasets['test']) if datasets['test'] else None
    
    print(f"\nCombined dataset sizes:")
    print(f"  Train: {len(train_dataset) if train_dataset else 0}")
    print(f"  Val:   {len(val_dataset) if val_dataset else 0}")
    print(f"  Test:  {len(test_dataset) if test_dataset else 0}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    ) if train_dataset and len(train_dataset) > 0 else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if val_dataset and len(val_dataset) > 0 else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    ) if test_dataset and len(test_dataset) > 0 else None
    
    return train_loader, val_loader, test_loader


def create_dataloaders(
    data_dir: str,
    config: Optional[DatasetConfig] = None,
    batch_size: int = 8,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for single data source.
    
    Args:
        data_dir: Path to data directory
        config: Dataset configuration
        batch_size: Batch size for dataloaders
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    config = config or DatasetConfig()
    
    train_dataset = CARLADataset(data_dir, config, split='train')
    val_dataset = CARLADataset(data_dir, config, split='val')
    test_dataset = CARLADataset(data_dir, config, split='test')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def main():
    """Test dataset functionality."""
    config = DatasetConfig(
        sequence_length=8,
        frame_skip=1,
        image_size=(256, 256),
        augment=False,
    )
    
    train_loader, val_loader, test_loader = create_combined_dataloaders(
        carla_dir='./data/raw',
        nuscenes_dir='./data/processed/nuscenes',
        config=config,
        batch_size=4,
        num_workers=0,
    )
    
    if train_loader:
        batch = next(iter(train_loader))
        print(f"\nSample batch:")
        print(f"  Frames shape: {batch['frames'].shape}")
        print(f"  Frames range: [{batch['frames'].min():.2f}, {batch['frames'].max():.2f}]")
        print("\nDataset test passed successfully!")
    else:
        print("\nWarning: No training data found!")


if __name__ == '__main__':
    main()