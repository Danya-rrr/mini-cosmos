import os
from glob import glob
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object


class TrajectoryDatasetNPZ:
    """
    Prosty loader trajektorii zapisanych jako .npz w data/raw.

    Każdy plik .npz powinien mieć:
      - images: (T+1, H, W, 3) uint8
      - actions: (T, 2) float32
    """

    def __init__(self, root_dir: str = "data/raw"):
        self.root_dir = root_dir
        pattern = os.path.join(root_dir, "traj_ap_*.npz")
        self.files: List[str] = sorted(glob(pattern))

        if not self.files:
            raise FileNotFoundError(
                f"No trajectory files found in {root_dir}. "
                f"Run scripts/collect_trajectories.py first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.files[idx]
        data = np.load(path)

        images = data["images"]      # (T+1, H, W, 3)
        actions = data["actions"]    # (T, 2)

        return {
            "images": images,
            "actions": actions,
            "path": path,
        }


class WorldModelDataset(Dataset):
    """
    Dataset pod trening world modelu frame-to-frame.

    Z trajektorii (images, actions) generuje próbki:
      (obs_t, action_t, obs_{t+1})
    """

    def __init__(
        self,
        traj_dataset: TrajectoryDatasetNPZ,
        transform=None,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for WorldModelDataset.")

        self.traj_dataset = traj_dataset
        self.transform = transform

        # indeksy (traj_idx, t) dla wszystkich par obs_t -> obs_{t+1}
        self.index: List[Tuple[int, int]] = []
        for traj_idx in range(len(traj_dataset)):
            data = traj_dataset[traj_idx]
            T_plus_1 = data["images"].shape[0]
            T = T_plus_1 - 1
            for t in range(T):
                self.index.append((traj_idx, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj_idx, t = self.index[idx]
        data = self.traj_dataset[traj_idx]

        images = data["images"]      # (T+1, H, W, 3)
        actions = data["actions"]    # (T, 2)

        obs_t = images[t]
        obs_tp1 = images[t + 1]
        act_t = actions[t]

        # konwersja do tensora (C, H, W), znormalizowane do [0,1]
        obs_t = obs_t.astype(np.float32) / 255.0
        obs_tp1 = obs_tp1.astype(np.float32) / 255.0

        obs_t = np.transpose(obs_t, (2, 0, 1))
        obs_tp1 = np.transpose(obs_tp1, (2, 0, 1))

        sample = {
            "obs_t": obs_t,
            "obs_tp1": obs_tp1,
            "action": act_t.astype(np.float32),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        if torch is not None:
            sample = {
                k: torch.from_numpy(v) for k, v in sample.items()
            }

        return sample
