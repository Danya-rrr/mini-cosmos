import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.mini_cosmos.data_pipeline import TrajectoryDatasetNPZ, WorldModelDataset


def main():
    traj_ds = TrajectoryDatasetNPZ(root_dir="data/raw")
    print("Num trajectories:", len(traj_ds))

    sample_traj = traj_ds[0]
    print("One traj images:", sample_traj["images"].shape)
    print("One traj actions:", sample_traj["actions"].shape)

    wmodel_ds = WorldModelDataset(traj_ds)
    print("WorldModelDataset len:", len(wmodel_ds))

    sample = wmodel_ds[0]
    print("obs_t:", sample["obs_t"].shape, sample["obs_t"].dtype)
    print("obs_tp1:", sample["obs_tp1"].shape, sample["obs_tp1"].dtype)
    print("action:", sample["action"].shape, sample["action"].dtype)


if __name__ == "__main__":
    main()
