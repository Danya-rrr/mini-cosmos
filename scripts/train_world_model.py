import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.mini_cosmos.data_pipeline import TrajectoryDatasetNPZ, WorldModelDataset
from src.mini_cosmos.models.world_model import WorldModel


def main():
    # --- CONFIG ---
    data_dir = os.path.join("data", "raw")
    batch_size = 16
    num_epochs = 15
    lr = 1e-4
    latent_dim = 128
    action_dim = 3
    img_height, img_width = 90, 160
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    # --- DATA ---
    traj_ds = TrajectoryDatasetNPZ(root_dir=data_dir)
    wmodel_ds = WorldModelDataset(traj_ds)
    loader = DataLoader(wmodel_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    print("Num trajectories:", len(traj_ds))
    print("Num (obs_t, act_t, obs_tp1) pairs:", len(wmodel_ds))

    # --- MODEL ---
    model = WorldModel(
        latent_dim=latent_dim,
        action_dim=3,
        img_height=img_height,
        img_width=img_width,
    ).to(device)


    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- TRAIN LOOP ---
    os.makedirs("models", exist_ok=True)

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in loader:
            obs_t = batch["obs_t"].to(device)       # (B, 3, H, W)
            obs_tp1 = batch["obs_tp1"].to(device)   # (B, 3, H, W)
            action = batch["action"].to(device)     # (B, 2)

            optimizer.zero_grad()

            obs_pred, z_t, z_next = model(obs_t, action)

            loss = criterion(obs_pred, obs_tp1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                print(f"[epoch {epoch+1} step {global_step}] loss = {avg_loss:.6f}")
                running_loss = 0.0

        # save checkpoint after each epoch
        ckpt_path = os.path.join("models", f"world_model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # final save
    final_path = os.path.join("models", "world_model_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
