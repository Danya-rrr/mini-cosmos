import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.mini_cosmos.data_pipeline import TrajectoryDatasetNPZ
from src.mini_cosmos.models.world_model import WorldModel


def to_torch_img(arr: np.ndarray, device):
    """
    arr: (H, W, 3) uint8 lub float w [0,255]
    return: (1, 3, H, W) float32 w [0,1]
    """
    if arr.dtype == np.uint8:
        x = arr.astype(np.float32) / 255.0
    else:
        x = arr.astype(np.float32)
    x = np.transpose(x, (2, 0, 1))  # -> (3, H, W)
    x = np.expand_dims(x, axis=0)   # -> (1, 3, H, W)
    return torch.from_numpy(x).to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 1) wczytaj trajektorie
    traj_ds = TrajectoryDatasetNPZ(root_dir="data/raw")
    print("Num trajectories:", len(traj_ds))

    traj = traj_ds[0]
    images = traj["images"]    # (T+1, H, W, 3)
    actions = traj["actions"]  # (T, 2)

    T_plus_1, H, W, _ = images.shape
    T = T_plus_1 - 1
    print(f"Using trajectory with T={T}, image size=({H},{W})")

    model = WorldModel(
        latent_dim=128,
        action_dim=3,
        img_height=H,
        img_width=W,
    ).to(device)

    # 1) Dummy forward TYLKO przez encoder, żeby encoder.fc się stworzył
    with torch.no_grad():
        dummy_obs = torch.zeros(1, 3, H, W, device=device)
        _ = model.encoder(dummy_obs)

    # 2) Dopiero teraz ładujemy state_dict
    ckpt_path = os.path.join("models", "world_model_final.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    load_info = model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("Loaded model from:", ckpt_path)





    # 3) wybierz kilka indeksów czasowych do pokazania
    # np. 4 kroki równomiernie rozłożone po trajektorii
    num_examples = 4
    indices = np.linspace(0, T - 1, num_examples, dtype=int)
    print("Time indices:", indices)

    fig, axes = plt.subplots(2, num_examples, figsize=(4 * num_examples, 6))
    if num_examples == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # ujednolicenie kształtu

    for col, t in enumerate(indices):
        obs_t_np = images[t]      # (H, W, 3)
        obs_tp1_np = images[t + 1]
        act_t_np = actions[t]     # (2,)

        obs_t = to_torch_img(obs_t_np, device)     # (1, 3, H, W)
        act_t = torch.from_numpy(act_t_np).float().unsqueeze(0).to(device)  # (1, 2)

        with torch.no_grad():
            obs_pred, z_t, z_next = model(obs_t, act_t)  # (1, 3, H, W)

        # konwersja z powrotem do (H, W, 3)
        obs_pred_np = obs_pred.cpu().numpy()[0]        # (3, H, W)
        obs_pred_np = np.transpose(obs_pred_np, (1, 2, 0))  # (H, W, 3)
        obs_pred_np = np.clip(obs_pred_np, 0.0, 1.0)

        # oryginalne normalize do [0,1] do wyświetlenia
        obs_t_vis = obs_t_np.astype(np.float32) / 255.0
        obs_tp1_vis = obs_tp1_np.astype(np.float32) / 255.0

        # rysujemy: góra - prawdziwe obs_{t+1}, dół - przewidziane
        ax_true = axes[0, col]
        ax_pred = axes[1, col]

        ax_true.imshow(obs_tp1_vis)
        ax_true.set_title(f"t={t} → t+1 (true)")
        ax_true.axis("off")

        ax_pred.imshow(obs_pred_np)
        ax_pred.set_title("predicted")
        ax_pred.axis("off")

    plt.tight_layout()

    os.makedirs("visualizations", exist_ok=True)
    out_path = os.path.join("visualizations", "world_model_examples.png")
    plt.savefig(out_path, dpi=150)
    print("Saved visualization to:", out_path)


if __name__ == "__main__":
    main()
