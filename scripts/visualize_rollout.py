# scripts/visualize_rollout.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.mini_cosmos.models.world_model import WorldModel



def load_trajectory_npz(root_dir: str, traj_index: int = 0):
    """
    Ładuje jedną trajektorię z pliku NPZ.
    Zakładamy format:
      - images: (T+1, H, W, 3) uint8
      - actions: (T, action_dim) float32
    """
    pattern = os.path.join(root_dir, "traj_ap_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"Nie znaleziono plików trajektorii w {root_dir}")

    if traj_index < 0 or traj_index >= len(files):
        raise IndexError(f"traj_index={traj_index}, ale dostępne jest {len(files)} plików")

    path = files[traj_index]
    data = np.load(path)
    images = data["images"]      # (T+1, H, W, 3)
    actions = data["actions"]    # (T,   A)

    print(f"Używam trajektorii: {path}")
    print("images shape:", images.shape, "actions shape:", actions.shape)
    return images, actions, path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- parametry rolloutu ---
    root_dir = "data/raw"
    traj_index = 0        # którą trajkę bierzemy (0 = pierwsza)
    start_t = 0           # od której klatki startujemy
    rollout_steps = 40    # ile kroków do przodu przewidujemy

    # --- 1. Ładujemy trajektorię ---
    images, actions, path = load_trajectory_npz(root_dir, traj_index)
    T_plus_1, H, W, C = images.shape
    T = actions.shape[0]
    action_dim = actions.shape[1]

    if start_t + rollout_steps >= T:
        rollout_steps = T - 1 - start_t
        print(f"Zmieniam rollout_steps na {rollout_steps} (ograniczenie długości trajektorii).")

    # --- 2. Ładujemy wytrenowany model ---
    latent_dim = 128
    model = WorldModel(
        latent_dim=latent_dim,
        action_dim=action_dim,
        img_height=H,
        img_width=W,
    ).to(device)

    ckpt_path = "models/world_model_final.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Załadowano model z: {ckpt_path}")

    # --- 3. Przygotowujemy pierwszą obserwację i akcje ---
    # normalizacja do [0,1], CHW
    obs0_np = images[start_t].astype(np.float32) / 255.0
    obs0_t = (
        torch.from_numpy(obs0_np)
        .permute(2, 0, 1)          # HWC -> CHW
        .unsqueeze(0)              # (1, C, H, W)
        .to(device)
    )

    # akcje od start_t do start_t + rollout_steps - 1
    acts_np = actions[start_t : start_t + rollout_steps].astype(np.float32)
    acts_t = torch.from_numpy(acts_np).to(device)   # (K, A)

    # --- 4. Rollout w latent space ---
    pred_frames = []
    true_frames = []

    with torch.no_grad():
        # zakoduj pierwszą obserwację
        z_t = model.encoder(obs0_t)  # (1, latent_dim)

        for k in range(rollout_steps):
            a_t = acts_t[k].unsqueeze(0)  # (1, A)
            # pojedynczy krok dynamiki
            z_next = model.dynamics(z_t, a_t)
            # dekodujemy przewidziany stan
            obs_pred = model.decoder(z_next)  # (1, 3, H, W)

            obs_pred_np = obs_pred.squeeze(0).cpu().permute(1, 2, 0).numpy()
            obs_pred_np = np.clip(obs_pred_np, 0.0, 1.0)

            pred_frames.append(obs_pred_np)
            true_frames.append(images[start_t + 1 + k].astype(np.float32) / 255.0)

            z_t = z_next

    print(f"Wygenerowano rollout {rollout_steps} kroków.")

    # --- 5. Wizualizacja (kilka wybranych kroków) ---
    os.makedirs("visualizations", exist_ok=True)

    # wybieramy 4 indeksy równomiernie z zakresu [0, rollout_steps-1]
    num_show = 4
    indices = np.linspace(0, rollout_steps - 1, num=num_show, dtype=int)

    fig, axes = plt.subplots(2, num_show, figsize=(4 * num_show, 4))
    for i, k in enumerate(indices):
        gt = true_frames[k]
        pred = pred_frames[k]

        ax_gt = axes[0, i]
        ax_pr = axes[1, i]

        ax_gt.imshow(gt)
        ax_gt.set_title(f"t+{k+1} (true)")
        ax_gt.axis("off")

        ax_pr.imshow(pred)
        ax_pr.set_title("predicted")
        ax_pr.axis("off")

    plt.tight_layout()
    out_path = "visualizations/world_model_rollout.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"Rollout zapisany do: {out_path}")


if __name__ == "__main__":
    main()
