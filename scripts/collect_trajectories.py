import os
import sys
import time
import glob
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.mini_cosmos.carla_env import CarlaEnv


def collect_one_trajectory(env: CarlaEnv, horizon: int = 300):
    imgs, acts = [], []

    obs = env.reset()
    imgs.append(obs)

    for t in range(horizon):
        obs, reward, done, info = env.step(action=None)
        imgs.append(obs)
        acts.append(info["action"])

        if done:
            print("Done=True – stopping trajectory at step", t)
            break

        if not env.vehicle.is_alive:
            print("Vehicle destroyed – stopping trajectory at step", t)
            break

    images = np.stack(imgs, axis=0)               # (T+1, H, W, 3)
    actions = np.asarray(acts, dtype=np.float32)  # (T, 3)
    return images, actions


def main():
    TOWN = "Town03"
    FPS = 10
    IMG_W, IMG_H = 160, 90
    HORIZON = 300
    NUM_TRAJ = 10

    output_dir = os.path.join("data", "raw")
    print("Output dir =", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(output_dir, "traj_ap_*.npz")))
    start_idx = len(existing)
    print(f"Znaleziono już {start_idx} trajektorii. Nowe będą od indeksu {start_idx}.")

    # 1 środowisko na wiele trajek
    env = CarlaEnv(
        town=TOWN,
        fps=FPS,
        image_width=IMG_W,
        image_height=IMG_H,
        autopilot=True,
        num_traffic_vehicles=10,
        num_pedestrians=0,
    )

    try:
        for i in range(NUM_TRAJ):
            global_idx = start_idx + i
            print(f"\n=== TRAJ {global_idx} (local {i}/{NUM_TRAJ-1}) ===")

            images, actions = collect_one_trajectory(env, horizon=HORIZON)
            print("Image shape:", images.shape, "Action shape:", actions.shape)

            filename = f"traj_ap_{global_idx:03d}.npz"
            path = os.path.join(output_dir, filename)
            np.savez_compressed(path, images=images, actions=actions)
            print("Saved:", path)

            time.sleep(0.5)
    finally:
        env.close()

    print("DONE.")


if __name__ == "__main__":
    main()
