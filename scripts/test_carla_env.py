import os
import sys

# dodaj katalog główny projektu do ścieżki
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.mini_cosmos.carla_env import CarlaEnv
import time
import numpy as np


from src.mini_cosmos.carla_env import CarlaEnv


def main():
    env = CarlaEnv(fps=10, image_width=320, image_height=180)

    obs = env.reset()
    print("Obs shape:", obs.shape, obs.dtype)

    # simple rollout: drive forward with slight steering
    for i in range(50):
        action = (0.5, 0.1)  # throttle, steer
        obs, reward, done, info = env.step(action)
        print(f"Step {i} frame={info['frame']}")
        time.sleep(0.05)

    env.close()


if __name__ == "__main__":
    main()

