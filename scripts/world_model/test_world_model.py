import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.mini_cosmos.models.world_model import WorldModel


def main():
    wm = WorldModel(latent_dim=128, action_dim=2, img_height=90, img_width=160)

    # losowa próbka
    obs = torch.rand(4, 3, 90, 160)
    act = torch.rand(4, 3)  # throttle, steer

    obs_pred, z, z_next = wm(obs, act)

    print("obs shape      :", obs.shape)
    print("obs_pred shape :", obs_pred.shape)
    print("z shape        :", z.shape)
    print("z_next shape   :", z_next.shape)
    print("min/max pred   :", float(obs_pred.min()), float(obs_pred.max()))


if __name__ == "__main__":
    main()
