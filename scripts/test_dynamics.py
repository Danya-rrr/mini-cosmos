import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.mini_cosmos.models.encoder import Encoder
from src.mini_cosmos.models.dynamics import LatentDynamics


def main():
    encoder = Encoder(latent_dim=128)
    dynamics = LatentDynamics(latent_dim=128, action_dim=2, hidden_dim=256)

    # sztuczna próbka: 4 obrazki + 4 akcje
    x = torch.rand(4, 3, 90, 160)        # niby CARLA frames
    a = torch.rand(4, 2) * torch.tensor([1.0, 2.0])  # throttle, steer w jakimś zakresie

    z = encoder(x)        # (B, 128)
    z_next = dynamics(z, a)

    print("z shape      :", z.shape)
    print("z_next shape :", z_next.shape)
    print("z[0,:5]      :", z[0, :5])
    print("z_next[0,:5] :", z_next[0, :5])


if __name__ == "__main__":
    main()
