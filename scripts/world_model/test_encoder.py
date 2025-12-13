import os
import sys

# dodaj katalog główny projektu do ścieżki importów
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.mini_cosmos.models.encoder import Encoder


def main():
    encoder = Encoder(latent_dim=128)

    # test dummy batch of images: (B, 3, H, W)
    x = torch.rand(4, 3, 90, 160)  # simulate your CARLA images

    z = encoder(x)

    print("Input shape :", x.shape)
    print("Latent shape:", z.shape)
    print("Latent sample:", z[0, :5])  # print first few dims

if __name__ == "__main__":
    main()
