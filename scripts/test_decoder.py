import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.mini_cosmos.models.encoder import Encoder
from src.mini_cosmos.models.decoder import Decoder


def main():
    encoder = Encoder(latent_dim=128)
    decoder = Decoder(latent_dim=128, out_height=90, out_width=160)

    # sztuczne obrazki jak z CARLI
    x = torch.rand(4, 3, 90, 160)

    # przejście przez encoder + decoder
    z = encoder(x)
    x_hat = decoder(z)

    print("Input shape   :", x.shape)
    print("Latent shape  :", z.shape)
    print("Recon shape   :", x_hat.shape)
    print("Recon min/max :", float(x_hat.min()), float(x_hat.max()))


if __name__ == "__main__":
    main()
