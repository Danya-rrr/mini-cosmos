import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Convolutional encoder: RGB image (C,H,W) -> latent vector (latent_dim).
    """

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),   # -> (32, H/2,   W/2)
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # -> (64, H/4,   W/4)
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # -> (128, H/8,  W/8)
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),# -> (256, H/16, W/16)
            nn.ReLU(True),
        )

        self.fc = None
        self._inited = False

    def _init_fc(self, shape, device):
        # shape: (B, C, Hc, Wc)
        flat_dim = shape[1] * shape[2] * shape[3]
        self.fc = nn.Linear(flat_dim, self.latent_dim).to(device)
        self._inited = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) in [0,1]
        returns: (B, latent_dim)
        """
        h = self.conv_net(x)  # (B, C, Hc, Wc)

        if not self._inited:
            self._init_fc(h.shape, h.device)

        h = h.flatten(start_dim=1)  # (B, C*Hc*Wc)
        z = self.fc(h)              # (B, latent_dim)
        return z
