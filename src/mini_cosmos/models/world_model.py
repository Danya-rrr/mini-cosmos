import torch
import torch.nn as nn

from .encoder import Encoder
from .dynamics import LatentDynamics
from .decoder import Decoder


class WorldModel(nn.Module):
    """
    Full world model:
      obs_t -> Encoder -> z_t
      (z_t, action_t) -> Dynamics -> z_{t+1}
      z_{t+1} -> Decoder -> obs_hat_{t+1}
    """

    def __init__(self, latent_dim=128, action_dim=3, img_height=90, img_width=160):
        super().__init__()

        self.encoder = Encoder(latent_dim=latent_dim)
        self.dynamics = LatentDynamics(latent_dim=latent_dim, action_dim=action_dim)
        self.decoder = Decoder(latent_dim=latent_dim, out_height=img_height, out_width=img_width)

    def forward(self, obs_t, action_t):
        """
        obs_t: (B, 3, H, W)
        action_t: (B, 2)
        return:
           obs_pred: (B, 3, H, W)
           z_t, z_next
        """

        z_t = self.encoder(obs_t)
        z_next = self.dynamics(z_t, action_t)
        obs_pred = self.decoder(z_next)

        return obs_pred, z_t, z_next
