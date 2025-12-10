import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentDynamics(nn.Module):
    """
    Simple latent dynamics model:
      z_{t+1} = f(z_t, a_t)

    z_t: (B, latent_dim)
    a_t: (B, action_dim)
    """

    def __init__(self, latent_dim: int = 128, action_dim: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        a: (B, action_dim)
        returns: (B, latent_dim)  ~ predicted z_{t+1}
        """
        x = torch.cat([z, a], dim=-1)
        z_next = self.net(x)
        return z_next
