import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Decoder: latent vector -> RGB image (C, H, W)

    Dostosowany do encodera:
    - Encoder daje cechy o kształcie (256, 6, 10) dla wejścia 3x90x160.
    - Tutaj startujemy z (latent_dim) i odtwarzamy obraz 3x90x160.
    """

    def __init__(self, latent_dim: int = 128, out_height: int = 90, out_width: int = 160):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_height = out_height
        self.out_width = out_width

        self.hidden_channels = 256
        self.h_feat = 6
        self.w_feat = 10

        # mapowanie latent -> mapy cech (256, 6, 10)
        self.fc = nn.Linear(latent_dim, self.hidden_channels * self.h_feat * self.w_feat)

        # odwrócone konwolucje
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 6x10 -> 12x20
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # 12x20 -> 24x40
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),    # 24x40 -> 48x80
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),     # 48x80 -> 96x160
            nn.Sigmoid(),  # wartości w [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, latent_dim)
        return: (B, 3, out_height, out_width) w [0,1]
        """
        h = self.fc(z)  # (B, 256*6*10)
        h = h.view(-1, self.hidden_channels, self.h_feat, self.w_feat)
        x = self.deconv_net(h)  # (B, 3, 96, 160)

        # przytnij wysokość do out_height (np. 90)
        if x.shape[2] > self.out_height:
            x = x[:, :, : self.out_height, :]
        return x
