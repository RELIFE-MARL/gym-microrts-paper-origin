from torch import Tensor, nn

from .utils import Transpose, layer_init


class Decoder(nn.Module):
    def __init__(self, output_channels: int):
        super().__init__()

        self.deconv = nn.Sequential(
            layer_init(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
            ),
            nn.ReLU(),
            layer_init(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
            ),
            nn.ReLU(),
            layer_init(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            ),
            nn.ReLU(),
            layer_init(
                nn.ConvTranspose2d(
                    32, output_channels, 3, stride=2, padding=1, output_padding=1
                )
            ),
            Transpose((0, 2, 3, 1)),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.deconv(x)
