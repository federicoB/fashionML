import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    def __init__(self, img_height, img_width):
        super(ConvAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features=7200, out_features=128)
        )
        # Decoder Network
        self.decoder = nn.Sequential(
            nn.Linear(in_features=128, out_features=7200),
            nn.Unflatten(dim=1, unflattened_size=(32, 15, 15)),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=3, stride=2, output_padding=1))

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out
