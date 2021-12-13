import torch.nn as nn


class FashionDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(FashionDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(in_features=1152, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)
