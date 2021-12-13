import torch.nn as nn


class FashionGeneator(nn.Module):
    def __init__(self, ngpu):
        super(FashionGeneator, self).__init__()
        self.ngpu = ngpu
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=4, out_channels=3, kernel_size=3, stride=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.generator(x)
