# Autoencoder that flattens the image before encoding and unflatten it at the end of decoding

import torch.nn as nn


class LinearAutoEncoder(nn.Module):
    def __init__(self, img_height, img_width):
        super(LinearAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(nn.Flatten(),
                                     nn.Linear(in_features=img_height * img_width * 3, out_features=2048),
                                     nn.ReLU(True),
                                     nn.Linear(2048, 128))
        # Decoder Network
        self.decoder = nn.Sequential(nn.Linear(128, 2048),
                                     nn.ReLU(True),
                                     nn.Linear(in_features=2048, out_features=img_height * img_width * 3),
                                     nn.Tanh(),
                                     nn.Unflatten(1, (3, img_height, img_width)))

    def forward(self, x):
        x = self.encoder(x)
        out = self.decoder(x)
        return out
