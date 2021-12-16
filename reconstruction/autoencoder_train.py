# add current working directory to package discovery path
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from load_dataset import load_dataset
from reconstruction.models.conv_autoencoder import ConvAutoEncoder
from reconstruction.models.linear_autoencoder import LinearAutoEncoder


def autoencoder_train(dataset, model_type=1, epochs=15, batch_size=5,
                      img_height=128, img_width=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    if model_type == 1:
        model = ConvAutoEncoder()
    else:
        model = LinearAutoEncoder(img_height, img_width)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            _, outputs = model(batch)
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch)
            # compute accumulated gradients
            train_loss.backward()
            # perform parameter update based on current gradients
            optimizer.step()
            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()
        # compute the epoch training loss
        loss = loss / len(dataloader)
        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train autoencoder model and save it to a file')
    parser.add_argument("-m", "--model_type", type=int, default=1,
                        help="1 for Convolutional Autoencoder and 0 for Linear Autoencoder")
    parser.add_argument("-e", "--epochs", type=int, default=15,
                        help="Training epochs")
    parser.add_argument("-p", "--dataset_percent", type=int, default=100,
                        help="Percent of the dataset to use")
    parser.add_argument("-b", "--batch_size", type=int, default=10,
                        help="Batch size")

    args = vars(parser.parse_args())
    feidegger = load_dataset(args["dataset_percent"])

    model = autoencoder_train(feidegger, args["model_type"], args["epochs"], args["batch_size"])

    torch.save(model.state_dict(), "autoencoder.pt")
