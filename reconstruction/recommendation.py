# add current working directory to package discovery path
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import spatial

sys.path.append(os.getcwd())

from load_dataset import img_height, img_width, load_dataset
from reconstruction.models.conv_autoencoder import ConvAutoEncoder
from reconstruction.models.linear_autoencoder import LinearAutoEncoder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Given a trained model and an item index suggest similar items')
    parser.add_argument("-m", "--model_type", type=int, default=1,
                        help="1 for Convolutional Autoencoder and 0 for Linear Autoencoder")
    parser.add_argument("-pa", "--model_file_path", type=str, default="reconstruction/autoencoder.pt",
                        help="Path file of the .pt file continating the trained model")
    parser.add_argument("-q", "--query_item", type=int, default=0,
                        help="Index of the item to query")
    parser.add_argument("-pe", "--dataset_percent", type=int, default=5,
                        help="percent of the dataset to load")

    args = vars(parser.parse_args())
    if args['model_type'] == 1:
        model = ConvAutoEncoder()
    else:
        model = LinearAutoEncoder(img_height, img_width)

    model.load_state_dict(torch.load(args['model_file_path']))

    feidegger = load_dataset(percentage=args['dataset_percent'])

    encodings = [model(element.view(-1, 3, img_height, img_width))[0] for element in feidegger]
    encodings_np = np.array([encoding.detach().numpy() for encoding in encodings])
    encodings_np = encodings_np.reshape(encodings_np.shape[0], -1)
    tree = spatial.KDTree(encodings_np)
    neightbors = tree.query(encodings_np[args['query_item']], 4)[1]
    neightbors = neightbors[1:]

    f, axarr = plt.subplots(2, 2, figsize=(8, 8))

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0, 0].imshow(feidegger[args['query_item']].permute(1, 2, 0))
    axarr[1, 0].imshow(feidegger[neightbors[0]].permute(1, 2, 0))
    axarr[0, 1].imshow(feidegger[neightbors[1]].permute(1, 2, 0))
    axarr[1, 1].imshow(feidegger[neightbors[2]].permute(1, 2, 0))
    axarr[0, 0].title.set_text('Query image')
    axarr[1, 0].title.set_text('Result 1')
    axarr[0, 1].title.set_text('Result 2')
    axarr[1, 1].title.set_text('Result 3')
    plt.show()
