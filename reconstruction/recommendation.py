import matplotlib.pyplot as plt
import numpy as np
import torch
from docopt import docopt
from scipy import spatial

from autoencoder_train_eval import img_height, img_width
from reconstruction.autoencoder_train_eval import load_dataset
from reconstruction.models.conv_autoencoder import ConvAutoEncoder
from reconstruction.models.linear_autoencoder import LinearAutoEncoder

if __name__ == '__main__':
    # usage pattern
    usage = '''
     
    Usage:
      reccommendation.py
      reccommendation.py --model_type <int> --model_file_path <path> --query_item <int> --dataset_percent <int>
     
    Options:
      -m, --model_type          1 for Convolutional Autoencoder and 0 for Linear Autoencoder
      -p, --model_file_path     Path file of the .pt file continating the trained model
      -q, --query_item          Index of the item to query
      -p, --dataset_percent     percent of the dataset to load
      
           
    '''

    args = docopt(usage)
    model_type = args['--model_type'] if args['--model_type'] else 1
    if model_type == 1:
        model = ConvAutoEncoder(img_height, img_width)
    else:
        model = LinearAutoEncoder(img_height, img_width)

    model_file_path = args['--model_file_path'] if args['--model_file_path'] else "autoencoder.pt"
    query_item = args['--query_item'] if args['--query_item'] else 0
    dataset_percent = args['--dataset_percent'] if args['--dataset_percent'] else 5

    model.load_state_dict(torch.load(model_file_path))

    feidegger = load_dataset(percentage=dataset_percent)

    encodings = [model(element.view(-1, 3, img_height, img_width))[0] for element in feidegger]
    encodings_np = np.array([encoding.detach().numpy() for encoding in encodings])
    encodings_np = encodings_np.reshape(encodings_np.shape[0], -1)
    tree = spatial.KDTree(encodings_np)
    neightbors = tree.query(encodings_np[query_item], 4)[1]
    neightbors = neightbors[1:]

    f, axarr = plt.subplots(2, 2, figsize=(8, 8))

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    axarr[0, 0].imshow(feidegger[query_item].permute(1, 2, 0))
    axarr[1, 0].imshow(feidegger[neightbors[0]].permute(1, 2, 0))
    axarr[0, 1].imshow(feidegger[neightbors[1]].permute(1, 2, 0))
    axarr[1, 1].imshow(feidegger[neightbors[2]].permute(1, 2, 0))
    axarr[0, 0].title.set_text('Query image')
    axarr[1, 0].title.set_text('Result 1')
    axarr[0, 1].title.set_text('Result 2')
    axarr[1, 1].title.set_text('Result 3')
    plt.show()
