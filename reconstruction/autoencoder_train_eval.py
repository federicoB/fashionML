import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from docopt import docopt
from torch.utils.data import Dataset, DataLoader

from reconstruction.models.conv_autoencoder import ConvAutoEncoder
from reconstruction.models.linear_autoencoder import LinearAutoEncoder
from utils.simple_image_folder_dataset import SimpleImageFolderDataset

# constant because of layer of fixed size in the network
img_height = 128
img_width = 128


def load_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        # TODO normalize?
        # transforms.Normalize([0.5], [0.5])
    ])
    image_folder_full_path = os.path.join(os.path.dirname(os.getcwd()), "data")
    feidegger = SimpleImageFolderDataset(image_folder_full_path, transform=transform)
    return feidegger


def autoencoder_train_eval(dataset, model_type=1,
                           subdataset_size=40, epochs=15, batch_size=5,
                           img_height=128, img_width=128):
    dataset_subset = torch.utils.data.Subset(dataset, list(range(0, subdataset_size)))

    dataloader = DataLoader(dataset_subset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = None
    if model_type == 1:
        model = ConvAutoEncoder(img_height, img_width)
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
    # usage pattern
    usage = '''
     
    Usage:
      autoencoder_train_eval.py
      autoencoder_train_eval.py --model_type <int> --epochs <epochs> --subdataset_size <size> --batch_size <size>
      autoencoder_train_eval.py --e <epochs> --s <size> --b <size>
     
    Options:
      -m, --model_type          1 for Convolutional Autoencoder and 0 for Linear Autoencoder
      -e, --epochs              Training epochs
      -s, --subdataset_size     Subsize of the dataset to use
      -b, --batch_size          Batch size
      
           
    '''

    args = docopt(usage)
    model_type = args['--model_type'] if args['--model_type'] else 1
    epochs = args['--epochs'] if args['--epochs'] else 15
    subdataset_size = args['--subdataset_size'] if args['--subdataset_size'] else 200
    batch_size = args['--batch_size'] if args['--batch_size'] else 10

    feidegger = load_dataset()

    model = autoencoder_train_eval(feidegger, model_type, subdataset_size, epochs, batch_size)

    torch.save(model.state_dict(), "autoencoder.pt")
