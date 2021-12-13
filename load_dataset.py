import os

from torchvision import transforms as transforms

from utils.simple_image_folder_dataset import SimpleImageFolderDataset

# constant because of layer of fixed size in the network
img_height = 128
img_width = 128


def load_dataset(percentage=100):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    image_folder_full_path = os.path.join(os.path.dirname(os.getcwd()), "data")
    feidegger = SimpleImageFolderDataset(percentage, image_folder_full_path, transform=transform)
    return feidegger
