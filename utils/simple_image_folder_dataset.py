import os

import torch
from skimage import io
from torch.utils.data import Dataset


class SimpleImageFolderDataset(Dataset):
    def __init__(self, full_path, transform=None):
        self.root_dir = full_path
        self.transform = transform
        self.file_list = os.listdir(full_path)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,
                                self.file_list[idx])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        return image
