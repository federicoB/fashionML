import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

from reconstruction.models.conv_autoencoder import ConvAutoEncoder
from utils.simple_image_folder_dataset import SimpleImageFolderDataset

epochs = 15

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
    # transforms.Normalize([0.5], [0.5])
])

image_folder_full_path = os.path.join(os.path.dirname(os.getcwd()), "data")
feidegger = SimpleImageFolderDataset(image_folder_full_path, transform=transform)
feidegger_subset = torch.utils.data.Subset(feidegger, list(range(0, 40)))

img_height = feidegger[0].shape[1]
img_width = feidegger[0].shape[2]

dataloader = DataLoader(feidegger_subset, batch_size=1, shuffle=False, num_workers=0)

model = ConvAutoEncoder(img_height, img_width)
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()
model.train()
for epoch in range(epochs):
    loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(batch)
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

img = model(feidegger[12].view(-1, 3, img_height, img_width))
# remove batch dimension and permute dimension in a shape suitable for imshow
img = img.reshape(3, img_height, img_width).permute(1, 2, 0).detach().numpy()
original_image = feidegger[12].permute(1, 2, 0)
print("original image shape", original_image.shape)
plt.imshow(original_image)
plt.show()
plt.imshow(img)

plt.show()
