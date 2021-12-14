# add current working directory to package discovery path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from dcgan_discriminator import FashionDiscriminator
from load_dataset import load_dataset
from reconstruction.models.conv_autoencoder import ConvAutoEncoder

# Size of z latent vector (i.e. size of generator input)
latent_space_dim = 128

# Learning rate for optimizers
# lr = 0.0002
lr = 1E-5

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 0


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def dcgan_train(dataset, epochs=15, batch_size=5):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Create the generator
    generator = ConvAutoEncoder().to(device)

    # Handle multi-gpu if desired
    # if (device.type == 'cuda') and (ngpu > 1):
    #    generator = nn.DataParallel(generator, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    # generator.apply(weights_init)

    # Create the Discriminator
    discriminator = FashionDiscriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        discriminator = nn.DataParallel(discriminator, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    discriminator.apply(weights_init)

    # Initialize BCELoss function
    criterion = nn.BCELoss()
    criterion2 = nn.MSELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, latent_space_dim, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    # optimizerD = optim.Adam(discriminator.parameters())
    # optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=0.1)

    # Training Loop

    # Lists to keep track of progress
    # img_list = []
    generator_losses = []
    discriminator_losses = []
    iters = 0

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # log(D(x)) = ability of D to classify real images as real
            # log(1 - D(G(z))) = ability of D to classify fake images as fake
            ###########################
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            real_cpu = data.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            # Forward pass real batch through D
            output_d = discriminator(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output_d, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output_d.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            # noise = torch.randn(b_size, latent_space_dim, 1, 1, device=device)
            # Generate fake image batch with G
            _, fake = generator(data)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output_d = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output_d, label)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            errD_fake.backward()
            D_G_z1 = output_d.mean().item()
            # Compute error of D as sum over the fake and the real batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output_d = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            # errG = 0.001* criterion(output_d, label) + 0.999*criterion2(fake,data)
            errG = criterion2(fake, data)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output_d.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            generator_losses.append(errG.item())
            discriminator_losses.append(errD.item())

            iters += 1

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(generator_losses, label="G")
    plt.plot(discriminator_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()

    # Grab a batch of real images from the dataloader
    real_batch = next(iter(dataloader))

    # Plot the real images
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    img = np.transpose(feidegger[0], (1, 2, 0)).detach().numpy()
    img = (img + 1) / 2
    img = (img * 255).astype(np.uint8)
    plt.imshow(img)

    # Plot the fake images from the last epoch
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Fake Images")
    img = generator(feidegger[0][None, :])[1].detach().numpy().reshape(3, 128, 128)
    img = (img + img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


feidegger = load_dataset(0.05)
dcgan_train(feidegger, epochs=100, batch_size=5)
