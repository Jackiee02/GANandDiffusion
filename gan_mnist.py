import os
from multiprocessing import freeze_support

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 1. Weight initialization
# ------------------------------------------------------------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# ------------------------------------------------------------------------------
# 2. Generator definition (DCGAN-style for MNIST)
# ------------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, feature_map_size=64, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_map_size * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            nn.Conv2d(feature_map_size, channels, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# ------------------------------------------------------------------------------
# 3. Discriminator definition
# ------------------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, feature_map_size=64, channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size * 2, 1, 7, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.net(x).view(-1)

# ------------------------------------------------------------------------------
# 4. Simple moving average function
# ------------------------------------------------------------------------------
def moving_average(x, window_size):
    return np.convolve(x, np.ones(window_size) / window_size, mode='valid')

# ------------------------------------------------------------------------------
# 5. Main training, logging, sampling, and plotting routine
# ------------------------------------------------------------------------------
def main():
    # Hyperparameters and paths
    data_root    = "./data"
    samples_root = "./samples"
    os.makedirs(samples_root, exist_ok=True)

    batch_size   = 128
    image_size   = 28
    channels     = 1
    latent_dim   = 100
    lr           = 2e-4
    beta1, beta2 = 0.5, 0.999
    epochs       = 50
    num_workers  = 4  # Set for Windows spawn mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare MNIST dataloader
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Instantiate models, optimizers, and loss function
    netG = Generator(latent_dim=latent_dim, channels=channels).to(device)
    netD = Discriminator(channels=channels).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion  = nn.BCEWithLogitsLoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)
    real_label, fake_label = 1.0, 0.0

    # Containers for loss tracking
    lossD_list  = []
    lossG_list  = []
    epoch_lossD = []
    epoch_lossG = []

    # Training loop
    for epoch in range(1, epochs + 1):
        sumD, sumG, count = 0.0, 0.0, 0
        for i, (real_imgs, _) in enumerate(dataloader, 1):
            real_imgs = real_imgs.to(device)
            b_size = real_imgs.size(0)

            # ---- Update Discriminator ----
            netD.zero_grad()
            labels = torch.full((b_size,), real_label, device=device)
            output = netD(real_imgs)
            lossD_real = criterion(output, labels)
            lossD_real.backward()

            noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
            fake_imgs = netG(noise)
            labels.fill_(fake_label)
            output = netD(fake_imgs.detach())
            lossD_fake = criterion(output, labels)
            lossD_fake.backward()

            lossD = lossD_real + lossD_fake
            optimizerD.step()

            # ---- Update Generator ----
            netG.zero_grad()
            labels.fill_(real_label)
            output = netD(fake_imgs)
            lossG = criterion(output, labels)
            lossG.backward()
            optimizerG.step()

            # Record batch-level losses
            lossD_list.append(lossD.item())
            lossG_list.append(lossG.item())
            sumD += lossD.item()
            sumG += lossG.item()
            count += 1

            if i % 200 == 0:
                print(f"[Epoch {epoch}/{epochs}] "
                      f"Batch {i}/{len(dataloader)}  "
                      f"Loss_D: {lossD.item():.4f}  "
                      f"Loss_G: {lossG.item():.4f}")

        # Record epoch-level average losses
        epoch_lossD.append(sumD / count)
        epoch_lossG.append(sumG / count)

        # Save sample images from fixed noise
        with torch.no_grad():
            samples = netG(fixed_noise).cpu()
            samples = (samples + 1.0) / 2.0
            utils.save_image(
                samples,
                os.path.join(samples_root, f"epoch_{epoch:03d}.png"),
                nrow=8,
                normalize=False
            )

    # Save model checkpoints
    torch.save(netG.state_dict(), "generator.pth")
    torch.save(netD.state_dict(), "discriminator.pth")
    print("Training complete. Checkpoints saved.")

    # ---- Plot and save loss curves ----
    window_size = 200
    sm_lossD = moving_average(lossD_list, window_size)
    sm_lossG = moving_average(lossG_list, window_size)
    iters = np.arange(len(sm_lossD)) + window_size // 2

    plt.figure(figsize=(10, 8))
    # Smoothed batch-level losses
    plt.subplot(2, 1, 1)
    plt.plot(iters, sm_lossD, label='D loss (smoothed)')
    plt.plot(iters, sm_lossG, label='G loss (smoothed)')
    plt.title(f'Batch-level Loss (MA window={window_size})')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()

    # Epoch-level average losses
    plt.subplot(2, 1, 2)
    epochs_arr = np.arange(1, len(epoch_lossD) + 1)
    plt.plot(epochs_arr, epoch_lossD, marker='o', label='D loss (per epoch)')
    plt.plot(epochs_arr, epoch_lossG, marker='o', label='G loss (per epoch)')
    plt.title('Epoch-level Average Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_curves_smoothed_and_epoch.png')
    plt.close()
    print("Loss curves saved to loss_curves_smoothed_and_epoch.png")

# ------------------------------------------------------------------------------
# 6. Windows multiprocessing entry point
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    freeze_support()
    main()