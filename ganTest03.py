#43 based on examples from https://github.com/pytorch/examples
# https://github.com/L1aoXingyu/pytorch-beginner/https://github.com/L1aoXingyu/pytorch-beginner/
# https://github.com/bfarzin/pytorch_aae/blob/master/main_aae.py
# https://github.com/artemsavkin/aae/blob/master/aae.ipynb
import argparse
import os
import torch
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torch.distributions as D

def save_reconstructs(encoder, decoder, x, epoch, device="cuda"):
        with torch.no_grad():
            x = x.to(device)
            sample = decoder(encoder(x))
            save_image(x.view(x.shape[0], 1, 28, 28),
                       'results/originals_' + str(epoch) + '.png')
            save_image(sample.view(x.shape[0], 1, 28, 28),
                       'results/reconstructs_' + str(epoch) + '.png')

def save_random_reconstructs(model, nz, epoch, device="cuda"):
        with torch.no_grad():
            #sample = torch.randn(64, nz).to(device)
            sample = torch.randn(64, nz,1,1).to(device)
            sample = model(sample).cpu()
            sample = denorm(sample)
            sample = transforms.Resize(32)(sample)
            save_image(sample,
                       'results/sample_' + str(epoch) + '.png')

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(64),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),  # 3 for RGB channels
    ]
)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
device = "cuda" if torch.cuda.is_available() else "cpu"


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

G = Generator().to(device)
G.apply(init_weights)
D = Discriminator().to(device)
D.apply(init_weights)
optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

bce = nn.BCELoss()


train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root="../data/", train=True, transform=transform,
        download=True),
    batch_size=batch_size,
    shuffle=True,
)

imgs, _ = iter(train_loader).next()

imgs = denorm(imgs)
imgs = transforms.Resize(32)(imgs)

save_image(imgs, "results/cifar.png")

epochs = 15
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train D
        optD.zero_grad()
        labels_real = torch.ones(batch_size, 1, 1, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1, 1, 1).to(device)
        xreal = data.to(device)
        predreal = D(xreal)
        lossDreal = bce(predreal, labels_real)
        lossDreal.backward()
        z = torch.randn(batch_size, nz, 1,1).to(device)
        xfake = G(z)
        predfake = D(xfake)
        lossDfake = bce(predfake, labels_fake)
        lossDfake.backward()
        lossD = lossDreal + lossDfake
        optD.step()
        # train G
        optG.zero_grad()
        labels_real = torch.ones(batch_size, 1, 1, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1, 1, 1).to(device)
        z = torch.randn(batch_size, nz, 1,1).to(device)
        y = G(z)
        pred = D(y)
        lossG = bce(pred, labels_real)
        lossG.backward()
        optG.step()
        if idx % 500 == 0:
            print(
                    lossD.mean().item(),
                    lossG.mean().item()
                    )


save_random_reconstructs(G, nz, "bar")



z = torch.randn(batch_size, nz, 1,1).to(device)
xfake = G(z)

predfake = D(xfake)

labels_fake = torch.zeros(batch_size, 1, 1, 1).to(device)
bce(predfake, labels_fake)

#################################################################################

image_size = 32
batch_size = 128
bce = nn.BCELoss()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Grayscale(1),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # 3 for RGB channels
    ]
)

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root="../data/", train=True, transform=transform,
        download=True),
    batch_size=batch_size,
    shuffle=True,
)

imgs, _ = iter(train_loader).next()

imgs = denorm(imgs)
save_image(imgs, "results/cifarg.png")

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nz, nz * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz*4, nz*16),
            nn.BatchNorm1d(nz*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz*16, nz*16),
            nn.BatchNorm1d(nz*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nz*16, image_size**2),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(-1, nz)
        output = self.main(input)
        return output.view(-1, 1, image_size, image_size)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Linear(image_size**2, image_size**2 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(image_size**2 * 2, image_size**2 * 4),
            nn.BatchNorm1d(image_size**2 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(image_size**2 * 4, image_size**2 * 6),
            nn.BatchNorm1d(image_size**2 * 6),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(image_size**2 * 6, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.view(-1, image_size**2)
        output = self.main(input)
        return output.view(-1, 1)

image_size = 28
nz = 2
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),  # 3 for RGB channels
    ]
)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/", train=True, download=True, transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
)

G = Generator().to(device)
G.apply(init_weights)
D = Discriminator().to(device)
D.apply(init_weights)
optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

epochs = 150
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train D
        optD.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        xreal = data.to(device)
        predreal = D(xreal)
        lossDreal = bce(predreal, labels_real)
        lossDreal.backward()
        z = torch.randn(batch_size, nz, 1,1).to(device)
        xfake = G(z)
        predfake = D(xfake)
        lossDfake = bce(predfake, labels_fake)
        lossDfake.backward()
        lossD = lossDreal + lossDfake
        optD.step()
        # train G
        optG.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz, 1,1).to(device)
        y = G(z)
        pred = D(y)
        lossG = bce(pred, labels_real)
        lossG.backward()
        optG.step()
        if idx % 500 == 0:
            print(
                    epoch, idx, "\n",
                    lossD.mean().item(),
                    lossG.mean().item()
                    )


save_random_reconstructs(G, nz, "barf2")









imgs, _ = iter(train_loader).next()
imgs = denorm(imgs)

imgs.shape
z = torch.randn(batch_size, nz)

G(z.cuda()).shape

D(imgs.cuda()).shape
