# 43 based on examples from https://github.com/pytorch/examples
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
from torchvision import models


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

class SimpleDecoder(nn.Module):
    def __init__(self, image_size=28, nz=20, nc=1, ngf=64):
        super(SimpleDecoder, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Flatten(),
            nn.Linear(nz, image_size**2 * 8),
            nn.LeakyReLU(),
            nn.Linear(image_size**2 * 8, image_size**2 * 4),
            nn.LeakyReLU(),
            nn.Linear(image_size**2 * 4, image_size**2 * 2),
            nn.ReLU(),
            nn.Linear(image_size**2 * 2, image_size**2),
            nn.Unflatten(1, (1, image_size, image_size)),
            # nn.MaxPool2d(2,2,0,1),
            nn.Sigmoid()
            # nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)


class Decoder(nn.Module):
    def __init__(self, image_size=28, nz=20, nc=1, ngf=64):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Flatten(),
            nn.Unflatten(1, (nz, 1, 1)),
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 64 x 64
            nn.Conv2d(nc, nc, 2, 2),
            nn.ReLU(True),
            # now ncx32x32
            nn.Flatten(),
            nn.Linear(32 ** 2, image_size ** 2),
            nn.Unflatten(1, (1, image_size, image_size)),
            # nn.MaxPool2d(2,2,0,1),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Encoder(nn.Module):
    def __init__(self, image_size=28, nz=20, nc=1, ngf=64):
        super(Encoder, self).__init__()
        # self.R = models.resnet18()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(image_size**2 * nc, image_size**2 * nc * 4),
            nn.LeakyReLU(),
            nn.Linear(image_size**2 * nc * 4, image_size**2 * nc * 8),
            nn.LeakyReLU(),
            nn.Linear(image_size**2 * nc * 8, nz),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)


def ResnetEncoder(image_size=28, nz=20, nc=1, ngf=64):
    R = models.resnet18()
    if nc != 3:
        R.conv1 = nn.Conv2d(
            nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
    lin = R.fc
    R.fc = nn.Sequential(lin, nn.ReLU(), nn.Linear(lin.out_features, nz), nn.Tanh())
    return R

class SimpleDiscriminator(nn.Module):
    def __init__(self, nc=1, nz=20, ndf=64):
        super(SimpleDiscriminator, self).__init__()
        self.main = nn.Sequential(
            # nn.Upsample(size=(64, 64)),
            nn.Flatten(),
            nn.Linear(nz, 64 ** 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64 ** 2, 64 ** 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64 ** 2, 1),
            # state size. (ndf*2) x 16 x 16
            nn.Sigmoid(),
        )
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, nc=1, nz=20, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # nn.Upsample(size=(64, 64)),
            nn.Flatten(),
            nn.Linear(nz, 64 ** 2),
            nn.ReLU(),
            nn.Linear(64 ** 2, 64 ** 2),
            nn.Unflatten(1, (nc, 64, 64)),
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
            nn.Flatten(),
            nn.Sigmoid(),
        )

    def forward(self, input):
        return self.main(input)


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize(64),
        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),  # 3 for RGB channels
    ]
)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def save_reconstructs(
    encoder, decoder, x, epoch, nz=20, device="cuda", normalize=False
):
    with torch.no_grad():
        x = x.to(device)
        y = encoder(x)
        sample = decoder(y).cpu()
        if normalize:
            sample = denorm(sample)
            x = denorm(x)
        # save_image(x.view(x.shape[0], 3, 28, 28),
        save_image(x, "results/originals_" + str(epoch) + ".png")
        save_image(sample, "results/reconstructs_" + str(epoch) + ".png")


def save_random_reconstructs(model, nz, epoch, device="cuda", normalize=False):
    with torch.no_grad():
        # sample = torch.randn(64, nz).to(device)
        sample = torch.randn(64, nz).to(device)
        sample = model(sample).cpu()
        if normalize:
            sample = denorm(sample)
        save_image(sample, "results/sample_" + str(epoch) + ".png")


# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 20
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

E = ResnetEncoder().to(device)
D = Decoder().to(device)
Dz = Discriminator().to(device)
optE = optim.Adam(E.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optDz = optim.Adam(Dz.parameters(), lr=lr, betas=(beta1, 0.999))


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/", train=True, download=True, transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data/", train=False, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

x, _ = iter(train_loader).next()
x = x.to(device)
x.shape
y = E(x)
y.shape
Dz(y).shape
D(y).shape

bce = nn.BCELoss()
mse = nn.MSELoss()

def trainAE(E, D, optE, optD, data, device, criterion=nn.BCELoss()):
    optE.zero_grad()
    optD.zero_grad()
    x = data.to(device)
    z = E(x)
    recon = D(z)
    lossED = criterion(recon, x)
    lossED.backward()
    optD.step()
    optE.step()
    return lossED

def trainE(E, Dz, optE, data, device, criterion=nn.BCELoss()):
    batch_size = data.shape[0]
    optE.zero_grad()
    labels_real = torch.ones(batch_size, 1).to(device)
    x = data.to(device)
    z = E(x)
    pred = Dz(z)
    lossE = bce(pred, labels_real)
    lossE.backward()
    optE.step()
    return lossE

def trainDz(Dz, E, optDz, data, device, criterion=nn.BCELoss()):
    # train Dz to disctiminate z 
    batch_size = data.shape[0]
    optDz.zero_grad()
    labels_real = torch.ones(batch_size, 1).to(device)
    labels_fake = torch.zeros(batch_size, 1).to(device)
    zreal = torch.randn(batch_size, nz).to(device)
    predZreal = Dz(zreal)
    lossZreal = bce(predZreal, labels_real)
    lossZreal.backward()
    xreal = data.to(device)
    zfake = E(xreal)
    predZfake = Dz(zfake)
    lossZfake = bce(predZfake, labels_fake)
    lossZfake.backward()
    lossZ = lossZfake + lossZreal
    optDz.step()
    return lossZ

start = 9
epochs = 3
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train encoder
        optE.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        x = data.to(device)
        z = E(x)
        pred = Dz(z)
        lossE = bce(pred, labels_real)
        lossE.backward()
        optE.step()
        # train encoder and decoder to autoencode
        optE.zero_grad()
        optD.zero_grad()
        x = data.to(device)
        z = E(x)
        recon = D(z)
        lossED = bce(recon, x)
        lossED.backward()
        optD.step()
        optE.step()
        # train Dz to disctiminate z 
        optDz.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        zreal = torch.randn(batch_size, nz).to(device)
        predZreal = Dz(zreal)
        lossZreal = bce(predZreal, labels_real)
        lossZreal.backward()
        xreal = data.to(device)
        zfake = E(xreal)
        predZfake = Dz(zfake)
        lossZfake = bce(predZfake, labels_fake)
        lossZfake.backward()
        lossZ = lossZfake + lossZreal
        optDz.step()
        if idx % 500 == 0:
            save_random_reconstructs(D, nz, "oooo" + str(epoch) +":" + str(idx))
            save_reconstructs(E, D, xreal, "oooo")
            print(
                    lossE.mean().item(),
                    lossED.mean().item(),
                    lossZ.mean().item()
                    )


#############
E = Encoder().to(device)
D = SimpleDecoder().to(device)
Dz = SimpleDiscriminator().to(device)
E.apply(init_weights)
D.apply(init_weights)
Dz.apply(init_weights)
optE = optim.Adam(E.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optDz = optim.Adam(Dz.parameters(), lr=lr, betas=(beta1, 0.999))

start = 0
epochs = 6
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        lossED =  trainAE(E, D, optE, optD, data, device, bce)
        if idx % 500 == 0:
            xreal = data.to(device)
            save_random_reconstructs(D, nz, "oooo" + str(epoch) +":" + str(idx))
            save_reconstructs(E, D, xreal, "oooo")
            print(
                    lossED.mean().item(),
                    #lossE.mean().item(),
                    #lossZ.mean().item()
                    )

Dz.apply(init_weights)
optDz = optim.Adam(Dz.parameters(), lr=lr, betas=(beta1, 0.999))
start = 6
epochs = 16
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        #lossED =  trainAE(E, D, optE, optD, data, device, bce)
        lossE = trainE(E, Dz, optE, data, device, bce)
        #lossDz = trainDz(Dz, E, optDz, data, device, bce)
        if idx % 500 == 0:
            xreal = data.to(device)
            save_random_reconstructs(D, nz, "oooo" + str(epoch) +":" + str(idx))
            save_reconstructs(E, D, xreal, "oooo")
            print(
                    lossED.mean().item(),
                    lossE.mean().item(),
                    lossDz.mean().item()
                    )
