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
            sample = torch.randn(64, nz).to(device)
            sample = model(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

class Discriminator(nn.Module):
    """
    Classifies real distribution and fake (generated) distribution.
    """
    def __init__(self, nin, nh1, nh2):
        super(Discriminator, self).__init__()
        self.nin = nin
        self.main = nn.Sequential(
                nn.Linear(nin, nh1),
                #nn.ReLU(),
                nn.LeakyReLU(),
                nn.Linear(nh1, nh2),
                #nn.ReLU(),
                nn.LeakyReLU(),
                nn.Linear(nh2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    """
    Encodes high dimensional data point into
    a low dimension latent space. 
    """

    def __init__(self, nin, nout, nh1, nh2):
        super(Encoder, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.ReLU(),
            nn.Linear(nh2, nout),
            #nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nin))


# the generator is the decoder
class Decoder(nn.Module):
    """
    Reconstructs a input from the latent, low dimensional space, into
    the original data space.
    """

    def __init__(self, nin, nout, nh1, nh2):
        super(Decoder, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.ReLU(),
            nn.Linear(nh2, nout),
            nn.Sigmoid()
            #nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


def trainG(G, D, optG, optD, x, criterion=nn.BCELoss(), device="cuda"):
    """
    Train G to 'fool' D on batch x.
    Maximize E[D(G(x))]
    """
    batch_size = x.shape[0]
    optG.zero_grad()
    optD.zero_grad()
    labels_real = torch.ones(batch_size, 1).to(device)
    #labels_fake = torch.zeros(batch_size, 1).to(device)
    z = G(x)
    pred = D(z)
    loss = criterion(pred, labels_real)
    loss.backward()
    optG.step()
    optG.zero_grad()
    optD.zero_grad()
    return loss

def trainD(D, optD, xreal, xfake, criterion=nn.BCELoss(), device="cuda"):
    """
    Train D to classify real and fake data.
    Minimize E[D(x)]
    """
    batch_size = xreal.shape[0]
    optD.zero_grad()
    labels_real = torch.ones(batch_size, 1).to(device)
    labels_fake = torch.zeros(batch_size, 1).to(device)
    predReal = D(xreal)
    predFake = D(xfake)
    lossReal = criterion(predReal, labels_real)
    lossFake = criterion(predFake, labels_fake)
    loss = lossReal + lossFake
    loss.backward()
    optD.step()
    optD.zero_grad()
    return loss

def trainAE(enc, dec, optE, optD, x, criterion=nn.BCELoss(), device="cuda"):
    """
    Train dec(enc(x)) to reconstruct x
    """
    optE.zero_grad()
    optD.zero_grad()
    z = enc(x)
    rec = dec(z)
    loss = criterion(rec, x)
    loss.backward()
    optD.step()
    optE.step()
    optE.zero_grad()
    optD.zero_grad()
    return loss

# parameters
nin = 28*28
nz = 64
batchSize = 256
epochs = 10
beta = 0.5
lr = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/", train=True, download=True, transform=transform
    ),
    batch_size=batchSize,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data/", train=False, transform=transform),
    batch_size=batchSize,
    shuffle=True,
)

Dx = Discriminator(nin, 128, 128).to(device)
Dz = Discriminator(nz, 128, 128).to(device)
Gx = Decoder(nz, nin, 4048, 2048).to(device)
Gz = Encoder(nin, nz, 4048 ,2048).to(device)
criterion = nn.BCELoss()
dx_optimizer = torch.optim.Adam(Dx.parameters(), )
dz_optimizer = torch.optim.Adam(Dz.parameters(), )
gx_optimizer = torch.optim.Adam(Gx.parameters(), )
gz_optimizer = torch.optim.Adam(Gz.parameters(), )
mse = nn.MSELoss()
bce = nn.BCELoss()

sample_dir = 'samples5'
epochs = 15
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train for reconstruction
        x = data.view(-1, nin).to(device)
        lossR = trainAE(Gz, Gx, gz_optimizer, gx_optimizer, x, bce, device)
        if idx % 3000 == 0:
            print(lossR.mean().item())

xs, ls = iter(test_loader).next()
save_reconstructs(Gz, Gx, xs, "foo")

save_random_reconstructs(Gx, nz, "foo")

for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train Gx against Dx
        #x = data.view(-1, nin).to(device)
        #z = Gz(x)
        z = torch.randn(batch_size, nz).to(device)
        lossGx = trainG(Gx, Dx, gx_optimizer, dx_optimizer, z, bce)
        if idx % 3000 == 0:
            print(lossGx.mean().item())

for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train Gz against Dz
        x = data.view(-1, nin).to(device)
        #z = Gz(x)
        lossGz = trainG(Gz, Dz, gz_optimizer, dz_optimizer, x, bce)
        if idx % 3000 == 0:
            print(lossGz.mean().item())

for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train Dx
        x = data.view(-1, nin).to(device)
        #z = Gz(x)
        #y = Gx(z)
        z = torch.randn(batch_size, nz).to(device)
        y = Gx(z)
        lossDx = trainD(Dx, dx_optimizer, x, y, bce)
        if idx % 3000 == 0:
            print(lossDx.mean().item())


for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train Dz
        x = data.view(-1, nin).to(device)
        zfake = Gz(x)
        #y = Gx(z)
        zreal = torch.randn(batch_size, nz).to(device)
        lossDz = trainD(Dz, dz_optimizer, zreal, zfake, bce)
        if idx % 3000 == 0:
            print(lossDz.mean().item())





for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train Dz
        x = data.view(-1, nin).to(device)
        zfake = Gz(x)
        #y = Gx(z)
        zreal = torch.randn(batch_size, nz).to(device)
        lossDz = trainD(Dz, dz_optimizer, zreal, zfake, bce)
        if idx % 3000 == 0:
            print(epoch, idx, lossDz.mean().item())
        # train Dx
        x = data.view(-1, nin).to(device)
        #z = Gz(x)
        #y = Gx(z)
        z = torch.randn(batch_size, nz).to(device)
        y = Gx(z)
        lossDx = trainD(Dx, dx_optimizer, x, y, bce)
        if idx % 3000 == 0:
            print(lossDx.mean().item())
        # train Gz against Dz
        x = data.view(-1, nin).to(device)
        #z = Gz(x)
        lossGz = trainG(Gz, Dz, gz_optimizer, dz_optimizer, x, bce)
        if idx % 3000 == 0:
            print(lossGz.mean().item())
        # train Gx against Dx
        #x = data.view(-1, nin).to(device)
        #z = Gz(x)
        z = torch.randn(batch_size, nz).to(device)
        lossGx = trainG(Gx, Dx, gx_optimizer, dx_optimizer, z, bce)
        if idx % 3000 == 0:
            print(lossGx.mean().item())

# GAN
Dx = Discriminator(nin, 4028, 4028).to(device)
Gx = Decoder(nz, nin, 4048, 4048).to(device)
gx_optimizer = torch.optim.Adam(Gx.parameters(),)
dx_optimizer = torch.optim.SGD(Dx.parameters(), lr=0.1, momentum=0.9)
mse = nn.MSELoss()
bce = nn.BCELoss()
criterion = nn.BCELoss()

for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train Gx
        labels_real = torch.ones(batch_size, 1).to(device)
        gx_optimizer.zero_grad()
        z = torch.randn(batch_size, nz).to(device)
        recon = Gx(z)
        pred = Dx(recon)
        lossGx = bce(pred, labels_real)
        lossGx.backward()
        gx_optimizer.step()
        if idx % 3000 == 0:
            print(lossGx.mean().item())
        # train Dx
        dx_optimizer.zero_grad()
        xreal = data.view(-1, nin).to(device)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        predreal = Dx(xreal)
        lossDxreal = bce(predreal, labels_real)
        lossDxreal.backward()
        z = torch.randn(batch_size, nz).to(device)
        xfake = Gx(z)
        predfake = Dx(xfake)
        lossDxfake = bce(predfake, labels_fake)
        lossDxfake.backward()
        lossDx = lossDxreal + lossDxfake
        dx_optimizer.step()
        if idx % 3000 == 0:
            print(lossDx.mean().item())






epochs = 15
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        # discriminator Dx turn
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        dx_optimizer.zero_grad()
        gx_optimizer.zero_grad()
        xreal = data.view(-1, nin).to(device)
        predreal = Dx(xreal)
        lossDreal = criterion(predreal, labels_real)
        D_x = predreal.mean().item()
        z = torch.randn(batch_size, nz).to(device)
        xfake = Gx(z)
        predfake = Dx(xfake)
        lossDfake = criterion(predfake, labels_fake)
        lossD = lossDreal + lossDfake
        lossD.backward()
        dx_optimizer.step()
        # generator Gx turn
        dx_optimizer.zero_grad()
        gx_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz).to(device)
        xfake = Gx(z)
        pred = Dx(xfake)
        lossG = criterion(pred, labels_real)
        lossG.backward()
        gx_optimizer.step()
        # discriminator Dz turn
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        dz_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        x = data.view(-1, nin).to(device)
        zreal = torch.randn(batch_size, nz).to(device)
        predreal = Dz(zreal)
        lossDreal = criterion(predreal, labels_real)
        D_x = predreal.mean().item()
        zfake = Gz(x)
        predfake = Dz(zfake)
        lossDfake = criterion(predfake, labels_fake)
        lossD = lossDreal + lossDfake
        lossD.backward()
        dz_optimizer.step()
        # generator Gz turn
        dz_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz).to(device)
        zfake = Gz(x)
        pred = Dz(zfake)
        lossG = criterion(pred, labels_real)
        lossG.backward()
        gx_optimizer.step()
