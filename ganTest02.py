# based on examples from https://github.com/pytorch/examples
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

def save_reconstructs(encoder, decoder, x, epoch):
        with torch.no_grad():
            x = x.to(device)
            sample = decoder(encoder(x))
            save_image(x.view(x.shape[0], 1, 28, 28),
                       'results/originals_' + str(epoch) + '.png')
            save_image(sample.view(x.shape[0], 1, 28, 28),
                       'results/reconstructs_' + str(epoch) + '.png')

def save_random_reconstructs(model, nz, epoch):
        with torch.no_grad():
            sample = torch.randn(64, nz).to(device)
            sample = model(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')



# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,),   # 3 for RGB channels
                                     std=(0.5,))])                

class Discriminator(nn.Module):
    """
    Classifies real distribution and fake (generated) distribution.
    """
    def __init__(self, nin, nh1, nh2):
        super(Discriminator, self).__init__()
        self.nin = nin
        self.main = nn.Sequential(
                nn.Linear(nin, nh1),
                nn.ReLU(),
                nn.Linear(nh1, nh2),
                nn.ReLU(),
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
            nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# parameters
nin = 28*28
nz = 64
batchSize = 32
epochs = 10
beta = 0.5
lr = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

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

bce = nn.BCELoss()
kld = lambda mu, logvar : -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
mse = nn.MSELoss()

encoder = Encoder(nin, nz, nin*4, 1024).to(device)
#decoder = Decoder(nz, nin, 1024, nin*4).to(device)
decoder = Decoder(nz, nin, 256, 256).to(device)
disGauss = Discriminator(nz, 1024, 512).to(device)
#disData = Discriminator(nin, nin*4, 1024).to(device)
disData = Discriminator(nin, 256, 256).to(device)
optimGauss = optim.Adam(disGauss.parameters(), lr=lr, betas=(beta, 0.999) )
optimData = optim.Adam(disData.parameters(), lr=lr, betas=(beta, 0.999))
optimEnc = optim.Adam(encoder.parameters(), lr=lr, betas=(beta, 0.999))
optimDec = optim.Adam(decoder.parameters(), lr=lr, betas=(beta, 0.999))


decoder = Decoder(nz, nin, 256, 256).to(device)
disData = Discriminator(nin, 256, 256).to(device)
optimDec = optim.Adam(decoder.parameters(), lr=lr, betas=(beta, 0.999))
optimData = optim.Adam(disData.parameters(), lr=lr, betas=(beta, 0.999))

epochs = 13
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        # train the discriminator
        optimDec.zero_grad()
        optimData.zero_grad()
        # train with real batch
        xreal = data.view(-1, nin).to(device)
        pred_real = disData(xreal)
        errDreal = bce(pred_real, labels_real)
        errDreal.backward()
        D_x = pred_real.mean().item()
        # train with fake
        z = torch.randn(batch_size, nz).to(device)
        xfake = decoder(z)
        pred_fake = disData(xfake.detach())
        errDfake = bce(pred_fake, labels_fake)
        errDfake.backward()
        D_z = pred_fake.mean().item()
        optimData.step()
        errD = errDreal + errDfake
        # train generator to maximiz logDG
        pred = disData(xfake)
        errG = bce(pred, labels_real)
        #errG = -1 * bce(pred, labels_fake)
        optimDec.step()
        if idx % 200 == 0:
            print(D_x, D_z, DG_z, errD.mean().item(), errD.mean().item(), errG.mean().item())

        break
    break

# Discriminator
D = nn.Sequential(
    nn.Linear(nin, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid())

latent_size = 64
hidden_size = 256
image_size = nin

# Generator 
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

D = Discriminator(nin, 256, 256).to(device)
G = Decoder(nz, nin, 256, 256).to(device)
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)

sample_dir = 'samples2'
epochs = 13
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        # discriminator turn
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        xreal = data.view(-1, nin).to(device)
        predreal = D(xreal)
        lossDreal = criterion(predreal, labels_real)
        D_x = predreal.mean().item()
        z = torch.randn(batch_size, nz).to(device)
        xfake = G(z)
        predfake = D(xfake)
        lossDfake = criterion(predfake, labels_fake)
        lossD = lossDreal + lossDfake
        lossD.backward()
        d_optimizer.step()
        D_z = predfake.mean().item()
        # generator
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz).to(device)
        xfake = G(z)
        pred = D(xfake)
        lossG = criterion(pred, labels_real)
        lossG.backward()
        g_optimizer.step()
        DG_z = pred.mean().item()
        if idx % 200 == 0:
            print(D_x, D_z, DG_z, lossD.mean().item(), lossG.mean().item(), )
            fake_images = G(z)
            fake_images = fake_images.view(data.size(0), 1, 28, 28)
            #fake_images = denorm(fake_images)
            save_image(denorm(fake_images.data), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))












save_random_reconstructs(G, nz, 334)
