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
from torchvision import models

def save_reconstructs(encoder, decoder, x, epoch, nz=20, device="cuda"):
        with torch.no_grad():
            x = x.to(device)
            y = encoder(x).view(-1, nz, 1, 1)
            sample = decoder(y).cpu()
            sample = denorm(sample)
            #save_image(x.view(x.shape[0], 3, 28, 28),
            save_image(denorm(x),
                       'results/originals_' + str(epoch) + '.png')
            save_image(sample,
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
        #transforms.Resize(64),
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
nz = 20
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
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root="../data/", train=True, transform=transform,
        download=True),
    batch_size=batch_size,
    shuffle=True,
)
imgs, _ = iter(train_loader).next()


encoder = models.resnet101()
encoder.apply(init_weights)
lin = encoder.fc
new_lin = nn.Sequential(
        lin,
        nn.ReLU(),
        nn.Linear(lin.out_features, nz), 
        )
encoder.fc = new_lin
encoder(imgs).shape


class Generator(nn.Module):
    def __init__(self, nz=20, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Flatten(),
            nn.Unflatten(1, (nz, 1, 1)),
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
            # state size. (nc) x 64 x 64
            nn.Conv2d(nc, nc, 2, 2),
            # now ncx32x32
            #nn.MaxPool2d(2,2,0,1),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

decoder = Generator()
x = torch.rand(128, 20, 1, 1)
decoder(x).shape

discx = models.resnet18()
temp = discx.fc
newfc = nn.Sequential(
        temp,
        nn.ReLU(),
        nn.Linear(temp.out_features, 1),
        nn.Sigmoid()
        )
discx.fc = newfc


## GAN training
G = Generator()
D = models.resnet18()
temp = D.fc
newfc = nn.Sequential(
        temp,
        nn.ReLU(),
        nn.Linear(temp.out_features, 1),
        nn.Sigmoid()
        )
D.fc = newfc

D.to(device)
G.to(device)
optG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
bce = nn.BCELoss()


epochs = 9
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train D
        optD.zero_grad()
        #labels_real = torch.ones(batch_size, 1, 1, 1).to(device)
        #labels_fake = torch.zeros(batch_size, 1, 1, 1).to(device)
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
        #labels_real = torch.ones(batch_size, 1, 1, 1).to(device)
        #labels_fake = torch.zeros(batch_size, 1, 1, 1).to(device)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz, 1,1).to(device)
        y = G(z)
        pred = D(y)
        lossG = bce(pred, labels_real)
        lossG.backward()
        optG.step()
        if idx % 500 == 0:
            save_random_reconstructs(G, nz, "arr" + str(epoch) +":" + str(idx))
            print(
                    lossD.mean().item(),
                    lossG.mean().item()
                    )

save_random_reconstructs(G, nz, "arr")



## train AA
decoder = Generator()
x = torch.rand(128, 20, 1, 1)
decoder(x).shape

encoder = models.resnet101()
encoder.apply(init_weights)
lin = encoder.fc
new_lin = nn.Sequential(
        lin,
        nn.ReLU(),
        nn.Linear(lin.out_features, nz), 
        )
encoder.fc = new_lin
encoder(imgs).shape

encoder.to(device)
decoder.to(device)
optEn = optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, 0.999))
optDe = optim.Adam(decoder.parameters(), lr=lr, betas=(beta1, 0.999))


mse = nn.MSELoss()
epochs = 9
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train D
        optDe.zero_grad()
        optEn.zero_grad()
        x = data.to(device)
        y = encoder(x)
        y = y.view(batch_size, nz, 1, 1)
        rec = decoder(y)
        loss = mse(rec, x)
        loss.backward()
        optDe.step()
        optEn.step()
        if idx % 500 == 0:
            save_random_reconstructs(G, nz, "barr" + str(epoch) +":" + str(idx))
            print(
                    loss.mean().item(),
                    )
save_reconstructs(encoder, decoder, imgs, "hahaha")



### AAE Training
batch_size = 128
image_size = 32
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

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(32),
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
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data/", train=False, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)


imgs, _ = iter(train_loader).next()

encoder = models.resnet101()
encoder.conv1 = nn.Conv2d(1,64,(7,7),(2,2),(3,3),bias=False)
lin = encoder.fc
new_lin = nn.Sequential(
        lin,
        nn.ReLU(),
        nn.Linear(lin.out_features, nz), 
        )
encoder.fc = new_lin
encoder.apply(init_weights)
encoder(imgs).shape

decoder = Generator(nc=nc, nz=nz)
x = torch.rand(128, 20, 1, 1)
decoder.apply(init_weights)
decoder(x).shape

Dx = models.resnet18()
Dx.conv1 = nn.Conv2d(1,64,(7,7),(2,2),(3,3),bias=False)
temp = Dx.fc
newfc = nn.Sequential(
        temp,
        nn.ReLU(),
        nn.Linear(temp.out_features, 1),
        nn.Sigmoid()
        )
Dx.fc = newfc

Dz = models.resnet18()
#Dx.conv1 = nn.Conv2d(1,64,(7,7),(2,2),(3,3),bias=False)
Dz.conv1 = nn.Sequential(
        nn.Linear(20,32**2),
        nn.Unflatten(1, (1, 32, 32)),
        nn.Conv2d(1,64,(7,7),(2,2),(3,3),bias=False),
        )

temp = Dz.fc
newfc = nn.Sequential(
        temp,
        nn.ReLU(),
        nn.Linear(temp.out_features, 1),
        nn.Sigmoid()
        )
Dz.fc = newfc
Dz.apply(init_weights)
z = torch.rand(128, 20)
Dz(z).shape


encoder.to(device)
decoder.to(device)
Dz.to(device)
Dx.to(device)
optE = optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, 0.999))
optD = optim.Adam(decoder.parameters(), lr=lr, betas=(beta1, 0.999))
optDz = optim.Adam(Dz.parameters(), lr=lr, betas=(beta1, 0.999))
optDx = optim.Adam(Dx.parameters(), lr=lr, betas=(beta1, 0.999))
bce = nn.BCELoss()
mse = nn.MSELoss()


epochs = 9
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train encoder
        optE.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        x = data.to(device)
        z = encoder(x)
        pred = Dz(z)
        lossE = bce(pred, labels_real)
        lossE.backward()
        optE.step()
        # train encoder and decoder to autoencode
        optE.zero_grad()
        optD.zero_grad()
        x = data.to(device)
        z = encoder(x)
        recon = decoder(z)
        lossED = mse(recon, x)
        lossED.backward()
        optD.step()
        optE.step()
        # train Dz
        optDz.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        zreal = torch.randn(batch_size, nz).to(device)
        predZreal = Dz(zreal)
        lossZreal = bce(predZreal, labels_real)
        lossZreal.backward()
        xreal = data.to(device)
        zfake = encoder(xreal)
        predZfake = Dz(zfake)
        lossZfake = bce(predZfake, labels_fake)
        lossZfake.backward()
        lossZ = lossZfake + lossZreal
        optDz.step()
        if idx % 500 == 0:
            save_random_reconstructs(decoder, nz, "oooo" + str(epoch) +":" + str(idx))
            save_reconstructs(encoder, decoder, xreal, "oooo")
            print(
                    lossE.mean().item(),
                    lossED.mean().item(),
                    lossZ.mean().item()
                    )

save_random_reconstructs(decoder, nz, "oooo")

save_reconstructs(encoder, decoder, imgs, "oooo")

x.shape
encoder(x).shape
z = encoder(x)

decoder(z).shape
