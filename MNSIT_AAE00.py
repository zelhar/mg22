# This is an attempt to reconstruct some of the networks and results from the
# original paper (Adversarial Autoencoders, Makhzani et. al)
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

class SimpleNet(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        nh=1000,
        activation=nn.Identity(),
        unflatten=False,
        image_size=28,
    ):
        """
        nin, nout, nh are (self explanatory) the dimensions of the input,
        output, and the hidden layers.
        dropout should be a real value between 0 and 1.
        activation is the the activation function used on the output layer.
        """
        super(SimpleNet, self).__init__()
        self.nin = nin
        self.nz = nout
        self.nh = nh
        self.unflatten = unflatten
        self.image_size = image_size
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nout),
            activation,
        )

    def forward(self, input):
        output = self.main(input)
        if self.unflatten:
            output = nn.Unflatten(1, (1, self.image_size, self.image_size))(output)
        return output

class Net(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        nh=3000,
        dropout=0,
        activation=nn.Identity(),
        unflatten=False,
        image_size=28,
    ):
        """
        nin, nout, nh are (self explanatory) the dimensions of the input,
        output, and the hidden layers.
        dropout should be a real value between 0 and 1.
        activation is the the activation function used on the output layer.
        """
        super(Net, self).__init__()
        self.nin = nin
        self.nz = nout
        self.nh = nh
        self.unflatten = unflatten
        self.image_size = image_size
        self.dropout = 1.0 * dropout
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.dropout),
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nout),
            activation,
        )

    def forward(self, input):
        output = self.main(input)
        if self.unflatten:
            output = nn.Unflatten(1, (1, self.image_size, self.image_size))(output)
        return output


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize(64),
        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),  # 3 for RGB channels
        # transforms.Normalize(mean=(0.0,), std=(0.5,)),  # 3 for RGB channels
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

def trainAE(E, D, optE, optD, data, device, criterion=nn.BCELoss()):
    E.train()
    D.train()
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

def trainG(G, Dz, optG, data, device, criterion=nn.BCELoss()):
    batch_size = data.shape[0]
    Dz.eval()
    G.train()
    optG.zero_grad()
    labels_real = torch.ones(batch_size, 1).to(device)
    x = data.to(device)
    z = G(x)
    pred = Dz(z)
    lossG = criterion(pred, labels_real)
    lossG.backward()
    optG.step()
    return lossG

def trainDz(Dz, G, optDz, data, nz, device, criterion=nn.BCELoss()):
    # train Dz to disctiminate z 
    batch_size = data.shape[0]
    G.eval()
    Dz.train()
    optDz.zero_grad()
    labels_real = torch.ones(batch_size, 1).to(device)
    labels_fake = torch.zeros(batch_size, 1).to(device)
    zreal = torch.randn(batch_size, nz).to(device)
    predZreal = Dz(zreal)
    lossZreal = criterion(predZreal, labels_real)
    lossZreal.backward()
    xreal = data.to(device)
    zfake = G(xreal)
    predZfake = Dz(zfake)
    lossZfake = criterion(predZfake, labels_fake)
    lossZfake.backward()
    lossZ = lossZfake + lossZreal
    optDz.step()
    return lossZ


# parameters
lr1 = 1e-1
lr2 = 1e-2
lr3 = 1e-3
lr4 = 1e-4
momentumAE = 0.9
momentumGD = 0.1
batch_size = 128
image_size = 28
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 10
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"
bce = nn.BCELoss()
mse = nn.MSELoss()

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

E = Net(image_size ** 2, nz, nh=3000, dropout=0.1, activation=nn.Identity()).to(device)
E.apply(init_weights)
D = Net(
    nz, image_size ** 2, nh=3000, dropout=0.1, activation=nn.Sigmoid(), unflatten=True
).to(device)
D.apply(init_weights)
Dz = Net(nz, 1, nh=3000, dropout=0.2, activation=nn.Sigmoid()).to(device)
Dz.apply(init_weights)
optE = optim.Adam(E.parameters(), lr=1e-3)
optD = optim.Adam(D.parameters(), lr=1e-3)
optG = optim.SGD(E.parameters(), lr=5e-5, momentum=0.1)
optDz = optim.SGD(Dz.parameters(), lr=5e-5, momentum=0.1)

x, _ = iter(train_loader).next()
x = x.to(device)
y = E(x)
y.shape
Dz(y).shape
D(y).shape
save_random_reconstructs(D, nz, "apu" + str(-1) +":" + str(-1))
save_reconstructs(E, D, x, "apu")

start = 1
epochs = 21
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        # reconstruction phase
        lossED =  trainAE(E, D, optE, optD, data, device, bce)
        # Dz discriminator
        lossDz = trainDz(Dz, E, optDz, data, nz, device, bce)
        # E as generator
        lossG = trainG(E, Dz, optG, data, device, bce)
        if idx % 500 == 0:
            xreal = data.to(device)
            save_random_reconstructs(D, nz, "apu" + str(epoch) +":" + str(idx))
            save_reconstructs(E, D, xreal, "apu")
            print(
                    lossED.mean().item(),
                    lossG.mean().item(),
                    lossDz.mean().item()
                    )


E = Net(image_size ** 2, nz, nh=1000, dropout=0.0, activation=nn.Identity()).to(device)
E.apply(init_weights)
D = Net(
    nz, image_size ** 2, nh=1000, dropout=0.0, activation=nn.Sigmoid(), unflatten=True
).to(device)
D.apply(init_weights)

Dz = Net(nz, 1, nh=1000, dropout=0.0, activation=nn.Sigmoid()).to(device)
Dz.apply(init_weights)

optE = optim.Adam(E.parameters())
optG = optim.Adam(E.parameters())
optD = optim.Adam(E.parameters())

optDz = optim.Adam(Dz.parameters())

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

Dz = Discriminator(nz=nz).to(device)
Dz = Net(10, 1, nh=50, activation=nn.Sigmoid()).to(device)

Dz = SimpleNet(10, 1, nh=1500, activation=nn.Sigmoid()).to(device)
Dz = Discriminator(nz=nz).to(device)
Dz.apply(init_weights)
optDz = optim.Adam(Dz.parameters(), lr=1e-4, betas=(beta1, 0.999))

start = 0
epochs = 5
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # reconstruction phase
        #lossED =  trainAE(E, D, optE, optD, data, device, bce)
        # Dz discriminator
        lossZ = trainDz(Dz, E, optDz, data, nz, device, bce)
        # E as generator
        lossG = trainG(E, Dz, optG, data, device, bce)
        # train Dz to disctiminate z 
        if idx % 500 == 0:
            xreal = data.to(device)
            #save_random_reconstructs(D, nz, "apu" + str(epoch) +":" + str(idx))
            save_reconstructs(E, D, xreal, "apu")
            print(
                    str(epoch) + ":" + str(idx),
                    #lossED.mean().item(),
                    lossG.mean().item(),
                    lossZ.mean().item()
                    )

E = Net(image_size ** 2, nz, nh=3000, dropout=0.1, activation=nn.Identity()).to(device)
E.apply(init_weights)
D = Net(
    nz, image_size ** 2, nh=3000, dropout=0.1, activation=nn.Sigmoid(), unflatten=True
).to(device)
D.apply(init_weights)
optE = optim.Adam(E.parameters())
optD = optim.Adam(D.parameters())
optG = optim.Adam(E.parameters())

Dz = Net(nz, 1, nh=3000, dropout=0.2, activation=nn.Sigmoid()).to(device)
Dz = Discriminator(nz=nz).to(device)
Dz.apply(init_weights)
optDz = optim.Adam(Dz.parameters())

start = 0
epochs = 9
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        # reconstruction phase
        lossED =  trainAE(E, D, optE, optD, data, device, bce)
        # Dz discriminator
        lossDz = trainDz(Dz, E, optDz, data, nz, device, bce)
        # E as generator
        lossG = trainG(E, Dz, optG, data, device, bce)
        if idx % 500 == 0:
            xreal = data.to(device)
            save_random_reconstructs(D, nz, "apu" + str(epoch) +":" + str(idx))
            save_reconstructs(E, D, xreal, "apu")
            print(
                    str(epoch) + ":" + str(idx) + ":\n",
                    lossED.mean().item(),
                    lossG.mean().item(),
                    lossDz.mean().item()
                    )

start = 9
epochs = 15
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        # reconstruction phase
        lossED =  trainAE(E, D, optE, optD, data, device, bce)
        # Dz discriminator
        lossDz = trainDz(Dz, E, optDz, data, nz, device, bce)
        # E as generator
        #lossG = trainG(E, Dz, optG, data, device, bce)
        if idx % 500 == 0:
            xreal = data.to(device)
            save_random_reconstructs(D, nz, "apu" + str(epoch) +":" + str(idx))
            save_reconstructs(E, D, xreal, "apu")
            print(
                    str(epoch) + ":" + str(idx) + ":\n",
                    lossED.mean().item(),
                    #lossG.mean().item(),
                    lossDz.mean().item()
                    )

E = Net(image_size ** 2, nz, nh=3000, dropout=0.1, activation=nn.Identity()).to(device)
E.apply(init_weights)
D = Net(
    nz, image_size ** 2, nh=3000, dropout=0.1, activation=nn.Sigmoid(), unflatten=True
).to(device)
D.apply(init_weights)
optE = optim.Adam(E.parameters())
optD = optim.Adam(D.parameters())
optG = optim.Adam(E.parameters())

Dz = Net(nz, 1, nh=3000, dropout=0.2, activation=nn.Sigmoid()).to(device)
#Dz = Discriminator(nz=nz).to(device)
Dz.apply(init_weights)
optDz = optim.Adam(Dz.parameters())

start = 0
epochs = 250
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        # reconstruction phase
        lossED =  trainAE(E, D, optE, optD, data, device, bce)
        # Dz discriminator
        lossDz = trainDz(Dz, E, optDz, data, nz, device, bce)
        # E as generator
        lossG = trainG(E, Dz, optG, data, device, bce)
        if (epoch % 11 == 0) and (idx % 313 == 0):
            xreal = data.to(device)
            save_random_reconstructs(D, nz, "rapapu" + str(epoch) +":" + str(idx))
            save_reconstructs(E, D, xreal, "rapapu")
            print(
                    str(epoch) + ":" + str(idx) + ":\n",
                    lossED.mean().item(),
                    lossG.mean().item(),
                    lossDz.mean().item()
                    )

torch.save(D.state_dict(), './results/D_rapapu.weights.pth')
torch.save(E.state_dict(), './results/E_rapapu.weights.pth')
torch.save(Dz.state_dict(), './results/Dz_rapapu.weights.pth')
