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
#import torch.distributions as D
from torch import distributions
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
        batchnorm=True,
    ):
        """
        nin, nout, nh are (self explanatory) the dimensions of the input,
        output, and the hidden layers.
        dropout should be a real value between 0 and 1.
        activation is the the activation function used on the output layer.
        If batcnorm=True, batch normalization is applied to the hidden layers,
        after activation.
        """
        super(Net, self).__init__()
        self.nin = nin
        self.nz = nout
        self.nh = nh
        self.unflatten = unflatten
        self.image_size = image_size
        self.dropout = 1.0 * dropout
        self.batchnorm = nn.Identity()
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(num_features=nh)
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=self.dropout),
            nn.Linear(nin, nh),
            nn.ReLU(),
            #nn.BatchNorm1d(num_features=nh),
            self.batchnorm,
            nn.Linear(nh, nh),
            nn.ReLU(),
            #nn.BatchNorm1d(num_features=nh),
            self.batchnorm,
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


def save_random_reconstructs(model, nz, epoch, device="cuda", normalize=False,
        sampler=None):
    with torch.no_grad():
        if sampler == None:
            sample = torch.randn(64, nz).to(device)
        else:
            sample = sampler((64, )).to(device)
        sample = model(sample).cpu()
        if normalize:
            sample = denorm(sample)
        save_image(sample, "results/sample_" + str(epoch) + ".png")

def test(x, f=None):
    if f==None:
        return x
    else:
        return f(x)
test(10, lambda x: x*7 + 7)

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
    #zreal = torch.randn(batch_size, nz).to(device)
    zreal = gmm.sample((batch_size,)).to(device)
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
momentumAE = 0.9
momentumGD = 0.1
batch_size = 128
image_size = 28
lr1 = 1e-4
lr2 = 1e-5
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 2
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

### Creating a sampler from mixed model
# gaussian mixture of 10 2-d gaussians
mix = distributions.Categorical(probs=torch.ones(10,))
comp = distributions.Independent(
        distributions.Normal(torch.rand(10,2), torch.rand(10,2)), 1)
gmm = distributions.MixtureSameFamily(mix, comp)

#lows = torch.ones((5,3)) * torch.Tensor([0,2,4])
#lows
#highs = torch.ones((5,3)) * torch.Tensor([1,3,5])
#mix = distributions.Categorical(probs=torch.ones(5,))
#comp = distributions.Independent(
#        distributions.Uniform(lows, highs), 1)
#mm = distributions.MixtureSameFamily(mix, comp)

z = torch.randn(128, nz).to(device)
z.shape
z = gmm.sample((128,))
z.shape

prior = distributions.MultivariateNormal(torch.zeros(nz), torch.eye(nz))

### Model and Ooptimizers Initiation
# Encoder
E = Net(image_size ** 2, nz, nh=3000, dropout=0.1, activation=nn.Identity()).to(device)
E.apply(init_weights)
# Decoder
D = Net(
    nz, image_size ** 2, nh=3000, dropout=0.1, activation=nn.Sigmoid(), unflatten=True
).to(device)
D.apply(init_weights)
optE = optim.Adam(E.parameters(), lr=1e-3, )
optD = optim.Adam(D.parameters(), lr=1e-3)
#optG = optim.Adam(E.parameters())
optG = optim.SGD(E.parameters(), lr=1e-2, momentum=0.9)
# discriminator for the latent space
Dz = Net(nz, 1, nh=3000, dropout=0.2, activation=nn.Sigmoid()).to(device)
#Dz = Discriminator(nz=nz).to(device)
Dz.apply(init_weights)
#optDz = optim.Adam(Dz.parameters())
optDz = optim.SGD(Dz.parameters(), lr=1e-2, momentum=0.9)

### Training


optE = optim.Adam(E.parameters(), )
optD = optim.Adam(D.parameters(), )
optG = optim.Adam(E.parameters())
optDz = optim.Adam(Dz.parameters())
start = 0
epochs = 19
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        # reconstruction phase
        lossED =  trainAE(E, D, optE, optD, data, device, bce)
        # Dz discriminator
        lossDz = trainDz(Dz, E, optDz, data, nz, device, bce)
        # E as generator
        lossG = trainG(E, Dz, optG, data, device, bce)
        if (epoch % 2 == 0) and (idx % 313 == 0):
            xreal = data.to(device)
            save_random_reconstructs(D, nz, "gmm" + str(epoch) +":" + str(idx), sampler=gmm.sample)
            save_reconstructs(E, D, xreal, "gmm")
            print(
                    str(epoch) + ":" + str(idx) + ":\n",
                    lossED.mean().item(),
                    lossG.mean().item(),
                    lossDz.mean().item()
                    )



save_random_reconstructs(D, nz, "gmm" + str(epoch) +":" + str(idx), sampler=gmm.sample)



