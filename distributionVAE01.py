import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision.utils as vutils
from math import pi, sin, cos, sqrt, log
from toolz import partial, curry
from torch import Tensor
from torch import nn, optim, distributions
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast

from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs


device = "cuda" if torch.cuda.is_available() else "cpu"

class GaussVAE(nn.Module):
    """
    VAE with Gauss (as opposed to Bernouli) decoder
    """
    def __init__(self, nin, nz, nh1, nh2, nh3, nh4):
        super(GaussVAE, self).__init__()

        self.nin = nin
        self.nz = nz

        self.encoder = nn.Sequential(
                nn.Linear(nin, nh1), 
                nn.ReLU(),
                nn.Linear(nh1, nh2),
                nn.ReLU(),
                )

        self.decoder = nn.Sequential(
                nn.Linear(nz, nh3),
                nn.ReLU(),
                nn.Linear(nh2, nh4),
                #nn.ReLU(),
                nn.Tanh(),
                #nn.Linear(nh4, nin),
                #nn.Sigmoid(),
                )

        self.mumap = nn.Linear(nh2, nz)
        self.logvarmap = nn.Linear(nh2, nz)

        self.dmu = nn.Linear(nh4, nin)
        self.dlv = nn.Linear(nh4, nin)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mumap(h)
        logvar = self.logvarmap(h)
        return mu, logvar
        #h1 = F.relu(self.fc1(x))
        #return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.decoder(z)
        mu = self.dmu(h)
        logvar = self.dlv(h)
        return mu, logvar

    def forward(self, x):
        x = x.view(-1, self.nin)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        dmu, dlogvar = self.decode(z)
        recon = self.reparameterize(dmu, dlogvar)
        return dmu, dlogvar, mu, logvar, recon


# Batch size during training
batch_size = 128
nz = 4
#nz = 20
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
device = "cuda" if torch.cuda.is_available() else "cpu"
#bce = nn.BCELoss()
mse = nn.MSELoss()
bce = nn.BCELoss(reduction="sum")
kld = lambda mu, logvar : -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#model = VAE(nin=2, nz=nz, nh1=2*1024, nh2=2*512, nh3=2*512, nh4=2*1024).to(device)
model = GaussVAE(nin=2, nz=nz, nh1=2*1024, nh2=2*512, nh3=2*512, nh4=2*1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)


def fnorm(x, mu=0, s=1):
    """
    normal distribution function.
    """
    x = (x - mu) / s
    x = x ** 2
    x = -0.5 * x
    x = x.exp()
    x = x / (s * 2 * sqrt(pi) )
    return torch.sum(torch.log(x))

for epoch in range(9000):
    model.zero_grad()
    x = torch.randn((128,2)).to(device)
    m,s , mu, logvar, r = model(x)
    recon = m + torch.randn_like(m) * (0.5 * s).exp()
    #loss_recon = bce(recon, x)
    loss_recon = -fnorm(x, m, (0.5 * s).exp())
    loss_kld = kld(mu, logvar)
    loss = loss_kld + loss_recon
    loss.backward()
    optimizer.step()
    if epoch % 250 == 0:
        print(
                "loss_kld = ", loss_kld.item(),
                "loss_recon = ",
                loss_recon.item(),
                )


x = torch.randn((3280,2)).to(device)
a,b,c,d,r = model(x)

z = a + torch.randn_like(a) * (0.5 * b).exp()

xs = x.detach().cpu().numpy()
zs = z.detach().cpu().numpy()
u = zs[:,0]
v = zs[:,1]
plt.scatter(u,v)

ux = xs[:,0]
vx = xs[:,1]
plt.scatter(ux,vx)

plt.cla()

# now shrinking test
nz=1
model = GaussVAE(nin=2, nz=nz, nh1=2*1024, nh2=2*512, nh3=2*512, nh4=2*1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)


for epoch in range(3000):
    model.zero_grad()
    x = torch.randn((128,2)).to(device)
    m,s , mu, logvar, r = model(x)
    recon = m + torch.randn_like(m) * (0.5 * s).exp()
    #loss_recon = bce(recon, x)
    loss_recon = -fnorm(x, m, (0.5 * s).exp())
    loss_kld = kld(mu, logvar)
    loss = loss_kld + loss_recon
    loss.backward()
    optimizer.step()
    if epoch % 250 == 0:
        print(
                "loss_kld = ", loss_kld.item(),
                "loss_recon = ",
                loss_recon.item(),
                )


x = torch.randn((3280,2)).to(device)
a,b,c,d,r = model(x)

z = a + torch.randn_like(a) * (0.5 * b).exp()

xs = x.detach().cpu().numpy()
zs = z.detach().cpu().numpy()
u = zs[:,0]
v = zs[:,1]
plt.scatter(u,v)

ux = xs[:,0]
vx = xs[:,1]
plt.scatter(ux,vx)

plt.cla()

# MNIST test
nz=2
nin=28**2

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
    ])


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data", train=True, download=True, transform=transform
    ),
    batch_size=128,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=False, transform=transform),
    batch_size=128,
    shuffle=True,
)


model = GaussVAE(28**2, nz, 2*1024, 2*512, 2*512, 2*1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)

mse = nn.MSELoss(reduction="sum")
bce = nn.BCELoss(reduction="sum")

for epoch in range(8):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        x = data.view(-1,nin).to(device)
        model.train()
        model.requires_grad_(True)
        optimizer.zero_grad()
        dmu, dvar, mu, logvar, recon = model(x)
        #loss_recon = mse(recon, x)
        s = (0.5 * dvar).exp()
        loss_recon = -torch.sum(-0.5 * ((x - dmu) / s).pow(2) - dvar - 0.5 * log(2 * pi))
        #m,s , mu, logvar, r = model(x)
        #loss_recon = -fnorm(x, m, (0.5 * s).exp())
        #loss_recon = -fnorm(x, dmu, (0.5 * dvar).exp())
        loss_kld = kld(mu, logvar)
        loss = loss_kld + loss_recon
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print("losses:\n",
                    "reconstruction loss:", loss_recon.item(),
                    "kld:", loss_kld.item()
                    )


imgs, labels = test_loader.__iter__().next()

plot_images(imgs)

plot_images(denorm(imgs))

dm,dv, m, v, ximgs = model(imgs.cuda())

ximgs = ximgs.view(-1, 1, 28, 28)

plot_images(ximgs.cpu(), nrow=16)

plot_images(denorm(ximgs).cpu(), nrow=16)

