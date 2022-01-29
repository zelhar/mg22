import argparse
import matplotlib.pyplot as plt
import my_torch_utils as ut
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import torch
import torch.distributions as dist
import torch.utils.data
import torchvision.utils as vutils
from math import pi, sin, cos, sqrt, log
from toolz import partial, curry
from torch import Tensor
from torch import nn, optim, distributions
from torch.nn.functional import one_hot
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast
# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import fnorm, replicate, logNorm
from my_torch_utils import scsimDataset

import scsim.scsim as scsim

data = torch.rand((1000, 2)) # 1k 2-dimensional data points 
data[:,0] = torch.sin(data[:,0])
data[:,1] = torch.cos(data[:,1])

# take a batch of 100 data points
x = data[:100]

loc=torch.tensor(0.3)
scale=torch.tensor(0.2)
var=scale.pow(2)
logvar=2 * scale.log()
g = dist.Normal(loc, scale)

g.log_prob(x) - ut.logNorm(x, loc, logvar )

# to compute elbo we need to sample l points for each input data point and
# calculate averages...

mu = torch.rand_like(x)
logvar = torch.rand_like(x)

l = 5 # 5 sampled points
eps = torch.randn(x.shape + (l,))
z = mu.unsqueeze(-1) + logvar.exp().unsqueeze(-1) * eps
# take mean over last axis
lz = z.mean(dim=2)
lz.shape


device = "cuda" if torch.cuda.is_available() else "cpu"
mse = nn.MSELoss(reduction="sum")
bce = nn.BCELoss(reduction = "sum")
kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class GVAE(nn.Module):
    """
    VAE with gaussian encoder and decoder.
    """

    def __init__(self, nz: int = 10, nh: int = 1024,
            imgsize : int = 28) -> None:
        super(GVAE, self).__init__()
        self.nin = nin = imgsize**2
        self.nz = nz
        self.imgsize = imgsize

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.Softplus(),
            nn.Linear(nh, nh),
            nn.Softplus(),
            nn.Linear(nh, nh),
            nn.Softplus(),
            #nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.Softplus(),
            nn.Linear(nh, nh),
            nn.Softplus(),
            nn.Linear(nh, nh),
            nn.Softplus(),
            #nn.Tanh(),
            #nn.Sigmoid(),
            #nn.Unflatten(1, (1, imgsize, imgsize)),
        )

        self.xmu = nn.Sequential(nn.Linear(nh, nin),
                #nn.Tanh(), 
                )
        self.xlogvar = nn.Sequential(nn.Linear(nh, nin),
                #nn.Tanh(),
                )
        self.zmu = nn.Sequential(nn.Linear(nh, nz),
                #nn.Tanh(),
                )
        self.zlogvar = nn.Sequential(nn.Linear(nh, nz),
                #nn.Tanh(),
                )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return mu, logvar

    def log_q_z_given_x(self, z, mu, logvar):
        """
        assume that z has one more dimension than mu,logvar
        b/c z is l samples from q( . | x)
        """
        temp = -0.5 * (
                (z - mu.unsqueeze(-1)).pow(2) / logvar.unsqueeze(-1).exp() + 
                logvar.unsqueeze(-1) + 
                log(2 * pi)) 
        return temp

    def log_p(self, z):
        """
        standard normap logprob of input z
        """
        temp = -0.5 * (log(2 * pi) + z.pow(2))
        return temp

    def log_p_x_given_z(self, x, mu, logvar):
        """
        logP(x | z) for the decoder, where
        z is samples of of size l from q(z | x)
        so we get l xmu and xlogvar for every x.
        """
        temp = -0.5 * (
                (x.unsqueeze(1) - mu).pow(2) / logvar.exp() + 
                logvar + 
                log(2 * pi)) 
        return temp

    def decode(self, z):
        h = self.decoder(z)
        mu = self.xmu(h)
        logvar = self.xlogvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, minisampleshape=(1,)):
        eps = torch.randn(mu.shape + minisampleshape).to(mu.device)
        sigma = (0.5 *logvar).exp()
        return mu.unsqueeze(-1) + sigma.unsqueeze(-1) * eps 

    def forward(self, x):
        zmu, zlogvar = self.encode(x)
        z = self.reparameterize(zmu, zlogvar, (1,)).to(x.device)
        w = z.permute(0,2,1)
        mu, logvar = self.decode(w)
        #return zmu, zlogvar, z
        return mu, logvar

def log_gaussian_prob(x : torch.Tensor, 
        mu : torch.Tensor = torch.tensor(0), 
        logvar : torch.Tensor = torch.tensor(0)
        ) -> torch.Tensor:
    """
    compute the log density function of a gaussian.
    user must make sure that the dimensions are aligned correctly.
    """
    return -0.5 * (
            log(2 * pi) + 
            logvar +
            (x - mu).pow(2) / logvar.exp()
            )


transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
    ])
test_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=False,
       download=True,
       transform=transform,
       ),
   batch_size=128,
   shuffle=True,
)
train_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=True,
       download=True,
       transform=transform,
       ),
   batch_size=128,
   shuffle=True,
)

imgs, labels = test_loader.__iter__().next()

model = GVAE()

hz = model.encoder(imgs)
hz.shape
zmu, zlogvar = model.encode(imgs)
z = model.reparameterize(zmu, zlogvar, (5,)).permute(0,2,1)
z.shape
mu = torch.tensor([0.5,0.9])
eps = torch.rand((2,5))

x = imgs.flatten(1)
x.shape
zmu, zlogvar = model.encode(x)
zmu.shape
zlogvar.shape
z = model.reparameterize(zmu, zlogvar)
z.shape
q1 = log_gaussian_prob(z.flatten(1), zmu, zlogvar)
q2 = model.log_q_z_given_x(z, zmu, zlogvar)
q1.shape
q2.shape
(q2.squeeze(2) - q1).abs().max()
(q2 - q1.unsqueeze(2)).abs().max()
xmu, xlogvar = model.decode(z.flatten(1))
xxmu, xxlogvar = model.decode(z.permute(0,2,1))
xmu.shape
xxlogvar.shape
xmu.unsqueeze(1).shape
xxlogvar.squeeze(1).shape
(xxmu.squeeze(1) - xmu).abs().max()
(xxlogvar - xlogvar.unsqueeze(1)).max()
z = z.flatten(1)
z.shape
pz1 = log_gaussian_prob(z.flatten(1))
pz2 = model.log_p(z)
(pz2.flatten(1) - pz1).abs().max()
px1 = log_gaussian_prob(x, xmu, xlogvar)
px2 = model.log_p_x_given_z(x, xxmu, xxlogvar)
(px1 - px2.squeeze(1)).abs().max()
kl = kld(zmu, zlogvar)
kl2 = q1.sum() - pz1.sum()
rec = -px1.sum()

l1loss = nn.L1Loss(reduction="sum")

z = model.reparameterize(zmu, zlogvar, (15,))
z.shape

xxmu, xxlogvar = model.decode(z.permute(0,2,1))
xxmu.shape

q2 = model.log_q_z_given_x(z, zmu, zlogvar)
q1 = log_gaussian_prob(z[:,:,0], zmu, zlogvar)
(q1 - q2[:,:,0]).abs().sum()
l1loss(q1, q2[:,:,0])

pz1 = log_gaussian_prob(z[:,:,0])
pz2 = model.log_p(z)
l1loss(pz1, pz2[:,:,0])

px1 = log_gaussian_prob(x, xxmu[:,0,:], xxlogvar[:,0,:])
px2 = model.log_p_x_given_z(x, xxmu, xxlogvar)
l1loss(px1, px2[:,0,:])
(px1 - px2[:,0,:]).abs().max()

kl = kld(zmu, zlogvar)

kl2 = (q2.sum() - pz2.sum())/15

rec = -px2.sum()/15
ms = mse(x, xxmu.mean(1))
(rec - ms).mean()


######## training procudure goes like this:
model.to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
optimizer.zero_grad()
model.train()

# input is a batch of images
imgs.shape
x = imgs.to(device).flatten(1)

# obtain zmu, zlogvar of q(z | x):
zmu, zlogvar = model.encode(x)
zmu.shape, zlogvar.shape

# sample noise from standard normal distribution, l=5 samples per data point
noise = torch.randn(zmu.shape + (5,)).to(device)
noise.shape
# reparametrize z: 
z = zmu.unsqueeze(-1) + zlogvar.unsqueeze(-1) * noise * 0.5
z.shape

# alternative: 
z = model.reparameterize(zmu, zlogvar, (5,))

loss_z = -model.log_q_z_given_x(z, zmu, zlogvar).mean(dim=2)
loss_z.shape
loss_noise = model.log_p(z).mean(dim=2)
loss_noise.shape # implicitly loss from zmu, zlogvar
w = z.permute(0,2,1)
w.shape
xmu, xlogvar = model.decode(w)
xmu.shape
xlogvar.shape
# so now we have l mus and logvar for every x input
loss_x = model.log_p_x_given_z(x, xmu, xlogvar).mean(dim=1)
loss_x.shape
loss = (loss_z.sum(1) + loss_noise.sum(1) + loss_x.sum(1)).mean()
loss.shape
loss.backward()
optimizer.step()

mu, logvar = model(x)
mu.squeeze(1).shape


zmu, zlogvar = model.encode(x)
z = model.reparameterize(zmu, zlogvar).to(x.device)
w = z.permute(0,2,1)
mu, logvar = model.decode(w)



model.to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
optimizer.zero_grad()
model.train()

#xs, ls = trainLoader.__iter__().next()
for epoch in range(10):
    for idx, (data, labels) in enumerate(train_loader):
        x = data.to(device).flatten(1)
        batch_size = data.shape[0]
        model.train()
        model.zero_grad()
        # obtain zmu, zlogvar of q(z | x):
        zmu, zlogvar = model.encode(x)
        # sample noise from the standard normal prior distribution, 
        # l=15 samples per data point
        l = 150
        #noise = torch.randn(zmu.shape + (l,)).to(device)
        # alternative: 
        z = model.reparameterize(zmu, zlogvar, (l,))
        # so now z has one more dimension then zmu, 
        # l z^i sampled for each
        # and calculate log_p(z)
        log_P_z =  model.log_p(z).mean(dim=2)
        # calculate log_q_z_given_x
        log_Q_z_given_x = model.log_q_z_given_x(z, zmu, zlogvar).mean(dim=2)
        # now calculate xmu, xlogvar (from the decoder)
        # need to make the last dim of z aligned with the decoder
        # so we geg l=15 results for each input
        w = z.permute(0,2,1)
        xmu, xlogvar = model.decode(w)
        # having problems with logvar
        xlogvar = torch.ones_like(xmu)
        # so now we have l mus and logvar for every x input
        # on the dim=1
        log_P_x_given_z = model.log_p_x_given_z(x, xmu, xlogvar).mean(dim=1)
        # calculate the loss
        loss = torch.sum(
                -log_P_x_given_z.sum(1) - log_P_z.sum(1) + 
                log_Q_z_given_x.sum(1)
                )/128
        loss.backward()
        optimizer.step()
        if idx % 300 == 0:
            print(
                "losses:\n",
                loss.item(),
                -log_P_z.sum().item()/128,
                -log_P_x_given_z.sum().item()/128,
                log_Q_z_given_x.sum().item()/128,
            )


model.cpu()
imgs, labels = iter(test_loader).next()

plot_images(imgs)

plot_images(denorm(imgs))

mu, logvar = model(imgs)
mu = mu.view(-1, 1, 28,28)
logvar = logvar.view(-1, 1, 28,28)

rec = mu + torch.randn_like(mu) * logvar

plot_images(denorm(rec))

log_P_x_given_z

#xs, ls = trainLoader.__iter__().next()
for epoch in range(10):
    for idx, (data, labels) in enumerate(train_loader):
        x = data.to(device).flatten(1)
        batch_size = data.shape[0]
        model.train()
        model.zero_grad()
        # obtain zmu, zlogvar of q(z | x):
        zmu, zlogvar = model.encode(x)
        # sample noise from the standard normal prior distribution, 
        # l=15 samples per data point
        l = 1
        #noise = torch.randn(zmu.shape + (l,)).to(device)
        # alternative: 
        z = model.reparameterize(zmu, zlogvar, (l,))
        # so now z has one more dimension then zmu, 
        # l z^i sampled for each
        # and calculate log_p(z)
        log_P_z =  model.log_p(z).mean(dim=2)
        # calculate log_q_z_given_x
        log_Q_z_given_x = model.log_q_z_given_x(z, zmu, zlogvar).mean(dim=2)
        # now calculate xmu, xlogvar (from the decoder)
        # need to make the last dim of z aligned with the decoder
        # so we geg l=15 results for each input
        w = z.permute(0,2,1)
        xmu, xlogvar = model.decode(w)
        # so now we have l mus and logvar for every x input
        # on the dim=1
        log_P_x_given_z = model.log_p_x_given_z(x, xmu, xlogvar).mean(dim=1)
        # calculate the loss
        loss = torch.sum(
                -log_P_x_given_z.sum(1) - log_P_z.sum(1) + 
                log_Q_z_given_x.sum(1)
                )/128
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(
                "losses:\n",
                loss.item(),
            )
