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

class VAE(nn.Module):
    """
    VAE with gaussian encoder and decoder.
    """

    def __init__(self, nz: int = 10, nh: int = 1024,
            imgsize : int = 28, is_Bernouli : bool = False) -> None:
        super(VAE, self).__init__()
        self.nin = nin = imgsize**2
        self.nz = nz
        self.imgsize = imgsize
        self.is_Bernouli = is_Bernouli

        self.encoder = nn.Sequential(
                nn.Flatten(),
                fclayer(nin, nh, False, 0.2, nn.LeakyReLU()),
                fclayer(nh,nh, True, 0.2, nn.LeakyReLU()),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                )

        self.decoder = nn.Sequential(
                fclayer(nz, nh, False, 0.2, nn.LeakyReLU()),
                fclayer(nh,nh, True, 0.2, nn.LeakyReLU()),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                )

        self.xmu = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nin),
                )

        # or if we prefer Bernoulu decoder
        self.bernouli = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nin),
                nn.Sigmoid(),
                )

        self.xlogvar = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nin),
                nn.Softplus(),
                )

        self.zmu = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )

        self.zlogvar = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Softplus(),
                )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, ):
        eps = torch.randn(mu.shape).to(mu.device)
        sigma = (0.5 *logvar).exp()
        return mu + sigma * eps

    def decode_bernouli(self, z):
        h = self.decoder(z)
        p = self.bernouli(h)
        return p, torch.tensor(0)

    def decode(self, z):
        h = self.decoder(z)
        mu = self.xmu(h)
        logvar = self.xlogvar(h)
        return mu, logvar

    def forward(self, x):
        zmu, zlogvar = self.encode(x)
        z = self.reparameterize(zmu, zlogvar)
        if self.is_Bernouli:
            xmu, xlogvar = self.decode_bernouli(z)
        else:
            xmu, xlogvar = self.decode(z)
        return xmu, xlogvar, z


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
x = imgs.flatten(1)
model = VAE()
model(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
mse = nn.MSELoss(reduction="sum")
bce = nn.BCELoss(reduction = "sum")
kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
l1loss = nn.L1Loss(reduction="sum")

model.is_Bernouli = True
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
        z = model.reparameterize(zmu, zlogvar)
        xmu, xlogvar = model.decode_bernouli(z)
        kld_loss = kld(zmu, zlogvar)
        bce_loss = bce(xmu, x)
        mse_loss = mse(xmu, x)
        p_x_given_z = log_gaussian_prob(x, xmu, xlogvar).sum(1).mean()
        p_z = log_gaussian_prob(z).sum(1).mean()
        q_z_given_x = log_gaussian_prob(z, zmu, zlogvar).sum(1).mean()
        loss = kld_loss + bce_loss
        loss.backward()
        optimizer.step()
        if idx % 300 == 0:
            print(
                "losses:\n",
                loss.item(),
            )


model.cpu()
imgs, labels = iter(test_loader).next()

plot_images(imgs)

rec, _, _ = model(imgs)

rec = rec.view(-1, 1,28,28)

plot_images(rec)

model = VAE(is_Bernouli=False)
model.is_Bernouli = False
model.to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
optimizer.zero_grad()
model.train()

#xs, ls = trainLoader.__iter__().next()
model.to(device)

for epoch in range(10):
    for idx, (data, labels) in enumerate(train_loader):
        x = data.to(device).flatten(1)
        batch_size = data.shape[0]
        model.train()
        model.zero_grad()
        # obtain zmu, zlogvar of q(z | x):
        zmu, zlogvar = model.encode(x)
        z = model.reparameterize(zmu, zlogvar)
        xmu, xlogvar = model.decode(z)
        kld_loss = kld(zmu, zlogvar)
        #bce_loss = bce(xmu, x)
        mse_loss = mse(xmu, x)
        log_p_x_given_z = log_gaussian_prob(x, xmu, xlogvar).sum(1).mean()
        log_p_z = log_gaussian_prob(z).sum(1).mean()
        log_q_z_given_x = log_gaussian_prob(z, zmu, zlogvar).sum(1).mean()
        #loss = kld_loss + mse_loss
        loss = kld_loss - log_p_x_given_z
        loss.backward()
        optimizer.step()
        if idx % 300 == 0:
            print(
                "losses:\n",
                loss.item(),
            )


model.cpu()
imgs, labels = iter(test_loader).next()

plot_images(imgs)

rec, _, _ = model(imgs)

rec = rec.view(-1, 1,28,28)

plot_images(rec)

