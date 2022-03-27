# sources:
# https://github.com/psanch21/VAE-GMVAE
# https://arxiv.org/abs/1611.02648
# http://ruishu.io/2016/12/25/gmvae/
# https://github.com/RuiShu/vae-clustering
# https://github.com/hbahadirsahin/gmvae

import gdown
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyro
import pyro.distributions as pyrodist
import scanpy as sc
import seaborn as sns
import time
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as vutils
import umap
from abc import abstractmethod
from anndata.experimental.pytorch import AnnLoader
from importlib import reload
from math import pi, sin, cos, sqrt, log
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from toolz import partial, curry
from torch import nn, optim, distributions, Tensor
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO

# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights, buildNetwork
from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import scsimDataset
import my_torch_utils as ut


import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder

print(torch.cuda.is_available())

class Autoencoder(nn.Module):

    def __init__(self):
        super(Autoencoder,self).__init__()
        
        #self.encoder = nn.Sequential(
        #    nn.Conv2d(1, 6, kernel_size=5),
        #    nn.ReLU(True),
        #    nn.Conv2d(6,16,kernel_size=5),
        #    nn.ReLU(True))

        #self.decoder = nn.Sequential(             
        #    nn.ConvTranspose2d(16,6,kernel_size=5),
        #    nn.ReLU(True),
        #    nn.ConvTranspose2d(6,1,kernel_size=5),
        #    nn.ReLU(True),
        #    nn.Sigmoid())

        nh=1024
        nx=28**2
        nz=16
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                #nn.Sigmoid(),
                #nn.Linear(nh, nz),
                )
        self.decoder = nn.Sequential(
                nn.Linear(nz, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                nn.Sigmoid(),
                )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                #x = data.flatten(1).to(device)
                x = data.to(device)
                self.train()
                self.requires_grad_(True)
                xhat = self.forward(x)
                loss = nn.MSELoss()(xhat, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        loss.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None



class AE(nn.Module):
    """
    simple AE
    """
    def __init__(self, nx : int=28**2,
            nclusters : int=16,
            nz : int=16,
            nh : int=3000,
            adversarial_loss_scale : float=0.9,
            ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclusters
        self.encoder = nn.Sequential(
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                #nn.Tanh(),
                nn.LeakyReLU(),
                #nn.Sigmoid(),
                #nn.Linear(nh, nz),
                )
        self.decoder = nn.Sequential(
                nn.Linear(nz, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                nn.Sigmoid(),
                )
        return

    def encode(self, x):
        z = self.encoder(x)
        return z
    def decode(self, z):
        x = self.decoder(z)
        return x
    def reconstruction_loss(self, x, xhat):
        loss = nn.MSELoss(reduction="none")(x, xhat)
        loss = loss.sum(dim=-1).mean()
        return loss

    def forward(self, input):
        #x = nn.Flatten()(input)
        x = input.view(-1,28**2)
        z = self.encode(x)
        xhat = self.decode(z)
        recon_loss = self.reconstruction_loss(x, xhat)
        loss = recon_loss
        output = {
                "xhat" : xhat,
                "z" : z,
                "loss" : loss,
                }
        return output
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                #x = data.flatten(1).to(device)
                x = data.view(-1,28**2).to(device)
                output = self.forward(x)
                loss = output['loss']
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        loss.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



class VAE2(nn.Module):
    """
    VAE
    """
    def __init__(self, nx : int=28**2,
            nclusters : int=20,
            nz : int=20,
            nh : int=400,
            ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclusters
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        #self.encoder = nn.Sequential(
        #        nn.Flatten(),
        #        nn.Linear(nx, nh),
        #        nn.ReLU(),
        #        )
        #self.mu_z = nn.Sequential(
        #        nn.Linear(nh, nz),
        #        )
        #self.logvar_z = nn.Sequential(
        #        nn.Linear(nh, nz),
        #        )
        #self.decoder = nn.Sequential(
        #        nn.Linear(nz, nh),
        #        nn.ReLU(),
        #        nn.Linear(nh,nx),
        #        nn.Sigmoid(),
        #        )
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                #nn.BatchNorm1d(nh),
                nn.BatchNorm1d(nh, affine=False, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh, affine=False, track_running_stats=False),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        self.mu_z = nn.Sequential(
                nn.Linear(nh, nh),
                #nn.BatchNorm1d(nh),
                nn.BatchNorm1d(nh, affine=False, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z = nn.Sequential(
                nn.Linear(nh, nh),
                #nn.BatchNorm1d(nh),
                nn.BatchNorm1d(nh, affine=False, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                )
        self.decoder = nn.Sequential(
                nn.Linear(nz, nh),
                #nn.BatchNorm1d(nh),
                nn.BatchNorm1d(nh, affine=False, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                #nn.BatchNorm1d(nh),
                nn.BatchNorm1d(nh, affine=False, track_running_stats=False),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                nn.Sigmoid(),
                )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        return
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_z(h)
        logvar = self.logvar_z(h)
        std = (logvar * 0.5).exp()
        q_z = pyrodist.Normal(loc=mu, scale=std).to_event(1)
        return mu, logvar, q_z
    def decode(self, z):
        x = self.decoder(z)
        return x
    def forward(self, input):
        x = nn.Flatten()(input)
        mu, logvar, q_z = self.encode(x)
        #z = q_z.rsample()
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(mu)
        z = mu + std*eps
        kld_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        x_hat = self.decode(z)
        recon_loss = nn.MSELoss(reduction="none")(x_hat, x).sum(-1).mean()
        #logsigma_x = ut.softclip(self.logsigma_x, -2, 2)
        #sigma_x = logsigma_x.exp()
        #q_x = pyrodist.Normal(loc=x_hat, scale=sigma_x).to_event(1)
        #recon_loss = -q_x.log_prob(x).mean()
        #recon_loss = -ut.log_gaussian_prob(x, x_hat, sigma_x).sum(-1).mean()
        loss = recon_loss + kld_z
        output = {
                "mu" : mu,
                "logvar" : logvar,
                "z" : z,
                "kld_z" : kld_z,
                "recon_loss" : recon_loss,
                "loss" : loss,
                "q_z" : q_z,
                "x_hat" : x_hat,
                #"sigma_x" : sigma_x,
                #"q_x" : q_x,
                }
        return output
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                x = data.flatten(1).to(device)
                output = self.forward(x)
                loss = output['loss']
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        loss.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None
