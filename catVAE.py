import gdown
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyro
import pyro.distributions as dist
import scanpy as sc
import seaborn as sns
import time
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
from abc import abstractmethod
from anndata.experimental.pytorch import AnnLoader
from importlib import reload
from math import pi, sin, cos, sqrt, log
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from toolz import partial, curry
from torch import nn, optim, distributions, Tensor
from torch.nn.functional import one_hot
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

print(torch.cuda.is_available())

import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder


class BaseVAE(nn.Module):
    def __init__(self, 
            nin : int,
            nz : int,
            nh : int,) -> None:
        super(BaseVAE, self).__init__()
        self.nin = nin
        self.nz = nz
        self.nh = nh
        return

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sampel(self, batch_size: int, device: str = "cuda:0", **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

class VAE(BaseVAE):
    def __init__(self,
            nin : int,
            nz : int,
            nh : int,
            **kwargs) -> None:
        super(VAE, self).__init__(nin, nz, nh)
        #self.nin = nin
        #self.nz = nz
        #self.nh = nh
        self.encoder = buildNetwork([nin, nh, nh], activation=nn.LeakyReLU(),)
        self.zmu =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        self.zlogvar =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        self.decoder = buildNetwork([nz, nh, nh], activation=nn.LeakyReLU(),)
        self.xmu = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nin),
                nn.Sigmoid(),
                )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    def encode(self, input: Tensor) -> List[Tensor]:
        x = nn.Flatten()(input)
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return [mu, logvar]

    def decode(self, z : Tensor) -> Tensor:
        h = self.decoder(z)
        mu = self.xmu(h)
        return mu

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return [recon, input, mu, logvar]


    def loss_function(self, *args, **kwargs) -> List[Tensor]:
        recon = args[0]
        input = args[1].reshape(recon.shape)
        mu = args[2]
        logvar = args[3]
        batch_size = mu.shape[0]
        kld_loss = self.kld_unreduced(mu, logvar).sum() / batch_size
        recon_loss = nn.MSELoss(reduction="sum")(recon, input) / batch_size
        loss = recon_loss + kld_loss
        return [loss, recon_loss.detach(), kld_loss.detach()]

    def generate(self, x : Tensor, **kwargs) -> Tensor:
        """
        returns just the reconstruction part from forward.
        """
        return self.forward(x)[0]

    def sample(self,
            num_samples : int,
            device : str = 'cuda:0',
            **kwargs,) -> Tensor:
        """
        sample from the latent space.
        """
        z = torch.randn(num_samples, self.nz).to(device)
        samples = self.decode(z)
        return samples

    def fit(self,
            train_loader : torch.utils.data.DataLoader,
            num_epochs=10,
            lr=1e-3,
            device : str = "cuda:0",) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                #log_sigma = ut.softclip(self.log_sigma, -2, 2)
                #self.log_sigma.fill_(log_sigma)
                x = data.flatten(1).to(device)
                recon, x, mu, logvar = self.forward(x)
                loss, rec_loss, kl_loss = self.loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print("loss = ",
                            loss.item(),
                            kl_loss.item(),
                            rec_loss.item(),
                            )
        self.cpu()
        optimizer = None
        print('done training')
        return None


class VAE_MC(BaseVAE):
    def __init__(self,
            nin : int,
            nz : int,
            nh : int,
            **kwargs) -> None:
        #super(VAE, self).__init__(nin, nz, nh)
        #super(self.__class__, self).__init__(nin, nz, nh)
        super().__init__(nin, nz, nh)
        #self.nin = nin
        #self.nz = nz
        #self.nh = nh
        self.encoder = buildNetwork([nin, nh, nh], activation=nn.LeakyReLU(),)
        self.zmu =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        self.zlogvar =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        self.decoder = buildNetwork([nz, nh, nh], activation=nn.LeakyReLU(),)
        self.xmu = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nin),
                nn.Sigmoid(),
                )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        self.logsigma = nn.Parameter(torch.ones(nin), requires_grad=True)

    def kl_mc(self, z, mu, logvar):
        """
        KLD computed by monte carlo sampling.
        """
        p = distributions.Normal(loc=torch.zeros_like(mu),
                scale=torch.ones_like(mu))
        q = distributions.Normal(loc=mu, scale=torch.exp(0.5 * logvar))
        log_pz = p.log_prob(z)
        log_qzx = q.log_prob(z)
        kl = (log_qzx - log_pz).sum(-1).mean()
        return kl

    def gaussian_likelihood(self, xhat, logsigma, x):
        scale = logsigma.exp()
        mean = xhat
        p = distributions.Normal(mean, scale)
        log_pxz = p.log_prob(x).sum(dim=-1).mean()
        return log_pxz

    def encode(self, input: Tensor) -> List[Tensor]:
        x = nn.Flatten()(input)
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return [mu, logvar]

    def decode(self, z : Tensor) -> Tensor:
        h = self.decoder(z)
        mu = self.xmu(h)
        return mu

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return [recon, input, mu, logvar]


    def loss_function(self, *args, **kwargs) -> List[Tensor]:
        recon = args[0]
        input = args[1].reshape(recon.shape)
        mu = args[2]
        logvar = args[3]
        z = self.reparameterize(mu, logvar)
        batch_size = mu.shape[0]
        #kld_loss = self.kld_unreduced(mu, logvar).sum() / batch_size
        kld_loss = self.kl_mc(z, mu, logvar)
        #recon_loss = nn.MSELoss(reduction="sum")(recon, input) / batch_size
        recon_loss = -1 * self.gaussian_likelihood(recon, self.logsigma, input)
        loss = recon_loss + kld_loss
        return [loss, recon_loss.detach(), kld_loss.detach()]

    def generate(self, x : Tensor, **kwargs) -> Tensor:
        """
        returns just the reconstruction part from forward.
        """
        return self.forward(x)[0]

    def sample(self,
            num_samples : int,
            device : str = 'cuda:0',
            **kwargs,) -> Tensor:
        """
        sample from the latent space.
        """
        z = torch.randn(num_samples, self.nz).to(device)
        samples = self.decode(z)
        return samples

    def fit(self,
            train_loader : torch.utils.data.DataLoader,
            num_epochs=10,
            lr=1e-3,
            device : str = "cuda:0",) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                #log_sigma = ut.softclip(self.log_sigma, -2, 2)
                #self.log_sigma.fill_(log_sigma)
                x = data.flatten(1).to(device)
                recon, x, mu, logvar = self.forward(x)
                loss, rec_loss, kl_loss = self.loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print("loss = ",
                            loss.item(),
                            kl_loss.item(),
                            rec_loss.item(),
                            )
        self.cpu()
        optimizer = None
        print('done training')
        return None


class catVAE(BaseVAE):
    def __init__(self,
            nin : int,
            nz : int,
            nh : int,
            nclass : int,
            temperature : float = 0.5,
            anneal_rate : float = 3e-5,
            anneal_interval : int = 100,
            alpha : float = 30.,
            **kwargs) -> None:
        super(self.__class__, self).__init__(nin, nz, nh)
        self.nclass = nclass
        self.temperature = temperature
        self.min_temperature = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        self.encoder = buildNetwork([nin, nh, nh], activation=nn.LeakyReLU(),)
        self.zmu =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz * nclass),
                )
        self.zlogvar =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz * nclass),
                )
        self.decoder = buildNetwork([nz, nh, nh], activation=nn.LeakyReLU(),)
        self.xmu = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nin),
                nn.Sigmoid(),
                )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        #self.sampling_dist = torch.distributions.OneHotCategorical(torch.ones((nclass, 1)))
        self.sampling_dist = torch.distributions.OneHotCategorical(torch.ones(nclass))

    def encode(self, input: Tensor) -> List[Tensor]:
        x = nn.Flatten()(input)
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return [mu, logvar]

    def decode(self, z : Tensor) -> Tensor:
        h = self.decoder(z)
        mu = self.xmu(h)
        return mu

