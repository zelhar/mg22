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

print(torch.cuda.is_available())

import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder


class BaseVAE(nn.Module):
    def __init__(
        self,
        nin: int,
        nz: int,
        nh: int,
    ) -> None:
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
    def __init__(self, nin: int, nz: int, nh: int, **kwargs) -> None:
        super(VAE, self).__init__(nin, nz, nh)
        # self.nin = nin
        # self.nz = nz
        # self.nh = nh
        self.encoder = buildNetwork(
            [nin, nh, nh],
            activation=nn.LeakyReLU(),
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
        )
        self.decoder = buildNetwork(
            [nz, nh, nh],
            activation=nn.LeakyReLU(),
        )
        self.xmu = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nin),
            nn.Sigmoid(),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        x = nn.Flatten()(input)
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return [mu, logvar]

    def decode(self, z: Tensor) -> Tensor:
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

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        returns just the reconstruction part from forward.
        """
        return self.forward(x)[0]

    def sample(
        self,
        num_samples: int,
        device: str = "cuda:0",
        **kwargs,
    ) -> Tensor:
        """
        sample from the latent space.
        """
        z = torch.randn(num_samples, self.nz).to(device)
        samples = self.decode(z)
        return samples

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                # log_sigma = ut.softclip(self.log_sigma, -2, 2)
                # self.log_sigma.fill_(log_sigma)
                x = data.flatten(1).to(device)
                recon, x, mu, logvar = self.forward(x)
                loss, rec_loss, kl_loss = self.loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        loss.item(),
                        kl_loss.item(),
                        rec_loss.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None


class VAE_MC(BaseVAE):
    # https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
    def __init__(self, nin: int, nz: int, nh: int, **kwargs) -> None:
        # super(VAE, self).__init__(nin, nz, nh)
        # super(self.__class__, self).__init__(nin, nz, nh)
        super().__init__(nin, nz, nh)
        # self.nin = nin
        # self.nz = nz
        # self.nh = nh
        self.encoder = buildNetwork(
            [nin, nh, nh],
            activation=nn.LeakyReLU(),
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
        )
        self.decoder = buildNetwork(
            [nz, nh, nh],
            activation=nn.LeakyReLU(),
        )
        self.xmu = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nin),
            nn.Sigmoid(),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.logsigma = nn.Parameter(torch.ones(nin), requires_grad=True)

    def kl_mc(self, z, mu, logvar):
        """
        KLD computed by monte carlo sampling.
        """
        p = distributions.Normal(loc=torch.zeros_like(mu), scale=torch.ones_like(mu))
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

    def decode(self, z: Tensor) -> Tensor:
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
        # kld_loss = self.kld_unreduced(mu, logvar).sum() / batch_size
        kld_loss = self.kl_mc(z, mu, logvar)
        # recon_loss = nn.MSELoss(reduction="sum")(recon, input) / batch_size
        recon_loss = -1 * self.gaussian_likelihood(recon, self.logsigma, input)
        loss = recon_loss + kld_loss
        return [loss, recon_loss.detach(), kld_loss.detach()]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        returns just the reconstruction part from forward.
        """
        return self.forward(x)[0]

    def sample(
        self,
        num_samples: int,
        device: str = "cuda:0",
        **kwargs,
    ) -> Tensor:
        """
        sample from the latent space.
        """
        z = torch.randn(num_samples, self.nz).to(device)
        samples = self.decode(z)
        return samples

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                # log_sigma = ut.softclip(self.log_sigma, -2, 2)
                # self.log_sigma.fill_(log_sigma)
                x = data.flatten(1).to(device)
                recon, x, mu, logvar = self.forward(x)
                loss, rec_loss, kl_loss = self.loss_function(recon, x, mu, logvar)
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        loss.item(),
                        kl_loss.item(),
                        rec_loss.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None


class VAE_MC_Gumbel(nn.Module):
    def __init__(
        self, nx: int, ny: int, nz: int, nh: int, tau: float, **kwargs
    ) -> None:
        super().__init__()
        self.nin = nin = nx + ny
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nh = nh
        self.tau = tau
        self.encoder = buildNetwork(
            [nin, nh, nh],
            activation=nn.LeakyReLU(),
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
        )
        # classifier produces logits =
        # un-normalized log probabilities
        # hence any real value is valid
        self.classifier = nn.Sequential(
            nn.Linear(nx, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, ny),
            nn.ReLU(),
        )
        self.decoder = buildNetwork(
            [nz, nh, nh],
            activation=nn.LeakyReLU(),
        )
        self.px_logit = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
            # nn.Sigmoid(),
        )
        self.mz_prior = nn.Sequential(
            nn.Linear(ny, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz),
        )
        self.logvarz_prior = nn.Sequential(
            nn.Linear(ny, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        # only used for gaussian decoder:
        self.logsigma = nn.Parameter(torch.ones(nx), requires_grad=True)

    def gaussian_likelihood(self, mu: Tensor, logvar: Tensor, x: Tensor) -> Tensor:
        logsigma = 0.5 * logvar
        scale = logsigma.exp()
        mean = mu
        p = distributions.Normal(mean, scale)
        log_pxz = p.log_prob(x).sum(dim=-1).mean()
        return log_pxz

    def encode(self, input: Tensor) -> List[Tensor]:
        x = nn.Flatten()(input)
        y_logits = self.classifier(x)  # logit of q(y|x)
        m = distributions.RelaxedOneHotCategorical(
            self.tau,
            logits=y_logits,
        )
        # sample from relaxed categorical distribution
        y = m.rsample()
        xy = torch.cat((x, y), dim=1)
        h = self.encoder(xy)
        mu = self.zmu(h)  # mu of q(z | x,y)
        logvar = self.zlogvar(h)  # logcar of q(z | x,y)
        return [mu, logvar, y_logits, y]

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z: Tensor, y: Tensor) -> List[Tensor]:
        # get the mu, logvar for the z-prior
        mz_prior = self.mz_prior(y)
        logvarz_prior = self.logvarz_prior(y)
        # decode
        h = self.decoder(z)
        px_logit = self.px_logit(h)
        return [px_logit, mz_prior, logvarz_prior]

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, logvar, y_logits, y = self.encode(input)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, y)
        return [*recon, mu, logvar, y_logits, y, z]
        # return [*recon, input, mu, logvar, z]

    def loss_function(
        self,
        input,
        y_logits,
        mu_zx,
        logvar_zx,
        mu_z,
        logvar_z,
        px_logit,
        z,
        *args,
        **kwargs,
    ) -> List[Tensor]:
        """
        loss function for unlabeled data.
        -logP(x|z) + logQ(z|x,y) - logP(z|y) + logQ(y|x) - logP(y)
        """
        batch_size = mu_z.shape[0]
        x = nn.Flatten()(input)
        # recon_loss = nn.MSELoss(reduction="sum")(recon, input) / batch_size
        # recon_loss = -1 * self.gaussian_likelihood(recon, self.logsigma, input)
        # -logP(x|z) term:
        recon_loss = (
            binary_cross_entropy_with_logits(input=px_logit, target=x, reduction="none")
            .sum(dim=-1)
            .mean()
        )
        # logit_loss = (y_p.log() - np.log(1 / self.ny) ).sum(dim=-1)
        # logQ(y|x) term:
        # convert logits to probabilities
        y_p = nn.Softmax(dim=-1)(y_logits)
        logQyx = y_p.log().sum(dim=-1).mean()
        # - logP(y) terms:
        # logPy = np.log(1 / self.ny)
        logPy = torch.tensor(1 / self.ny).log()
        # logQ(z|x,y) - logP(z|y) terms:
        q_zxy = distributions.Normal(loc=mu_zx, scale=torch.exp(0.5 * logvar_zx))
        p_zy = distributions.Normal(loc=mu_z, scale=torch.exp(0.5 * logvar_z))
        z_loss = (
            q_zxy.log_prob(z).sum(dim=-1).mean() - p_zy.log_prob(z).sum(dim=-1).mean()
        )
        # z_loss = self.gaussian_likelihood(mu_zx, logvar_zx, z)
        loss = recon_loss - logPy + logQyx + z_loss
        return [loss, recon_loss.detach(), logQyx.detach(), z_loss.detach()]

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                # log_sigma = ut.softclip(self.log_sigma, -2, 2)
                # self.log_sigma.fill_(log_sigma)
                x = data.flatten(1).to(device)
                (
                    px_logit,
                    mz_prior,
                    logvarz_prior,
                    mu,
                    logvar,
                    y_logits,
                    y,
                    z,
                ) = self.forward(x)
                # recon, x, mu, logvar = self.forward(x)
                # loss, rec_loss, kl_loss = self.loss_function(recon, x, mu, logvar)
                loss, _, _, _ = self.loss_function(
                    x,
                    y_logits,
                    mu,
                    logvar,
                    mz_prior,
                    logvarz_prior,
                    px_logit,
                    z,
                )
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


class catVAE(BaseVAE):
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/cat_vae.py
    def __init__(
        self,
        nin: int,
        nz: int,
        nh: int,
        nclass: int,
        temperature: float = 0.5,
        anneal_rate: float = 3e-5,
        anneal_interval: int = 100,
        alpha: float = 30.0,
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__(nin, nz, nh)
        self.nclass = nclass
        self.temperature = temperature
        self.min_temperature = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        self.encoder = buildNetwork(
            [nin, nh, nh],
            activation=nn.LeakyReLU(),
        )
        self.zmu = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz * nclass),
        )
        self.zlogvar = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz * nclass),
        )
        self.decoder = buildNetwork(
            [nz, nh, nh],
            activation=nn.LeakyReLU(),
        )
        self.xmu = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nin),
            nn.Sigmoid(),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        # self.sampling_dist = torch.distributions.OneHotCategorical(torch.ones((nclass, 1)))
        self.sampling_dist = torch.distributions.OneHotCategorical(torch.ones(nclass))

    def encode(self, input: Tensor) -> List[Tensor]:
        x = nn.Flatten()(input)
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return [mu, logvar]

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder(z)
        mu = self.xmu(h)
        return mu


class SoftCatVAE(nn.Module):
    # https://github.com/AntixK/PyTorch-VAE/blob/master/models/cat_vae.py
    def __init__(
        self,
        nin: int,
        nz: int,
        nh: int,
        nclass: int,
        temperature: float = 0.5,
        anneal_rate: float = 3e-5,
        anneal_interval: int = 100,
        alpha: float = 30.0,
        **kwargs,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin
        self.nz = nz
        self.nh = nh
        self.nclass = nclass
        self.temperature = temperature
        self.min_temperature = temperature
        self.anneal_rate = anneal_rate
        self.anneal_interval = anneal_interval
        self.alpha = alpha
        self.encoder = buildNetwork(
            [nin, nh, nh],
            activation=nn.LeakyReLU(),
        )
        self.zmu = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz * nclass),
        )
        self.zlogvar = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz * nclass),
        )
        self.decoder = buildNetwork(
            [nz, nh, nh],
            activation=nn.LeakyReLU(),
        )
        self.xmu = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nin),
            nn.Sigmoid(),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        # self.sampling_dist = torch.distributions.OneHotCategorical(torch.ones((nclass, 1)))
        self.sampling_dist = torch.distributions.OneHotCategorical(torch.ones(nclass))

    def encode(self, input: Tensor) -> List[Tensor]:
        x = nn.Flatten()(input)
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return [mu, logvar]

    def decode(self, z: Tensor) -> Tensor:
        h = self.decoder(z)
        mu = self.xmu(h)
        return mu


######
# applying
# https://github.com/leequant761/Gumbel-SSVAE/blob/main/models.py


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.logvar = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        return

    def forward(self, x: Tensor):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        return

    def forward(self, x: Tensor):
        h = self.decoder(x)
        mu = self.mu(h)
        return mu


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        return

    def forward(self, x):
        logit = self.classifier(x)
        return logit


class GumbelSSAE(nn.Module):
    # https://github.com/leequant761/Gumbel-SSVAE/blob/main/models.py
    def __init__(
        self,
        input_dim: int = 28 ** 2,
        nclasses: int = 10,
        latent_dim: int = 20,
        hidden_dim: int = 1024,
        tau: Tensor = torch.tensor(0.99),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nclasses = nclasses
        self.tau = tau  # initial temperature
        self.r = 3e-5  # anneal rate
        self.tau_min = 0.5
        self.encoder = Encoder(
            input_dim=input_dim + nclasses, latent_dim=latent_dim, hidden_dim=hidden_dim
        )
        self.decoder = Decoder(
            input_dim=latent_dim + nclasses, output_dim=input_dim, hidden_dim=hidden_dim
        )
        self.classifier = Classifier(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=nclasses
        )
        self.prior_y = distributions.RelaxedOneHotCategorical(
            self.tau, probs=torch.ones(self.nclasses)
        )
        self.prior_z = pyrodist.Normal(
            loc=torch.zeros(latent_dim), scale=torch.ones(latent_dim)
        ).to_event(1)
        # self.apply(init_weights)
        # q(y,z | x) = q(y | x) * q(z | x,y)
        # p(x,y,z) = p(y) * p(z) * p(x | y,z)
        return

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def sample_y(self, log_alpha, tau, y=None):
        v_dist = distributions.RelaxedOneHotCategorical(tau, logits=log_alpha)
        concrete = v_dist.rsample() if y is None else y
        return concrete, v_dist

    def sample_z(self, mu, logvar):
        mean = mu
        sigma = 0.5 * logvar
        std = sigma.exp()
        v_dist = distributions.Normal(loc=mean, scale=std)
        z_q = v_dist.rsample()
        return z_q, v_dist

    def forward(self, input, y=None):
        x = nn.Flatten()(input)
        log_alpha = self.classifier(x)
        y_q, y_q_dist = self.sample_y(log_alpha, self.tau, y)
        z_mu, z_logvar = self.encoder(torch.cat((x, y_q), dim=-1))
        z_q, z_q_dist = self.sample_z(z_mu, z_logvar)
        x_hat = self.decode(torch.cat((z_q, y_q), dim=-1))
        return x_hat, z_q, y_q, z_q_dist, y_q_dist

    def loss_MC(
        self,
        x,
        x_hat,
        z_q,
        y_q,
        z_q_dist,
        y_q_dist,
        is_observed=False,
    ):
        """
        Monte Carlo estimation of the loss
        """
        n_batch = x.shape[0]
        x = nn.Flatten()(x)
        # bce = F.binary_cross_entropy(x_hat, x, reduction='none').sum(-1).mean()
        bce = (
            F.binary_cross_entropy_with_logits(x_hat, x, reduction="none")
            .sum(dim=-1)
            .mean()
        )
        # prior_y = self.prior_y.expand((n_batch,))
        # prior_z = self.prior_z.expand((n_batch,))
        prior_y = distributions.RelaxedOneHotCategorical(
            temperature=self.tau,
            probs=torch.ones_like(y_q, device=y_q.device),
        )
        prior_z = distributions.Normal(
            loc=torch.zeros_like(z_q, device=y_q.device),
            scale=torch.ones_like(z_q, device=y_q.device),
        )
        if not is_observed:
            kl_y = (y_q_dist.log_prob(y_q.softmax(-1)) - prior_y.log_prob(y_q.softmax(-1))).sum(dim=-1).mean()
        else:
            kl_y = 0.0
        kl_z = (z_q_dist.log_prob(z_q) - prior_z.log_prob(z_q)).sum(dim=-1).mean()
        kl = kl_y + kl_z
        return bce, kl

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                x = data.flatten(1).to(device)
                x_hat, z_q, y_q, z_q_dist, y_q_dist = self.forward(x)
                bce, kl = self.loss_MC(
                    x.cuda(),
                    x_hat,
                    z_q,
                    y_q,
                    z_q_dist,
                    y_q_dist,
                )
                loss = bce + kl
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        bce.item(),
                        kl.item(),
                        loss.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None

class GumbelGMSSAE(nn.Module):
    # http://ruishu.io/2016/12/25/gmvae/
    # https://github.com/leequant761/Gumbel-SSVAE/blob/main/models.py
    def __init__(
        self,
        input_dim: int = 28 ** 2,
        nclasses: int = 10,
        latent_dim: int = 20,
        hidden_dim: int = 1024,
        tau: Tensor = torch.tensor(0.99),
        num_epochs : int = 10
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.nclasses = nclasses
        self.tau = tau  # initial temperature
        self.r = 3e-5  # anneal rate
        self.tau_min = 0.5
        self.num_epochs = num_epochs
        self.encoder = Encoder(
            input_dim=input_dim + nclasses, latent_dim=latent_dim, hidden_dim=hidden_dim
        )
        self.decoder = Decoder(
            input_dim=latent_dim, output_dim=input_dim, hidden_dim=hidden_dim
        )
        self.classifier = Classifier(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=nclasses
        )
        self.prior_y = distributions.RelaxedOneHotCategorical(
            self.tau, probs=torch.ones(self.nclasses)
        )
        self.prior_z = pyrodist.Normal(
            loc=torch.zeros(latent_dim), scale=torch.ones(latent_dim)
        ).to_event(1)
        self.mu_prior = nn.Sequential(
                nn.Linear(nclasses, latent_dim),
                )
        self.logvar_prior = nn.Sequential(
                nn.Linear(nclasses, latent_dim),
                )
        # self.apply(init_weights)
        # inference model
        # q(y,z | x) = q(y | x) * q(z | x,y)
        # where:
        # q(y | x) ~ relaxed-ca(logits(x))
        # q(z | x,y) ~ N(m(x,y), v(x,y))
        # generative model:
        # p(x,y,z) = p(y) * p(z|y) * p(x | z)
        # where
        # p(y) ~ relaxed 1/k-cat
        # p(z|y) ~ N(m(y),v(y))
        # p(x|z) ~ Bernoulli(m(z))
        # Loss:
        # Expectation over Q(z,y|x) of:
        # -logP(x,y,z) + logQ(y,z|x)
        
        return

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def sample_y(self, log_alpha, tau, y=None):
        v_dist = distributions.RelaxedOneHotCategorical(tau, logits=log_alpha)
        concrete = v_dist.rsample() if y is None else y
        return concrete, v_dist

    def sample_z(self, mu, logvar):
        mean = mu
        sigma = 0.5 * logvar
        std = sigma.exp()
        v_dist = distributions.Normal(loc=mean, scale=std)
        z_q = v_dist.rsample()
        return z_q, v_dist

    def forward(self, input, y=None):
        x = nn.Flatten()(input)
        log_alpha = self.classifier(x)
        y_q, y_q_dist = self.sample_y(log_alpha, self.tau, y)
        z_mu, z_logvar = self.encoder(torch.cat((x, y_q), dim=-1))
        z_q, z_q_dist = self.sample_z(z_mu, z_logvar)
        #x_hat = self.decode(torch.cat((z_q, y_q), dim=-1))
        x_hat = self.decode(z_q)
        z_mu_prior = self.mu_prior(y_q)
        z_logvar_prior = self.logvar_prior(y_q)
        return x_hat, z_q, y_q, z_q_dist, y_q_dist, z_mu_prior, z_logvar_prior

    def labeled_loss(
            self, x, x_hat,
            z_mu, z_logvar, z, 
            z_mu_prior, z_logvar_prior,
            ):
        """
        loss function for a fixed labeled y
        """
        bce = (
            F.binary_cross_entropy_with_logits(x_hat, x, reduction="none")
            .sum(dim=-1)
            .mean()
        )
        prior_z = distributions.Normal(
            loc=z_mu_prior,
            scale=(0.5 * z_logvar_prior).exp(),
        )
        post_z = distributions.Normal(
            loc=z_mu,
            scale=(0.5 * z_logvar).exp(),
        )
        kl_z = (post_z.log_prob(z) - prior_z.log_prob(z)).sum(dim=-1).mean()
        return bce, kl_z

    def loss_MC(
        self,
        x,
        x_hat,
        z_q,
        y_q,
        z_q_dist,
        y_q_dist,
        z_mu_prior,
        z_logvar_prior,
        is_observed=False,
    ):
        """
        Monte Carlo estimation of the loss
        """
        n_batch = x.shape[0]
        x = nn.Flatten()(x)
        # bce = F.binary_cross_entropy(x_hat, x, reduction='none').sum(-1).mean()
        bce = (
            F.binary_cross_entropy_with_logits(x_hat, x, reduction="none")
            .sum(dim=-1)
            .mean()
        )
        # prior_y = self.prior_y.expand((n_batch,))
        # prior_z = self.prior_z.expand((n_batch,))
        prior_y = distributions.RelaxedOneHotCategorical(
            temperature=self.tau,
            probs=torch.ones_like(y_q, device=y_q.device),
        )
        prior_z = distributions.Normal(
            loc=z_mu_prior,
            scale=(0.5 * z_logvar_prior).exp(),
        )
        if not is_observed:
            #kl_y = (y_q_dist.log_prob(y_q.softmax(-1)) - prior_y.log_prob(y_q.softmax(-1))).sum(dim=-1).mean()
            #kl_y = (y_q_dist.log_prob(y_q) - prior_y.log_prob(y_q)).sum(dim=-1).mean()
            probs = y_q_dist.probs
            eps = 1e-3
            kl_y = (probs * 
                    (self.nclasses * (probs + eps)).log()).sum(dim=-1).mean()
        else:
            kl_y = 0.0
        kl_z = (z_q_dist.log_prob(z_q) - prior_z.log_prob(z_q)).sum(dim=-1).mean()
        kl = kl_y + kl_z
        # I suspect that kl_y isn't important and we can just ignore it.
        #kl = kl_z
        return bce, kl

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
                x_hat, z_q, y_q, z_q_dist, y_q_dist, z_mu_prior, z_logvar_prior = self.forward(x)
                bce, kl = self.loss_MC(
                    x.cuda(),
                    x_hat,
                    z_q,
                    y_q,
                    z_q_dist,
                    y_q_dist,
                    z_mu_prior,
                    z_logvar_prior,
                )
                loss = bce + kl
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        bce.item(),
                        kl.item(),
                        loss.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None



# wtf
