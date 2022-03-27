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
        self.categorical_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                nn.Softmax(dim=-1),
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
        c = self.categorical_encoder(x)
        return z,c
    def decode(self, z):
        x = self.decoder(z)
        return x
    def reconstruction_loss(self, x, xhat):
        loss = nn.MSELoss(reduction="none")(x, xhat)
        loss = loss.sum(dim=-1).mean()
        return loss

    def forward(self, input):
        x = nn.Flatten()(input)
        z,c = self.encode(x)
        xhat = self.decode(z+c)
        recon_loss = self.reconstruction_loss(x, xhat)
        cat_loss = -c.max(dim=-1)[0].mean() #min is 1
        loss = recon_loss + 10*cat_loss
        output = {
                "xhat" : xhat,
                "z" : z,
                "loss" : loss,
                "cat loss" : cat_loss
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

class AAE(nn.Module):
    """
    simple AAE
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
        self.adversarial_loss_scale = adversarial_loss_scale
        self.y_prior = distributions.OneHotCategorical(
                probs=torch.ones(nclusters))
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        self.mu_z = nn.Sequential(
                #nn.Linear(nh, nh),
                #nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z = nn.Sequential(
                #nn.Linear(nh, nh),
                #nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.gauss_discriminator = nn.Sequential(
                nn.Linear(nz, nh),
                nn.ReLU(),
                nn.Linear(nh,1),
                nn.Sigmoid(),
                )
        self.categorical_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nclusters),
                nn.Softmax(dim=-1),
                )
        self.categorical_discriminator = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nclusters, nh),
                nn.ReLU(),
                nn.Linear(nh,1),
                nn.Sigmoid(),
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
        h = self.encoder(x)
        mu = self.mu_z(h)
        logvar = self.logvar_z(h)
        z = torch.randn_like(mu, requires_grad=True, device=mu.device)
        std = (0.5 * logvar).exp()
        z = mu + z * std
        clusterhead = self.categorical_encoder(x)
        z = z + clusterhead
        return mu, logvar, z
    def decode(self, z):
        x = self.decoder(z)
        return x
    def adversarial_loss(self, z):
        gen_adversarial_score = self.gauss_discriminator(z)
        true_labels = torch.ones_like(gen_adversarial_score,
                requires_grad=False, device=z.device)
        return F.binary_cross_entropy(gen_adversarial_score, true_labels)
    def categorical_adversarial_loss(self, x):
        c = self.categorical_encoder(x)
        gen_adversarial_score = self.categorical_discriminator(c)
        true_labels = torch.ones_like(gen_adversarial_score,
                requires_grad=False, device=c.device)
        return F.binary_cross_entropy(gen_adversarial_score, true_labels)
    def discriminator_loss(self, z):
        w = torch.randn_like(z, device=z.device, requires_grad=True)
        prior_adversarial_score = self.gauss_discriminator(w)
        true_labels = torch.ones_like(prior_adversarial_score,
                requires_grad=False, device=z.device)
        fake_labels = torch.zeros_like(prior_adversarial_score,
                requires_grad=False, device=z.device)
        z_ = z.clone().detach().requires_grad_(True)
        gen_adversarial_score_ = self.gauss_discriminator(z_)
        discriminator_loss = 0.5 * (
            F.binary_cross_entropy(
                prior_adversarial_score, true_labels
            )  # prior is true
        ) + 0.5 * (
            F.binary_cross_entropy(
                gen_adversarial_score_, fake_labels
            )  # generated are false
        )
        return discriminator_loss
    def reconstruction_loss(self, x, xhat):
        loss = nn.MSELoss(reduction="none")(x, xhat)
        loss = loss.sum(dim=-1).mean()
        return loss

    def categorical_discriminator_loss(self, x):
        y = self.y_prior.sample((x.shape[0],)).to(x.device)
        c = self.categorical_encoder(x)
        prior_adversarial_score = self.categorical_discriminator(y)
        gen_adversarial_score_ = self.categorical_discriminator(c)
        true_labels = torch.ones_like(prior_adversarial_score,
                requires_grad=False, device=c.device)
        fake_labels = torch.zeros_like(prior_adversarial_score,
                requires_grad=False, device=c.device)
        discriminator_loss = 0.5 * (
            F.binary_cross_entropy(
                prior_adversarial_score, true_labels
            )  # prior is true
        ) + 0.5 * (
            F.binary_cross_entropy(
                gen_adversarial_score_, fake_labels
            )  # generated are false
        )
        return discriminator_loss

    def forward(self, input):
        x = nn.Flatten()(input)
        mu, logvar, z = self.encode(x)
        xhat = self.decode(z)
        #xhat = self.decode(mu)
        adversarial_loss = self.adversarial_loss(z) * self.adversarial_loss_scale
        recon_loss = self.reconstruction_loss(x, xhat) * (1 - self.adversarial_loss_scale)
        disc_loss = self.discriminator_loss(z)
        cat_disc_loss = self.categorical_discriminator_loss(x)
        cat_adv_loss = self.categorical_adversarial_loss(x)
        loss = adversarial_loss + recon_loss + disc_loss
        loss = loss + cat_adv_loss + cat_disc_loss
        #loss = recon_loss
        output = {
                "xhat" : xhat,
                "mu" : mu,
                "logvar" : logvar,
                "loss" : loss,
                "advloss" : adversarial_loss,
                "discloss" : disc_loss,
                "recloss" : recon_loss,
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

class VAE(nn.Module):
    """
    VAE
    """
    def __init__(self, nx : int=28**2,
            nclusters : int=16,
            nz : int=16,
            nh : int=3000,
            ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclusters
        #self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        self.mu_z = nn.Sequential(
                #nn.Linear(nh, nh),
                #nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z = nn.Sequential(
                #nn.Linear(nh, nh),
                #nn.LeakyReLU(),
                nn.Linear(nh, nz),
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
        z = q_z.rsample()
        std = (logvar * 0.5).exp()
        eps = torch.randn_like(mu)
        #z = mu + std*eps
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

class VanillaVAE(nn.Module):
    """
    just your vanilla type VAE.
    """
    def __init__(self, nx : int=28**2,
            nclusters : int=16,
            nz : int=16,
            nh : int=3000,
            ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclusters
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        self.mu_z = nn.Sequential(
                #nn.Linear(nh, nh),
                #nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z = nn.Sequential(
                #nn.Linear(nh, nh),
                #nn.LeakyReLU(),
                nn.Linear(nh, nz),
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
        z = q_z.rsample()
        kld_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        x_hat = self.decode(z)
        #recon_loss = nn.MSELoss(reduction="none")(x_hat, x).sum(-1).mean()
        logsigma_x = ut.softclip(self.logsigma_x, -2, 2)
        sigma_x = logsigma_x.exp()
        q_x = pyrodist.Normal(loc=x_hat, scale=sigma_x).to_event(1)
        recon_loss = -q_x.log_prob(x).mean()
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
                "q_x" : q_x,
                "x_hat" : x_hat,
                "sigma_x" : sigma_x,
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

    def fit2(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer1 = optim.Adam(self.parameters(), lr=lr)
        optimizer2 = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                x = data.flatten(1).to(device)
                output = self.forward(x)
                recon_loss = output['recon_loss']
                recon_loss.backward()
                optimizer1.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                x = data.flatten(1).to(device)
                output = self.forward(x)
                kld_z = output['kld_z']
                kld_z.backward()
                optimizer2.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        recon_loss.item(),
                        kld_z.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None



class VAE_with_Clusterheads(nn.Module):
    """
    simple VAE with categorical clusterhead encoder.
    """
    def __init__(self, nx : int=28**2,
            nclusters : int=16,
            nz : int=16,
            nh : int=3000,
            ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclusters
        self.sigma_x = torch.nn.Parameter(torch.ones(nx), requires_grad=True)
        self.y_prior = distributions.OneHotCategorical(
                probs=torch.ones(nclusters))
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        self.mu_z = nn.Sequential(
                #nn.Linear(nh, nh),
                #nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z = nn.Sequential(
                #nn.Linear(nh, nh),
                #nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.categorical_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nclusters),
                nn.Softmax(dim=-1),
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
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        return
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_z(h)
        logvar = self.logvar_z(h)
        clusterhead = self.categorical_encoder(x)
        #z = z + clusterhead
        return mu, logvar, clusterhead
    def decode(self, z):
        x = self.decoder(z)
        return x
    def cluster_distribution(self, x):
        c = self.categorical_encoder(x)
        prior = distributions.OneHotCategorical(probs=torch.ones_like(c)/self.nclasses)
        posterior = distributions.OneHotCategorical(probs=c)
        return prior, posterior

    def forward(self, input):
        x = nn.Flatten()(input)
        mu, logvar, c = self.encode(x)
        z = torch.randn_like(mu, requires_grad=True, device=mu.device)
        std = (0.5 * logvar).exp()
        z = mu + z * std
        kld_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        x_hat = self.decode(z + c)
        recon_loss = nn.MSELoss(reduction="none")(x_hat, x).sum(-1).mean()
        prior = distributions.OneHotCategorical(probs=torch.ones_like(c)/self.nclasses)
        posterior = distributions.OneHotCategorical(probs=c)
        # kld_y = (q.probs * (q.logits - p.logits)), q=posterior...
        kld_y = nn.KLDivLoss(reduction="none")(
                input = prior.logits, target=posterior.probs).sum(-1).mean()
        cat_loss = -c.max(dim=-1)[0].mean() #min is 1
        loss = recon_loss + kld_z + kld_y + cat_loss
        output = {
                "mu" : mu,
                "logvar" : logvar,
                "z" : z,
                "kld_z" : kld_z,
                "kld_y" : kld_y,
                "recon_loss" : recon_loss,
                "loss" : loss,
                "y_prior" : prior,
                "y_posterior" : posterior,
                "cat_loss" : cat_loss,
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


class VAE_with_Clusterheadsv2(nn.Module):
    """
    simple VAE with categorical clusterhead encoder.
    """
    def __init__(self, nx : int=28**2,
            nclusters : int=16,
            nz : int=16,
            nh : int=3000,
            ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclusters
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        #self.y_prior = distributions.OneHotCategorical(
        #        probs=torch.ones(nclusters))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                temperature=0.2, probs=torch.ones(nclusters))
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        self.mu_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.categorical_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nclusters),
                nn.Softmax(dim=-1),
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

        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        return
    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu_z(h)
        logvar = self.logvar_z(h)
        clusterhead = self.categorical_encoder(x)
        #z = z + clusterhead
        return mu, logvar, clusterhead
    def decode(self, z):
        x = self.decoder(z)
        return x
    def cluster_distribution(self, x):
        c = self.categorical_encoder(x)
        #prior = distributions.OneHotCategorical(probs=torch.ones_like(c)/self.nclasses)
        #posterior = distributions.OneHotCategorical(probs=c)
        prior = distributions.RelaxedOneHotCategorical(temperature=0.2, probs=torch.ones_like(c)/self.nclasses)
        posterior = distributions.RelaxedOneHotCategorical(temperature=0.2, probs=c)
        return prior, posterior

    def forward(self, input):
        x = nn.Flatten()(input)
        mu, logvar, c = self.encode(x)
        z = torch.randn_like(mu, requires_grad=True, device=mu.device)
        std = (0.5 * logvar).exp()
        z = mu + z * std
        kld_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        x_hat = self.decode(z + c)
        recon_loss = nn.MSELoss(reduction="none")(x_hat, x).sum(-1).mean()
        prior = distributions.OneHotCategorical(probs=torch.ones_like(c)/self.nclasses)
        posterior = distributions.OneHotCategorical(probs=c)
        # kld_y = (q.probs * (q.logits - p.logits)), q=posterior...
        batch_size = x.shape[0]
        p = prior.probs.sum(0)/batch_size
        eps=1e-6
        q = posterior.probs.sum(0)/batch_size
        q += eps
        q /= q.sum()
        kld_y = torch.sum(p * (p.log() - q.log())
                )
        #kld_y = nn.KLDivLoss(reduction="none")(
        #        input = prior.logits, target=posterior.probs).sum(-1).mean()
        cat_loss = -c.max(dim=-1)[0].mean() #min is -1
        loss = recon_loss + kld_z + kld_y + cat_loss
        output = {
                "mu" : mu,
                "logvar" : logvar,
                "z" : z,
                "kld_z" : kld_z,
                "kld_y" : kld_y,
                "recon_loss" : recon_loss,
                "loss" : loss,
                "y_prior" : prior,
                "y_posterior" : posterior,
                "cat_loss" : cat_loss,
                "x_hat" : x_hat,
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






class GMMClustering(nn.Module):
    """
    simple model for clustering using mixed gaussian model.
    essentially the M2 model from Kingma's deep semi supervised...
    generative model: P(x,y,z) = P(x | y,z) P(y) P(z)
    P(x | y,z) ~ GMM, meaning for a fixed y=i P(x | z,y=i) ~ N(mu(z_i), sig(z_i)
    P(z) ~ N(0,I), P(y) = (relaxed)Cat(pi), pi ~ (symmetric) Dirichlet prior.

    inference model: Q(z,y | x) = Q(z | x) Q(y | x)
    Q(z | x) ~ Normal, Q(z | y) ~ (relaxed) Categircal.
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nclasses: int = 10,
        nh: int = 1024,
        nz: int = 20,
        tau: Tensor = torch.tensor(0.3),
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclasses
        self.tau = tau
        self.enc_z_x = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                )
        self.mu_z_x = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z_x = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logits_y_x = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nclasses),
                )
        self.mus_xs_z = nn.Sequential(
                nn.Linear(nz,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nx * nclasses),
                nn.Unflatten(dim=-1, unflattened_size=(nclasses, nx)),
                )
        self.logvars_xs_z = nn.Sequential(
                nn.Linear(nz,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nx),
                #nn.Linear(nh, nx * nclasses),
                #nn.Unflatten(dim=-1, unflattened_size=(nclasses, nx)),
                )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        return
    def forward(self, x, tau=0.3):
        h_x = self.enc_z_x(x)
        mu_z_x = self.mu_z_x(h_x)
        logvar_z_x = self.logvar_z_x(h_x)
        logits_y_x = self.logits_y_x(x)
        q_z = pyrodist.Normal(loc=mu_z_x, scale=torch.exp(0.5*logvar_z_x)).to_event(1)
        z = q_z.rsample()
        mus_xs_z = self.mus_xs_z(z)
        #logvars_xs_z = self.logvars_xs_z(z)
        q_y = distributions.RelaxedOneHotCategorical(tau, logits=logits_y_x,)
        return mu_z_x, logvar_z_x, logits_y_x, q_z, z, mus_xs_z, q_y
    def loss_v1(self, x, mu_z_x, logvar_z_x, logits_y_x, q_y, q_z, z, mus_xs_z, ):
        """
        Summation method (over labels y)
        """
        target = x.flatten(1).unsqueeze(1).expand(-1, self.nclasses, -1)
        # nclasses pointwise bcelosses per sample [batch, C, nx]
        bce = nn.BCEWithLogitsLoss(reduction='none',)(input=mus_xs_z,
                target=target)
        # log q(z|x) - log p(z):
        # not dependent on y, can also be calculated analytically
        kld_z = self.kld_unreduced(mu_z_x, logvar_z_x)
        # log q(y|x) - log p(y):
        p_y = distributions.OneHotCategorical(probs=torch.ones_like(logits_y_x))
        kld_y = (q_y.logits - p_y.logits) * q_y.probs
        return bce, kld_y, kld_z, target, 
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
                mu_z_x, logvar_z_x, logits_y_x, q_z, z, mus_xs_z, q_y = self.forward(x)
                bce , kld_y, kld_z, target = self.loss_v1(x, mu_z_x, logvar_z_x, logits_y_x, q_y, q_z, z, mus_xs_z)
                #recon_loss = (bce.sum(-1) * q_y.probs).sum(-1).mean()
                y = q_y.rsample()
                recon_loss = (bce.sum(-1) * y).sum(-1).mean()
                kld_y = kld_y.sum(-1).mean()
                #kld_y = torch.max(torch.tensor(10), kld_y)
                #kld_y = torch.max(torch.tensor(10.0, device=kld_y.device, requires_grad=True), kld_y)
                kld_z = kld_z.sum(-1).mean()
                loss = recon_loss + 100*kld_y + kld_z
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        recon_loss.item(),
                        kld_y.item(),
                        kld_z.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None

class GMMKoolooloo(nn.Module):
    """
    Following Dilokanthankul's model for clustering using mixed gaussian model.
    P(x,z,w,y) = P(x|z)P(z|w,y)P(w)P(y)
    P(y) ~ Cat(pi)
    P(w) ~ N(0,I)
    P(z |w, y=i) ~ Normal
    P(x | x) ~ Bernoulli or normal with fixed/global variance

    Q(y,w,z | x) = Q(y | w,z)Q(w|x)Q(z|x)
    Q(w|x), Q(z|x) are normal gaussian posterior
    Q(y | w,z) is categorical posterio even though theoretically it is
    tractable.
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nclasses: int = 10, #ny
        nh: int = 1024,
        nz: int = 20,
        nw: int= 30,
        tau: Tensor = torch.tensor(0.2),
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.tau = tau
        # Q graph (encoder)
        self.enc_z_x = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                )
        self.mu_z_x = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z_x = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.enc_w_x = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                )
        self.mu_w_x = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nw),
                )
        self.logvar_w_x = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nw),
                )
        self.logits_y_wz = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nz+nw,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nclasses),
                )
        # P graph (decoder)
        self.y_prior = distributions.RelaxedOneHotCategorical(temperature=tau,
                probs=torch.ones(nclasses))
        self.w_prior = distributions.Normal(loc=0, scale=1)
        self.dec_h_w = nn.Sequential(
                nn.Linear(nw,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                )
        self.mus_zs_hw = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz * nclasses),
                nn.Unflatten(dim=-1, unflattened_size=(nclasses, nz)),
                )
        self.logvars_zs_hw = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz * nclasses),
                nn.Unflatten(dim=-1, unflattened_size=(nclasses, nz)),
                #nn.Linear(nh, nx * nclasses),
                #nn.Unflatten(dim=-1, unflattened_size=(nclasses, nx)),
                )
        self.dec_h_z = nn.Sequential(
                nn.Linear(nz,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                )
        self.logits_x_hz = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.sigma_x = torch.nn.Parameter(torch.ones(nx), requires_grad=True)
        return

    def encode_Qz_x(self, x):
        hz = self.enc_z_x(x)
        mu_z = self.mu_z_x(hz)
        logvar_z = self.logvar_z_x(hz)
        q_z = pyrodist.Normal(loc=mu_z, scale=(0.5*logvar_z).exp()).to_event(1)
        return q_z

    def encode_Qw_x(self, x):
        hw = self.enc_w_x(x)
        mu_w = self.mu_w_x(hw)
        logvar_w = self.logvar_w_x(hw)
        q_w = pyrodist.Normal(loc=mu_w, scale=(0.5*logvar_w).exp()).to_event(1)
        return q_w

    def encode_Qy_wz(self, z, w):
        #z = q_z.rsample()
        #w = q_w.rsample()
        zw = torch.cat((z,w), dim=-1)
        y_logit = self.logits_y_wz(zw)
        q_y = distributions.RelaxedOneHotCategorical(temperature=self.tau, logits=y_logit,)
        return q_y

    def encode(self, x):
        hw = self.enc_w_x(x)
        hz = self.enc_z_x(x)
        mu_w = self.mu_w_x(hw)
        logvar_w = self.logvar_w_x(hw)
        mu_z = self.mu_z_x(hz)
        logvar_z = self.logvar_z_x(hz)
        q_z = pyrodist.Normal(loc=mu_z, scale=(0.5*logvar_z).exp()).to_event(1)
        q_w = pyrodist.Normal(loc=mu_w, scale=(0.5*logvar_w).exp()).to_event(1)
        z = q_z.rsample()
        w = q_w.rsample()
        zw = torch.cat((z,w), dim=-1)
        y_logit = self.logits_y_wz(zw)
        return mu_z, logvar_z, z, mu_w, logvar_w, w, y_logit

    def decode_zs_w(self, w):
        hw = self.dec_h_w(w)
        mus = self.mus_zs_hw(hw)
        logvars = self.logvars_zs_hw(hw)
        #[batchdim=128, eventdim=(10,nz)]
        return mus, logvars

    def decode_x_z(self, z):
        hz = self.dec_h_z(z)
        x_logit = self.logits_x_hz(hz)
        return x_logit

    def forward(self, input):
        x = nn.Flatten()(input)
        q_z = self.encode_Qz_x(x)
        q_w = self.encode_Qw_x(x)
        z = q_z.rsample()
        w = q_w.rsample()
        q_y = self.encode_Qy_wz(z,w)
        mus_z, logvars_z = self.decode_zs_w(w)
        y = q_y.rsample()
        logit_x = self.decode_x_z(z)
        # hard sample:
        #yhard = F.gumbel_softmax(logits=q_y.logits, tau=self.tau, hard=True, )
        return q_z, q_w, z, w, q_y, mus_z, logvars_z, y, logit_x

    def reconstruction_loss(self, x_logit,x,isgaussian=False):
        target = nn.Flatten()(x)
        if not isgaussian:
            recon_loss = nn.BCEWithLogitsLoss(reduction='none')(input=x_logit,
                    target=target).sum(-1).mean()
        else:
            sigma_x = ut.softclip(self.sigma_x, -1, 1)
            #p_x = pyrodist.Normal(loc=x_logit, scale=sigma_x)
            #recon_loss = -p_x.log_prob(x)
            recon_loss = (x - x_logit.sigmoid()).pow(2)/sigma_x
            recon_loss = recon_loss.sum(dim=-1).mean()
        return recon_loss

    def kld_z_loss(self, q_z, mus_zs_w, logvars_zs_w, q_y):
        z = q_z.rsample()
        log_Qz_x = q_z.log_prob(z)
        P_zs_w = pyrodist.Normal(loc=mus_zs_w,
                scale=(0.5*logvars_zs_w).exp(),).to_event(1)
        log_Pzs_wy = P_zs_w.log_prob(z.unsqueeze(1)) * q_y.probs
        # normalize log probs with the y distribution
        # shape is [B, Class, ]
        log_Pz = log_Pzs_wy.sum(-1)
        loss = log_Qz_x - log_Pz
        return loss.mean()

    def kld_w_loss(self, q_w):
        # KLD(q_w || prior_w)
        # recall prior_w ~ N(0,I)
        mu = q_w.mean
        var = q_w.variance
        logvar = var.log()
        loss = self.kld_unreduced(mu=mu, logvar=logvar).sum(-1).mean()
        return loss

    def kld_y_loss(self, q_y):
        # KLD(q_y || p_y)
        # recall p_y = 1/k fixed prior
        q = q_y.probs
        p = torch.ones_like(q) / self.nclasses
        loss = nn.KLDivLoss(reduction='none')(input=q.log(), target=p)
        return loss.sum(-1).mean()
        
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
                q_z, q_w, z, w, q_y, mus_z, logvars_z, y, logit_x = self.forward(x)
                #loss_rec = self.reconstruction_loss(logit_x, x, False)
                loss_rec = self.reconstruction_loss(logit_x, x, True)
                loss_z = self.kld_z_loss(q_z, mus_z, logvars_z, q_y)
                loss_w = self.kld_w_loss(q_w)
                loss_y = self.kld_y_loss(q_y)
                loss = loss_rec + loss_z  + loss_w + loss_y 
                #loss = loss_rec + loss_z * 1e1 + loss_w + loss_y * 1e3
                #loss_y = F.threshold(loss_y, 1, 1)
                #loss = loss_rec + loss_z * 1e1 + loss_w + loss_y
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print(
                        "loss = ",
                        loss.item(),
                        loss_z.item(),
                        loss_w.item(),
                        loss_y.item(),
                    )
        self.cpu()
        optimizer = None
        print("done training")
        return None

    def generate_class(self, c : int, batch : int = 5):
        w = self.w_prior.sample((batch, self.nw))
        mus_z, logvars_z = self.decode_zs_w(w)
        mu = mus_z[:,c,:]
        logvar = logvars_z[:,c,:]
        eps = torch.randn_like(mu)
        z = mu + eps * (logvar * 0.5).exp()
        return z



    #def decode_zs(self, w):
    #    hw = self.dec_h_w(w)
    #    mus = self.mus_zs_hw(hw)
    #    logvars = self.logvars_zs_hw(hw)
    #    #[batchdim=128, eventdim=(10,nz)]
    #    p_zs = pyrodist.Normal(loc=mus, scale=(0.5*logvars).exp()).to_event(2)
    #    zs = p_zs.rsample()
    #    return mus, logvars, p_zs, zs


    #def decode_x_wy(self, w, y_logits=None):
    #    hw = self.dec_h_w(w)
    #    mus = self.mus_zs_hw(hw)
    #    logvars = self.logvars_zs_hw(hw)
    #    #[batchdim=128, eventdim=(10,nz)]
    #    p_zs = pyrodist.Normal(loc=mus, scale=(0.5*logvars).exp()).to_event(2)
    #    zs = p_zs.rsample()
    #    if y_logits is not None:
    #        # draw classes
    #        choice = F.gumbel_softmax(logits=y_logits, tau=self.tau.item(), hard=True)
    #        zs = zs * choice.unsqueeze(-1)
    #        zs = zs.sum(dim=1)
    #    hz = self.dec_h_z(zs)
    #    x_logit = self.logits_x_hz(hz)
    #    return mus, logvars, p_zs, zs, x_logit, choice

    #def forward(self, x):
    #    mu_z, logvar_z, z, mu_w, logvar_w, w, y_logit = encodings = self.encode(x)
    #    mus, logvars, p_zs, zs = self.decode_zs(w)
    #    choice = F.gumbel_softmax(logits=y_logit, tau=self.tau.item(), hard=True)
    #    zhat = (zs * choice.unsqueeze(-1)).sum(dim=1)
    #    hz = self.dec_h_z(zhat)
    #    x_logit = self.logits_x_hz(hz)
    #    out_dict = {
    #            "x_logit" : x_logit,
    #            "z_mus" : mus,
    #            "z_logvars" : logvars,
    #            "p_zs" : p_zs,
    #            "zs" : zs,
    #            "choice" : choice,
    #            "zhat" : zhat,
    #            "mu_z_x" : mu_z,
    #            "logvar_z_x" : logvar_z,
    #            "z" : z,
    #            "mu_w_x" : mu_w,
    #            "logvar_w_x" : logvar_w,
    #            "w" : w,
    #            "y_logit" : y_logit,
    #            }
    #    #return x_logit, mus, logvars, p_zs, zs, choice, zhat, z, *encodings
    #    return out_dict




class SigmaVAE(nn.Module):
    def __init__(
        self,
        nz: int = 20,
        nh: int = 2*1024,
        nin: int = 28 ** 2,
        imgsize: Optional[int] = 28,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin
        self.nz = nz
        self.imgsize = imgsize
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
        )
        self.xmu = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nin),
                )
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
        #self.log_sigma = torch.nn.Parameter(torch.zeros(1)[0], requires_grad=True)
        # per pixel sigma:
        self.log_sigma = torch.nn.Parameter(torch.zeros(nin), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )

    def reparameterize(self, mu, logsig, ):
        eps = torch.randn(mu.shape).to(mu.device)
        sigma = (logsig).exp()
        return mu + sigma * eps

    def encode(self, x):
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder(z)
        mu = self.xmu(h)
        return mu

    def forward(self, x):
        zmu, zlogvar = self.encode(x)
        z = self.reparameterize(zmu, 0.5*zlogvar)
        xmu = self.decode(z)
        return zmu, zlogvar, xmu

    def reconstruction_loss(self, x, xmu, log_sigma):
        # log_sigma is the parameter for 'global' variance on x
        #result = gaussian_nll(xmu, xlogsig, x).sum()
        #result = -ut.log_gaussian_prob(x, xmu, log_sigma).sum()
        q_x = pyrodist.Normal(loc=xmu, scale=log_sigma.exp()).to_event(1)
        result = -q_x.log_prob(x).sum(-1).mean()
        return result
    
    def loss_function(self, x, xmu, log_sigma, zmu, zlogvar):
        #batch_size = x.size(0)
        #rec = self.reconstruction_loss(x, xmu, log_sigma) / batch_size
        rec = self.reconstruction_loss(x, xmu, log_sigma)
        kl = self.kld_unreduced(zmu, zlogvar).sum(-1).mean()
        return rec, kl

    def init_kmeans(self, nclusters, data):
        """
        initiate the kmeans cluster heads
        """
        self.cpu()
        lattent_data, _ = self.encode(data)
        kmeans = KMeans(nclusters, n_init=20)
        y_pred = kmeans.fit_predict(lattent_data.detach().numpy())
        #self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        self.y_pred = y_pred
        #self.q = q = self.soft_assign(lattent_data)
        #self.p = p = self.target_distribution(q)
        self.kmeans = kmeans

    def fit(self, train_loader, num_epochs=10, lr=1e-3,
            optimizer = None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        if not optimizer:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                log_sigma = ut.softclip(self.log_sigma, -2, 2)
                #self.log_sigma.fill_(log_sigma)
                x = data.flatten(1).to(device)
                zmu, zlogvar, xmu = self.forward(x)
                rec, kl = self.loss_function(x, xmu, log_sigma, zmu, zlogvar)
                loss = rec + kl
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print("loss = ",
                            loss.item(),
                            kl.item(),
                            rec.item(),
                            )
        self.cpu()
        optimizer = None
        print('done training')
        return None
