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


class BaseAE(nn.Module):
    """
    a template class for all AE/VAE garden variaety
    """

    def __init__(
        self,
        nx: int,
        nh: int,
        nz: int,
        nclasses: int,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        # encoder from nx to hidden nh
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                #nn.Linear(nh, nz),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.Linear(nh, nz),
                )
        # map from nh into nz
        self.mu_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                #nn.Tanh(),
                )
        # second map from nh into nz
        self.logvar_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                )
        # cat encoder continues from nh to nclasses
        self.categorical_encoder = nn.Sequential(
                #nn.Flatten(),
                #nn.Linear(nx, nh),
                #nn.BatchNorm1d(nh),
                #nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nclasses),
                nn.Softmax(dim=-1),
                )
        # Bernoulli decoder
        self.decoder = nn.Sequential(
                nn.Linear(nz, nh),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                nn.Sigmoid(),
                )
        return

    def printOutputDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, *args, **kwargs):
        x = nn.Flatten()(input) 
        z = self.encode(x)
        xhat = self.decode(z)
        #loss = nn.MSELoss(reduction="none")(x, xhat).sum(-1).mean()
        lossdict = self.lossFunction(x, xhat)
        output = {
                "x" : x,
                "xhat" : xhat,
                "z" : z,
                "losses" : lossdict,
                }
        return output

    def lossFunction(self, input, target):
        criterion = nn.MSELoss(reduction="none")
        loss = criterion(input, target).sum(-1).mean()
        lossdict = { "total_loss" : loss}
        return lossdict

    def _initWeights(
        self,
    ):
        self.apply(init_weights)

    def encode(
        self, x,
    ):
        h = self.encoder(x)
        mu = self.mu_z(h)
        return mu

    def decode(
        self, z,
    ):
        xhat = self.decoder(z)
        return xhat

    def generate(self, sampler: distributions.Distribution, size: torch.Size):
        raise NotImplementedError()

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
                # x = data.flatten(1).to(device)
                x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    self.printOutputDict(output["losses"])
                    # print(
                    #    "loss = ",
                    #    loss.item(),
                    # )
        self.cpu()
        optimizer = None
        print("done training")
        return None

class AE(BaseAE):
    """
    Vanilla autoencoder model.
    """
    def __init__(self,
            nx : int=28**2,
            nh : int=1024,
            nz : int=16,
            nclasses: int=16,
            ) -> None:
        super().__init__(nx, nh, nz, nclasses,)

class BaseGAN(nn.Module):
    """
    a basic categorical GAN model.
    nh: dim of hidden layer,
    nz: dim of the generator's space
    nx: dim of the target space
    """
    def __init__(self, nx : int=3, nh : int=124, nz : int=6, nsamples : int=64):
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nsamples = nsamples
        self.generator = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nz, nh),
                #nn.Linear(nz*nsamples, nh),
                nn.Dropout(p=0.25),
                nn.ReLU(),
                nn.Linear(nh,nh),
                nn.Dropout(p=0.25),
                nn.ReLU(),
                #nn.Linear(nh,nx*nsamples),
                nn.Linear(nh,nx),
                #nn.Unflatten(dim=-1, unflattened_size=(nsamples, nx)),
                nn.Softmax(dim=-1),
                )
        self.discriminator = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.25),
                nn.Linear(nx, nh),
                #nn.Linear(nx*nsamples, nh*nsamples),
                nn.ReLU(),
                nn.Dropout(p=0.25),
                #nn.Linear(nh*nsamples,nh),
                nn.Linear(nh,nh),
                nn.ReLU(),
                nn.Linear(nh,1),
                nn.Sigmoid(),
                )
        self.prior_z = distributions.Normal(
                loc=torch.zeros(nz),
                scale=torch.ones(nz),
                )
        #self.target_x = distributions.OneHotCategorical(
        #        probs=torch.ones(nx),
        #        )
        self.target_x = distributions.RelaxedOneHotCategorical(
                temperature=torch.tensor(0.2),
                probs=torch.ones(nx),
                )
        return
    def fit(
        self,
        num_epochs=1,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizerGen = optim.Adam(self.generator.parameters(), lr=lr)
        optimizerDisc = optim.Adam(self.discriminator.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for round in range(10**4):
                self.train()
                self.requires_grad_(True)
                # sample from the target distribution
                #x = self.target_x.sample((128,self.nsamples)).to(device)
                x = self.target_x.sample((128,)).to(device)
                # sample form the latent space
                z = self.prior_z.sample((128,)).to(device)
                # generate a sample
                xgen = self.generator(z)
                predict_real = self.discriminator(x)
                true_labels = torch.ones_like(predict_real)
                fake_labels = torch.zeros_like(predict_real)
                predict_fake = self.discriminator(xgen)
                criterion = nn.BCELoss(reduction="none")
                optimizerDisc.zero_grad()
                disc_loss_real = criterion(
                        input=predict_real, target=true_labels).sum(-1).mean()
                disc_loss_fake = criterion(
                        input=predict_fake, target=fake_labels).sum(-1).mean()
                disc_loss = 0.5*(disc_loss_fake + disc_loss_real)
                disc_loss.backward()
                optimizerDisc.step()
                # now train the generator to fool the discriminator
                xgen = self.generator(z)
                predicts = self.discriminator(xgen)
                true_labels = torch.ones_like(predicts)
                fake_labels = torch.zeros_like(predicts)
                optimizerGen.zero_grad()
                gen_loss = criterion(
                        input=predicts,
                        target=true_labels,
                        ).sum(-1).mean()
                gen_loss.backward()
                optimizerGen.step()
                if round % 10**3 == 0:
                    print(
                            "discriminator loss:",
                            disc_loss.item(),
                            "generator loss:",
                            gen_loss.item(),
                            )
        self.cpu()
        optimizerGen = None
        optimizerDisc = None
        print("done training")
        return

class AAE(nn.Module):
    """
    AAE with normal and categorical gans for mixed model.
    """
    def __init__(self, nx : int=28**2,
            nclasses : int=16,
            nz : int=16,
            nh : int=1024,
            adversarial_loss_scale : float=0.9,
            ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.adversarial_loss_scale = adversarial_loss_scale
        self.y_prior = distributions.OneHotCategorical(
                probs=torch.ones(nclasses))
        self.z_prior = distributions.Normal(
                loc=torch.zeros(nz),
                scale=torch.ones(nz),
                )
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.25),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.Dropout(p=0.25),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        self.mu_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.Dropout(p=0.25),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                )
        self.logvar_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.Dropout(p=0.25),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                )
        self.gauss_discriminator = nn.Sequential(
                nn.Linear(nz, nh),
                nn.Dropout(p=0.25),
                nn.ReLU(),
                nn.Linear(nh,1),
                nn.Sigmoid(),
                )
        self.categorical_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.Dropout(p=0.25),
                nn.BatchNorm1d(nh),
                nn.Linear(nh,nh),
                nn.Dropout(p=0.25),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nclasses),
                nn.Softmax(dim=-1),
                )
        self.clusterhead_embedding = nn.Sequential(
                nn.Linear(nclasses, nz),
                )
        self.categorical_discriminator = nn.Sequential(
                nn.Linear(nclasses, nh),
                nn.Dropout(p=0.25),
                nn.ReLU(),
                nn.Linear(nh,1),
                nn.Sigmoid(),
                )
        self.decoder = nn.Sequential(
                nn.Linear(nz, nh),
                nn.Dropout(p=0.25),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                nn.Sigmoid(),
                )
        return

    def reconstruction_loss(self, x, xhat):
        logsigma_x = ut.softclip(self.logsigma_x, -2, 2)
        sigma_x = logsigma_x.exp()
        #q_x = pyrodist.Normal(loc=x_hat, scale=sigma_x).to_event(1)
        q_x = distributions.Normal(loc=xhat, scale=sigma_x)
        #recon_loss = -q_x.log_prob(x).sum(-1).mean()
        recon_loss = nn.MSELoss(reduction="none")(
                x, xhat).sum(-1).mean()
        #recon_loss = -ut.log_gaussian_prob(x, x_hat, sigma_x).sum(-1).mean()
        return recon_loss

    def forward(self, x):
        output = {}
        h = self.encoder(x)
        output["mu_z"] = mu_z = self.mu_z(h)
        output["logvar_z"] = logvar_z = self.logvar_z(h)
        #output["c"] = c = self.categorical_encoder(h)
        output["c"] = c = self.categorical_encoder(x)
        std_z = (0.5 * logvar_z).exp()
        eps = torch.randn_like(mu_z)
        output["z"] = z = mu_z + std_z * eps
        output["cluster_head"] = cz = self.clusterhead_embedding(c)
        output["xhat"] = xhat = self.decoder(z+cz)
        output["rec_loss"] = rec_loss = self.reconstruction_loss(x.flatten(1), xhat)
        output["q_z"] = distributions.Normal(loc=mu_z, scale=std_z)
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
        optimizerGen = optim.Adam(self.parameters(), lr=lr)
        optimizerDisc = optim.Adam(self.parameters(), lr=lr)
        optimizerGenCat = optim.Adam(self.parameters(), lr=lr)
        optimizerDiscCat = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                h = self.encoder(x)
                mu_z = self.mu_z(h)
                logvar_z = self.logvar_z(h)
                #c = self.categorical_encoder(h)
                c = self.categorical_encoder(x)
                cz = self.clusterhead_embedding(c)
                std_z = (0.5 * logvar_z).exp()
                eps = torch.randn_like(mu_z)
                z = mu_z + std_z * eps
                xhat = self.decoder(z + cz)
                rec_loss = self.reconstruction_loss(x, xhat)
                #q_z = distributions.Normal(loc=mu_z, scale=std_z)
                #x = data.to(device)
                self.train()
                self.requires_grad_(True)
                #output = self.forward(x)
                optimizer.zero_grad()
                #rec_loss = output["rec_loss"]
                rec_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # now train gauss GAN
                #q_z = output["q_z"]
                #q_z = distributions.Normal(loc=mu_z, scale=std_z)
                #zfake = q_z.rsample().to(device)
                #zfake = output["z"]
                zfake = z.detach()
                zreal = torch.randn_like(zfake).to(device)
                predict_real = self.gauss_discriminator(zreal.detach())
                predict_fake = self.gauss_discriminator(zfake.detach())
                true_labels = torch.ones_like(predict_real).to(device)
                fake_labels = torch.ones_like(predict_real).to(device)
                criterion = nn.BCELoss(reduction="none")
                optimizerDisc.zero_grad()
                disc_loss_real = criterion(
                        input=predict_real, target=true_labels).sum(-1).mean()
                disc_loss_fake = criterion(
                        input=predict_fake, target=fake_labels).sum(-1).mean()
                disc_loss = 0.5*(disc_loss_fake + disc_loss_real)
                disc_loss.backward()
                optimizerDisc.step()

                # now train the generator to fool the discriminator
                #zfake = q_z.rsample().to(device)
                h = self.encoder(x)
                mu_z = self.mu_z(h)
                logvar_z = self.logvar_z(h)
                std_z = (0.5 * logvar_z).exp()
                eps = torch.randn_like(mu_z)
                z = mu_z + std_z * eps
                zfake = z
                predicts = self.gauss_discriminator(zfake)
                true_labels = torch.ones_like(predicts)
                optimizerGen.zero_grad()
                gen_loss = criterion(
                        input=predicts,
                        target=true_labels,
                        ).sum(-1).mean()
                gen_loss.backward()
                optimizerGen.step()

                # now train cat gan
                criterion = nn.BCELoss(reduction="none")
                cfake = self.categorical_encoder(x)
                creal = self.y_prior.sample((x.shape[0],)).to(device)
                predict_real = self.categorical_discriminator(creal.detach())
                predict_fake = self.categorical_discriminator(cfake.detach())
                true_labels = torch.ones_like(predict_real).to(device)
                fake_labels = torch.ones_like(predict_real).to(device)
                optimizerDiscCat.zero_grad()
                disc_cat_loss_real = criterion(
                        input=predict_real, target=true_labels).sum(-1).mean()
                disc_cat_loss_fake = criterion(
                        input=predict_fake, target=fake_labels).sum(-1).mean()
                disc_cat_loss = 0.5*(disc_cat_loss_fake + disc_cat_loss_real)
                disc_cat_loss.backward()
                optimizerDiscCat.step()

                ## now train the generator to fool the discriminator
                cfake = self.categorical_encoder(x)
                predicts = self.categorical_discriminator(cfake)
                true_labels = torch.ones_like(predicts).to(device)
                optimizerGenCat.zero_grad()
                gen_cat_loss = criterion(
                        input=predicts,
                        target=true_labels,
                        ).sum(-1).mean()
                gen_cat_loss.backward()
                optimizerGenCat.step()
                if idx % 300 == 0:
                    print(
                            "rec_loss:", rec_loss.item(),
                            )
        self.cpu()
        print("done training")
        return

class VAEGMM(nn.Module):
    """
    a VAE with the following generative (P) and inference (Q)
    models:
    P(x,y,z,w) = P(x|z)P(z|w,y)P(w)P(y)
    P(x|z) ~ N(0,I), P(z|w,y=k) ~ N(mu(w,y), sigma(w,y)), 
    P(w) ~ N(0,I), P(y) ~ Cat(pi), pi=1/k 
    Q(y,w,z|x) = Q(z|x)Q(w|z)Q(y|z)
    Q(z|x) ~ N(mu(x), sigma(x)),
    Q(w|z) ~ N(mu(z), sigma(z)),
    Q(y|z) ~ Cat(pi(z))
    Basically it is Kingmas M1+M2 stacked model,
    where M1 is a vannilla VAE between x and z,
    and M2 is a GMM on the reduced z-space.
    """
    def __init__(
        self,
        nx: int=28**2,
        nh: int=1024,
        nz: int=16,
        nclasses: int=10,
        nw: int=16,
    ) -> None:
        super().__init__()
        self.nx=nx
        self.nh=nh
        self.nz=nz
        self.nw=nw
        self.nclasses=nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        # map from nh into nz
        self.mu_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                #nn.Tanh(),
                )
        # second map from nh into nz
        self.logvar_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                )
        self.categorical_encoder_y_z = nn.Sequential(
                nn.Linear(nz, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nclasses),
                nn.Softmax(dim=-1),
                )
        self.encoder_w_z = nn.Sequential(
                nn.Linear(nz, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                #nn.Linear(nh, nz),
                #nn.Tanh(),
                #nn.Sigmoid(),
                #nn.Linear(nh, nz),
                )
        self.mu_w_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nw),
                nn.Tanh(),
                )
        self.logvar_w_z = nn.Sequential(
                nn.Linear(nh, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nw),
                nn.Tanh(),
                )
        self.clusterhead_embedding_w_y = nn.Sequential(
                nn.Linear(nclasses, nw),
                )
        self.mu_z_w = nn.Sequential(
                nn.Linear(nw, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z_w = nn.Sequential(
                nn.Linear(nw, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                )
        self.y_prior = distributions.OneHotCategorical(
                probs=torch.ones(nclasses))
        self.z_prior = distributions.Normal(
                loc=torch.zeros(nz),
                scale=torch.ones(nz),
                )
        self.w_prior = distributions.Normal(
                loc=torch.zeros(nw),
                scale=torch.ones(nw),
                )
        # Bernoulli decoder
        self.decoder = nn.Sequential(
                nn.Linear(nz, nh),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                nn.Sigmoid(),
                )
        return
    def generate_class(self, y : Tensor, nsamples : int=7):
        """
        y is a tensor representing an array of 1-hot encoded vectors,
        each row is a one-hot class representation.
        """
        w = self.w_prior.sample((y.shape[0],))
        cw = self.clusterhead_embedding_w_y(y)
        mu_z = self.mu_z_w(w + cw)
        logvar_z = self.logvar_z_w(w + cw)
        q_z = distributions.Normal(
                loc=mu_z, 
                scale=(0.5 * logvar_z).exp(),
                )
        z = q_z.rsample()
        xhat = self.decoder(z)
        return xhat
    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return
    def forward(self, input):
        x = nn.Flatten()(input)
        output = {}
        losses = output["losses"] = {}
        h = self.encoder(x)
        output["mu_z_x"] = mu_z_x = self.mu_z(h)
        output["logvar_z_x"] = logvar_z_x = self.logvar_z(h)
        std_z_x = (0.5 * logvar_z_x).exp()
        output["q_z"]= q_z = distributions.Normal(loc=mu_z_x, scale=std_z_x)
        z = output["z"] = q_z.rsample().to(x.device)
        output["xhat"] = xhat = self.decoder(z)
        loss_rec = losses["loss_rec"] = nn.MSELoss(reduction="none")(
                x, xhat).sum(-1).mean()
        loss = loss_rec
        h_w_z = self.encoder_w_z(z)
        output["mu_w"] = mu_w_z = self.mu_w_z(h_w_z)
        output["logvar_w"] = logvar_w_z = self.logvar_w_z(h_w_z)
        std_w_z = (0.5 * logvar_w_z).exp()
        output["q_w"]= q_w = distributions.Normal(loc=mu_w_z, scale=std_w_z)
        w = output["w"] = q_w.rsample().to(x.device)
        output["y"] = y_z = self.categorical_encoder_y_z(z)
        losses["y_divergence_loss"] = y_divergence_loss = - y_z.max(-1)[0].mean()
        loss = loss + y_divergence_loss 
        output["q_y"] = q_y = y_z.mean(0)
        losses["y_cat_loss"] = y_cat_loss = (q_y * 
                (q_y.log() - self.y_prior.logits.to(x.device))).sum()
        loss = loss + y_cat_loss
        losses["kld_w"] = kld_w = self.kld_unreduced(mu_w_z, logvar_w_z).sum(-1).mean()
        loss = loss + kld_w
        output["cw"] = cw = self.clusterhead_embedding_w_y(y_z)
        output["mu_z_wy"] = mu_z_wy = self.mu_z_w(w + cw)
        output["logvar_z_wy"] = logvar_z_wy = self.logvar_z_w(w + cw)
        losses["kld_z"] = kld_z = self.kld_unreduced(mu_z_wy, logvar_z_wy).sum(-1).mean()
        loss = loss + kld_z
        losses["total_loss"] = loss
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
                x = data.flatten(1).to(device)
                #x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    self.printDict(output["losses"])
                    # print(
                    #    "loss = ",
                    #    loss.item(),
                    # )
        self.cpu()
        optimizer = None
        print("done training")
        return None

class VAEM2(nn.Module):
    """
    a VAE with the following generative (P) and inference (Q)
    models:
    P(x,y,z) = P(x|z,y)P(z)P(y)
    P(x|z,y) ~ N(mu(z,y),I), P(z) ~ N(0,I), P(y)~Cat(1/K) 
    Q(y,z|x) = Q(z|x)Q(y|x)
    Q(z|x) ~ N(mu(x), sigma(x)),
    Q(y|x) ~ Cat(pi(x))
    Basically it is Kingmas M2 model,
    where M1 is a vannilla VAE between x and z,
    and M2 is a GMM.
    """
    def __init__(
        self,
        nx: int=28**2,
        nh: int=1024,
        nz: int=16,
        nclasses: int=10,
        nw: int=16,
    ) -> None:
        super().__init__()
        self.nx=nx
        self.nh=nh
        self.nz=nz
        #self.nw=nw
        self.nclasses=nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                )
        # map from nh into nz
        self.mu_z = nn.Sequential(
                nn.Linear(nh, nh),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                )
        # second map from nh into nz
        self.logvar_z = nn.Sequential(
                nn.Linear(nh, nh),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                nn.Tanh(),
                )
        self.categorical_encoder = nn.Sequential(
                nn.Linear(nx, nh),
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                #nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nclasses),
                nn.Softmax(dim=-1),
                )
        self.clusterhead_embedding_z_y = nn.Sequential(
                nn.Linear(nclasses, nz),
                )
        self.y_prior = distributions.OneHotCategorical(
                probs=torch.ones(nclasses))
        self.z_prior = distributions.Normal(
                loc=torch.zeros(nz),
                scale=torch.ones(nz),
                )
        # Bernoulli decoder
        self.decoder = nn.Sequential(
                nn.Linear(nz, nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                nn.Sigmoid(),
                )
        return
    def generate_class(self, y : Tensor, nsamples : int=7):
        """
        y is a tensor representing an array of 1-hot encoded vectors,
        each row is a one-hot class representation.
        """
        z = self.z_prior.sample((y.shape[0],))
        cw = self.clusterhead_embedding_z_y(y)
        xhat = self.decoder(z+cw)
        return xhat
    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return
    def forward(self, input):
        x = nn.Flatten()(input)
        output = {}
        output["losses"] = losses = {}
        h = self.encoder(x)
        output["mu_z_x"] = mu_z_x = self.mu_z(h)
        output["logvar_z_x"] = logvar_z_x = self.logvar_z(h)
        std_z_x = (0.5 * logvar_z_x).exp()
        output["q_z"]= q_z = distributions.Normal(loc=mu_z_x, scale=std_z_x)
        output["z"] = z = q_z.rsample().to(x.device)
        output["y"] = y = self.categorical_encoder(x)
        output["q_y"] = q_y = y.mean(0)
        output["cz"] = cz = self.clusterhead_embedding_z_y(y)
        output["xhat"] = xhat = self.decoder(z + cz)
        losses["loss_rec"] = loss_rec = nn.MSELoss(reduction="none")(
                x, xhat).sum(-1).mean()
        loss = loss_rec
        losses["kld_z"] = kld_z = self.kld_unreduced(mu_z_x, logvar_z_x).sum(-1).mean()
        loss = loss + kld_z
        losses["y_divergence_loss"] = y_divergence_loss = -1e1 * y.max(-1)[0].mean()
        loss = loss + y_divergence_loss 
        losses["y_cat_loss"] = y_cat_loss = 1e1*(q_y * 
                (q_y.log() - self.y_prior.logits.to(x.device))).sum()
        loss = loss + y_cat_loss
        losses["total_loss"] = loss
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
                x = data.flatten(1).to(device)
                #x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    self.printDict(output["losses"])
                    # print(
                    #    "loss = ",
                    #    loss.item(),
                    # )
        self.cpu()
        optimizer = None
        print("done training")
        return None

        
class VAE_Dilo(nn.Module):
    """
    a VAE with the following generative (P) and inference (Q)
    models as described by
    https://arxiv.org/abs/1611.02648:
    P(x,y,z) = P(x|z,y)P(z)P(y)
    P(x|z,y) ~ N(mu(z,y),I), P(z) ~ N(0,I), P(y)~Cat(1/K) 
    Q(y,z|x) = Q(z|x)Q(y|x)
    Q(z|x) ~ N(mu(x), sigma(x)),
    Q(y|x) ~ Cat(pi(x))
    Basically it is Kingmas M2 model,
    where M1 is a vannilla VAE between x and z,
    and M2 is a GMM.
    """
    def __init__(
        self,
        nx: int=28**2,
        nh: int=3024,
        nz: int=200,
        nw: int=150,
        nclasses: int=10,
    ) -> None:
        super().__init__()
        self.nx=nx
        self.nh=nh
        self.nz=nz
        self.nw=nw
        self.nclasses=nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.y_prior = distributions.OneHotCategorical(
                probs=torch.ones(nclasses))
        self.w_prior = distributions.Normal(
                loc=torch.zeros(nw),
                scale=torch.ones(nw),
                )
        # P network
        self.Px_z = nn.Sequential(
                nn.Linear(nz, nh),
                nn.ReLU(),
                nn.Linear(nh,nx),
                nn.Sigmoid(),
                )
        self.P_z_wy = nn.Sequential(
                nn.Linear(nw, nh),
                nn.Tanh(),
                nn.Linear(nh, 2*nclasses*nz),
                )
        # Q network
        self.Qwz_x = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.ReLU(),
                nn.Linear(nh,2*(nz + nw)), 
                )
        self.Q_y_wz = nn.Sequential(
                nn.Linear(nw+nz, nh),
                nn.ReLU(),
                nn.Linear(nh, nclasses),
                nn.Softmax(dim=-1),
                )

        return
    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return
    def forward(self, input):
        x = nn.Flatten()(input)
        wzs = self.Qwz_x(x)
        mu_z_x = wzs[:,:self.nz]
        logvar_z_x =wzs[:,self.nz:self.nz + self.nz]
        mu_w_x = wzs[:,2*self.nz:2*self.nz + self.nw]
        logvar_w_x = wzs[:,2*self.nz + self.nw : ]
        std_z_x = (0.5 * logvar_z_x).exp()
        std_w_x = (0.5 * logvar_w_x).exp()
        q_z = distributions.Normal(
                loc=mu_z_x, scale=std_z_x)
        q_w = distributions.Normal(
                loc=mu_w_x, scale=std_w_x)
        z = q_z.rsample()
        w = q_w.rsample()
        mus_logvars_z_w = self.P_z_wy(w).reshape(-1, self.nclasses, 2*self.nz)
        mus_z_w = mus_logvars_z_w[:,:,:self.nz]
        logvars_z_w = mus_logvars_z_w[:,:,self.nz:]
        p_z = distributions.Normal(
                loc=mus_z_w,
                scale=(0.5*logvars_z_w).exp(),
                )
        rec = self.Px_z(z)
        rec_loss = nn.MSELoss(reduction="none")(x,rec).sum(-1).mean()
        losses = {}
        losses["rec_loss"] = rec_loss
        total_loss = rec_loss
        wz = torch.cat([w,z], dim=-1)
        q_y = self.Q_y_wz(wz)
        z_loss = q_z.log_prob(z).unsqueeze(1).sum(-1)
        z_loss = z_loss - p_z.log_prob(z.unsqueeze(1)).sum(-1)
        z_loss = q_y * z_loss
        z_loss = z_loss.sum(-1).mean()
        total_loss = total_loss + z_loss
        losses["z_loss"] = z_loss
        w_loss = self.kld_unreduced(
                mu=mu_w_x,
                logvar=logvar_w_x,).sum(-1).mean()
        losses["w_loss"] = w_loss
        total_loss = total_loss + w_loss
        #y_loss = q_y * (q_y.log() - self.y_prior.logits.to(x.device))
        #y_loss = 10*y_loss.sum(-1).mean()
        y_loss = -1e0 * q_y.max(-1)[0].mean()
        y_loss = y_loss + 1e1 * (
                q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device))
                ).sum(-1).mean()
        losses["y_loss"] = y_loss
        total_loss = total_loss + y_loss

        losses["total_loss"] = total_loss
        output = {
                "x" : x,
                "mu_z_x" : mu_z_x,
                "logvar_z_x" : logvar_z_x,
                "mu_w_x" : mu_w_x,
                "logvar_w_x" : logvar_w_x,
                "q_z": q_z,
                "q_w": q_w,
                "z": z,
                "w": w,
                "mus_logvars_z_w" : mus_logvars_z_w,
                "mus_z_w" : mus_z_w,
                "logvars_z_w" : logvars_z_w,
                "p_z" : p_z,
                "rec": rec,
                "losses" : losses,
                "q_y": q_y,
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
                x = data.flatten(1).to(device)
                #x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    self.printDict(output["losses"])
                    # print(
                    #    "loss = ",
                    #    loss.item(),
                    # )
        self.cpu()
        optimizer = None
        print("done training")
        return None




                
    

# test demo for class inherritance properties
class Aclass():
    def __init__(self,):
        super().__init__()
        return
    def methodA(self, x):
        return x*2
    def methodB(self, ):
        raise NotImplementedError()
    def sayhi(self,):
        print("hi")
        return
    def printOutputDict(self, d : dict):
        for k,v in d.items():
            print(k + ":", v)
        return

class Bclass(Aclass):
    def __init__(self,):
        super().__init__()
        return
    def methodA(self, x,y):
        return y+x*2
    def methodB(self, x):
        print(x)
        return
