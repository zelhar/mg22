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
                nn.BatchNorm1d(nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.BatchNorm1d(nh),
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
        recon_loss = -q_x.log_prob(x).sum(-1).mean()
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
