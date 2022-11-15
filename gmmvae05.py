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


#import pytorch_lightning as pl
#from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder

print(torch.cuda.is_available())

class AE_Type02(nn.Module):
    """
    autoencoder with cluster embeder.
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=0.1,)
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        ## P network
        self.Px = ut.buildNetworkv2(
                [nz, nh, nh, nx],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        ## Q network
        self.Qz = ut.buildNetworkv2(
                [nx, nh, nh, nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Qy = ut.buildNetworkv2(
                [nx, nh, nh, nclasses],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Ez = ut.buildNetworkv2(
                [nclasses, nh, nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        #z = self.Qz(x).tanh()
        z = self.Qz(x)
        output["z"] = z
        rec = self.Px(z)
        output["rec"]= rec
        logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
        sigma_x = logsigma_x.exp()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
        losses["rec"] = loss_rec
        #losses["rec"] = loss_rec * 1e2
        q_y = self.Qy(x).softmax(-1)
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        output["q_y"] = q_y
        loss_cat = -1e1 * q_y.max(-1)[0].mean()
        losses["cat"] = loss_cat
        loss_l = (
            1e1
            * (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
            .sum(-1)
            .mean()
        )
        losses["l"] = loss_l
        cz = self.Ez(q_y)
        output["cz"] = cz
        loss_cluster = nn.MSELoss(reduction='none')(cz,z).sum(-1).mean()
        losses["cluster"] = loss_cluster
        total_loss = (
                loss_rec
                + 0e0
                + loss_cat
                + loss_l * 1e2
                + loss_cluster
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output


    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                # x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
                # loss = output["losses"]["rec_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 1500 == 0:
                    self.printDict(output["losses"])
                    print()
                    # print(
                    #    "loss = ",
                    #    loss.item(),
                    # )
        self.cpu()
        optimizer = None
        print("done training")
        return None

class VAE_Type00(nn.Module):
    """
    vanilla variational autoencoder.
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        ## P network
        self.Px = ut.buildNetworkv2(
                [nz, nh, nh, nx],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        ## Q network
        self.Qz = ut.buildNetworkv2(
                [nx, nh, nh, nz * 2],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        temp_z = self.Qz(x)
        mu = temp_z[:,:self.nz]
        logvar = temp_z[:,self.nz:]
        std = (0.5 * logvar).exp()
        noise = torch.randn_like(mu)
        z = mu + std * noise
        output["z"] = z
        output["mu"] = mu
        output["logvar"] = logvar
        rec = self.Px(z)
        output["rec"]= rec
        logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
        sigma_x = logsigma_x.exp()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        losses["rec"] = loss_rec
        loss_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        losses["z"] = loss_z
        total_loss = (
                loss_rec * 1e-0
                + loss_z * 1e-0
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                # x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
                # loss = output["losses"]["rec_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 1500 == 0:
                    self.printDict(output["losses"])
                    print()
        self.cpu()
        optimizer = None
        print("done training")
        return None

class AE_Type03(nn.Module):
    """
    variational autoencoder with cluster embeder.
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=0.1,)
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        ## P network
        self.Px = ut.buildNetworkv2(
                [nz, nh, nh, nx],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        ## Q network
        self.Qz = ut.buildNetworkv2(
                [nx, nh, nh, nz * 2],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Qy = ut.buildNetworkv2(
                [nx, nh, nh, nclasses],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Ez = ut.buildNetworkv2(
                [nclasses, nh, nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        temp_z = self.Qz(x)
        mu = temp_z[:,:self.nz]
        logvar = temp_z[:,self.nz:]
        std = (0.5 * logvar).exp()
        noise = torch.randn_like(mu)
        z = mu + std * noise
        output["z"] = z
        output["mu"] = mu
        output["logvar"] = logvar
        rec = self.Px(z)
        output["rec"]= rec
        logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
        sigma_x = logsigma_x.exp()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        losses["rec"] = loss_rec
        q_y = self.Qy(x).softmax(-1)
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        output["q_y"] = q_y
        loss_cat = -1e1 * q_y.max(-1)[0].mean()
        losses["cat"] = loss_cat
        loss_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        losses["z"] = loss_z
        loss_l = (
            1e1
            * (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
            .sum(-1)
            .mean()
        )
        losses["l"] = loss_l
        cz = self.Ez(q_y)
        output["cz"] = cz
        loss_cluster = nn.MSELoss(reduction='none')(cz,z).sum(-1).mean()
        losses["cluster"] = loss_cluster
        total_loss = (
                loss_rec
                + 0e0
                + loss_cat
                + loss_l
                + loss_cluster
                + loss_z
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output
    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                # x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
                # loss = output["losses"]["rec_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 1500 == 0:
                    self.printDict(output["losses"])
                    print()
                    # print(
                    #    "loss = ",
                    #    loss.item(),
                    # )
        self.cpu()
        optimizer = None
        print("done training")
        return None

class VAE_Dilo_Type04(nn.Module):
    """
    P(x,y,z,w,l) = P(x|z)P(z|w,y)P(w)P(y|l)P(l)
    using P(y|l)=P(l|y)=delta(x,y),
    P(l)~Cat(1/K), P(w)~N(0,I)
    Q(y,z,w,l|x) = Q(z|x)Q(w|z)Q(y|z,w)Q(l|y)
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        nw: int = 150,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=0.1,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv2(
                [nz,nh,nh,nx],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Pz = ut.buildNetworkv2(
                [nw, nh, nh, 2*nclasses*nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv2(
                [nx,nh,nh,2*nw + 2*nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv2(
                [nw + nz, nh, nh, nclasses],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Qy.add_module( "softmax", nn.Softmax(dim=-1))

        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None):
        x = nn.Flatten()(input)
        #logsigma_x = ut.softclip(self.logsigma_x, -2, 2)
        losses = {}
        output = {}
        #sigma_x = logsigma_x.exp()
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        #logvar_w = wz[:,1,:self.nw]
        logvar_w = wz[:,1,:self.nw].tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        mu_z = wz[:,0,self.nw:]
        #logvar_z = wz[:,1,self.nw:]
        logvar_z = wz[:,1,self.nw:].tanh()
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        #q_y = self.Qy(wz[:,0,:])
        q_y = self.Qy(torch.cat([w,z], dim=1))
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        Qy = distributions.RelaxedOneHotCategorical(
                temperature=0.1, probs=q_y)
        #y = Qy.rsample().to(x.device)
        output["q_y"] = q_y
        output["Qy"] = Qy
        output["w"]=w
        output["z"]=z
        #output["y"] = y
        rec = self.Px(z)
        logsigma_x = ut.softclip(self.logsigma_x, -9, 9)
        sigma_x = logsigma_x.exp()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        #Qx = distributions.Normal(loc=rec, scale=sigma_x)
        #loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        #logvar_z_w = z_w[:,:,self.nz:]
        logvar_z_w = z_w[:,:,self.nz:].tanh()
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        output["Qz"] = Qz
        loss_z = ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            lp_y = self.y_prior.logits.to(x.device)
            loss_l = (q_y.mean(0) * (
                    q_y.mean(0).log() - lp_y)).sum()
        else:
            loss_z = (q_y*loss_z).sum(-1).mean()
            Qy = distributions.OneHotCategorical(probs=q_y)
            loss_l = -Qy.log_prob(y).mean()
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_z"] = loss_z
        losses["loss_l"] = loss_l
        losses["loss_y"] = loss_y
        loss_w = self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        #loss_y = -(q_y * q_y.log() ).sum(-1).mean()
        losses["loss_w"]=loss_w
        total_loss = (
                loss_rec * 1e0
                + loss_z * 1e0
                + loss_w * 1e0
                #+ 1e1 * loss_y
                + 1e1 * loss_l
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output


    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                y = labels.to(device)
                # x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x,)
                #output = self.forward(x,y=y)
                loss = output["losses"]["total_loss"]
                # loss = output["losses"]["rec_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 1500 == 0:
                    self.printDict(output["losses"])
                    print()
                    # print(
                    #    "loss = ",
                    #    loss.item(),
                    # )
        self.cpu()
        optimizer = None
        print("done training")
        return None

class VAE_Dirichlet_Type05(nn.Module):
    """
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        nw: int = 150,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        # Dirichlet constant prior:
        self.l = 1e-3
        self.dir_prior = distributions.Dirichlet(1e-3*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=0.1,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv2(
                [nz,nh,nh,nx],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Pz = ut.buildNetworkv2(
                [nw, nh, nh, 2*nclasses*nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv2(
                [nx,nh,nh,2*nw + 2*nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv2(
                [nw + nz, nh, nh, nclasses],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qp = ut.buildNetworkv2(
                [nw + nz, nh, nh, nclasses],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        #self.Qp.add_module( "softmax", nn.Softmax(dim=-1))

        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        eps=1e-5
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw].tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:].tanh()
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        #q_y = self.Qy(wz[:,0,:])
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qp(torch.cat([w,z], dim=1))
        D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        Qy = distributions.RelaxedOneHotCategorical(
                temperature=0.1, probs=q_y)
        y = Qy.rsample().to(x.device)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        output["y"] = y
        rec = self.Px(z)
        logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
        sigma_x = logsigma_x.exp()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        #logvar_z_w = z_w[:,:,self.nz:]
        logvar_z_w = z_w[:,:,self.nz:].tanh()
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        output["Qz"] = Qz
        loss_z = ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        loss_z = (q_y*loss_z).sum(-1).mean()
        losses["loss_z"] = loss_z
        loss_w = self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        lp_y = self.y_prior.logits.to(x.device)
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                (eps+q_y).log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        loss_d = 1e1 * distributions.kl_divergence(D_y, Pd).mean()
        #loss_d = (D_y.log_prob(q_y) - Pd.log_prob(q_y)).mean()
        losses["loss_d"]=loss_d
        p_y = D_y.rsample()
        loss_p = (p_y * (
            p_y.log() - q_y.log())).sum(-1).mean()
        losses["loss_p"] = loss_p
        # alt loss_y:
        #Qy = distributions.OneHotCategorical(probs=q_y)
        p_y = D_y.rsample()
        loss_y_alt = (q_y * (
                (eps+q_y).log() - (eps+p_y).log())).sum(-1).mean()
        losses["loss_y_alt"] = loss_y_alt
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                #+ 1e1 * loss_y
                #+ 1e1 * loss_l
                #+ 1e0 * loss_l_alt
                #+ loss_p
                + loss_y_alt
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                # x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
                # loss = output["losses"]["rec_loss"]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 1500 == 0:
                    self.printDict(output["losses"])
                    print()
                    # print(
                    #    "loss = ",
                    #    loss.item(),
                    # )
        self.cpu()
        optimizer = None
        print("done training")
        return None

def trainSemiSuper(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_epochs=10,
    lr=1e-3,
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled : bool = True,
) -> None:
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wt,)
    for epoch in range(num_epochs):
        print("labeled phase")
        for idx, (data, labels) in enumerate(train_loader_labeled):
            x = data.flatten(1).to(device)
            y = labels.to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            #output = model.forward(x,)
            output = model.forward(x,y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 1500 == 0:
                model.printDict(output["losses"])
                print()
        print("unlabeled phase")
        for idx, (data, labels) in enumerate(train_loader_unlabeled):
            if do_unlabeled == False:
                break
            x = data.flatten(1).to(device)
            y = labels.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(x,)
            #output = model.forward(x,y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 1500 == 0:
                model.printDict(output["losses"])
                print()
            if idx >= len(train_loader_labeled):
                break
        print("eval phase")
        for idx, (data, labels) in enumerate(test_loader):
            x = data.flatten(1).to(device)
            y = labels.to(device)
            model.eval()
            #output = model.forward(x,)
            output = model.forward(x,y=y)
            loss = output["losses"]["total_loss"]
            q_y = output["q_y"]
            ce_loss = (y * q_y.log()).sum(-1).mean()
            if idx % 1500 == 0:
                model.printDict(output["losses"])
                print("ce loss:", ce_loss.item())
                print()
    model.cpu()
    optimizer = None
    print("done training")
    return None

