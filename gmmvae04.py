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

class AE_Type00(nn.Module):
    """
    simple autoencoder.
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
        ## P network
        self.Px = nn.Sequential(
            nn.Linear(nz, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
        )
        ## Q network
        self.Qz = nn.Sequential(
            nn.Linear(nx, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz),
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
        z = self.Qz(x)
        output["z"] = z
        rec = self.Px(z)
        output["rec"]= rec
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        loss_rec = nn.MSELoss(reduction='none')(rec,x).mean()
        losses["rec"] = loss_rec
        #losses["rec"] = loss_rec * 1e2
        total_loss = (
                loss_rec
                + 0e0
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
        #optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        optimizer = optim.Adam(self.parameters(), lr=lr,)
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
                [nz, nh, nh, nx],)
        ## Q network
        self.Qz = ut.buildNetworkv2(
                [nx, nh, nh, nz],)
        self.Qy = ut.buildNetworkv2(
                [nx, nh, nh, nclasses],)
        self.Ez = ut.buildNetworkv2(
                [nclasses, nh, nz],)
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
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
                + loss_l
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
                [nz, nh, nh, nx],)
        ## Q network
        self.Qz = ut.buildNetworkv2(
                [nx, nh, nh, nz * 2],)
        self.Qy = ut.buildNetworkv2(
                [nx, nh, nh, nclasses],)
        self.Ez = ut.buildNetworkv2(
                [nclasses, nh, nz],)
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
                loss_rec * 1e2
                + 0e0
                + loss_cat
                + loss_l
                + loss_cluster
                + loss_z * 1e2
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

class AAE_Type01(nn.Module):
    """
    AAE with categorical and Gaussian encoders.
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
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=0.1,)
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.Qh = ut.buildNetworkv2([nx,nh,nh,nh],)
        self.Qz = ut.buildNetworkv2([nh,nz],)
        self.Qy = ut.buildNetworkv2([nh,nclasses],)
        self.Px = ut.buildNetworkv2(
                [nz + nclasses, nh, nh, nx],
                )
        self.Dz = ut.buildNetworkv2(
                [nz, nh, nh, 1],
                )
        self.Dy = ut.buildNetworkv2(
                [nclasses, nh, nh, 1],
                )
        return
    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        losses = {}
        output = {}
        x = nn.Flatten()(input)
        h = self.Qh(x)
        y_logit = self.Qy(h)
        y = y_logit.softmax(-1)
        z = self.Qz(h)
        dy_logit = self.Dy(y)
        dy = dy_logit.sigmoid()
        dz_logit = self.Dz(z)
        dz = dz_logit.sigmoid()
        zy = torch.cat([z,y], dim=-1)
        rec = self.Px(zy)
        output["y"] = y
        output["q_y"] = y
        output["z"] = z
        output["dy"] = dy
        output["dz"] = dz
        output["rec"] = rec
        return output

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        #optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        optimizerD = optim.Adam([
            {'params' : self.Dy.parameters()},
            {'params' : self.Dz.parameters()},
            ],
                lr=lr*1e-1,
                weight_decay=0e-4)
        optimizerQ = optim.Adam([
            {'params' : self.Qh.parameters()},
            {'params' : self.Qz.parameters()},
            {'params' : self.Qy.parameters()},
            ],
                lr=lr*3e-1,
                weight_decay=0e-4)
        optimizerPQ = optim.Adam([
            {'params' : self.Qh.parameters()},
            {'params' : self.Qz.parameters()},
            {'params' : self.Qy.parameters()},
            {'params' : self.Px.parameters()},
            ],
                lr=lr,
                weight_decay=0e-4)
        losses = {}
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                bce = nn.BCELoss(reduction="none")
                batch_size = x.shape[0]
                true_labels = torch.ones((batch_size, 1), device=x.device,)
                false_labels = torch.zeros((batch_size, 1), device=x.device,)
                ### Reconstruction phase
                output = self.forward(x)
                rec = output["rec"]
                loss_rec = 1e3 * nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
                #loss_rec = nn.MSELoss(reduction='none')(rec,x).mean()
                optimizerPQ.zero_grad()
                loss_rec.backward()
                optimizerPQ.step()
                losses["rec"] = loss_rec
                ## Regularization phases
                ### Discriminators
                output = self.forward(x)
                dz = output["dz"]
                loss_dz_false = bce(dz, false_labels).mean()
                dy = output["dy"]
                loss_dy_false = bce(dy, false_labels).mean()
                #y_sample = self.y_prior.sample((batch_size,)).to(x.device)
                #z_sample = self.z_prior.sample((batch_size,)).to(x.device)
                y_sample = self.y_prior.rsample((batch_size,)).to(x.device)
                z_sample = self.z_prior.rsample((batch_size,)).to(x.device)
                dy_true = self.Dy(y_sample).sigmoid()
                dz_true = self.Dz(z_sample).sigmoid()
                loss_dy_true = bce(dy_true, true_labels).mean()
                loss_dz_true = bce(dz_true, true_labels).mean()
                loss_d = (
                        loss_dy_false * 1e2
                        + loss_dz_false
                        + loss_dy_true * 1e2
                        + loss_dz_true
                        ) * 1e-1
                optimizerD.zero_grad()
                loss_d.backward()
                optimizerD.step()
                losses["loss_d"] = loss_d
                ### Generators
                output = self.forward(x)
                dz = output["dz"]
                dy = output["dy"]
                loss_gz = bce(dz, true_labels).mean()
                loss_gy = bce(dy, true_labels).mean()
                loss_g = (
                        loss_gz
                        + loss_gy
                        ) * 5e-1
                optimizerQ.zero_grad()
                loss_g.backward()
                optimizerQ.step()
                losses["loss_g"] = loss_g
                if idx % 1500 == 0:
                    self.printDict(losses)
                    print()
        self.cpu()
        print("done training")
        return

    def fitv2(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr, )
        losses = {}
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                batch_size = x.shape[0]
                true_labels = torch.ones((batch_size, 1), device=x.device,)
                false_labels = torch.zeros((batch_size, 1), device=x.device,)
                bce = nn.BCELoss(reduction="none")
                mse = nn.MSELoss(reduction="none")
                h = self.Qh(x)
                y_logit = self.Qy(h)
                y = y_logit.softmax(-1)
                z = self.Qz(h)
                dy_logit = self.Dy(y)
                dy = dy_logit.sigmoid()
                dz_logit = self.Dz(z)
                dz = dz_logit.sigmoid()
                zy = torch.cat([z,y], dim=-1)
                rec = self.Px(zy)
                loss_rec = mse(rec, x).sum(-1).mean()
                losses["loss_rec"] = loss_rec
                loss = loss_rec
                y_sample = self.y_prior.rsample((batch_size,)).to(x.device)
                z_sample = self.z_prior.rsample((batch_size,)).to(x.device)
                #y_sample = self.y_prior.sample((batch_size,)).to(x.device)
                #z_sample = self.z_prior.sample((batch_size,)).to(x.device)
                loss_dy_true = -self.Dy(y_sample).clamp(-1e2,1e2).mean()
                loss_dz_true = -self.Dz(z_sample).clamp(-1e2,1e2).mean()
                loss_dy_false = self.Dy(y.clone().detach()).clamp(-1e2,1e2).mean()
                loss_dz_false = self.Dz(z.clone().detach()).clamp(-1e2,1e2).mean()
                loss_d = (
                        loss_dy_true * 1e2
                        + loss_dy_false * 1e2
                        + loss_dz_true
                        + loss_dz_false
                        )
                losses["loss_d"] = loss_d
                loss = loss + loss_d
                loss_gz = -self.Dz(z).clamp(-1e2,1e2).mean()
                loss_gy = -self.Dy(y).clamp(-1e2,1e2).mean()
                loss_g = (
                        loss_gz * 1e2
                        + loss_gy * 1e2
                        )
                losses["loss_g"] = loss_g
                loss = loss + loss_g
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses["loss"] = loss
                if idx % 1500 == 0:
                    self.printDict(losses)
                    print()
        self.cpu()
        print("done training")
        return


class VAE_dirichlet_type04(nn.Module):
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
        self.Px = ut.buildNetworkv2(
                [nz, nh, nh, nx],)
        self.Pz = ut.buildNetworkv2(
                [nw, nh, nh, 2 * nclasses * nz],)
        self.Pz.add_module(
                        "unflatten",
                        nn.Unflatten(1, (nclasses, 2*nz)),
                        )
        self.Qwz = ut.buildNetworkv2(
                [nx, nh, nh, 2*nz + 2*nw],)
        self.Qwz.add_module("unflatten", nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv2(
                [nw + nz, nh, nh, nclasses],)

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return



class VAE_Dilo3(nn.Module):
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
        self.Px = nn.Sequential(
            nn.Linear(nz, nh),
            #nn.Dropout(p=0.2),
            #nn.BatchNorm1d(
            #    num_features=nh,
            #),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            #nn.BatchNorm1d(
            #    num_features=nh,
            #),
            #nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
            #nn.Sigmoid(),
        )
        self.Pz = nn.Sequential(
            nn.Linear(nw, nh),
            #nn.Dropout(p=0.2),
            #nn.BatchNorm1d(
            #    num_features=nh,
            #),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            #nn.Dropout(p=0.2),
            #nn.BatchNorm1d(
            #    num_features=nh,
            #),
            nn.LeakyReLU(),
            nn.Linear(nh, 2 * nclasses * nz),
            nn.Unflatten(1, (nclasses, 2*nz)),
            # nn.Tanh(),
        )
        ## Q network
        self.Qwz = nn.Sequential(
            nn.Linear(nx, nh),
            #nn.Dropout(p=0.2),
            #nn.BatchNorm1d(
            #    num_features=nh,
            #),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.Linear(nh, nh),
            #nn.Dropout(p=0.2),
            #nn.BatchNorm1d(
            #    num_features=nh,
            #),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.Linear(nh, 2 * nw + 2 * nz),
            nn.Unflatten(1, (2, nz + nw)),
        )
        self.Qy = nn.Sequential(
            nn.Linear(nw + nz, nh),
            #nn.Dropout(p=0.2),
            #nn.BatchNorm1d(
            #    num_features=nh,
            #),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            #nn.Dropout(p=0.2),
            #nn.BatchNorm1d(
            #    num_features=nh,
            #),
            nn.LeakyReLU(),
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
        logsigma_x = ut.softclip(self.logsigma_x, -2, 2)
        losses = {}
        output = {}
        sigma_x = logsigma_x.exp()
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
        Qy = distributions.RelaxedOneHotCategorical(
                temperature=0.1, probs=q_y)
        y = Qy.rsample().to(x.device)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        output["y"] = y
        rec = self.Px(z)
        loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        #Qx = distributions.Normal(loc=rec, scale=sigma_x)
        #loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec * 1e2
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
        #loss_z = Qz.log_prob(z).unsqueeze(1).sum(-1) 
        #loss_z = loss_z - Pz.log_prob(z.unsqueeze(1)).sum(-1)
        #loss_z = (q_y*loss_z).sum(-1).mean()
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
        lp_y = self.y_prior.logits.to(x.device)
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        ## bad idea:
        #loss_l = (q_y * (
        #        q_y.log() - lp_y)).mean(-1).sum()
        losses["loss_l"] = loss_l
        #loss_y = -(q_y * q_y.log() ).sum(-1).mean()
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        losses["loss_w"]=loss_w
        total_loss = (
                loss_rec
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

class VAE_2moons_type005(nn.Module):
    """
    """
    def __init__(
        self,
        nx: int = 2,
        nh: int = 1024,
        nz: int = 10,
        nclasses: int = 2,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=0.1,)
        #self.w_prior = distributions.Normal(
        #    loc=torch.zeros(nw),
        #    scale=torch.ones(nw),
        #)
        self.z_prior = distributions.Uniform(low=-1,high=1)
        self.Px = ut.buildNetworkv2(
                [nz, nh, nh, nx * nclasses],)
        self.Px.add_module(
                        "unflatten",
                        nn.Unflatten(1, (nclasses, nx)),
                        )
        self.Qy = ut.buildNetworkv2(
                [nx, nh, nh, nclasses],)
        self.Qz = ut.buildNetworkv2(
                [nx + nclasses, nh, nh, nz * 2],)
        #self.Pz = ut.buildNetworkv2(
        #        [nw, nh, nh, 2 * nclasses * nz],)
        #self.Pz.add_module(
        #                "unflatten",
        #                nn.Unflatten(1, (nclasses, 2*nz)),
        #                )
        #self.Qwz = ut.buildNetworkv2(
        #        [nx, nh, nh, 2*nz + 2*nw],)
        #self.Qwz.add_module("unflatten", nn.Unflatten(1, (2, nz + nw)))
        #self.Qy = ut.buildNetworkv2(
        #        [nw + nz, nh, nh, nclasses],)

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        x = nn.Flatten()(input)
        logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
        losses = {}
        output = {}
        sigma_x = logsigma_x.exp()
        q_y_logits = self.Qy(x)
        q_y = q_y_logits.softmax(-1)
        xy = torch.hstack([x,q_y])
        zmlv = self.Qz(xy)
        mu = zmlv[:,:self.nz]
        logvar = zmlv[:,self.nz:]
        std = (0.5 * logvar).exp()
        Qz = distributions.Normal(mu, std)
        z = Qz.rsample().to(x.device)
        loss_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        #Pz = distributions.Uniform(
        #        low = -torch.ones_like(z),
        #        high = torch.ones_like(z),
        #        )
        #loss_z = (
        #        Qz.log_prob(z) 
        #        - Pz.log_prob(z)
        #        ).sum(-1).mean()
        losses["loss_z"] = loss_z
        xs = self.Px(z)
        output["q_y"] = q_y
        output["mu"]=z
        output["logvar"]=z
        output["z"]=z
        output["xs"] = xs
        lp_y = self.y_prior.logits.to(x.device)
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_l"] = loss_l
        Qx = distributions.Normal(loc=xs, scale=sigma_x)
        loss_rec = (-Qx.log_prob(x.unsqueeze(1)).sum(-1) 
                * q_y).mean()
        losses["loss_rec"] = loss_rec
        total_loss = (
                loss_rec
                + loss_z * 1e0
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
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["total_loss"]
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

