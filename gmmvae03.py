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

class AE2(nn.Module):
    """
    AE, clusterhead map
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 10,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nclasses = nclasses
        self.nz = nz
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.y_prior = distributions.OneHotCategorical(
            probs=torch.ones(nclasses),
        )
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.Qh = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(nx, nh),
            nn.BatchNorm1d(nh),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
        )
        self.Qz = nn.Sequential(
            nn.Linear(nh, nz),
            # nn.Softmax(dim=-1),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )
        self.LVz = nn.Sequential(
            nn.Linear(nh, nz),
            # nn.Tanh(),
        )
        self.Qy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nx, nh),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(nh),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nclasses),
            nn.Softmax(dim=-1),
        )
        self.clusterhead_embedding = nn.Sequential(
            nn.Linear(nclasses, nz),
        )
        self.Px = nn.Sequential(
            nn.Linear(nz, nh),
            nn.Dropout(p=0.25),
            # nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            # nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
            nn.Sigmoid(),
            # nn.Unflatten(1, (1,28,28)),
        )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def assignCluster(self, z):
        q_y = torch.eye(self.nclasses).to(z.device)
        q_y = self.clusterhead_embedding(q_y)
        c = (q_y - z.unsqueeze(1)).abs().sum(-1).argmin(-1)
        return c

    def forward(self, input):
        x = nn.Flatten()(input)
        h = self.Qh(x)
        q_y = self.Qy(x)
        # q_y = torch.eye(self.nclasses).to(x.device)
        c_y = self.clusterhead_embedding(q_y)
        # mu_z = self.Qz(h) + q_y
        mu_z = self.Qz(h)
        logvar_z = self.LVz(h)
        std_z = (0.5 * logvar_z).exp()
        eps = torch.randn_like(mu_z).to(x.device)
        z = mu_z + std_z * eps
        # rec = self.Px(mu_z)
        rec = self.Px(z)
        loss_cat = -1e1 * q_y.max(-1)[0].mean()
        loss_l = (
            1e1
            * (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
            .sum(-1)
            .mean()
        )
        #loss_l = (
        #    1e0
        #    * (q_y * (q_y.log() - self.y_prior.logits.to(x.device)))
        #    .sum(-1)
        #    .mean()
        #)
        loss_rec = nn.MSELoss(reduction="none")(rec, x).sum(-1).mean()
        loss_cluster = nn.MSELoss(reduction="none")(c_y, mu_z).sum(-1).mean()
        # loss_cluster = 1e0 * (q_y - mu_z.unsqueeze(1)).abs().pow(2).sum(-1).min(-1)[0].mean()
        # loss_z = mu_z.mean(0)
        mu = mu_z.mean(0)
        # mu = mu_z.mean(0) - 1/self.nclasses
        std = std_z.mean(0)
        # loss_z = self.kld_unreduced(mu,logvar=2*std.log()).sum()
        loss_z = self.kld_unreduced(mu, logvar=logvar_z).sum(-1).mean()
        # total_loss = loss_cat + loss_l + loss_rec + loss_cluster + loss_z
        # total_loss = loss_rec + loss_cluster + loss_z
        total_loss = loss_cat + loss_l + loss_rec + loss_cluster + loss_z
        losses = {
            "loss_cat": loss_cat,
            "loss_l": loss_l,
            "loss_cluster": loss_cluster,
            "loss_rec": loss_rec,
            "loss_z": loss_z,
            "total_loss": total_loss,
        }
        output = {
            "x": x,
            "h": h,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "q_y": q_y,
            "rec": rec,
            "losses": losses,
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
                if idx % 350 == 0:
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

class Encoder(nn.Module):
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 10,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.Qh = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(nx, nh),
            nn.BatchNorm1d(nh),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
        )
        self.Qz = nn.Sequential(
            nn.Linear(nh, nz),
            # nn.Softmax(dim=-1),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )
        self.LVz = nn.Sequential(
            nn.Linear(nh, nz),
            # nn.Tanh(),
        )
        return
    def forward(self, x):
        h = self.Qh(x)
        mu = self.Qz(h)
        logvar = self.LVz(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 10,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.Px = nn.Sequential(
            nn.Linear(nz, nh),
            nn.Dropout(p=0.25),
            # nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            # nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
            nn.Sigmoid(),
            # nn.Unflatten(1, (1,28,28)),
        )
        return
    def forward(self, z):
        rec = self.Px(z)
        return rec

class Clusterer(nn.Module):
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 10,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.Qy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nx, nh),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(nh),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nclasses),
            nn.Softmax(dim=-1),
        )
        self.clusterhead_embedding = nn.Sequential(
            nn.Linear(nclasses, nz),
        )
        return
    def forward(self, x):
        y = self.Qy(x)
        c = self.clusterhead_embedding(y)
        return c

class AE3(nn.Module):
    """
    AE, clusterhead map
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 10,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nclasses = nclasses
        self.nz = nz
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.y_prior = distributions.OneHotCategorical(
            probs=torch.ones(nclasses),
        )
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.Q = Encoder(nx, nh, nz, nclasses,)
        self.C = Clusterer(nx, nh, nz, nclasses,)
        self.P = Decoder(nx, nh, nz, nclasses,)
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def assignCluster(self, z):
        q_y = torch.eye(self.nclasses).to(z.device)
        q_y = self.C.clusterhead_embedding(q_y)
        c = (q_y - z.unsqueeze(1)).abs().sum(-1).argmin(-1)
        return c

    def forward(self, input):
        x = nn.Flatten()(input)
        mu_z, logvar_z = self.Q(x)
        c_z = self.C(x)
        std_z = (0.5 * logvar_z).exp()
        eps = torch.randn_like(mu_z).to(x.device)
        z = mu_z + std_z * eps
        rec = self.P(z)
        output = {
            "x": x,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "c_z": c_z,
            "rec": rec,
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
        optimAE = optim.Adam([
            {'params' : self.Q.parameters(), 'lr' : lr}, 
            {'params' : self.P.parameters(), 'lr' : lr},
            {'params' : self.C.parameters(), 'lr' : lr}, 
            ])
        optimReg = optim.Adam([
            {'params' : self.Q.parameters(), 'lr' : lr*8e-1}, 
            ])
        optimC = optim.Adam([
            {'params' : self.C.parameters(), 'lr' : lr*2}, 
            ])
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                losses = {}
                # train autoencoder
                self.Q.train()
                self.P.train()
                self.C.eval()
                self.P.requires_grad_(True)
                self.Q.requires_grad_(True)
                #self.C.requires_grad_(False)
                self.C.requires_grad_(True)
                output = self.forward(x)
                loss_rec = nn.MSELoss(reduction="none")(output["rec"], x).sum(-1).mean()
                cz = self.C(x)
                loss_cluster = nn.MSELoss(reduction="none")(cz, output["mu_z"]).sum(-1).mean()
                loss_rec = loss_rec + loss_cluster
                optimAE.zero_grad()
                loss_rec.backward()
                optimAE.step()
                losses["loss_rec"] = loss_rec
                # train Q to fit normal_Z and clusterhead
                self.Q.train()
                self.P.eval()
                self.C.eval()
                self.P.requires_grad_(False)
                self.Q.requires_grad_(True)
                self.C.requires_grad_(False)
                mu, logvar = self.Q(x)
                loss_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
                cz = self.C(x)
                loss_cluster = nn.MSELoss(reduction="none")(cz, mu).sum(-1).mean()
                loss_cz = loss_z + loss_cluster
                optimReg.zero_grad()
                loss_cz.backward()
                optimReg.step()
                losses["loss_z"] = loss_z
                losses["loss_cluster"] = loss_cluster
                losses["loss_cz"] = loss_cz
                # train clusterrer
                self.Q.eval()
                self.P.eval()
                self.C.train()
                self.P.requires_grad_(False)
                self.Q.requires_grad_(False)
                self.C.requires_grad_(True)
                mu, logvar = self.Q(x)
                q_y = self.C.Qy(x)
                cz = self.C.clusterhead_embedding(q_y)
                loss_cat = -1e1 * q_y.max(-1)[0].mean()
                loss_l = (
                    1e1
                    * (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
                    .sum(-1)
                    .mean()
                )
                loss_clustermap = nn.MSELoss(reduction="none")(cz, mu).sum(-1).mean()
                loss_c = loss_cat + loss_l + loss_clustermap
                optimC.zero_grad()
                loss_c.backward()
                optimC.step()
                losses["loss_cat"] = loss_cat
                losses["loss_l"] = loss_l
                losses["loss_c"] = loss_c
                losses["loss_clustermap"] = loss_clustermap
                if idx % 350 == 0:
                    self.printDict(losses)
                    print()
        self.cpu()
        optimizer = None
        print("done training")
        return None

    def fit_v2(
        self,
        train_loader: torch.utils.data.DataLoader,
        num_epochs=10,
        lr=1e-3,
        device: str = "cuda:0",
    ) -> None:
        self.to(device)
        optimizerRec = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        optimizerCluster = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        optimizerReg = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                # x = data.to(device)
                self.train()
                self.requires_grad_(True)
                output = self.forward(x)
                loss = output["losses"]["loss_rec"]
                loss = loss + output["losses"]["loss_z"]
                optimizerRec.zero_grad()
                loss.backward()
                optimizerRec.step()
                output = self.forward(x)
                loss = output["losses"]["loss_cluster"]
                optimizerCluster.zero_grad()
                loss.backward()
                optimizerCluster.step()
                output = self.forward(x)
                loss = output["losses"]["loss_cat"]
                loss = loss + output["losses"]["loss_l"]
                optimizerReg.zero_grad()
                loss.backward()
                optimizerReg.step()
                if idx % 350 == 0:
                    self.printDict(output["losses"])
                    print()
        self.cpu()
        print("done training")
        return None

class AE4(nn.Module):
    """
    AE, clusterhead map
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 10,
        nclasses: int = 10,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nclasses = nclasses
        self.nz = nz
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.y_prior = distributions.OneHotCategorical(
            probs=torch.ones(nclasses),
        )
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.Qh = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(nx, nh),
            nn.BatchNorm1d(nh),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
        )
        self.Qz = nn.Sequential(
            nn.Linear(nh, nz),
            # nn.Softmax(dim=-1),
            # nn.Tanh(),
            # nn.Sigmoid(),
        )
        self.LVz = nn.Sequential(
            nn.Linear(nh, nz),
            #nn.Tanh(),
        )
        self.Qy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nx, nh),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(nh),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nclasses),
            nn.Softmax(dim=-1),
        )
        self.clusterhead_embedding = nn.Sequential(
            nn.Linear(nclasses, nz),
        )
        self.Px = nn.Sequential(
            nn.Linear(nz, nh),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.25),
            nn.BatchNorm1d(nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
            nn.Sigmoid(),
            # nn.Unflatten(1, (1,28,28)),
        )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def assignCluster(self, z):
        q_y = torch.eye(self.nclasses).to(z.device)
        q_y = self.clusterhead_embedding(q_y)
        c = (q_y - z.unsqueeze(1)).abs().sum(-1).argmin(-1)
        return c

    def forward(self, input):
        x = nn.Flatten()(input)
        h = self.Qh(x)
        q_y = self.Qy(x)
        # q_y = torch.eye(self.nclasses).to(x.device)
        c_y = self.clusterhead_embedding(q_y)
        #y = F.gumbel_softmax(
        #        logits=q_y.log(), tau=0.3, hard=True)
        #c_y = self.clusterhead_embedding(y)
        f = nn.Threshold(threshold=0.51, value=0.)
        g = nn.Threshold(threshold=-0.1, value=1.)
        y = g(-f(q_y))
        c_y = self.clusterhead_embedding(y)
        # mu_z = self.Qz(h) + q_y
        mu_z = self.Qz(h)
        logvar_z = self.LVz(h)
        std_z = (0.5 * logvar_z).exp()
        eps = torch.randn_like(mu_z).to(x.device)
        z = mu_z + std_z * eps
        # rec = self.Px(mu_z)
        rec = self.Px(z)
        loss_cat = -1e1 * q_y.max(-1)[0].mean()
        loss_l = (
            1e1
            * (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
            .sum(-1)
            .mean()
        )
        loss_rec = nn.MSELoss(reduction="none")(rec, x).sum(-1).mean()
        loss_cluster = nn.MSELoss(reduction="none")(c_y, mu_z).sum(-1).mean()
        # loss_cluster = 1e0 * (q_y - mu_z.unsqueeze(1)).abs().pow(2).sum(-1).min(-1)[0].mean()
        # loss_z = mu_z.mean(0)
        mu = mu_z.mean(0)
        # mu = mu_z.mean(0) - 1/self.nclasses
        std = std_z.mean(0)
        # loss_z = self.kld_unreduced(mu,logvar=2*std.log()).sum()
        #loss_z = self.kld_unreduced(mu, logvar=logvar_z).sum(-1).mean()
        loss_z = self.kld_unreduced(mu_z, logvar_z).sum(-1).mean()
        total_loss = loss_cat + loss_l + loss_rec + loss_cluster + loss_z
        #total_loss = loss_l + loss_rec + loss_cluster + loss_z
        losses = {
            "loss_cat": loss_cat,
            "loss_l": loss_l,
            "loss_cluster": loss_cluster,
            "loss_rec": loss_rec,
            "loss_z": loss_z,
            "total_loss": total_loss,
        }
        output = {
            "x": x,
            "h": h,
            "mu_z": mu_z,
            "logvar_z": logvar_z,
            "q_y": q_y,
            "rec": rec,
            "losses": losses,
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
                if idx % 350 == 0:
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
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
            nn.Sigmoid(),
        )
        self.Pz = nn.Sequential(
            nn.Linear(nw, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, 2 * nclasses * nz),
            nn.Unflatten(1, (nclasses, 2*nz)),
            # nn.Tanh(),
        )
        ## Q network
        self.Qwz = nn.Sequential(
            nn.Linear(nx, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.Linear(nh, 2 * nw + 2 * nz),
            nn.Unflatten(1, (2, nz + nw)),
        )
        self.Qy = nn.Sequential(
            nn.Linear(nw + nz, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
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
        total_loss = (loss_rec
                + loss_z 
                + loss_w
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

class VAE_Dilo3AnnData(nn.Module):
    """
    P(x,y,z,w,l) = P(x|z)P(z|w,y)P(w)P(y|l)P(l)
    using P(y|l)=P(l|y)=delta(x,y),
    P(l)~Cat(1/K), P(w)~N(0,I)
    Q(y,z,w,l|x) = Q(z|x)Q(w|z)Q(y|z,w)Q(l|y)
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
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
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
            #nn.Sigmoid(),
        )
        self.Pz = nn.Sequential(
            nn.Linear(nw, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, 2 * nclasses * nz),
            nn.Unflatten(1, (nclasses, 2*nz)),
            # nn.Tanh(),
        )
        ## Q network
        self.Qwz = nn.Sequential(
            nn.Linear(nx, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.Linear(nh, 2 * nw + 2 * nz),
            nn.Unflatten(1, (2, nz + nw)),
        )
        self.Qy = nn.Sequential(
            nn.Linear(nw + nz, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
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
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
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
        total_loss = (loss_rec
                + loss_z 
                + loss_w
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
            for idx, data in enumerate(train_loader):
                x = data.layers['logcounts'].float().to(device)
                #x = data.X.float().to(device)
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

class VAE_Dirichlet(nn.Module):
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
        self.Px = nn.Sequential(
            nn.Linear(nz, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(),
            nn.Linear(nh, nx),
            nn.Sigmoid(),
        )
        self.Pz = nn.Sequential(
            nn.Linear(nw, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, 2 * nclasses * nz),
            nn.Unflatten(1, (nclasses, 2*nz)),
            # nn.Tanh(),
        )
        ## Q network
        self.Qwz = nn.Sequential(
            nn.Linear(nx, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            #nn.Tanh(),
            nn.Linear(nh, 2 * nw + 2 * nz),
            nn.Unflatten(1, (2, nz + nw)),
        )
        self.Qy = nn.Sequential(
            nn.Linear(nw + nz, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nclasses),
            #nn.Softmax(dim=-1),
        )
        self.Qp = nn.Sequential(
            nn.Linear(nw + nz, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nh),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(
                num_features=nh,
            ),
            nn.LeakyReLU(),
            nn.Linear(nh, nclasses),
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
        eps=1e-5
        sigma_x = logsigma_x.exp()
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
        #d = self.Qp(torch.cat([w,z], dim=1)).exp()
        #d = self.Qp(torch.cat([w,z,q_y], dim=1)).clamp(1e-3,1e3)
        d_logits = self.Qp(torch.cat([w,z], dim=1))
        D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #q_y = D_y.rsample().to(x.device)
        Qy = distributions.RelaxedOneHotCategorical(
                temperature=0.1, probs=q_y)
        y = Qy.rsample().to(x.device)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        output["y"] = y
        rec = self.Px(z)
        loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        #loss_rec = nn.BCELoss(reduction="none")(x, rec).sum(-1).mean()
        #loss_rec = nn.BCELoss(reduction="none")(rec, x).sum(-1).mean()
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
        total_loss = (loss_rec
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

    def forward_old(self, input):
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
        #d = self.Qp(torch.cat([w,z], dim=1)).exp()
        d = self.Qp(torch.cat([w,z], dim=1)).clamp(1e-3,1e3)
        D_y = distributions.Dirichlet(d)
        #q_y = D_y.rsample().to(x.device)
        Qy = distributions.RelaxedOneHotCategorical(
                temperature=0.1, probs=q_y)
        y = Qy.rsample().to(x.device)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        output["y"] = y
        rec = self.Px(z)
        loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
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
        #loss_z = Qz.log_prob(z).unsqueeze(1).sum(-1) 
        #loss_z = loss_z - Pz.log_prob(z.unsqueeze(1)).sum(-1)
        #loss_z = (q_y*loss_z).sum(-1).mean()
        loss_z = ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        #loss_z = (q_y*loss_z).sum(-1).mean()
        f = nn.Threshold(threshold=0.51, value=0.)
        g = nn.Threshold(threshold=-0.1, value=1.)
        y = g(-f(q_y.exp().exp().exp().softmax(-1)))
        loss_z = (y*loss_z).sum(-1).mean()
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
        #loss_l = (q_y * (
        #        q_y.log() - lp_y)).sum(-1).mean()
        losses["loss_l"] = loss_l
        loss_l_alt = (1e1/self.nclasses * (
                q_y.mean(0).log() - lp_y)).sum()
        loss_l_alt = (1e1*q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_l_alt"] = loss_l_alt
        #loss_y = -(q_y * q_y.log() ).sum(-1).mean()
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        losses["loss_w"]=loss_w
        Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-3)
        loss_d = distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
        p_y = Pd.rsample()
        loss_p = (q_y * (
            q_y.log() - p_y.log())).sum(-1).mean()
        losses["loss_p"] = loss_p
        total_loss = (loss_rec
                + loss_z 
                + loss_w
                #+ loss_d
                #+ 1e1 * loss_y
                #+ 1e1 * loss_l
                + 1e0 * loss_l_alt
                #+ loss_p
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
