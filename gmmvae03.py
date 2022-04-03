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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx % 350 == 0:
                    self.printDict(output["losses"])
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


class VAE_DiloModified(nn.Module):
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
        self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        # P network
        self.Px_z = nn.Sequential(
            nn.Linear(nz, nh),
            # nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(nh, nx),
            nn.Sigmoid(),
        )
        self.Pz_w = nn.Sequential(
            nn.Linear(nw, nh),
            # nn.ReLU(),
            nn.Tanh(),
            nn.Linear(nh, 2 * nclasses * nz),
            nn.Tanh(),
        )
        # Q network
        self.Qwz_x = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nx, nh),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(nh, 2 * (nz + nw)),
            nn.Tanh(),
        )
        self.Qy_wz = nn.Sequential(
            nn.Linear(nw + nz, nh),
            nn.ReLU(),
            nn.Linear(nh, nclasses),
            nn.Softmax(dim=-1),
        )
        self.Py_x = nn.Sequential(
            nn.Linear(nx, nh),
            nn.ReLU(),
            nn.Linear(nh, nclasses),
            nn.Softmax(),
        )
        ## P network
        # self.Px_z = nn.Sequential(
        #        nn.Linear(nz, nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh,nh),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        #nn.Dropout(p=0.2),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh,nx),
        #        nn.Sigmoid(),
        #        )
        # self.Pz_w = nn.Sequential(
        #        nn.Linear(nw, nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh,nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh, 2*nclasses*nz),
        #        #nn.Tanh(),
        #        )
        ## Q network
        # self.Qz_x = nn.Sequential(
        #        nn.Flatten(),
        #        nn.Linear(nx, nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh,nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh,2*nz),
        #        #nn.Tanh(),
        #        )
        # self.Qw_x = nn.Sequential(
        #        nn.Linear(nx, nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh,nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh,2*nw),
        #        )
        # self.Qy_wz = nn.Sequential(
        #        nn.Linear(nw+nz, nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh,nh),
        #        #nn.Dropout(p=0.2),
        #        #nn.BatchNorm1d(num_features=nh,),
        #        nn.LeakyReLU(),
        #        nn.Linear(nh, nclasses),
        #        nn.Softmax(dim=-1),
        #        )

        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        logsigma_x = ut.softclip(self.logsigma_x, -2, 2)
        sigma_x = logsigma_x.exp()
        x = nn.Flatten()(input)
        wzs = self.Qwz_x(x)
        mu_z_x = wzs[:, : self.nz]
        logvar_z_x = wzs[:, self.nz : self.nz + self.nz]
        mu_w = wzs[:, 2 * self.nz : 2 * self.nz + self.nw]
        logvar_w = wzs[:, 2 * self.nz + self.nw :]
        # zs = self.Qz_x(x)
        # mu_z_x = zs[:,:self.nz]
        # logvar_z_x =zs[:,self.nz:]
        std_z_x = (0.5 * logvar_z_x).exp()
        q_z = distributions.Normal(loc=mu_z_x, scale=std_z_x)
        z = q_z.rsample()
        # ws = self.Qw_x(x)
        # mu_w = ws[:,:self.nw]
        # logvar_w = ws[:,self.nw: ]
        std_w = (0.5 * logvar_w).exp()
        q_w = distributions.Normal(loc=mu_w, scale=std_w)
        w = q_w.rsample()
        mus_logvars_z_w = self.Pz_w(w).reshape(-1, self.nclasses, 2 * self.nz)
        mus_z_w = mus_logvars_z_w[:, :, : self.nz]
        logvars_z_w = mus_logvars_z_w[:, :, self.nz :]
        p_z = distributions.Normal(
            loc=mus_z_w,
            scale=(0.5 * logvars_z_w).exp(),
        )
        rec = self.Px_z(z)
        q_x = distributions.Normal(loc=rec, scale=sigma_x)
        # rec_loss = 1e0 * nn.MSELoss(reduction="none")(x,rec).sum(-1).mean()
        # rec_loss = 1e0 * nn.BCELoss(reduction="none")(rec, x).sum(-1).mean()
        rec_loss = -q_x.log_prob(x).sum(-1).mean().relu()
        losses = {}
        losses["rec_loss"] = rec_loss
        total_loss = rec_loss
        wz = torch.cat([w, z], dim=-1)
        q_y = self.Qy_wz(wz)
        z_loss = q_z.log_prob(z).unsqueeze(1).sum(-1)
        z_loss = z_loss - p_z.log_prob(z.unsqueeze(1)).sum(-1)
        z_loss = q_y * z_loss
        z_loss = z_loss.sum(-1).mean()
        total_loss = total_loss + z_loss
        losses["z_loss"] = z_loss
        w_loss = (
            1e0
            * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w,
            )
            .sum(-1)
            .mean()
        )
        losses["w_loss"] = w_loss
        total_loss = total_loss + w_loss
        # y_loss = -1e0 * q_y.max(-1)[0].log().mean()
        # y_loss = -1e1 * (q_y * q_y.log()).sum(-1).mean()
        y_loss = -1e0 * q_y.max(-1)[0].mean()
        losses["y_loss"] = y_loss
        total_loss = total_loss + y_loss
        l_loss = (
            1e1
            * (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
            .sum(-1)
            .mean()
        )
        losses["l_loss"] = l_loss
        total_loss = total_loss + l_loss
        y_predict = self.Py_x(x)
        eps = torch.tensor(1e-6).to(x.device)
        predict_loss = (
            1
            * 1e1
            * (q_y * ((q_y + eps).log() - (y_predict + eps).log())).sum(-1).mean()
        )
        losses["predict_loss"] = predict_loss
        total_loss = total_loss + predict_loss
        losses["total_loss"] = total_loss
        output = {
            "x": x,
            "mu_z_x": mu_z_x,
            "logvar_z_x": logvar_z_x,
            "mu_w": mu_w,
            "logvar_w": logvar_w,
            "q_z": q_z,
            "q_w": q_w,
            "z": z,
            "w": w,
            "mus_logvars_z_w": mus_logvars_z_w,
            "mus_z_w": mus_z_w,
            "logvars_z_w": logvars_z_w,
            "p_z": p_z,
            "rec": rec,
            "losses": losses,
            "q_y": q_y,
            "y_pred": y_predict,
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
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-6)
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
                if idx % 300 == 0:
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
