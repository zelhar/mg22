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

class Generic_Net(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        return
    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return
    def forward(self, input):
        raise NotImplementedError()


class AE_Primer_Type701(Generic_Net):
    """
    Vanila AE.
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        bn : bool = True,
        dropout : float = 0.2,
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
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Px = ut.buildNetworkv3(
        #        [nz, nh, nh, nh, nx],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        layernorm=bn,
        #        )
        ## Q network
        self.Qz = ut.buildNetworkv2(
                [nx, nh, nh, nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qz = ut.buildNetworkv3(
        #        [nx, nh, nh, nh, nz*2],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        layernorm=bn,
        #        )
        return

    def studentize(self, m, z):
        q = 1 + (m - z.unsqueeze(1)).pow(2).sum(-1)
        q = 1 / q
        s = q.sum(-1, keepdim=True,)
        q = q/s
        return q

    def forward(self, input, y=None):
        x = nn.Flatten()(input)
        losses = {}
        output = {}

        z = self.Qz(x)
        output["z"] = z
        rec = self.Px(z)
        output["rec"]= rec
        logsigma_x = ut.softclip(self.logsigma_x, -9, 9)
        sigma_x = logsigma_x.exp()
        loss_rec = nn.MSELoss(reduction='none')(rec,x).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        #Qx = distributions.Normal(loc=rec, scale=sigma_x)
        #loss_rec = -Qx.log_prob(x).sum(-1).mean()
        losses["rec"] = loss_rec
        #losses["rec"] = loss_rec * 1e2
        total_loss = 1e0 * (
                loss_rec
                + 0e0
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output



class VAE_Primer_Type700(Generic_Net):
    """
    Vanila VAE.
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        bn : bool = True,
        dropout : float = 0.2,
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
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        ## P network
        #self.Px = ut.buildNetworkv2(
        #        [nz, nh, nh, nx],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        self.Px = ut.buildNetworkv3(
                [nz, nh, nh, nh, nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                layernorm=bn,
                )
        ## Q network
        #self.Qz = ut.buildNetworkv2(
        #        [nx, nh, nh, nz*2],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        self.Qz = ut.buildNetworkv3(
                [nx, nh, nh, nh, nz*2],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                layernorm=bn,
                )
        return

    def studentize(self, m, z):
        q = 1 + (m - z.unsqueeze(1)).pow(2).sum(-1)
        q = 1 / q
        s = q.sum(-1, keepdim=True,)
        q = q/s
        return q

    def forward(self, input, y=None):
        x = nn.Flatten()(input)
        losses = {}
        output = {}

        temp_z = self.Qz(x)
        mu = temp_z[:,:self.nz]
        logvar = temp_z[:,self.nz:]
        std = (0.5 * logvar).exp()
        noise = torch.randn_like(mu)
        z = mu + std * noise
        loss_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        losses["z"] = loss_z
        output["z"] = z
        output["mu"] = mu
        output["logvar"] = logvar
        rec = self.Px(z)
        output["rec"]= rec
        logsigma_x = ut.softclip(self.logsigma_x, -9, 9)
        sigma_x = logsigma_x.exp()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
        losses["rec"] = loss_rec
        #losses["rec"] = loss_rec * 1e2
        total_loss = 1e0 * (
                loss_rec
                + 0e0
                + loss_z
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output


class CVAE_anndata_Type703(Generic_Net):
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        bn : bool = True,
        dropout : float = 0.2,
        n_conds : int = 4,
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
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        ## P network
        #self.Px = ut.buildNetworkv2(
        #        [nz, nh, nh, nx],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        self.Px = ut.buildNetworkv3(
                [nz + n_conds, nh, nh, nh, nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                layernorm=bn,
                )
        ## Q network
        #self.Qz = ut.buildNetworkv2(
        #        [nx, nh, nh, nz*2],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        self.Qz = ut.buildNetworkv3(
                [nx + n_conds, nh, nh, nh, nz*2],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                layernorm=bn,
                )
        return

    pass



class VAE_Stacked_Dilo_Anndata_Type701(nn.Module):
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
        nz: int = 32,
        nw: int = 15,
        nclasses: int = 10,
        bn : bool = True,
        dropout : float = 0.2,
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
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv2(
                [nw, nh, nh, 2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv2(
                [nx,nh,nh,2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv2(
                [nw + nz, nh, nh, nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qy.add_module( "softmax", nn.Softmax(dim=-1))

        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        #logvar_w = wz[:,1,:self.nw].tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        #logvar_z = wz[:,1,self.nw:].tanh()
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
        output["q_y"] = q_y
        output["Qy"] = Qy
        output["w"]=w
        output["z"]=z
        rec = self.Px(z)
        #rec = self.Px(mu_z)
        logsigma_x = ut.softclip(self.logsigma_x, -9, 9)
        sigma_x = logsigma_x.exp()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        #logvar_z_w = z_w[:,:,self.nz:].tanh()
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
            #loss_l = (q_y * (
            #        q_y.log() - lp_y)).sum(-1).mean()
        else:
            #loss_z = (q_y*loss_z).sum(-1).mean()
            loss_z = (y*loss_z).sum(-1).mean()
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
                #+ 1e1 * loss_l
                + self.nclasses * loss_l
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
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

class VAE_Dirichlet_Type705(nn.Module):
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

    def forward(self, input, y=None,):
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
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        Qy = distributions.RelaxedOneHotCategorical(
                temperature=0.1, probs=q_y)
        #y = Qy.rsample().to(x.device)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        #output["y"] = y
        rec = self.Px(z)
        logsigma_x = ut.softclip(self.logsigma_x, -9, 9)
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
        lp_y = self.y_prior.logits.to(x.device)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        loss_p = (p_y * (
            p_y.log() - q_y.log())).sum(-1).mean()
        losses["loss_p"] = loss_p
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #lp_y = self.y_prior.logits.to(x.device)
            loss_l = (q_y.mean(0) * (
                    q_y.mean(0).log() - lp_y)).sum()
            #loss_y_alt = (q_y * (
            #        (eps+q_y).log() - (eps+p_y).log())).sum(-1).mean()
            loss_y_alt = (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
        else:
            #loss_z = (q_y*loss_z).sum(-1).mean()
            loss_z = (y*loss_z).sum(-1).mean()
            Qy = distributions.OneHotCategorical(probs=q_y)
            loss_y_alt = loss_l = -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
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
        # alt loss_y:
        #Qy = distributions.OneHotCategorical(probs=q_y)
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
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

#class MLP(nn.Module):
#    def __init__(self, input_dim, hidden_dims, out_dim):
#        super().__init__()
#        modules = []
#        for in_size, out_size in zip([input_dim]+hidden_dims, hidden_dims):
#            modules.append(nn.Linear(in_size, out_size))
#            modules.append(nn.LayerNorm(out_size))
#            modules.append(nn.ReLU())
#            modules.append(nn.Dropout(p=0.05))
#        modules.append(nn.Linear(hidden_dims[-1], out_dim))
#        self.fc = nn.Sequential(*modules)
#    def forward(self, *inputs):
#        input_cat = torch.cat(inputs, dim=-1)
#        return self.fc(input_cat)
#
#class CVAE(nn.Module):
#    # The code is based on the scarches trVAE model
#    # https://github.com/theislab/scarches/blob/v0.3.5/scarches/models/trvae/trvae.py
#    # and on the pyro.ai Variational Autoencoders tutorial
#    # http://pyro.ai/examples/vae.html
#    def __init__(self, input_dim, n_conds, n_classes, hidden_dims, latent_dim):
#        super().__init__()
#        self.encoder = MLP(input_dim+n_conds, hidden_dims, 2*latent_dim) # output - mean and logvar of z
#        self.decoder = MLP(latent_dim+n_conds, hidden_dims[::-1], input_dim)
#        self.theta = nn.Linear(n_conds, input_dim, bias=False)
#        self.classifier = nn.Linear(latent_dim, n_classes)
#        self.latent_dim = latent_dim
#    def model(self, x, batches, classes, size_factors):
#        pyro.module("cvae", self)
#        batch_size = x.shape[0]
#        with pyro.plate("data", batch_size):
#            z_loc = x.new_zeros((batch_size, self.latent_dim))
#            z_scale = x.new_ones((batch_size, self.latent_dim))
#            z = pyro.sample("latent", pyrodist.Normal(z_loc, z_scale).to_event(1))
#            classes_probs = self.classifier(z).softmax(dim=-1)
#            pyro.sample("class", pyrodist.Categorical(probs=classes_probs), obs=classes)
#            dec_mu = self.decoder(z, batches).softmax(dim=-1) * size_factors[:, None]
#            dec_theta = torch.exp(self.theta(batches))
#            logits = (dec_mu + 1e-6).log() - (dec_theta + 1e-6).log()
#            pyro.sample("obs", pyrodist.NegativeBinomial(total_count=dec_theta, logits=logits).to_event(1), obs=x.int())
#    def guide(self, x, batches, classes, size_factors):
#        batch_size = x.shape[0]
#        with pyro.plate("data", batch_size):
#            z_loc_scale = self.encoder(x, batches)
#            z_mu = z_loc_scale[:, :self.latent_dim]
#            z_var = torch.sqrt(torch.exp(z_loc_scale[:, self.latent_dim:]) + 1e-4)
#            pyro.sample("latent", pyrodist.Normal(z_mu, z_var).to_event(1))
#
#class PyroGMMVAE(nn.Module):
#    """
#    """
#    def __init__(
#        self,
#        nx: int = 28 ** 2,
#        nh: int = 1024,
#        nz: int = 32,
#        nw: int = 15,
#        nclasses: int = 10,
#        bn : bool = True,
#        dropout : float = 0.2,
#    ) -> None:
#        super().__init__()
#        self.nx = nx
#        self.nh = nh
#        self.nz = nz
#        self.nw = nw
#        self.nclasses = nclasses
#        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
#        self.kld_unreduced = lambda mu, logvar: -0.5 * (
#            1 + logvar - mu.pow(2) - logvar.exp()
#        )
#        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
#        self.y_prior = distributions.RelaxedOneHotCategorical(
#                probs=torch.ones(nclasses), temperature=0.1,)
#        self.w_prior = distributions.Normal(
#            loc=torch.zeros(nw),
#            scale=torch.ones(nw),
#        )
#        ## P network
#        self.Px = ut.buildNetworkv2(
#                [nz,nh,nh,nx],
#                dropout=dropout, 
#                activation=nn.LeakyReLU(),
#                batchnorm=bn,
#                )
#        self.Pz = ut.buildNetworkv2(
#                [nw, nh, nh, 2*nclasses*nz],
#                dropout=dropout, 
#                activation=nn.LeakyReLU(),
#                batchnorm=bn,
#                )
#        self.Pz.add_module(
#                "unflatten", 
#                nn.Unflatten(1, (nclasses, 2*nz)))
#        ## Q network
#        self.Qwz = ut.buildNetworkv2(
#                [nx,nh,nh,2*nw + 2*nz],
#                dropout=dropout, 
#                activation=nn.LeakyReLU(),
#                batchnorm=bn,
#                )
#        self.Qwz.add_module(
#                "unflatten", 
#                nn.Unflatten(1, (2, nz + nw)))
#        self.Qy = ut.buildNetworkv2(
#                [nw + nz, nh, nh, nclasses],
#                dropout=dropout, 
#                activation=nn.LeakyReLU(),
#                batchnorm=bn,
#                )
#        self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
#        return
#
#    def model(self, x, y, z, w,):
#        pyro.module("cvae", self)
#        batch_size = x.shape[0]
#        with pyro.plate("data", batch_size):
#            y = pyro.sample("y", 
#                    pyrodist.Categorical(
#                        probs=torch.ones_like(x.mean(-1)),)
#                    )
#            w = pyro.sample("w", 
#                    pyrodist.Normal(
#                        loc=torch.zeros(batch_size, self.nw),
#                        scale=torch.ones(batch_size, self.nw),
#                        ))
#            z_w = self.Pz(w)
#            mu_z_w = z_w[:,:,:self.nz]
#            logvar_z_w = z_w[:,:,self.nz:]
#
#
#        def guide(self, x, y, z, w,):
#            wz = self.Qwz(x)
#            mu_w = wz[:,0,:self.nw]
#            logvar_w = wz[:,1,:self.nw]
#            std_w = (0.5 * logvar_w).exp()
#            w =  pyro.sample("latent_w",
#                    pyrodist.Normal(mu_w, std_w,).to_event(1))
#            #pyro.sample("latent_w",
#            #        pyrodist.Normal(mu_w, std_w,).to_event(1))
#            mu_z = wz[:,0,self.nw:]
#            logvar_z = wz[:,1,self.nw:]
#            std_z = (0.5 * logvar_z).exp()
#            z =  pyro.sample("latent_z",
#                    pyrodist.Normal(mu_z, std_z,).to_event(1))
#
#            z_loc = x.new_zeros((batch_size, self.latent_dim))
#            z_scale = x.new_ones((batch_size, self.latent_dim))
#            z = pyro.sample("latent", pyrodist.Normal(z_loc, z_scale).to_event(1))
#            classes_probs = self.classifier(z).softmax(dim=-1)
#            pyro.sample("class", pyrodist.Categorical(probs=classes_probs), obs=classes)
#            dec_mu = self.decoder(z, batches).softmax(dim=-1) * size_factors[:, None]
#            dec_theta = torch.exp(self.theta(batches))
#            logits = (dec_mu + 1e-6).log() - (dec_theta + 1e-6).log()
#            pyro.sample("obs", pyrodist.NegativeBinomial(total_count=dec_theta, logits=logits).to_event(1), obs=x.int())
#
#
#        noise = torch.randn_like(mu_z).to(x.device)
#        z = mu_z + noise * std_z
#        Qz = distributions.Normal(loc=mu_z, scale=std_z)
#        output["wz"] = wz
#        output["mu_z"] = mu_z
#        output["mu_w"] = mu_w
#        output["logvar_z"] = logvar_z
#        output["logvar_w"] = logvar_w
#        #q_y = self.Qy(wz[:,0,:])
#        q_y = self.Qy(torch.cat([w,z], dim=1))
#        eps = 1e-6
#        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
#        Qy = distributions.RelaxedOneHotCategorical(
#                temperature=0.1, probs=q_y)
#        output["q_y"] = q_y
#        output["Qy"] = Qy
#        output["w"]=w
#        output["z"]=z
#        rec = self.Px(z)
#        #rec = self.Px(mu_z)
#        logsigma_x = ut.softclip(self.logsigma_x, -9, 9)
#        sigma_x = logsigma_x.exp()
#        Qx = distributions.Normal(loc=rec, scale=sigma_x)
#        loss_rec = -Qx.log_prob(x).sum(-1).mean()
#        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
#        output["rec"]= rec
#        losses["rec"] = loss_rec
#        #logvar_z_w = z_w[:,:,self.nz:].tanh()
#        std_z_w = (0.5*logvar_z_w).exp()
#        Pz = distributions.Normal(
#                loc=mu_z_w,
#                scale=std_z_w,
#                )
#        output["Pz"] = Pz
#        output["Qz"] = Qz
#        loss_z = ut.kld2normal(
#                mu=mu_z.unsqueeze(1),
#                logvar=logvar_z.unsqueeze(1),
#                mu2=mu_z_w,
#                logvar2=logvar_z_w,
#                ).sum(-1)
#        if y == None:
#            loss_z = (q_y*loss_z).sum(-1).mean()
#            lp_y = self.y_prior.logits.to(x.device)
#            loss_l = (q_y.mean(0) * (
#                    q_y.mean(0).log() - lp_y)).sum()
#            #loss_l = (q_y * (
#            #        q_y.log() - lp_y)).sum(-1).mean()
#        else:
#            #loss_z = (q_y*loss_z).sum(-1).mean()
#            loss_z = (y*loss_z).sum(-1).mean()
#            Qy = distributions.OneHotCategorical(probs=q_y)
#            loss_l = -Qy.log_prob(y).mean()
#        loss_y = -1e0 * q_y.max(-1)[0].mean()
#        losses["loss_z"] = loss_z
#        losses["loss_l"] = loss_l
#        losses["loss_y"] = loss_y
#        loss_w = self.kld_unreduced(
#                mu=mu_w,
#                logvar=logvar_w).sum(-1).mean()
#        #loss_y = -(q_y * q_y.log() ).sum(-1).mean()
#        losses["loss_w"]=loss_w
#        total_loss = (
#                loss_rec * 1e0
#                + loss_z * 1e0
#                + loss_w * 1e0
#                #+ 1e1 * loss_y
#                + 1e1 * loss_l
#                )
#        losses["total_loss"] = total_loss
#        output["losses"] = losses
#        return output
