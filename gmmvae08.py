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

class AE_Primer_Type801(Generic_Net):
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
        logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
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
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type805(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        nw: int = 150,
        nclasses: int = 10,
        dirscale : float = 1e1,
        concentration : float = 1e-4,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        # Dirichlet constant prior:
        self.l = 1e-3
        self.dirscale = dirscale
        self.concentration = concentration
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
        #logvar_w = wz[:,1,:self.nw]
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:].tanh()
        #logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qp(torch.cat([w,z], dim=1))
        output["d_logits"] = d_logits
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        #output["y"] = y
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
        logvar_z_w = z_w[:,:,self.nz:].tanh()
        #logvar_z_w = z_w[:,:,self.nz:]
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
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y_alt = (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = -Py.log_prob(y).mean()
            loss_y_alt2 = -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                (eps+q_y).log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = 1e0 * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d * self.dirscale
                + loss_y_alt
                + loss_y_alt2
                #+ loss_l
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type806(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 64,
        nw: int = 15,
        nclasses: int = 10,
        dirscale : float = 1e1,
        concentration : float = 1e-4,
        clamp : float = 1e-2,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        # Dirichlet constant prior:
        self.l = 1e-3
        self.dirscale = dirscale
        self.clamp = clamp
        self.concentration = concentration
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
                #[nx, nh, nh, nclasses],
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
        #logvar_w = wz[:,1,:self.nw].clamp(-1e2,1e0)
        #logvar_w = wz[:,1,:self.nw]
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:].tanh()
        #logvar_z = wz[:,1,self.nw:].clamp(-1e2,1e0)
        #logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        #q_y_logits = self.Qy(x, dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qp(torch.cat([w,z], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        #D_y = distributions.Dirichlet(d_logits.exp())
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(self.clamp, 1e1))
        output["D_y"] = D_y
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        #output["y"] = y
        rec = self.Px(z)
        logsigma_x = ut.softclip(self.logsigma_x, -6, 6)
        sigma_x = logsigma_x.exp()
        Qx = distributions.Normal(loc=rec, scale=sigma_x)
        loss_rec = -Qx.log_prob(x).sum(-1).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:].tanh()
        #logvar_z_w = z_w[:,:,self.nz:].clamp(-1e2,1e0)
        #logvar_z_w = z_w[:,:,self.nz:]
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
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = -Py.log_prob(y).mean()
            loss_y_alt2 = -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                q_y.log() - lp_y)).sum(-1).mean()
        #loss_l_alt = (q_y * (
        #        (eps+q_y).log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = 1e0 * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        #loss_q_y = -D_y.log_prob(q_y).mean()
        #losses["loss_q_y"] = loss_q_y
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d * self.dirscale
                + loss_y_alt
                #+ loss_y_alt2
                #+ loss_l
                #+ loss_l_alt
                #+ loss_q_y
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type807(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dirscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        nw: int = 150,
        nclasses: int = 10,
        dirscale : float = 1e1,
        zscale : float = 1e0,
        concentration : float = 1e-3,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        # Dirichlet constant prior:
        self.l = 1e-3
        self.dirscale = dirscale
        self.zscale = zscale
        self.concentration = concentration
        self.dir_prior = distributions.Dirichlet(dirscale*torch.ones(nclasses))
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
        #logvar_w = wz[:,1,:self.nw]
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:].tanh()
        #logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qp(torch.cat([w,z], dim=1))
        #d_logits = self.Qp(torch.cat([w,z], dim=1)).tanh()
        output["d_logits"] = d_logits
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        D_y = distributions.Dirichlet(d_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        #output["y"] = y
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
        logvar_z_w = z_w[:,:,self.nz:].tanh()
        #logvar_z_w = z_w[:,:,self.nz:]
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        output["Qz"] = Qz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        lp_y = self.y_prior.logits.to(x.device)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y_alt = (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = -Py.log_prob(y).mean()
            loss_y_alt2 = -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                (eps+q_y).log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dirscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                #+ loss_d * self.dirscale
                + loss_y_alt
                + loss_y_alt2
                #+ loss_l
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def autoencode(self, input):
        x = input.flatten(1)
        wz = self.Qwz(x)
        z = wz[:,0,self.nw:]
        rec = self.Px(z)
        return z, rec

    def fitAE(self,
            train_loader : torch.utils.data.DataLoader,
            num_epochs : int = 10,
            lr : float = 1e-3,
            device : str = "cuda:0",
            wt : float = 0e-4,
            report_interval : int = 3,
            ) -> None:
        self.train()
        self.to(device)
        self.requires_grad_(True)
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wt,)
        for epoch in range(num_epochs):
            for idx, (data, labels) in enumerate(train_loader):
                x = data.flatten(1).to(device)
                y = labels.to(device)
                wz = self.Qwz(x)
                z = wz[:,0,self.nw:]
                rec = self.Px(z)
                logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
                sigma_x = logsigma_x.exp()
                Qx = distributions.Normal(loc=rec, scale=sigma_x)
                loss_rec = -Qx.log_prob(x).sum(-1).mean()
                optimizer.zero_grad()
                loss_rec.backward()
                optimizer.step()
                if epoch % report_interval == 0 and idx % 1500 == 0:
                    print("training phase")
                    print("epoch" + str(epoch), "loss_rec : ", loss_rec.item(), "\n")
        self.cpu()
        return

class VAE_Dirichlet_Type808(nn.Module):
    """
    deterministic version.
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dirscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        nw: int = 150,
        nclasses: int = 10,
        dirscale : float = 1e1,
        zscale : float = 1e0,
        concentration : float = 1e-3,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        # Dirichlet constant prior:
        self.l = 1e-3
        self.dirscale = dirscale
        self.zscale = zscale
        self.concentration = concentration
        self.dir_prior = distributions.Dirichlet(dirscale*torch.ones(nclasses))
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
                [nw, nh, nh, 1*nclasses*nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 1*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv2(
                [nx,nh,nh,1*nw + 1*nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (1, nz + nw)))
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
        logvar_w = -1e1 * torch.ones_like(mu_w)
        #logvar_w = wz[:,1,:self.nw].tanh()
        #logvar_w = wz[:,1,:self.nw]
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = -1e1 * torch.ones_like(mu_z)
        #logvar_z = wz[:,1,self.nw:].tanh()
        #logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qp(torch.cat([w,z], dim=1))
        #d_logits = self.Qp(torch.cat([w,z], dim=1)).tanh()
        output["d_logits"] = d_logits
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        D_y = distributions.Dirichlet(d_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        #output["y"] = y
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
        logvar_z_w = -1e1 * torch.ones_like(mu_z_w)
        #logvar_z_w = z_w[:,:,self.nz:].tanh()
        #logvar_z_w = z_w[:,:,self.nz:]
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        output["Qz"] = Qz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        lp_y = self.y_prior.logits.to(x.device)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y_alt = (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = -Py.log_prob(y).mean()
            loss_y_alt2 = -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                (eps+q_y).log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dirscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                #+ loss_d * self.dirscale
                + loss_y_alt
                + loss_y_alt2
                + loss_l * self.nclasses
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type809(nn.Module):
    """
    deterministic clustering autoencoder.
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dirscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
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
        #logvar_w = -2e1 * torch.ones_like(mu_w)
        logvar_w = wz[:,1,:self.nw]
        #logvar_w = wz[:,1,:self.nw].tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        mu_z = wz[:,0,self.nw:]
        logvar_z = -2e1 * torch.ones_like(mu_z)
        #logvar_z = wz[:,1,self.nw:]
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
        lp_y = self.y_prior.logits.to(x.device)
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
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
                + 1e1 * loss_y
                #+ 1e1 * loss_l
                + self.nclasses * loss_l
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type810(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dirscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dirscale : float = 1e1,
        zscale : float = 1e0,
        concentration : float = 1e-3,
        numhidden : int = 3,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        # Dirichlet constant prior:
        self.dirscale = dirscale
        self.zscale = zscale
        self.concentration = concentration
        self.dir_prior = distributions.Dirichlet(dirscale*torch.ones(nclasses))
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
        self.Px = ut.buildNetworkv4(
                #[nz, nh, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Pz = ut.buildNetworkv4(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhidden * [nh] + [2*nclasses*nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv4(
                #[nx, nh, nh, nh, 2*nw + 2*nz],
                [nx] + numhidden*[nh] + [2*nw + 2*nz],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv4(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
                dropout=0.2, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qp = ut.buildNetworkv4(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
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
        #logvar_w = wz[:,1,:self.nw].tanh()
        logvar_w = wz[:,1,:self.nw]
        logvar_w = ut.softclip(logvar_w, -2.5, 1.5)
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        #logvar_z = wz[:,1,self.nw:].tanh()
        logvar_z = wz[:,1,self.nw:]
        logvar_z = ut.softclip(logvar_z, -2.5, 1.5)
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        eps = 1e-6
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qp(torch.cat([w,z], dim=1))
        #d_logits = self.Qp(torch.cat([w,z], dim=1)).tanh()
        output["d_logits"] = d_logits
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        D_y = distributions.Dirichlet(d_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
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
        #logvar_z_w = z_w[:,:,self.nz:].tanh()
        logvar_z_w = z_w[:,:,self.nz:]
        logvar_z_w = ut.softclip(logvar_z_w, -2.5, 1.5)
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        output["Qz"] = Qz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        lp_y = self.y_prior.logits.to(x.device)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y_alt = (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = -Py.log_prob(y).mean()
            loss_y_alt2 = -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                q_y.log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dirscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                #+ loss_d * self.dirscale
                + loss_y_alt
                + loss_y_alt2
                #+ loss_l
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output


def basicTrain(
        model,
        train_loader : torch.utils.data.DataLoader,
        test_loader : Optional[torch.utils.data.DataLoader] = None,
        num_epochs : int = 10,
        lr : float = 1e-3,
        device : str = "cuda:0",
        wt : float = 1e-4,
        loss_type : str = "total_loss",
        report_interval : int = 3,
        ) -> None:
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wt,)
    for epoch in range(num_epochs):
        #print("training phase")
        for idx, (data, labels) in enumerate(train_loader):
            x = data.flatten(1).to(device)
            y = labels.to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(x,)
            #output = model.forward(x,y)
            loss = output["losses"][loss_type]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("training phase")
                model.printDict(output["losses"])
                print()
    model.cpu()
    optimizer = None
    print("done training")
    return



def trainSemiSuper(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    num_epochs=15,
    lr=1e-3,
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled : bool = True,
    do_eval: bool = True,
    report_interval : int = 3,
) -> None:
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wt,)
    for epoch in range(num_epochs):
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
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("labeled phase")
                model.printDict(output["losses"])
                print()
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
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("unlabeled phase")
                model.printDict(output["losses"])
                print()
            if idx >= len(train_loader_labeled):
                break
        for idx, (data, labels) in enumerate(test_loader):
            if do_eval == False:
                break
            x = data.flatten(1).to(device)
            y = labels.to(device)
            model.eval()
            #output = model.forward(x,)
            output = model.forward(x,y=y)
            loss = output["losses"]["total_loss"]
            q_y = output["q_y"]
            ce_loss = (y * q_y.log()).sum(-1).mean()
            #if idx % 1500 == 0:
            #    model.printDict(output["losses"])
            #    print("ce loss:", ce_loss.item())
            #    print()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("eval phase")
                model.printDict(output["losses"])
                print("ce loss:", ce_loss.item())
                print()
    model.cpu()
    optimizer = None
    print("done training")
    return None


