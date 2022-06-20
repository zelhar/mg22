# best of models
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
#from pyro.optim import Adam
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

# dud
class VAE_Dirichlet_Type1100(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
        temperature : float = 0.1,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.concentration = concentration
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.temperature = torch.tensor([temperature])
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv5(
                #[nx, nh, nh, nh, 2*nw + 2*nz],
                [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qy1 = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nw + nz] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qy2 = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd1 = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qd2 = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qd.add_module( "softmax", nn.Softmax(dim=-1))

        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        eps=1e-6
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        #logvar_w = wz[:,1,:self.nw].tanh()
        logvar_w = wz[:,1,:self.nw]
        if self.applytanh:
            logvar_w = 3 * logvar_w.tanh()
        #logvar_w = ut.softclip(logvar_w, -2.5, 1.5)
        std_w = (0.5 * logvar_w).exp()
        #std_w = (0.5 * logvar_w).exp() + eps
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        #Qw = distributions.Normal(loc=mu_w, scale=std_w)
        #output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        #logvar_z = wz[:,1,self.nw:].tanh()
        logvar_z = wz[:,1,self.nw:]
        if self.applytanh:
            logvar_z = 3 * logvar_z.tanh()
        #logvar_z = ut.softclip(logvar_z, -2.5, 1.5)
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        #Qz = distributions.Normal(loc=mu_z, scale=std_z)
        #output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        q_y1_logits = self.Qy1(torch.cat([w,z], dim=1))
        q_y1 = nn.Softmax(dim=-1)(q_y1_logits)
        q_y1 = (eps/self.nclasses +  (1 - eps) * q_y1)
        q_y2_logits = self.Qy2(torch.cat([x], dim=1))
        q_y2 = nn.Softmax(dim=-1)(q_y2_logits)
        q_y2 = (eps/self.nclasses +  (1 - eps) * q_y2)
        d1_logits = self.Qd1(torch.cat([w,z], dim=1))
        d2_logits = self.Qd2(torch.cat([x], dim=1))
        #output["d_logits"] = d_logits
        D_y1 = distributions.Dirichlet(d1_logits.exp())
        D_y2 = distributions.Dirichlet(d2_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
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
        if self.applytanh:
            logvar_z_w = 3 * logvar_z_w.tanh()
        #logvar_z_w = ut.softclip(logvar_z_w, -2.5, 1.5)
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        p_y1 = D_y1.rsample()
        p_y1 = (eps/self.nclasses +  (1 - eps) * p_y1)
        p_y2 = D_y2.rsample()
        p_y2 = (eps/self.nclasses +  (1 - eps) * p_y2)
        #p_y = 0.5*(p_y1 + p_y2)
        p_y = 0.5*(q_y1 + q_y2)
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
            loss_y1_alt = self.yscale * (q_y1 * (
                    q_y1.log() - p_y1.log())).sum(-1).mean()
            loss_y1_alt2 = torch.tensor(0)
            loss_y2_alt = self.yscale * (q_y2 * (
                    q_y2.log() - p_y2.log())).sum(-1).mean()
            loss_y2_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
            # dummy for now
            loss_y1_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y1_alt2 = self.yscale * -Qy.log_prob(y).mean()
            loss_y2_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y2_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d1 = self.dscale * distributions.kl_divergence(D_y1, Pd).mean()
        loss_d2 = self.dscale * distributions.kl_divergence(D_y2, Pd).mean()
        losses["loss_d"] = loss_d = loss_d1 + loss_d2
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        losses["loss_y1_alt"] = loss_y1_alt
        losses["loss_y1_alt2"] = loss_y1_alt2
        losses["loss_y2_alt"] = loss_y2_alt
        losses["loss_y2_alt2"] = loss_y2_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                + loss_y_alt
                + loss_y_alt2
                + loss_y1_alt
                + loss_y1_alt2
                + loss_y2_alt
                + loss_y2_alt2
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

# dud
class VAE_Dirichlet_Type1100a(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
        temperature : float = 0.1,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv5(
                #[nx, nh, nh, nh, 2*nw + 2*nz],
                [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                #[nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qy1 = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nw + nz] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qy2 = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd1 = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qd2 = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qd.add_module( "softmax", nn.Softmax(dim=-1))

        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        eps=1e-6
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        #logvar_w = wz[:,1,:self.nw].tanh()
        logvar_w = wz[:,1,:self.nw]
        if self.applytanh:
            logvar_w = 3 * logvar_w.tanh()
        #logvar_w = ut.softclip(logvar_w, -2.5, 1.5)
        std_w = (0.5 * logvar_w).exp()
        #std_w = (0.5 * logvar_w).exp() + eps
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        #Qw = distributions.Normal(loc=mu_w, scale=std_w)
        #output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        #logvar_z = wz[:,1,self.nw:].tanh()
        logvar_z = wz[:,1,self.nw:]
        if self.applytanh:
            logvar_z = 3 * logvar_z.tanh()
        #logvar_z = ut.softclip(logvar_z, -2.5, 1.5)
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        #Qz = distributions.Normal(loc=mu_z, scale=std_z)
        #output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        #q_y_logits = self.Qy(torch.cat([w,z,x], dim=1))
        q_y_logits = self.Qy(torch.cat([w,z,x], dim=1))
        #q_y_logits = self.Qy(torch.cat([x], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        q_y1_logits = self.Qy1(torch.cat([w,z], dim=1))
        q_y1 = nn.Softmax(dim=-1)(q_y1_logits)
        q_y1 = (eps/self.nclasses +  (1 - eps) * q_y1)
        q_y2_logits = self.Qy2(torch.cat([x], dim=1))
        q_y2 = nn.Softmax(dim=-1)(q_y2_logits)
        q_y2 = (eps/self.nclasses +  (1 - eps) * q_y2)
        d1_logits = self.Qd1(torch.cat([w,z], dim=1))
        d2_logits = self.Qd2(torch.cat([x], dim=1))
        #output["d_logits"] = d_logits
        D_y1 = distributions.Dirichlet(d1_logits.exp())
        D_y2 = distributions.Dirichlet(d2_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.applytanh:
            logvar_z_w = 3 * logvar_z_w.tanh()
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        p_y1 = D_y1.rsample()
        p_y1 = (eps/self.nclasses +  (1 - eps) * p_y1)
        p_y2 = D_y2.rsample()
        p_y2 = (eps/self.nclasses +  (1 - eps) * p_y2)
        #p_y = 0.5*(p_y1 + p_y2)
        #p_y = 0.5*(q_y1 + q_y2)
        p_y =q_y1
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
            loss_y1_alt = self.yscale * (q_y1 * (
                    q_y1.log() - p_y1.log())).sum(-1).mean()
            loss_y1_alt2 = torch.tensor(0)
            loss_y2_alt = self.yscale * (q_y2 * (
                    q_y2.log() - p_y2.log())).sum(-1).mean()
            loss_y2_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
            # dummy for now
            loss_y1_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y1_alt2 = self.yscale * -Qy.log_prob(y).mean()
            loss_y2_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y2_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d1 = self.dscale * distributions.kl_divergence(D_y1, Pd).mean()
        loss_d2 = self.dscale * distributions.kl_divergence(D_y2, Pd).mean()
        losses["loss_d"] = loss_d = loss_d1 + loss_d2
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        losses["loss_y1_alt"] = loss_y1_alt
        losses["loss_y1_alt2"] = loss_y1_alt2
        losses["loss_y2_alt"] = loss_y2_alt
        losses["loss_y2_alt2"] = loss_y2_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                #+ loss_d
                + loss_y_alt
                + loss_y_alt2
                #+ loss_y1_alt
                #+ loss_y1_alt2
                #+ loss_y2_alt
                #+ loss_y2_alt2
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type1101(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), Q(d|w,z,x)
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
        temperature : float = 0.1,
        relax : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv5(
                [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                #[nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetworkv5(
                #[nw + nz] + numhiddenq*[nhq] + [nclasses],
                #[nx] + numhiddenq*[nhq] + [nclasses],
                [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        #eps=1e-6
        eps=1e-8
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.applytanh:
            logvar_w = 3 * logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        #std_w = (0.5 * logvar_w).exp() + eps
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.applytanh:
            logvar_z = 3 * logvar_z.tanh()
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x], dim=1))
        #q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        #q_y_logits = self.Qy(torch.cat([x], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        #d_logits = self.Qd(torch.cat([x], dim=1))
        #d_logits = self.Qd(torch.cat([w,z], dim=1))
        d_logits = self.Qd(torch.cat([w,z,x], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        #Qy = distributions.OneHotCategorical(probs=q_y)
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.applytanh:
            logvar_z_w = 3 * logvar_z_w.tanh()
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        #Py = distributions.OneHotCategorical(probs=p_y)
        #if (y != None) and self.relax:
        #    y = (eps/self.nclasses +  (1 - eps) * y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=p_y,
                    )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            #if self.relax:
            #    y = (eps/self.nclasses +  (1 - eps) * y)
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                + loss_y_alt
                + loss_y_alt2
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output


class VAE_Dirichlet_Type1102(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|x), Q(d|w,z)
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
        temperature : float = 0.1,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv5(
                #[nx, nh, nh, nh, 2*nw + 2*nz],
                [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                [nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetworkv5(
                [nw + nz] + numhiddenq*[nhq] + [nclasses],
                #[nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        eps=1e-6
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.applytanh:
            logvar_w = 3 * logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        #std_w = (0.5 * logvar_w).exp() + eps
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.applytanh:
            logvar_z = 3 * logvar_z.tanh()
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        #q_y_logits = self.Qy(torch.cat([w,z,x], dim=1))
        #q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y_logits = self.Qy(torch.cat([x], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        #d_logits = self.Qd(torch.cat([x], dim=1))
        d_logits = self.Qd(torch.cat([w,z], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.applytanh:
            logvar_z_w = 3 * logvar_z_w.tanh()
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                + loss_y_alt
                + loss_y_alt2
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output



class VAE_Dirichlet_Type1103(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z), Q(d|x)
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
        temperature : float = 0.1,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv5(
                [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                [nw + nz] + numhidden*[nh] + [nclasses],
                #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                #[nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                #[nw + nz] + numhiddenq*[nhq] + [nclasses],
                [nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        eps=1e-6
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.applytanh:
            logvar_w = 3 * logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        #std_w = (0.5 * logvar_w).exp() + eps
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.applytanh:
            logvar_z = 3 * logvar_z.tanh()
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        #q_y_logits = self.Qy(torch.cat([w,z,x], dim=1))
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        #q_y_logits = self.Qy(torch.cat([x], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([x], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.applytanh:
            logvar_z_w = 3 * logvar_z_w.tanh()
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                + loss_y_alt
                + loss_y_alt2
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type1104(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), Q(d|w,z)
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
        temperature : float = 0.1,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv5(
                [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                #[nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetworkv5(
                [nw + nz] + numhiddenq*[nhq] + [nclasses],
                #[nx] + numhiddenq*[nhq] + [nclasses],
                #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        eps=1e-6
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.applytanh:
            logvar_w = 3 * logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        #std_w = (0.5 * logvar_w).exp() + eps
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.applytanh:
            logvar_z = 3 * logvar_z.tanh()
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x], dim=1))
        #q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        #q_y_logits = self.Qy(torch.cat([x], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        #d_logits = self.Qd(torch.cat([x], dim=1))
        d_logits = self.Qd(torch.cat([w,z], dim=1))
        #d_logits = self.Qd(torch.cat([w,z,x], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(w)
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.applytanh:
            logvar_z_w = 3 * logvar_z_w.tanh()
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                + loss_y_alt
                + loss_y_alt2
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output


class VAE_Dirichlet_Classifier_Type1105(nn.Module):
    """
    no reconstruct, just classifier and dirichlet.
    minimize E_q[log(q(y|x) - log(q(y|d) + log(q(d|x)) - log(p(d))]
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
        temperature : float = 0.1,
        relax : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        ## Q network
        self.Qy = ut.buildNetworkv5(
                #[nw + nz] + numhidden*[nh] + [nclasses],
                #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                [nx] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetworkv5(
                #[nw + nz] + numhiddenq*[nhq] + [nclasses],
                [nx] + numhiddenq*[nhq] + [nclasses],
                #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        #eps=1e-6
        eps=1e-8
        q_y_logits = self.Qy(torch.cat([x], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([x], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        #Qy = distributions.OneHotCategorical(probs=q_y)
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        #Py = distributions.OneHotCategorical(probs=p_y)
        #if (y != None) and self.relax:
        #    y = (eps/self.nclasses +  (1 - eps) * y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=p_y,
                    )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = -D_y.log_prob(q_y).mean()
            #loss_y_alt2 = torch.tensor(0)
        else:
            #if self.relax:
            #    y = (eps/self.nclasses +  (1 - eps) * y)
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_d
                + loss_y_alt
                + loss_y_alt2
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

def tandemTrain(
        model1,
        model2,
        train_loader : torch.utils.data.DataLoader,
        test_loader : Optional[torch.utils.data.DataLoader] = None,
        num_epochs : int = 10,
        lr : float = 1e-3,
        device : str = "cuda:0",
        wt : float = 1e-4,
        loss_type : str = "total_loss",
        report_interval : int = 3,
        best_loss : float = 1e6,
        do_plot : bool = False,
        ) -> None:
    model1.train()
    model1.to(device)
    model2.train()
    model2.to(device)
    optimizer = optim.Adam([
        {'params' : model1.parameters()},
        {'params' : model2.parameters()},
        ],
        lr=lr, weight_decay=wt,)
    #best_result = model1.state_dict()
    for epoch in range(num_epochs):
        #print("training phase")
        for idx, (data, labels) in enumerate(train_loader):
            x = data.flatten(1).to(device)
            y = labels.to(device)
            # x = data.to(device)
            model1.train()
            model1.requires_grad_(True)
            model2.train()
            model2.requires_grad_(True)
            if idx % 3 == 0:
                # train both unsupervised
                output1 = model1.forward(x,)
                output2 = model2.forward(x,)
            elif idx % 3 == 1:
                # train model1 unsupervised
                output1 = model1.forward(x,)
                q_y1 = output1["q_y"].detach()
                output2 = model2.forward(x,q_y1)
            else:
                # train model2 unsupervised
                output2 = model2.forward(x,)
                q_y2 = output2["q_y"].detach()
                output1 = model2.forward(x,q_y2)
            #output = model.forward(x,y)
            loss1 = output1["losses"][loss_type]
            loss2 = output2["losses"][loss_type]
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if loss < best_loss:
            #    best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model1.printDict(output1["losses"])
                model2.printDict(output2["losses"])
                print()
                if do_plot:
                    model1.cpu()
                    model1.eval()
                    w = model1.w_prior.sample((5, ))
                    z = model1.Pz(w)
                    mu = z[:,:,:model1.nz].reshape(5*model1.nclasses, model1.nz)
                    rec = model1.Px(mu).reshape(-1,1,28,28)
                    if model1.reclosstype == "Bernoulli":
                        rec = rec.sigmoid()
                    ut.plot_images(rec, model1.nclasses)
                    plt.pause(0.05)
                    plt.savefig("tmp.png")
                    model1.train()
                    model1.to(device)
    model1.cpu()
    model2.cpu()
    optimizer = None
    #model.load_state_dict(best_result)
    print("done training")
    return 

def tandemTrainV2(
        model1,
        model2,
        train_loader : torch.utils.data.DataLoader,
        num_epochs : int = 10,
        lr : float = 1e-3,
        device : str = "cuda:0",
        wt : float = 1e-4,
        loss_type : str = "total_loss",
        report_interval : int = 3,
        best_loss : float = 1e6,
        do_plot : bool = False,
        ) -> None:
    model1.train()
    model1.to(device)
    model2.train()
    model2.to(device)
    optimizer = optim.Adam([
        {'params' : model1.parameters()},
        {'params' : model2.parameters()},
        ],
        lr=lr, weight_decay=wt,)
    #best_result = model1.state_dict()
    for epoch in range(num_epochs):
        #print("training phase")
        for idx, (data, labels) in enumerate(train_loader):
            x = data.flatten(1).to(device)
            y = labels.to(device)
            # x = data.to(device)
            model1.train()
            model1.requires_grad_(True)
            model2.train()
            model2.requires_grad_(True)
            output1 = model1.forward(x,)
            output2 = model2.forward(x,)
            #if idx % 3 == 0:
            #    # train both unsupervised
            #    output1 = model1.forward(x,)
            #    output2 = model2.forward(x,)
            #elif idx % 3 == 1:
            #    # train model1 unsupervised
            #    output1 = model1.forward(x,)
            #    q_y1 = output1["q_y"].detach()
            #    output2 = model2.forward(x,q_y1)
            #else:
            #    # train model2 unsupervised
            #    output2 = model2.forward(x,)
            #    q_y2 = output2["q_y"].detach()
            #    output1 = model2.forward(x,q_y2)
            #output = model.forward(x,y)
            loss1 = output1["losses"][loss_type]
            loss2 = output2["losses"][loss_type]
            q_y1 = output1["q_y"]
            q_y2 = output2["q_y"]
            Qy1 = distributions.OneHotCategorical(probs=q_y1)
            Qy2 = distributions.OneHotCategorical(probs=q_y2)
            loss3 = distributions.kl_divergence(Qy1, Qy2).mean()
            loss4 = distributions.kl_divergence(Qy2, Qy1).mean()
            loss = loss1 + loss2 + loss3 + loss4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #if loss < best_loss:
            #    best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model1.printDict(output1["losses"])
                model2.printDict(output2["losses"])
                print()
                if do_plot:
                    model1.cpu()
                    model1.eval()
                    w = model1.w_prior.sample((5, ))
                    z = model1.Pz(w)
                    mu = z[:,:,:model1.nz].reshape(5*model1.nclasses, model1.nz)
                    rec = model1.Px(mu).reshape(-1,1,28,28)
                    if model1.reclosstype == "Bernoulli":
                        rec = rec.sigmoid()
                    ut.plot_images(rec, model1.nclasses)
                    plt.pause(0.05)
                    plt.savefig("tmp.png")
                    model1.train()
                    model1.to(device)
    model1.cpu()
    model2.cpu()
    optimizer = None
    #model.load_state_dict(best_result)
    print("done training")
    return 

