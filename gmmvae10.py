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


#import pytorch_lightning as pl
#from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder

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

class AE_Primer_Type1001(Generic_Net):
    """
    Vanila AE.
    batchnorm.
    reclosstype: 'gauss' is default. other option: 'Bernoulli'
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        bn : bool = True,
        dropout : float = 0.2,
        numhidden : int = 2,
        reclosstype : str = "Gauss",
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.numhidden=numhidden
        self.reclosstype = reclosstype
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
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        ## Q network
        self.Qz = ut.buildNetworkv5(
                #[nx, nh, nh, nz],
                [nx] + numhidden * [nh] + [nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
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

        z = self.Qz(x)
        output["z"] = z
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        losses["rec"] = loss_rec
        #losses["rec"] = loss_rec * 1e2
        total_loss = 1e0 * (
                loss_rec
                + 0e0
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

class VAE_Primer_Type1002(Generic_Net):
    """
    Vanila VAE.
    batchnorm.
    reclosstype: 'gauss' is default. other option: 'Bernoulli'
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        bn : bool = True,
        dropout : float = 0.2,
        numhidden : int = 2,
        reclosstype : str = "Gauss",
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.numhidden=numhidden
        self.reclosstype = reclosstype
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
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        ## Q network
        self.Qz = ut.buildNetworkv5(
                #[nx, nh, nh, nz],
                [nx] + numhidden * [nh] + [nz * 2],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
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

        tempz = self.Qz(x)
        mu_z = tempz[:,:self.nz]
        logvar_z = tempz[:,self.nz:]
        std_z = (0.5 * logvar_z).exp()
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        output["z"] = z
        output["logvar_z"] = logvar_z
        output["mu_z"] = mu_z
        loss_z = self.kld_unreduced(
                mu=mu_z,
                logvar=logvar_z).sum(-1).mean()
        losses["loss_z"] = loss_z
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        losses["rec"] = loss_rec
        #losses["rec"] = loss_rec * 1e2
        total_loss = 1e0 * (
                loss_rec
                + loss_z
                + 0e0
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

class VAE_Stacked_Dilo_Anndata_Type1003(nn.Module):
    """
    P(x,y,z,w,l) = P(x|z)P(z|w,y)P(w)P(y|l)P(l)
    using P(y|l)=P(l|y)=delta(x,y),
    P(l)~Cat(1/K), P(w)~N(0,I)
    Q(y,z,w,l|x) = Q(z|x)Q(w|z)Q(y|z,w)Q(l|y)
    reclosstype: 'gauss' is default. other option: 'Bernoulli'
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
        numhidden : int = 3,
        reclosstype : str = "Gauss",
        loss_l_type : str = "heuristic1",
        l_scale : float = 1e0,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden=numhidden
        self.lscale = l_scale
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.loss_l_type = loss_l_type
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
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv4(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhidden * [nh] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=True,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildNetworkv2(
                #[nx,nh,nh,2*nw + 2*nz],
                [nx] + numhidden*[nh] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv2(
                #[nw + nz, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
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
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
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
            if self.loss_l_type == "heuristic1":
                #loss_l = self.nclasses * (q_y.mean(0) * (
                #        q_y.mean(0).log() - lp_y)).sum()
                loss_l = self.lscale * (q_y.mean(0) * (
                        q_y.mean(0).log() - lp_y)).sum()
            elif self.loss_l_type == "heuristic2":
                loss_l = self.nclasses * (q_y.mean(0) * (
                        q_y.mean(0).log() - lp_y)).sum()
            else:
                loss_l = self.lscale * (q_y * (
                        q_y.log() - lp_y)).sum(-1).mean()
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
                + 1e0 * loss_l
                #+ self.nclasses * loss_l
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type1004(nn.Module):
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
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 3,
        dropout : float = 0.2,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
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
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Px.add_module("sigmoid",
        #        nn.Sigmoid(),
        #        )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhidden * [nh] + [2*nclasses*nz],
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
                [nx] + numhidden*[nh] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
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
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z], dim=1))
        #d_logits = self.Qd(torch.cat([w,z], dim=1)).tanh()
        output["d_logits"] = d_logits
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        D_y = distributions.Dirichlet(d_logits.exp())
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-1,1e1))
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
        lp_y = self.y_prior.logits.to(x.device)
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
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                q_y.log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
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

class VAE_Dirichlet_Type1004S(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    q(y|x,w,z)
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 3,
        dropout : float = 0.2,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
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
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Px.add_module("sigmoid",
        #        nn.Sigmoid(),
        #        )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhidden * [nh] + [2*nclasses*nz],
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
                [nx] + numhidden*[nh] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                #[nw + nz] + numhidden*[nh] + [nclasses],
                [nx + nw + nz] + numhidden*[nh] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
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
        #q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y_logits = self.Qy(torch.cat([x,w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z], dim=1))
        #d_logits = self.Qd(torch.cat([w,z], dim=1)).tanh()
        output["d_logits"] = d_logits
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        D_y = distributions.Dirichlet(d_logits.exp())
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-1,1e1))
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
        lp_y = self.y_prior.logits.to(x.device)
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
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                q_y.log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
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



class VAE_Dirichlet_Type1004A(nn.Module):
    """
    experimenting with different priors
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other option: 'Bernoulli'
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 3,
        dropout : float = 0.2,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
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
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhidden * [nh] + [2*nclasses*nz],
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
                [nx] + numhidden*[nh] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qd = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
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
        mse = nn.MSELoss(reduction="none")
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.applytanh:
            logvar_w = 3 * logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        #Qw = distributions.Normal(loc=mu_w, scale=std_w)
        #output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.applytanh:
            logvar_z = 3 * logvar_z.tanh()
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
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
        #loss_z = self.zscale * ut.kld2normal(
        #        mu=mu_z.unsqueeze(1),
        #        #logvar=logvar_z.unsqueeze(1),
        #        logvar=torch.tensor(-10),
        #        mu2=mu_z_w,
        #        logvar2=torch.tensor(-2),
        #        #logvar2=logvar_z_w,
        #        ).sum(-1)
        lp_y = self.y_prior.logits.to(x.device)
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
            #loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
            loss_y_alt2 = torch.tensor(0)
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
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
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
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


class VAE_Dirichlet_Type1004C(nn.Module):
    """
    convolution version.
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other option: 'Bernoulli'
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 3,
        dropout : float = 0.2,
        bn : bool = True,
        nf : int = 8,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.nf = nf
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
        #self.Px = ut.buildNetworkv5(
        #        #[nz, nh, nh, nh, nx],
        #        [nz] + numhidden * [nh] + [nx],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        self.Px = ut.buildTCNetworkv2(
                nin=nz, nf=nf, nout=nx, dropout=dropout,)
        self.Pz = ut.buildTCNetworkv2(
                nin=nw, 
                nf=nf,
                nout=2*nclasses*nz,
                dropout=dropout,
                )
        #self.Pz = ut.buildNetworkv5(
        #        #[nw, nh, nh, nh, 2*nclasses*nz],
        #        [nw] + numhidden * [nh] + [2*nclasses*nz],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        self.Qwz = ut.buildCNetworkv2(
                nin=nx,
                nf=nf,
                nout=2*nw + 2*nz,
                dropout=dropout,
                )
        #self.Qwz = ut.buildNetworkv5(
        #        #[nx, nh, nh, nh, 2*nw + 2*nz],
        #        [nx] + numhidden*[nh] + [2*nw + 2*nz],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        #self.Qy = ut.buildTCNetworkv2(
        #        nin=nw+nz,
        #        nout=nclasses,
        #        nf=nf,
        #        dropout=dropout,
        #        )
        self.Qy = ut.buildCNetworkv2(
                nin=nw+nz,
                nout=nclasses,
                nf=nf,
                dropout=dropout,
                )
        #self.Qy = ut.buildNetworkv5(
        #        #[nw + nz, nh, nh, nh, nclasses],
        #        [nw + nz] + numhidden*[nh] + [nclasses],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        #self.Qd = ut.buildTCNetworkv2(
        #        nin=nw+nz,
        #        nout=nclasses,
        #        nf=nf,
        #        dropout=dropout,
        #        )
        self.Qd = ut.buildCNetworkv2(
                nin=nw+nz,
                nout=nclasses,
                nf=nf,
                dropout=dropout,
                )
        #self.Qd = ut.buildNetworkv5(
        #        #[nw + nz, nh, nh, nh, nclasses],
        #        [nw + nz] + numhidden*[nh] + [nclasses],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )

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
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z], dim=1))
        #d_logits = self.Qd(torch.cat([w,z], dim=1)).tanh()
        output["d_logits"] = d_logits
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        D_y = distributions.Dirichlet(d_logits.exp())
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-1,1e1))
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
        lp_y = self.y_prior.logits.to(x.device)
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
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                q_y.log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
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

class VAE_Dirichlet_Type1004R(nn.Module):
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
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 3,
        dropout : float = 0.2,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
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
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Px.add_module("sigmoid",
        #        nn.Sigmoid(),
        #        )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhidden * [nh] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        resnet_wz = models.resnet18()
        #resnet_wz = models.resnet152()
        #resnet_wz = models.resnet34()
        resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
        self.Qwz = nn.Sequential(
                nn.Flatten(1),
                nn.Unflatten(1, (1,28,28)),
                resnet_wz,
                nn.Linear(1000, 2*nw + 2*nz),
                )
        #self.Qwz = ut.buildNetworkv5(
        #        #[nx, nh, nh, nh, 2*nw + 2*nz],
        #        [nx] + numhidden*[nh] + [2*nw + 2*nz],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        #resnet_y = models.resnet101()
        resnet_y = models.resnet18()
        #resnet_y = models.resnet34()
        resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
        self.Qy = nn.Sequential(
                nn.Linear(nx+nz+nw, 64**2),
                #nn.Flatten(1),
                nn.Unflatten(1, (1,64,64)),
                resnet_y,
                nn.Linear(1000, nclasses),
                )
        #self.Qy = ut.buildNetworkv5(
        #        #[nw + nz, nh, nh, nh, nclasses],
        #        [nw + nz] + numhidden*[nh] + [nclasses],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        #resnet_d = models.resnet101()
        resnet_d = models.resnet18()
        #resnet_d = models.resnet34()
        resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
        self.Qd = nn.Sequential(
                nn.Linear(nz+nw, 64**2),
                #nn.Flatten(1),
                nn.Unflatten(1, (1,64,64)),
                resnet_d,
                nn.Linear(1000, nclasses),
                )
        #self.Qd = ut.buildNetworkv5(
        #        #[nw + nz, nh, nh, nh, nclasses],
        #        [nw + nz] + numhidden*[nh] + [nclasses],
        #        dropout=dropout, 
        #        activation=nn.LeakyReLU(),
        #        batchnorm=bn,
        #        )
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
        d_logits = self.Qd(torch.cat([w,z], dim=1))
        #d_logits = self.Qd(torch.cat([w,z], dim=1)).tanh()
        output["d_logits"] = d_logits
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-2, 1e8))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-8, 1e9))
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-5, 1e5))
        D_y = distributions.Dirichlet(d_logits.exp())
        #D_y = distributions.Dirichlet(d_logits.exp().clamp(1e-1,1e1))
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
        lp_y = self.y_prior.logits.to(x.device)
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
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                q_y.log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        loss_y = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
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





class VAE_Type1005(nn.Module):
    """
    variational autoencoder with cluster embeder.
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 3024,
        nz: int = 200,
        bn : bool = True,
        nclasses: int = 10,
        dropout : float = 0.2,
        numhidden : int = 3,
        dscale : float = 1e1,
        concentration : float = 5e-3,
        reclosstype : str = "Gauss",
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.numhidden=numhidden
        self.nclasses = nclasses
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.dscale = dscale
        self.concentration = concentration
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
        self.Px = ut.buildNetworkv5(
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        ## Q network
        self.Qz = ut.buildNetworkv5(
                [nx] + numhidden*[nh] + [2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qy = ut.buildNetworkv5(
                [nx] + numhidden*[nh] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Ez = ut.buildNetworkv5(
                [nclasses] + numhidden * [nh] + [nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

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
        output["z"] = z
        output["mu"] = mu
        output["logvar"] = logvar
        rec = self.Px(z)
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        #loss_rec = (-Qx.log_prob(x).sum(-1)).clamp(-10,).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        losses["rec"] = loss_rec
        #q_y = self.Qy(x).softmax(-1)
        q_y = self.Qy(x).tanh().exp().exp().softmax(-1)
        eps = 1e-7
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        output["q_y"] = q_y
        Qy = distributions.OneHotCategorical(probs=q_y)
        loss_cat = -1e1 * q_y.max(-1)[0].mean()
        losses["cat"] = loss_cat
        loss_z = self.kld_unreduced(mu, logvar).sum(-1).mean()
        losses["z"] = loss_z
        if y == None:
            loss_l = (
                    #self.nclasses
                    #* (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
                (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
                .sum(-1)
                .mean()
            )
            cz = self.Ez(q_y)
        else:
            loss_l = -Qy.log_prob(y).mean()
            cz = self.Ez(y)
        #loss_l = (
        #    1e1
        #    * (q_y.mean(0) * (q_y.mean(0).log() - self.y_prior.logits.to(x.device)))
        #    .sum(-1)
        #    .mean()
        #)
        losses["l"] = loss_l
        cz = self.Ez(q_y)
        output["cz"] = cz
        #loss_cluster = nn.MSELoss(reduction='none')(cz,z).sum(-1).mean()
        loss_cluster = ut.kld2normal(
                mu = mu,
                logvar = logvar,
                mu2 = cz,
                logvar2 = torch.zeros_like(cz),
                ).sum(-1).mean()
        losses["cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_q = -Pd.log_prob(q_y).mean()
        losses["loss_q"] = loss_q
        total_loss = (
                loss_rec
                #+ loss_cat
                + loss_l * self.nclasses
                + loss_cluster
                #+ loss_z
                + loss_q * self.dscale
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class VAE_Dirichlet_Type1006(nn.Module):
    """
    dirichlet without the q_y classifier.
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
        nz: int = 64,
        nw: int = 32,
        nclasses: int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 3,
        dropout : float = 0.2,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.numhidden = numhidden
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
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Px.add_module("sigmoid",
        #        nn.Sigmoid(),
        #        )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhidden * [nh] + [2*nclasses*nz],
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
                [nx] + numhidden*[nh] + [2*nw + 2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        self.Qy = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        self.Qd = ut.buildNetworkv5(
                #[nw + nz, nh, nh, nh, nclasses],
                [nw + nz] + numhidden*[nh] + [nclasses],
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
        q_y_logits = self.Qy(torch.cat([w,z], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z], dim=1))
        d_y = nn.Softmax(dim=-1)(d_logits)
        d_y = (eps/self.nclasses +  (1 - eps) * d_y)
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["d_y"] = d_y
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
        lp_y = self.y_prior.logits.to(x.device)
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            #loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_z = (d_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            loss_y_alt2 = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        loss_l = (q_y.mean(0) * (
                q_y.mean(0).log() - lp_y)).sum()
        losses["loss_w"]=loss_w
        losses["loss_l"] = loss_l
        loss_l_alt = (q_y * (
                q_y.log() - lp_y)).sum(-1).mean()
        losses["loss_l_alt"] = loss_l_alt
        #loss_y = -1e0 * q_y.max(-1)[0].mean()
        loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_y"] = loss_y
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-2)
        #Pd = distributions.Dirichlet(torch.ones_like(q_y)*1e-4)
        Pd = distributions.Dirichlet(torch.ones_like(p_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"]=loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                #+ loss_y_alt
                + loss_y_alt2
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(p_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class CVAE_Type1007(nn.Module):
    """
    a vanilla cvae
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        ny: int = 10, #number of conditioned categories
        numhidden : int = 3,
        dropout : float = 0.2,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.ny = ny
        self.numhidden = numhidden
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.Px = ut.buildNetworkv5(
                [nz] + numhidden*[nh] + [ny * nx],
                dropout=dropout,
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Px.add_module(
                "unflatten", 
                nn.Unflatten(1, (ny, nx)))
        self.Qz = ut.buildNetworkv5(
                [nx] + numhidden*[nh] + [ny * nz * 2],
                #[nx + ny] + numhidden*[nh] + [nz * 2],
                dropout=dropout,
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qz.add_module(
                "unflatten", 
                nn.Unflatten(1, (ny, nz * 2)))

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        eps=1e-6
        mu_logvars = (self.Qz(x) * y.unsqueeze(-1)).sum(1)
        mu_z = mu_logvars[:,:self.nz]
        logvar_z =mu_logvars[:,self.nz:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        loss_z = self.kld_unreduced(mu_z, logvar_z).sum(-1).mean()
        losses["loss_z"] = loss_z
        output["z"] = z
        output["mu_z"] = mu_z
        output["logvar_z"] = logvar_z
        rec = (self.Px(z) * y.unsqueeze(-1)).sum(1)
        output["rec"] = rec
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
        losses["loss_rec"] = loss_rec
        total_loss = (
                loss_rec
                + loss_z 
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

class CVAE_Type1008(nn.Module):
    """
    a vanilla cvae with concatenation as the conditioning method
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 64,
        ny: int = 10, #number of conditioned categories
        numhidden : int = 3,
        dropout : float = 0.2,
        bn : bool = True,
        reclosstype : str = "Gauss",
        applytanh : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.ny = ny
        self.numhidden = numhidden
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.Px = ut.buildNetworkv5(
                [nz + ny] + numhidden*[nh] + [nx],
                dropout=dropout,
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Qz = ut.buildNetworkv5(
                [nx + ny] + numhidden*[nh] + [nz * 2],
                #[nx + ny] + numhidden*[nh] + [nz * 2],
                dropout=dropout,
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        xy = torch.cat([x,y], dim=-1)
        mu_logvars = self.Qz(xy)
        mu_z = mu_logvars[:,:self.nz]
        logvar_z =mu_logvars[:,self.nz:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        loss_z = self.kld_unreduced(mu_z, logvar_z).sum(-1).mean()
        losses["loss_z"] = loss_z
        output["z"] = z
        output["mu_z"] = mu_z
        output["logvar_z"] = logvar_z
        zy = torch.cat([z,y], dim=-1)
        rec = self.Px(zy)
        output["rec"] = rec
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
        losses["loss_rec"] = loss_rec
        total_loss = (
                loss_rec
                + loss_z 
                )
        losses["total_loss"] = total_loss
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
        best_loss : float = 1e6,
        do_plot : bool = False,
        test_accuracy : bool = False,
        ) -> None:
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wt,)
    best_result = model.state_dict()
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
            if loss < best_loss:
                best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    model.cpu()
                    model.eval()
                    w = model.w_prior.sample((16, ))
                    z = model.Pz(w)
                    mu = z[:,:,:model.nz].reshape(16*model.nclasses, model.nz)
                    rec = model.Px(mu).reshape(-1,1,28,28)
                    if model.reclosstype == "Bernoulli":
                        rec = rec.sigmoid()
                    ut.plot_images(rec, model.nclasses)
                    plt.pause(0.05)
                    plt.savefig("tmp.png")
                    model.train()
                    model.to(device)
                if test_accuracy:
                    model.eval()
                    #model.cpu()
                    #r,p,s = estimateClusterImpurityLoop(model, x.cpu(), y.cpu(), "cpu", )
                    r,p,s = estimateClusterImpurityLoop(model, x, y, device, )
                    print(p, "\n", r.mean(), "\n", r)
                    print((r*s).sum() / s.sum(), "\n",)
                    model.train()
                    model.to(device)

    model.cpu()
    optimizer = None
    model.load_state_dict(best_result)
    print("done training")
    return 

def trainSemiSuper(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    #test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lr=1e-3,
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled : bool = True,
    do_eval: bool = True,
    report_interval : int = 3,
    do_plot : bool = False,
    best_loss : float = 1e6,
) -> None:
    model.train()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wt,)
    best_result = model.state_dict()
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
            if not do_unlabeled:
                if loss < best_loss:
                    best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("labeled phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    model.cpu()
                    model.eval()
                    w = model.w_prior.sample((16, ))
                    z = model.Pz(w)
                    mu = z[:,:,:model.nz].reshape(16*model.nclasses, model.nz)
                    rec = model.Px(mu).reshape(-1,1,28,28)
                    if model.reclosstype == "Bernoulli":
                        rec = rec.sigmoid()
                    ut.plot_images(rec, model.nclasses)
                    plt.pause(0.05)
                    plt.savefig("tmp.png")
                    model.train()
                    model.to(device)
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
            if loss < best_loss:
                best_result = model.state_dict()
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
    #optimizer = None
    del optimizer
    print("done training")
    return None


def basicTrainLoop(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot : bool = False,
    test_accuracy : bool = False,
) -> None:
    for lr in lrs:
        print("epoch's lr = ", lr,)
        basicTrain(
            model,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot = do_plot,
            test_accuracy=test_accuracy,
        )


def trainSemiSuperLoop(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_validation: bool = True,
    report_interval: int = 3,
    do_plot : bool = False,
) -> None:
    for lr in lrs:
        trainSemiSuper(
            model,
            train_loader_labeled,
            train_loader_unlabeled,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            do_unlabeled,
            do_validation,
            report_interval,
            do_plot = do_plot,
        )


def preTrainAE(
        model,
        train_loader : torch.utils.data.DataLoader,
        num_epochs : int = 10,
        lr : float = 1e-3,
        device : str = "cuda:0",
        wt : float = 1e-4,
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
            model.train()
            model.requires_grad_(True)
            output = model.forward(x,)
            mu = output["mu_z"]
            logvar = output["logvar_z"]
            logvar = ut.softclip(logvar, -4.5, 4.5)
            std = (0.5 * logvar).exp()
            noise = torch.randn_like(mu).to(x.device)
            z = mu + noise * std
            Qz = distributions.Normal(
                    loc=mu,
                    scale=std,
                    )
            loss_z = -Qz.log_prob(z).sum(-1).mean()
            #rec = output["rec"]
            rec = model.Px(z)
            #rec = model.Px(mu)
            if model.reclosstype == "Bernoulli":
                logits = rec
                rec = logits.sigmoid()
                bce = nn.BCEWithLogitsLoss(reduction="none")
                loss_rec = bce(logits, x).sum(-1).mean()
            elif model.reclosstype == "mse":
                mse = nn.MSELoss(reduction="none")
                loss_rec = mse(rec, x).sum(-1).mean()
            else:
                logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
                sigma_x = logsigma_x.exp()
                Qx = distributions.Normal(loc=rec, scale=sigma_x)
                loss_rec = -Qx.log_prob(x).sum(-1).mean()
            loss = loss_rec + loss_z
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                print(loss_z.item(), loss_rec.item(), loss.item())
    model.cpu()
    optimizer = None
    print("done training")
    return 


def estimateClusterImpurity(
        model,
        x,
        labels,
        device : str = "cpu",
        ):
    model.eval()
    model.to(device)
    output = model(x.to(device))
    model.cpu()
    y = output["q_y"].detach().to("cpu")
    del output
    n = y.shape[1] # number of clusters
    r = -np.ones(n) # homogeny index
    #r = np.zeros(n) # homogeny index
    #p = np.zeros(n) # label assignments to the clusters
    p = -np.ones(n) # label assignments to the clusters
    s = -np.ones(n) # label assignments to the clusters
    for i in range(n):
        c = labels[y.argmax(-1) == i]
        if c.shape[0] > 0:
            r[i] = c.sum(0).max().item() / c.shape[0] 
            p[i] = c.sum(0).argmax().item()
            s[i] = c.shape[0]
    return r, p, s

def estimateClusterImpurityHelper(
        model,
        x,
        labels,
        device : str = "cpu",
        ):
    model.eval()
    model.to(device)
    output = model(x.to(device))
    model.cpu()
    y = output["q_y"].detach().to("cpu")
    del output
    return y


def estimateClusterImpurityLoop(
        model,
        xs,
        labels,
        device : str = "cpu",
        ):
    y = []
    model.eval()
    model.to(device)
    data_loader = torch.utils.data.DataLoader(
            dataset=ut.SynteticDataSet(
                data=xs,
                labels=labels,
                ),
            batch_size=128,
            shuffle=False,
            )
    for x, label in data_loader.__iter__():
        x.to(device)
        q_y = estimateClusterImpurityHelper(model, x, label, device,)
        y.append(q_y.cpu())
    y = torch.concat(y, dim=0)
    n = y.shape[1] # number of clusters
    r = -np.ones(n) # homogeny index
    p = -np.ones(n) # label assignments to the clusters
    s = -np.ones(n) # label assignments to the clusters
    for i in range(n):
        c = labels[y.argmax(-1) == i]
        if c.shape[0] > 0:
            r[i] = c.sum(0).max().item() / c.shape[0] 
            p[i] = c.sum(0).argmax().item()
            s[i] = c.shape[0]
    return r, p, s

        

