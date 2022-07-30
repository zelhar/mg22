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

# dud
class VAE_Dirichlet_Type1100(nn.Module):
    pass

# dud
class VAE_Dirichlet_Type1100a(nn.Module):
    pass


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
    relax option
    resnet option
    softargmax option
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
        use_resnet : bool = False,
        softargmax : bool = False,
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
        self.softargmax = softargmax
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.applytanh = applytanh
        self.use_resnet = use_resnet
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
        if use_resnet:
            #resnet_wz = models.resnet18()
            resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            #resnet_y = models.resnet18()
            resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    #[nw + nz] + numhidden*[nh] + [nclasses],
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    #[nx] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #if self.softargmax:
        #    self.Qy.add_module(
        #            "softargmax",
        #            ut.SoftArgMaxOneHot(1e4),
        #            )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            #resnet_d = models.resnet18()
            resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nx+nw+nz, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    #[nw + nz] + numhiddenq*[nhq] + [nclasses],
                    #[nx] + numhiddenq*[nhq] + [nclasses],
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #if self.softargmax:
        #    self.Qy.add_module(
        #            "softargmax",
        #            ut.SoftArgMaxOneHot(2e2),
        #            )
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
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y_logits, factor=1e3)
        else:
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

class VAE_Dirichlet_Type1101RC(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), Q(d|w,z,x)
    relax option
    resnet option
    condition (batch etc.) option
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nb: int = 0, # batch categories
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
        use_resnet : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.nclasses = nclasses
        self.nb = nb
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
        self.use_resnet = use_resnet
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
                [nz + nb] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw + nb] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            #resnet_wz = models.resnet18()
            resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx + nb, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx + nb] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            #resnet_y = models.resnet18()
            resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    #[nw + nz] + numhidden*[nh] + [nclasses],
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    #[nx] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            #resnet_d = models.resnet18()
            resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nx+nw+nz, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
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

    def forward(self, input, y=None, b=None):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        #eps=1e-6
        eps=1e-8
        if b == None:
            wz = self.Qwz(x)
        else:
            wz = self.Qwz(torch.cat([x,b], dim=-1))
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
        if b == None:
            rec = self.Px(z)
        else:
            rec = self.Px(torch.cat([x,b], dim=-1))
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
        if b == None:
            z_w = self.Pz(w)
        else:
            z_w = self.Pz(torch.cat([w,b], dim=-1))
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
    relax option
    """
    pass



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
    relax option
    """
    pass

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
    relax option
    resnet option
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
        use_resnet : bool = False,
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
        self.use_resnet = use_resnet
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
        if use_resnet:
            #resnet_wz = models.resnet18()
            resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            #resnet_y = models.resnet18()
            resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    #[nw + nz] + numhidden*[nh] + [nclasses],
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    #[nx] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            #resnet_d = models.resnet18()
            resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nw+nz, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    [nw + nz] + numhiddenq*[nhq] + [nclasses],
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
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=p_y,
                    )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
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


# dud
class VAE_Dirichlet_Classifier_Type1105(nn.Module):
    """
    no reconstruct, just classifier and dirichlet.
    minimize E_q[log(q(y|x) - log(q(y|d) + log(q(d|x)) - log(p(d))]
    """
    pass

# dud
class VAE_Dirichlet_Tandem_Type1106(nn.Module):
    """
    Tandem model
    """
    pass

class AAE_GMM_Type1107(nn.Module):
    """
    Trying to force the same distribution as the GMMVAE but
    using discriminators.
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
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
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
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
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
        if use_resnet:
            #resnet_wz = models.resnet18()
            resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            #resnet_y = models.resnet18()
            resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    #[nw + nz] + numhidden*[nh] + [nclasses],
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    #[nx] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            #resnet_d = models.resnet18()
            resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nw+nz, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    [nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        # D network
        self.Dy = ut.buildNetworkv5(
                [nclasses] + numhidden*[nh] + [1],
                dropout=0,
                batchnorm=False,
                )
        self.Dw = ut.buildNetworkv5(
                [nw] + numhidden*[nh] + [1],
                dropout=0,
                batchnorm=False,
                )
        self.Dz = ut.buildNetworkv5(
                [nw] + numhidden*[nh] + [1],
                dropout=0,
                batchnorm=False,
                )
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        eps=1e-8
        wz = self.Qwz(x)
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        q_y_logits = self.Qy(torch.cat([w,z,x], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        output["Qw"] = Qw
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        output["w"]=w
        output["z"]=z
        output["q_y"] = q_y
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
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
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_certainty = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_certainty"] = loss_certainty
        total_loss = (
                loss_rec
                #+ loss_z 
                #+ loss_w
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

def trainAAE(
        model,
        train_loader,
        lr : float = 1e-3,
        device : str = "cuda:0",
        wt : float = 1e-4,
        loss_type : str = "total_loss",
        report_interval : int = 3,
        best_loss : float = 1e6,
        do_plot : bool = False,
        ) -> None:
    # reconstruction
    # generation
    # discrimination
    return

# dud
def basicTandemModelTrain(
        tandem_model,
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
    pass




# dud
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
    pass

# dud
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
    pass
