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
#from my_torch_utils import denorm, normalize, mixedGaussianCircular
#from my_torch_utils import fclayer, init_weights
#from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
#from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
#from my_torch_utils import scsimDataset
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

class ICC_AE_Type1200(nn.Module):
    """
    AE with 2 cluster heads,
    computes and tries to maximize
    mutual information
    of q(y|x) and q(z|x)
    """

    def __init__(
        self,
        nx : int = 28**2,
        nclasses: int = 10,
        nz: int = 64,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        xscale: float = 1e0,
        yscale: float = 1e0,
        zscale: float = 1e0,
        mi_scale: float = 1e0,
        concentration: float = 5e-1,
        numhidden: int = 2,
        numhiddenq: int = 2,
        numhiddenp: int = 2,
        dropout: float = 0.3,
        bn: bool = True,
        reclosstype: str = "Gauss",
        temperature: float = 0.1,
        relax: bool = False,
        use_resnet: bool = False,
        softargmax: bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.xscale = xscale
        self.yscale = yscale
        self.zscale = zscale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.softargmax = softargmax
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.use_resnet = use_resnet
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        ## Q network
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nz, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    #[nw + nz] + numhidden*[nh] + [nclasses],
                    [nz] + numhiddenq*[nhq] + [nclasses],
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
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nx, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    #[nw + nz] + numhiddenq*[nhq] + [nclasses],
                    #[nx] + numhiddenq*[nhq] + [nclasses],
                    [nx] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #if self.softargmax:
        #    self.Qy.add_module(
        #            "softargmax",
        #            ut.SoftArgMaxOneHot(2e2),
        #            )
        if use_resnet:
            resnet_z = models.resnet18()
            #resnet_z = models.resnet34()
            resnet_z.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_z,
                    nn.Linear(1000, nz),
                    )
        else:
            self.Qz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [nz],
                    dropout=dropout, 
                    #dropout=0, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qz.add_module(
        #        "unflatten", 
        #        nn.Unflatten(1, (2, nz + nw)))
        # P network
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz] + numhiddenp * [nhp] + [nx],
                #dropout=dropout, 
                dropout=0, 
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
        z = self.Qz(x)
        output["z"] = z
        rec = self.Px(z)
        output["rec"] = rec
        q_y_logits = self.Qy(torch.cat([z], dim=1))
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y_logits, factor=1e3)
        else:
            q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        output["q_y_logits"] = q_y_logits
        output["q_y"] = q_y
        q_d_logits = self.Qd(torch.cat([x], dim=1))
        q_d = nn.Softmax(dim=-1)(q_d_logits)
        q_d = (eps/self.nclasses +  (1 - eps) * q_d)
        output["q_d_logits"] = q_d_logits
        output["q_d"] = q_d
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
        losses["rec"] = self.xscale * loss_rec
        loss_mi = -ut.mutualInfo(q_y, q_d)
        losses["mi"] = self.mi_scale * loss_mi
        losses["clusterer"] = -1e0 * q_y.max(-1)[0].mean()
        total_loss = (
                loss_rec
                + loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output


class VAE_Dirichlet_Type1201(nn.Module):
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
        self.Px = ut.buildNetworkv6(
                #[nz, nh, nh, nh, nx],
                [nz] + numhiddenp * [nhp] + [nx],
                #dropout=dropout, 
                dropout=0, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv6(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                #dropout=dropout, 
                dropout=0, 
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
            self.Qwz = ut.buildNetworkv6(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    #dropout=0, 
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
            self.Qy = ut.buildNetworkv6(
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
            self.Qd = ut.buildNetworkv6(
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

class VAE_GMM_Type1202(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    version 5 nw
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
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
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 32, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 32**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,32,32)),
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
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 32, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 32**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,32,32)),
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
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None, just_predict=False,):
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
        if just_predict:
            return output
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
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_z"] = loss_z
        losses["loss_y"] = loss_y
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output



class VAE_Dirichlet_Type1203(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    version 5 nw
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
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
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
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
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

    def forward(self, input, y=None, just_predict=False,):
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
        if just_predict:
            return output
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


class VAE_Dirichlet_Type1204(nn.Module):
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
    only Q has dropout, only first layer
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
        self.Px = ut.buildNetworkv6(
                #[nz, nh, nh, nh, nx],
                [nz] + numhiddenp * [nhp] + [nx],
                #dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv6(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                #dropout=dropout, 
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
            self.Qwz = ut.buildNetworkv6(
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
            self.Qy = ut.buildNetworkv6(
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
            self.Qd = ut.buildNetworkv6(
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

    def forward(self, input, y=None, just_predict=False,):
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
        if just_predict:
            return output
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


class VAE_MI_Type1205(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), Q(d|w,z)
    MI
    relax option
    resnet option
    softargmax option
    """
#
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
        mi_scale : float = 1e0,
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
        noiseLevel : float = 0.1,
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
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.softargmax = softargmax
        self.noiseLevel = noiseLevel
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
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 32, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 32**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,32,32)),
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
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 32, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nz+nw, 32**2),
                    #nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,32,32)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nw + nz] + numhidden*[nh] + [nclasses],
                    #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    #[nx] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        if use_resnet:
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 32, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    #nn.Linear(nw+nz, 64**2),
                    nn.Linear(nx, 32**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,32,32)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    #[nw + nz] + numhiddenq*[nhq] + [nclasses],
                    [nx] + numhiddenq*[nhq] + [nclasses],
                    #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None, training : bool = True):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        #eps=1e-6
        eps=1e-8
        if training:
            xnoise = self.noiseLevel * torch.randn_like(x) 
            wz = self.Qwz(x + xnoise)
        else:
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
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y_logits, factor=1e3)
        else:
            q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([x], dim=1))
        #d_logits = self.Qd(torch.cat([w,z], dim=1))
        #d_logits = self.Qd(torch.cat([w,z,x], dim=1))
        output["d_logits"] = d_logits
        d_y = nn.Softmax(dim=-1)(d_logits)
        d_y = (eps/self.nclasses +  (1 - eps) * d_y)
        #D_y = distributions.Dirichlet(d_logits.exp())
        #Qy = distributions.OneHotCategorical(probs=q_y)
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["q_y_logits"] = q_y_logits
        output["d_y"] = d_y
        output["d_logits"] = d_logits
        output["w"]=w
        output["z"]=z
        loss_mi = -ut.mutualInfo(q_y, d_y) * self.mi_scale
        losses["loss_mi"] = loss_mi
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
        #Pz = distributions.Normal(
        #        loc=mu_z_w,
        #        scale=std_z_w,
        #        )
        #output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        #p_y = D_y.rsample()
        #p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        #Py = distributions.OneHotCategorical(probs=p_y)
        #if (y != None) and self.relax:
        #    y = (eps/self.nclasses +  (1 - eps) * y)
        #if self.relax:
        #    Py = distributions.RelaxedOneHotCategorical(
        #            temperature=self.temperature.to(x.device),
        #            probs=p_y,
        #            )
        #else:
        #    Py = distributions.OneHotCategorical(probs=p_y)
        #output["Py"] = Py
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            #loss_z = (q_y*loss_z).sum(-1).mean()
            loss_z = (0.5*(q_y + d_y) *loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            #loss_y_alt = self.yscale * (q_y * (
            #        q_y.log() - p_y.log())).sum(-1).mean()
            #loss_y_alt2 = torch.tensor(0)
        else:
            #if self.relax:
            #    y = (eps/self.nclasses +  (1 - eps) * y)
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
            #loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            #loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        losses["loss_y"] = loss_y
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        #loss_y = -1e0 * p_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        #Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        #loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        #losses["loss_d"] = loss_d = loss_d
        #losses["loss_y_alt"] = loss_y_alt
        #losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                #+ loss_d
                #+ loss_y_alt
                #+ loss_y_alt2
                + loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

class CVAE_Dirichlet_Type1206(nn.Module):
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
    version 5 nw
    MI?
    hyrarchical?
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nc1: int = 10, # level 1 categories
        nc2: int = 10, # level 2 categories
        nclasses : int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        #applytanh : bool = False,
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
        self.nc1 = nc1
        self.nc2 = nc2
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.use_resnet = use_resnet
        #self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        #self.applytanh = applytanh
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        #self.y_prior = distributions.RelaxedOneHotCategorical(
        #        probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                #[nz, nh, nh, nh, nx],
                [nz + nc1] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                #[nw, nh, nh, nh, 2*nclasses*nz],
                [nw + nc1] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx + nc1, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx + nc1] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw + nc1, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    #[nw + nz] + numhidden*[nh] + [nclasses],
                    [nx + nw + nz + nc1] + numhiddenq*[nhq] + [nclasses],
                    #[nx] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nw+nz + nc1, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    [nw + nz + nc1] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, cond1=None, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        eps=1e-6
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
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
        q_y_logits = self.Qy(torch.cat([w,z,x,cond1], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,cond1], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        #q_d = nn.Softmax(dim=-1)(d_logits)
        #q_d = (eps/self.nclasses +  (1 - eps) * q_d)
        #output["q_d"] = q_d
        #loss_mi = -ut.mutualInfo(q_y, q_d) * self.mi_scale
        #losses["loss_mi"] = loss_mi
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
        rec = self.Px(torch.cat([z, cond1], dim=1))
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
        z_w = self.Pz(torch.cat([w,cond1], dim=-1))
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
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
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
                #+ loss_mi
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
            output = model.forward(x, training=True)
            #output = model.forward(x,y, training=True)
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
                    r,p,s =ut.estimateClusterImpurityLoop(model, x, y, device, )
                    print(p, "\n", r.mean(), "\n", r)
                    print((r*s).sum() / s.sum(), "\n",)
                    model.train()
                    model.to(device)
    model.cpu()
    optimizer = None
    model.load_state_dict(best_result)
    return 

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

    print("done training")
    return 

