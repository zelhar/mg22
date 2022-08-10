#### Based on the models from gmmvae15.py, I want to convert them to
# zero-inflation models for the reconstruction and see if it improves the

# best of models
# sources:
# https://github.com/psanch21/VAE-GMVAE
# https://arxiv.org/abs/1611.02648
# http://ruishu.io/2016/12/25/gmvae/
# https://github.com/RuiShu/vae-clustering
# https://github.com/hbahadirsahin/gmvae
# https://docs.pyro.ai/en/1.5.0/distributions.html#zeroinflatednegativebinomial
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

# from pyro.optim import Adam
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
# from my_torch_utils import denorm, normalize, mixedGaussianCircular
# from my_torch_utils import fclayer, init_weights
# from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
# from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
# from my_torch_utils import scsimDataset
import my_torch_utils as ut

print(torch.cuda.is_available())

class AE_Type1600N(nn.Module):
    """
    vanilla AE
    z: unconstraint
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        zscale : float = 1e0,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        use_resnet : bool = False,
        eps : float = 1e-9,
        restrict_z : Union[bool, str] = False,
        activation = nn.LeakyReLU(),
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.eps = eps
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.zscale = zscale
        self.use_resnet = use_resnet
        self.restrict_z = restrict_z
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx *3],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        ## Q network
        self.Qz = ut.buildNetworkv5(
                [nx] + numhiddenq*[nhq] + [nz],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        mse = nn.MSELoss(reduction="none")
        losses = {}
        output = {}
        eps = self.eps
        z = self.Qz(torch.cat([x,], dim=-1))
        output["z"] = z
        xs = self.Px(torch.cat([z,], dim=1))
        rec_total_count_logits = xs[:,:self.nx]
        rec_total_count = rec_total_count_logits.exp()
        #rec_total_count = ut.softclip(rec_total_count_logits, 0.05, 1e2)
        rec_logits = xs[:,self.nx:2 * self.nx]
        #rec_probs = rec_logits.sigmoid()
        rec_gate_logits = xs[:,self.nx*2:]
        rec_gate = rec_gate_logits.sigmoid()
        rec_Dist = pyrodist.ZeroInflatedNegativeBinomial(
                total_count=rec_total_count,
                logits=rec_logits,
                gate=rec_gate,
                )
        rec = rec_Dist.sample()
        loss_rec = - rec_Dist.log_prob(x).sum(-1).mean()
        #if self.reclosstype == "Bernoulli":
        #    logits = rec
        #    rec = logits.sigmoid()
        #    bce = nn.BCEWithLogitsLoss(reduction="none")
        #    loss_rec = bce(logits, x).sum(-1).mean()
        #elif self.reclosstype == "mse":
        #    mse = nn.MSELoss(reduction="none")
        #    loss_rec = mse(rec, x).sum(-1).mean()
        #else:
        #    logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
        #    #logsigma_x = ut.softclip(self.logsigma_x, -2, 8)
        #    sigma_x = logsigma_x.exp()
        #    Qx = distributions.Normal(loc=rec, scale=sigma_x)
        #    loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        output["rec_logits"]= rec_logits
        output["rec_gate"]= rec_gate
        output["rec_total_count"]= rec_total_count
        losses["rec"] = loss_rec
        total_loss = (
                loss_rec
                + 0e0
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

class AE_Type1600G(nn.Module):
    """
    vanilla AE
    z: unconstraint
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        zscale : float = 1e0,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        use_resnet : bool = False,
        eps : float = 1e-9,
        restrict_z : Union[bool, str] = False,
        activation = nn.LeakyReLU(),
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.eps = eps
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.zscale = zscale
        self.use_resnet = use_resnet
        self.restrict_z = restrict_z
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx * 2 + 1],
                #[nz] + numhiddenp * [nhp] + [nx + 2],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        ## Q network
        self.Qz = ut.buildNetworkv5(
                [nx] + numhiddenq*[nhq] + [nz],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        mse = nn.MSELoss(reduction="none")
        losses = {}
        output = {}
        eps = self.eps
        z = self.Qz(torch.cat([x,], dim=-1))
        output["z"] = z
        xs = self.Px(torch.cat([z,], dim=1))
        rec_mu = xs[:,:self.nx]
        rec_logvar = ut.softclip(xs[:,self.nx:2*self.nx], -8, 8)
        #rec_logvar = xs[:,self.nx]
        rec_gate = xs[:,2*self.nx].sigmoid()
        #rec_gate = xs[:,self.nx+1].sigmoid()
        rec_std = (0.5 * rec_logvar).exp()
        #Qx = distributions.Normal(loc=rec_mu, scale=rec_std[:,None])
        Qx = distributions.Normal(loc=rec_mu, scale=rec_std)
        logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
        sigma_x = logsigma_x.exp()
        #    logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
        #    #logsigma_x = ut.softclip(self.logsigma_x, -2, 8)
        #    sigma_x = logsigma_x.exp()
        #    Qx = distributions.Normal(loc=rec, scale=sigma_x)
        #    loss_rec = -Qx.log_prob(x).sum(-1).mean()
        rec_Dist = pyrodist.ZeroInflatedDistribution(
                Qx, gate=rec_gate[:,None])
        loss_rec = - rec_Dist.log_prob(x).sum(-1).mean()
        rec = rec_Dist.sample()
        output["rec"]= rec
        output["rec_mu"]= rec_mu
        output["rec_logvar"]= rec_logvar
        output["rec_gate"]= rec_gate
        losses["rec"] = loss_rec
        total_loss = (
                loss_rec
                + 0e0
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output
# reconstruiction error.

class VAE_Dirichlet_GMM_Type1602(nn.Module):
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
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        cc_radius : float = 1e-1,
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
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : Union[bool, str] = False,
        restrict_z : Union[bool, str] = False,
        do_cc : bool = False,
        activation = nn.LeakyReLU(),
        recloss_min : float = 0,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.cc_radius = cc_radius
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.do_cc = do_cc
        self.softargmax = softargmax
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
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
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
                    #nn.Linear(nx, 64**2),
                    nn.Linear(nx, nhq),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    #activation=nn.Sigmoid(),
                    activation=activation,
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
                    #nn.Linear(nx+nz+nw, 64**2),
                    nn.Linear(nx+nz+nw, nhq),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    #activation=nn.Sigmoid(),
                    activation=activation,
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    #nn.Linear(nw+nz, 64**2),
                    nn.Linear(nw+nz, nhq),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    [nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    activation=activation,
                    batchnorm=bn,
                    )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w:
            #mu_w = ut.softclip(mu_w, -1,1)
            #logvar_w = ut.softclip(logvar_w, -5, 1)
            mu_w = ut.softclip(mu_w, -5,5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            #mu_z = ut.softclip(mu_z, -5,5)
            #logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5,5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
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
        rec = self.Px(torch.cat([z, ], dim=1))
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
            #logsigma_x = ut.softclip(self.logsigma_x, -2.0, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            #mu_z_w = ut.softclip(mu_z_w, -5,5)
            #logvar_z_w = ut.softclip(logvar_z_w, -5, 2)
            mu_z_w = ut.softclip(mu_z_w, -5,5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
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
        #losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
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
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
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
        #if self.do_cc:
        #    #ww = torch.zeros_like(w)
        #    ww = torch.randn_like(w) * self.cc_radius
        #    zz = self.Pz(ww)[0,:,:self.nz]
        #    rr = self.Px(zz)
        #    yy = self.justPredict(rr).to(x.device)
        #    cc = torch.eye(self.nclasses, device=x.device)
        #    loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
        #    total_loss = total_loss + loss_cc
        if self.do_cc:
            #ww = torch.zeros_like(w)
            #ww = torch.randn_like(w) * self.cc_radius
            ww = torch.zeros(10, self.nw, device=x.device) * self.cc_radius
            zz = self.Pz(ww)[:,:,:self.nz].reshape(-1, self.nz)
            rr = self.Px(zz)
            yy = self.justPredict(rr).to(x.device)
            cc = torch.eye(self.nclasses, device=x.device).tile(10,1)
            cc = (eps/self.nclasses + (1 - eps) * cc)
            #loss_cc = losses["loss_cc"] = self.cc_scale * (
            #        yy * (
            #            yy.log() - cc.log() ) ).sum(-1).mean()
            loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum(-1).mean()
            #loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
            total_loss = total_loss + loss_cc
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        #if self.restrict_w:
        #    mu_w = mu_w.tanh()
        #    #logvar_w = logvar_w.tanh() * 5
        #    logvar_w = ut.softclip(logvar_w, -5, 1)
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            #mu_w = mu_w.tanh()
            #mu_w = ut.softclip(mu_w, -5,5)
            #logvar_w = ut.softclip(logvar_w, -5, 2)
            mu_w = ut.softclip(mu_w, -5,5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            #mu_z = ut.softclip(mu_z, -5,5)
            #logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5,5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_Dirichlet_GMM_Type1602Temp(nn.Module):
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
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        cc_radius : float = 1e-1,
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
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : Union[bool, str] = False,
        restrict_z : Union[bool, str] = False,
        do_cc : bool = False,
        activation = nn.LeakyReLU(),
        recloss_min : float = 0,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.cc_radius = cc_radius
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.do_cc = do_cc
        self.softargmax = softargmax
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
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
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
                    #nn.Linear(nx, 64**2),
                    nn.Linear(nx, nhq),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    #activation=nn.Sigmoid(),
                    activation=activation,
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
                    #nn.Linear(nx+nz+nw, 64**2),
                    #nn.Linear(nx+nz+nw, nhq),
                    #nn.Linear(nx+nz, nhq),
                    nn.Linear(nz, nhq),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    #[nx + nz] + numhiddenq*[nhq] + [nclasses],
                    [nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    #activation=nn.Sigmoid(),
                    activation=activation,
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    #nn.Linear(nw+nz, 64**2),
                    #nn.Linear(nw+nz, nhq),
                    nn.Linear(nz, nhq),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    #[nw + nz] + numhiddenq*[nhq] + [nclasses],
                    [nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    activation=activation,
                    batchnorm=bn,
                    )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        elif self.restrict_w:
            #mu_w = ut.softclip(mu_w, -1,1)
            #logvar_w = ut.softclip(logvar_w, -5, 1)
            mu_w = ut.softclip(mu_w, -5,5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            #mu_z = ut.softclip(mu_z, -5,5)
            #logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5,5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
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
        #q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        #q_y_logits = self.Qy(torch.cat([z,x,], dim=1))
        q_y_logits = self.Qy(torch.cat([z,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        #d_logits = self.Qd(torch.cat([w,z,], dim=1))
        d_logits = self.Qd(torch.cat([z,], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
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
        rec = self.Px(torch.cat([z, ], dim=1))
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
            #logsigma_x = ut.softclip(self.logsigma_x, -2.0, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            #mu_z_w = ut.softclip(mu_z_w, -5,5)
            #logvar_z_w = ut.softclip(logvar_z_w, -5, 2)
            mu_z_w = ut.softclip(mu_z_w, -5,5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
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
        #losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
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
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
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
        #if self.do_cc:
        #    #ww = torch.zeros_like(w)
        #    ww = torch.randn_like(w) * self.cc_radius
        #    zz = self.Pz(ww)[0,:,:self.nz]
        #    rr = self.Px(zz)
        #    yy = self.justPredict(rr).to(x.device)
        #    cc = torch.eye(self.nclasses, device=x.device)
        #    loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
        #    total_loss = total_loss + loss_cc
        if self.do_cc:
            #ww = torch.zeros_like(w)
            #ww = torch.randn_like(w) * self.cc_radius
            ww = torch.zeros(50, self.nw, device=x.device) * self.cc_radius
            zz = self.Pz(ww)[:,:,:self.nz].reshape(-1, self.nz)
            rr = self.Px(zz)
            yy = self.justPredict(rr).to(x.device)
            cc = torch.eye(self.nclasses, device=x.device).tile(50,1)
            cc = (eps/self.nclasses + (1 - eps) * cc)
            #loss_cc = losses["loss_cc"] = self.cc_scale * (
            #        yy * (
            #            yy.log() - cc.log() ) ).sum(-1).mean()
            loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum(-1).mean()
            #loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
            total_loss = total_loss + loss_cc
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        #if self.restrict_w:
        #    mu_w = mu_w.tanh()
        #    #logvar_w = logvar_w.tanh() * 5
        #    logvar_w = ut.softclip(logvar_w, -5, 1)
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            #mu_w = mu_w.tanh()
            #mu_w = ut.softclip(mu_w, -5,5)
            #logvar_w = ut.softclip(logvar_w, -5, 2)
            mu_w = ut.softclip(mu_w, -5,5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            #mu_z = ut.softclip(mu_z, -5,5)
            #logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5,5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
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
        #q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        #q_y_logits = self.Qy(torch.cat([z,x,], dim=1))
        q_y_logits = self.Qy(torch.cat([z,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_GMM_Type1603(nn.Module):
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
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        cc_radius : float = 1e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : Union[bool, str] = False,
        restrict_z : Union[bool, str] = False,
        do_cc : bool = False,
        activation = nn.LeakyReLU(),
        recloss_min : float = 0,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.cc_radius = cc_radius
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.do_cc = do_cc
        self.softargmax = softargmax
        self.use_resnet = use_resnet
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
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=activation,
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=activation,
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
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
                    #nn.Linear(nx, 64**2),
                    nn.Linear(nx, nhq),
                    #nn.LeakyReLU(),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    #activation=nn.Sigmoid(),
                    activation=activation,
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
                    #nn.Linear(nx+nz+nw, 64**2),
                    nn.Linear(nx+nz+nw, nhq),
                    #nn.LeakyReLU(),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    #activation=nn.Sigmoid(),
                    activation=activation,
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            #mu_w = ut.softclip(mu_w, -5,5)
            #logvar_w = ut.softclip(logvar_w, -5, 2)
            mu_w = ut.softclip(mu_w, -5,5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            #mu_z = ut.softclip(mu_z, -5,5)
            #logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5,5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
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
        rec = self.Px(torch.cat([z, ], dim=1))
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
            #logsigma_x = ut.softclip(self.logsigma_x, -2.0, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            #mu_z_w = ut.softclip(mu_z_w, -5,5)
            #logvar_z_w = ut.softclip(logvar_z_w, -5, 2)
            mu_z_w = ut.softclip(mu_z_w, -5,5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
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
        #losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        #p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
            #loss_y = self.yscale * (
            #        q_y * (
            #            q_y.log() - p_y.log()
            #            )).sum(-1).mean()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        losses["loss_y"] = loss_y
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                )
        #if self.do_cc:
        #    #ww = torch.zeros_like(w)
        #    ww = torch.randn_like(w) * self.cc_radius
        #    zz = self.Pz(ww)[0,:,:self.nz]
        #    rr = self.Px(zz)
        #    yy = self.justPredict(rr).to(x.device)
        #    cc = torch.eye(self.nclasses, device=x.device)
        #    loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
        #    total_loss = total_loss + loss_cc
        if self.do_cc:
            #ww = torch.zeros_like(w)
            #ww = torch.randn_like(w) * self.cc_radius
            ww = torch.zeros(10, self.nw, device=x.device) * self.cc_radius
            zz = self.Pz(ww)[:,:,:self.nz].reshape(-1, self.nz)
            rr = self.Px(zz)
            yy = self.justPredict(rr).to(x.device)
            cc = torch.eye(self.nclasses, device=x.device).tile(10,1)
            cc = (eps/self.nclasses + (1 - eps) * cc)
            #loss_cc = losses["loss_cc"] = self.cc_scale * (
            #        yy * (
            #            yy.log() - cc.log() ) ).sum(-1).mean()
            loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum(-1).mean()
            #loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
            total_loss = total_loss + loss_cc
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justCC(self,batch_size=128):
        eps = self.eps
        ww = torch.randn(batch_size, self.nw) * self.cc_radius
        zz = self.Pz(ww)[:,:,:self.nz].reshape(-1, self.nz)
        rr = self.Px(zz)
        yy = self.justPredict(rr)
        cc = torch.eye(self.nclasses, ).tile(batch_size,1)
        #print(cc.shape, yy.shape, rr.shape, zz.shape, ww.shape)
        cc = (eps/self.nclasses + (1 - eps) * cc)
        loss_cc = self.cc_scale * (
                yy * (
                    yy.log() - cc.log() ) ).sum(-1).mean()
        return loss_cc, cc, yy, rr, zz, ww

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            #mu_w = ut.softclip(mu_w, -5,5)
            #logvar_w = ut.softclip(logvar_w, -5, 2)
            mu_w = ut.softclip(mu_w, -5,5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            #mu_z = ut.softclip(mu_z, -5,5)
            #logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5,5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_GMM_Type1603Temp(nn.Module):
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
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        cc_radius : float = 1e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : Union[bool, str] = False,
        restrict_z : Union[bool, str] = False,
        do_cc : bool = False,
        activation = nn.LeakyReLU(),
        recloss_min : float = 0,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.cc_radius = cc_radius
        self.mi_scale = mi_scale
        self.recloss_mii = recloss_min
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.restrict_z = restrict_z
        self.do_cc = do_cc
        self.softargmax = softargmax
        self.use_resnet = use_resnet
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
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=activation,
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=activation,
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
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
                    #nn.Linear(nx, 64**2),
                    nn.Linear(nx, nhq),
                    #nn.LeakyReLU(),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    #activation=nn.Sigmoid(),
                    activation=activation,
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
                    #nn.Linear(nx+nz+nw, 64**2),
                    #nn.Linear(nx+nz+nw, nhq),
                    nn.Linear(nz, nhq),
                    #nn.LeakyReLU(),
                    #nn.Sigmoid(),
                    activation,
                    nn.Linear(nhq, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    #[nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    [nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    #activation=nn.LeakyReLU(),
                    #activation=nn.Sigmoid(),
                    activation=activation,
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            #mu_w = ut.softclip(mu_w, -5,5)
            #logvar_w = ut.softclip(logvar_w, -5, 2)
            mu_w = ut.softclip(mu_w, -5,5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            #mu_z = ut.softclip(mu_z, -5,5)
            #logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5,5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
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
        #q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y_logits = self.Qy(torch.cat([z,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
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
        rec = self.Px(torch.cat([z, ], dim=1))
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
            #logsigma_x = ut.softclip(self.logsigma_x, -2.0, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        if self.restrict_z == "lv":
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z_w = mu_z_w.tanh()
            logvar_z_w = logvar_z_w.tanh() * 9
        elif self.restrict_z:
            #mu_z_w = ut.softclip(mu_z_w, -5,5)
            #logvar_z_w = ut.softclip(logvar_z_w, -5, 2)
            mu_z_w = ut.softclip(mu_z_w, -5,5)
            logvar_z_w = ut.softclip(logvar_z_w, -9, 1)
        else:
            pass
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
        #losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        #p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
            #loss_y = self.yscale * (
            #        q_y * (
            #            q_y.log() - p_y.log()
            #            )).sum(-1).mean()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        losses["loss_y"] = loss_y
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                )
        #if self.do_cc:
        #    #ww = torch.zeros_like(w)
        #    ww = torch.randn_like(w) * self.cc_radius
        #    zz = self.Pz(ww)[0,:,:self.nz]
        #    rr = self.Px(zz)
        #    yy = self.justPredict(rr).to(x.device)
        #    cc = torch.eye(self.nclasses, device=x.device)
        #    loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
        #    total_loss = total_loss + loss_cc
        if self.do_cc:
            #ww = torch.zeros_like(w)
            #ww = torch.randn_like(w) * self.cc_radius
            ww = torch.zeros(10, self.nw, device=x.device) * self.cc_radius
            zz = self.Pz(ww)[:,:,:self.nz].reshape(-1, self.nz)
            rr = self.Px(zz)
            yy = self.justPredict(rr).to(x.device)
            cc = torch.eye(self.nclasses, device=x.device).tile(10,1)
            cc = (eps/self.nclasses + (1 - eps) * cc)
            #loss_cc = losses["loss_cc"] = self.cc_scale * (
            #        yy * (
            #            yy.log() - cc.log() ) ).sum(-1).mean()
            loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum(-1).mean()
            #loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
            total_loss = total_loss + loss_cc
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justCC(self,batch_size=128):
        eps = self.eps
        ww = torch.randn(batch_size, self.nw) * self.cc_radius
        zz = self.Pz(ww)[:,:,:self.nz].reshape(-1, self.nz)
        rr = self.Px(zz)
        yy = self.justPredict(rr)
        cc = torch.eye(self.nclasses, ).tile(batch_size,1)
        #print(cc.shape, yy.shape, rr.shape, zz.shape, ww.shape)
        cc = (eps/self.nclasses + (1 - eps) * cc)
        loss_cc = self.cc_scale * (
                yy * (
                    yy.log() - cc.log() ) ).sum(-1).mean()
        return loss_cc, cc, yy, rr, zz, ww

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w == "lv":
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w == "mlv":
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 9
        elif self.restrict_w:
            #mu_w = ut.softclip(mu_w, -5,5)
            #logvar_w = ut.softclip(logvar_w, -5, 2)
            mu_w = ut.softclip(mu_w, -5,5)
            logvar_w = ut.softclip(logvar_w, -9, 1)
        else:
            pass
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        if self.restrict_z == "lv":
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z == "mlv":
            mu_z = mu_z.tanh()
            logvar_z = logvar_z.tanh() * 9
        elif self.restrict_z:
            #mu_z = ut.softclip(mu_z, -5,5)
            #logvar_z = ut.softclip(logvar_z, -5, 2)
            mu_z = ut.softclip(mu_z, -5,5)
            logvar_z = ut.softclip(logvar_z, -9, 1)
        else:
            pass
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
        #q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y_logits = self.Qy(torch.cat([z,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

