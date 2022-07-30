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


class GenericNet(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        raise NotImplementedError()

class GenericClusterAE(GenericNet):
    def __init__(
        self,
    ) -> None:
        """
        Contains all the common features of most the models 
        I am working with:
        w,z,y latent variables.
        """
        super().__init__()
        return

    def forward(self, input):
        raise NotImplementedError()

    def justPredict(self, input):
        raise NotImplementedError()



class VAE_Dirichlet_Type1400(nn.Module):
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
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,], dim=1))
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
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
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
                #+ loss_mi
                )
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
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_Dirichlet_Type1400_MNIST(nn.Module):
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
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    #nn.Linear(nx, 64**2),
                    nn.Unflatten(1, (1,28,28)),
                    #nn.Unflatten(1, (1,64,64)),
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
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,], dim=1))
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
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
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
                #+ loss_mi
                )
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
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y



class VAE_Dirichlet_Type1400D(nn.Module):
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
    deterministic
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
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,], dim=1))
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
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
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
                #+ loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        ww = torch.zeros_like(w)
        zz = self.Pz(ww)[0,:,:self.nz]
        rr = self.Px(zz)
        yy = self.justPredict(rr).to(x.device)
        cc = torch.eye(self.nclasses, device=x.device)
        loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
        total_loss = total_loss + loss_cc
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
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y


class CVAE_Dirichlet_Type1401(nn.Module):
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
        cc_scale : float = 1e1,
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
        restrict_w : bool = False,
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
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
                [nz + nc1] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
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
                    [nx + nw + nz + nc1] + numhiddenq*[nhq] + [nclasses],
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
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
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
                #+ loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output
    def justPredict(self, input, cond1=None, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_GMM_Type1402(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
    relax option
    resnet option
    version 5 nw
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
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        rec = self.Px(torch.cat([z,], dim=1))
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
        z_w = self.Pz(torch.cat([w,], dim=-1))
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
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

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_GMM_Type1402D(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
    relax option
    resnet option
    version 5 nw
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
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        rec = self.Px(torch.cat([z,], dim=1))
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
        z_w = self.Pz(torch.cat([w,], dim=-1))
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
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
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        ww = torch.randn(10,self.nw).to(x.device)
        zz = self.Pz(ww)[:,:,:self.nz]
        rr = self.Px(zz.reshape(10 * self.nclasses, self.nz))
        yy = self.justPredict(rr).to(x.device)
        cc = torch.eye(self.nclasses, device=x.device)
        cc = cc.repeat(10,1)
        loss_cc = losses["loss_cc"] = (yy - cc).abs().sum() * self.cc_scale
        total_loss = total_loss + loss_cc
        losses["total_loss"] = total_loss
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
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_GMM_Type1402_MNIST(nn.Module):
    """
    only suitable for 28×28 input.
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
    relax option
    resnet option
    version 5 nw
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
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    #nn.Linear(nx, 64**2),
                    nn.Unflatten(1, (1,28,28)),
                    #nn.Unflatten(1, (1,64,64)),
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
                    #nn.Unflatten(1, (1,28,28)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        rec = self.Px(torch.cat([z,], dim=1))
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
        z_w = self.Pz(torch.cat([w,], dim=-1))
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
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

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_GMM_Type1402D_MNIST(nn.Module):
    """
    only suitable for 28×28 input
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
    relax option
    resnet option
    version 5 nw
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
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    #nn.Linear(nx, 64**2),
                    nn.Unflatten(1, (1,28,28)),
                    #nn.Unflatten(1, (1,64,64)),
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
                    #nn.Unflatten(1, (1,28,28)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        rec = self.Px(torch.cat([z,], dim=1))
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
        z_w = self.Pz(torch.cat([w,], dim=-1))
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
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
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        ww = torch.randn(10,self.nw).to(x.device)
        zz = self.Pz(ww)[:,:,:self.nz]
        rr = self.Px(zz.reshape(10 * self.nclasses, self.nz))
        yy = self.justPredict(rr).to(x.device)
        cc = torch.eye(self.nclasses, device=x.device)
        cc = cc.repeat(10,1)
        loss_cc = losses["loss_cc"] = (yy - cc).abs().sum() * self.cc_scale
        total_loss = total_loss + loss_cc
        losses["total_loss"] = total_loss
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
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y


class CVAE_GMM_Type1403(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
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
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
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
        restrict_w : bool = False,
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
        self.nc1 = nc1
        self.nc2 = nc2
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
                [nz + nc1] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
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
                    [nx + nw + nz + nc1] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, cond1=None, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
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

    def justPredict(self, input, cond1=None, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_MI_Type1404(nn.Module):
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
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
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
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        #w = mu_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        #logvar_z = wz[:,1,self.nw:]
        #std_z = (0.5 * logvar_z).exp()
        #noise = torch.randn_like(mu_z).to(x.device)
        #z = mu_z + noise * std_z
        z = mu_z
        #Qz = distributions.Normal(loc=mu_z, scale=std_z)
        #output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        #output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,], dim=1))
        output["d_logits"] = d_logits
        q_d = nn.Softmax(dim=-1)(d_logits)
        q_d = (eps/self.nclasses +  (1 - eps) * q_d)
        output["q_d"] = q_d
        loss_mi = -ut.mutualInfo(q_y, q_d) * self.mi_scale
        losses["loss_mi"] = loss_mi
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
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        #logvar_z_w = z_w[:,:,self.nz:]
        #std_z_w = (0.5*logvar_z_w).exp()
        #Pz = distributions.Normal(
        #        loc=mu_z_w,
        #        scale=std_z_w,
        #        )
        #output["Pz"] = Pz
        #loss_z = self.zscale * ut.kld2normal(
        #        mu=mu_z.unsqueeze(1),
        #        logvar=logvar_z.unsqueeze(1),
        #        mu2=mu_z_w,
        #        logvar2=logvar_z_w,
        #        ).sum(-1)
        mse = nn.MSELoss(reduction="none")
        loss_z = self.zscale * mse(mu_z.unsqueeze(1), mu_z_w).sum(-1)
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_d,
                    )
        else:
            Py = distributions.OneHotCategorical(probs=q_d)
        output["Py"] = Py
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
            #loss_y_alt = self.yscale * (q_y * (
            #        q_y.log() - q_d.log())).sum(-1).mean()
            #loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            #loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            #loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
            loss_y = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        losses["loss_y"] = loss_y
        #losses["loss_y_alt"] = loss_y_alt
        #losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                #+ loss_y_alt
                #+ loss_y_alt2
                + loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
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
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class AE_Type1405(nn.Module):
    """
    AE
    w : gauss
    z: unconstraint
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
        #mi_scale : float = 1e0,
        cz_scale : float = 1e0,
        #concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        #temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cz_scale = cz_scale
        #self.mi_scale = mi_scale
        #self.concentration = concentration
        #self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
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
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
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
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            #logvar_w = logvar_w.tanh() * 5
            logvar_w = ut.softclip(logvar_w, -6, 1)
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
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z,], dim=1))
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
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:self.nz]
        logvar_z_w = z_w[:,self.nz:]
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
                ).sum(-1).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output



class AE_Type1406(nn.Module):
    """
    AE
    w : gauss
    z: unconstraint
    with lame clustering attmept
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
        #mi_scale : float = 1e0,
        cz_scale : float = 1e0,
        #concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        #temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
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
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cz_scale = cz_scale
        #self.mi_scale = mi_scale
        #self.concentration = concentration
        #self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
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
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
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
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        ## M network
        if use_resnet:
            resnet_m = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_m.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Mz = nn.Sequential(
                    nn.Linear(nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_m,
                    nn.Linear(1000, nclasses * nz),
                    )
        else:
            self.Mz = ut.buildNetworkv5(
                    [nw + nz] + numhiddenq*[nhq] + [nclasses * nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Mz.add_module(
                "unflatten",
                nn.Unflatten(1,(nclasses, nz)),
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
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            #logvar_w = logvar_w.tanh() * 5
            logvar_w = ut.softclip(logvar_w, -6, 1)
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
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z,], dim=1))
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
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:self.nz]
        logvar_z_w = z_w[:,self.nz:]
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
                ).sum(-1).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        output["q_y"] = q_y
        output["q_y_logits"] = q_y_logits
        #q_y.unsqueeze_(1) # (batch,1,nclasses)
        Mz = self.Mz(torch.cat([w,z], dim=1)) #(batch, nclasses, nz)
        cz = q_y.unsqueeze(1) @ Mz #(batch, 1, nz)
        cz.squeeze_(1) #(batch, nz)
        output["cz"] = cz
        output["Mz"] = Mz
        loss_cz = mse(z, cz).sum(-1).mean() * self.cz_scale
        losses["loss_cz"] = loss_cz
        p_y = torch.ones_like(q_y) / self.nclasses
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        if y == None:
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_cz
                + loss_y
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        output["losses"] = losses
        return output


class AE_Type1407(nn.Module):
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
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        ## Q network
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
        batch_size = x.shape[0]
        mse = nn.MSELoss(reduction="none")
        losses = {}
        output = {}
        eps = self.eps
        z = self.Qz(torch.cat([x,], dim=-1))
        output["z"] = z
        rec = self.Px(torch.cat([z,], dim=1))
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
        total_loss = (
                loss_rec
                + 0e0
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

class VAE_GMM_Type1408(nn.Module):
    """
    minibatch version of 1402. forget it for now.
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
        restrict_w : bool = False,
        mini_batch : int = 16,
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
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.mini_batch = mini_batch
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
                [nz * mini_batch] + numhiddenp * [nhp] + [nx * mini_batch],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw * mini_batch] + numhiddenp * [nhp] + [2*nclasses*nz * mini_batch],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (mini_batch, nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx * mini_batch, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, (2*nw + 2*nz) * mini_batch),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx * mini_batch] + numhiddenq*[nhq] + [(2*nw + 2*nz) * mini_batch],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (mini_batch, 2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear((nx+nz+nw) * mini_batch, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses * mini_batch),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [(nx + nw + nz) * mini_batch] + numhiddenq*[nhq] + [nclasses * mini_batch],
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

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        # make the batch size divide by the mini batch size
        n = self.mini_batch * (batch_size // self.mini_batch)
        x = x[:n]
        x = x.reshape(
                n // self.mini_batch, self.mini_batch * self.nx)
        losses = {}
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1)).flatten(0,1)
        mu_w = wz[:,0,:self.nw] #(batch, nw)
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w #(batch,nw)
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:] #(batch, nz)
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z #(batch, nz)
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        #wzx_minibatch = torch.cat([w,z,x,], dim=1).reshape(
        #        n // self.mini_batch, self.mini_batch, self.nx + self.nw + self.nz)
        #wzx_minibatch = wzx_minibatch.flatten(1)
        #q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        wzx_minibatch = torch.cat(
                [w.reshape(-1,self.mini_batch * self.nw),
                    z.reshape(-1,self.mini_batch * self.nz),
                    x.reshape(-1,self.mini_batch * self.nx),], dim=1)
        q_y_logits = self.Qy(wzx_minibatch).reshape(
                n // self.mini_batch, self.mini_batch, self.nclasses) 
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = q_y.reshape(n, self.nclasses) #(batch, nclasses)
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
        rec = self.Px(torch.cat([z.reshape(-1,self.mini_batch * self.nz),], dim=1))
        rec = rec.reshape(n, self.nx)
        x = x.reshape(n, self.nx)
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
        z_w = self.Pz(torch.cat([w.reshape(-1, self.mini_batch * self.nw),], dim=-1))
        z_w = z_w.flatten(0,1) #(batch, nclasses, 2*nz)
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
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
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

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        # make the batch size divide by the mini batch size
        n = self.mini_batch * (batch_size // self.mini_batch)
        x = x[:n]
        x = x.reshape(
                n // self.mini_batch, self.mini_batch * self.nx)
        output = {}
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1)).flatten(0,1)
        mu_w = wz[:,0,:self.nw] #(batch, nw)
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh() * 5
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:] #(batch, nz)
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
        wzx_minibatch = torch.cat(
                [w.reshape(-1,self.mini_batch * self.nw),
                    z.reshape(-1,self.mini_batch * self.nz),
                    x.reshape(-1,self.mini_batch * self.nx),], dim=1)
        q_y_logits = self.Qy(wzx_minibatch).reshape(
                n // self.mini_batch, self.mini_batch, self.nclasses) 
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        q_y = q_y.reshape(n, self.nclasses) #(batch, nclasses)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

