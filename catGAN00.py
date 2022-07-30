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

import my_torch_utils as ut

class catGAN(nn.Module):
    """
    one-hot categorical GAN.
    """
    def __init__(
        self,
        nx: int = 12,
        nz: int = 5,
        nh: int = 1024,
        concentration : float = 5e-1,
        numhidden : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nz = nz
        self.nh = nh
        self.numhidden = numhidden
        # Dirichlet constant prior:
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.x_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nx), temperature=self.temperature,)
        self.z_prior = distributions.Normal(
                loc=torch.ones(nz),
                scale=torch.tensor([1]),
                )
        # G network
        self.Gx = ut.buildNetworkv5(
                [nz] + numhidden*[nh] + [nx],
                dropout=0,
                #batchnorm=True,
                batchnorm=False,
                )
        self.Gx.add_module( "softmax", nn.Softmax(dim=-1))
        # D network
        self.Dx = ut.buildNetworkv5(
                [nx] + numhidden*[nh] + [1],
                dropout=0,
                batchnorm=False,
                #batchnorm=True,
                )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def generate(self, z=None, batch_size=128):
        if not z:
            z = torch.randn((batch_size, self.nz))
        y = self.Gx(z)
        return y

    def discriminate(self, x, ):
        d_logits = self.Dx(x) # logits
        return d_logits

    def fit(
        self,
        num_epochs=10000,
        lr=1e-3,
        device: str = "cuda:0",
        batch_size : int = 128,
    ) -> None:
        self.to(device)
        bce = nn.BCEWithLogitsLoss(reduction="none")
        optimizerD = optim.Adam([
            {"params" : self.Dx.parameters()},
            ],
                lr=lr*1e-1,
                weight_decay=0e-4,
                )
        optimizerG = optim.Adam([
            {"params" : self.Gx.parameters()},
            ],
                lr=lr*1e-1,
                weight_decay=0e-4,
                )
        for epoch in range(num_epochs):
            # train generator to decieve the discriminator
            self.Dx.eval()
            self.Gx.train()
            z_sample = self.z_prior.rsample((batch_size,50)).to(device) 
            x_gen = self.Gx(z_sample)
            true_labels = torch.ones((batch_size, 50, 1), device=device,)
            pred_x_logits = self.Dx(x_gen)
            loss_g = bce(pred_x_logits, true_labels).mean()
            optimizerG.zero_grad()
            loss_g.backward()
            optimizerG.step()
            # train the discriminator
            self.Dx.train()
            self.Gx.eval()
            z_sample = self.z_prior.rsample((batch_size,)).to(device) 
            fake_sample = self.Gx(z_sample)
            real_sample = self.x_prior.rsample((batch_size,)).to(device)
            pred_real_logits = self.Dx(real_sample)
            pred_fake_logits = self.Dx(fake_sample)
            true_labels = torch.ones((batch_size, 1), device=device,)
            false_labels = torch.zeros((batch_size, 1), device=device,)
            loss_d = (bce(pred_real_logits, true_labels).mean() +
                    bce(pred_fake_logits, false_labels).mean()
                    ) * 0.5
            optimizerD.zero_grad()
            loss_d.backward()
            optimizerD.step()
            if epoch % 100 == 0:
                print("loss_g: ", loss_g.item(), "loss_d:", loss_d.item(),
                        "\n",)
        self.cpu()
        return


    def forward(self, input, y=None,):
        pass
        #x = nn.Flatten()(input)
        #losses = {}
        #output = {}
        #eps=1e-6


model = catGAN()
model.fit()

model.generate()
model.generate().max(-1)

model.discriminate(model.generate(batch_size=10)).sigmoid()

model.x_prior.sample((10,))

model.discriminate(model.x_prior.sample((10,3))).sigmoid()
