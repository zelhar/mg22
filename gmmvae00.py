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

print(torch.cuda.is_available())

import pytorch_lightning as pl
from pl_bolts.models.autoencoders.components import resnet18_decoder, resnet18_encoder
class GMMClustering(nn.Module):
    """
    simple model for clustering using mixed gaussian model.
    generative model: P(x,y,z) = P(x | y,z) P(y) P(z)
    P(x | y,z) ~ GMM, meaning for a fixed y=i P(x | z,y=i) ~ N(mu(z_i), sig(z_i)
    P(z) ~ N(0,I), P(y) = (relaxed)Cat(pi), pi ~ (symmetric) Dirichlet prior.

    inference model: Q(z,y | x) = Q(z | x) Q(y | x)
    Q(z | x) ~ Normal, Q(z | y) ~ (relaxed) Categircal.
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nclasses: int = 10,
        nh: int = 1024,
        nz: int = 20,
        tau: Tensor = torch.tensor(0.7),
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.nclasses = nclasses
        self.tau = tau
        self.enc_z_x = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx, nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                )
        self.mu_z_x = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logvar_z_x = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.logits_y_x = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nx,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nclasses),
                )
        self.mus_xs_z = nn.Sequential(
                nn.Linear(nz,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nx * nclasses),
                nn.Unflatten(dim=-1, unflattened_size=(nclasses, nx)),
                )
        self.logvars_xs_z = nn.Sequential(
                nn.Linear(nz,nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nx),
                #nn.Linear(nh, nx * nclasses),
                #nn.Unflatten(dim=-1, unflattened_size=(nclasses, nx)),
                )
        return
    def forward(self, x, tau=0.5):
        h_x = self.enc_z_x(x)
        mu_z_x = self.mu_z_x(h_x)
        logvar_z_x = self.logvar_z_x(h_x)
        logits_y_x = self.logits_y_x(x)
        q_z = pyrodist.Normal(loc=mu_z_x, scale=torch.exp(0.5*logvar_z_x)).to_event(1)
        z = q_z.rsample()
        mus_xs_z = self.mus_xs_z(z)
        #logvars_xs_z = self.logvars_xs_z(z)
        q_y = distributions.RelaxedOneHotCategorical(tau, logits=logits_y_x,)
        return mu_z_x, logvar_z_x, logits_y_x, q_z, z, mus_xs_z, q_y
    def loss_v1(self, x, mu_z_x, logvar_z_x, logits_y_x, q_z, z, mus_xs_z, ):
        """
        Summation method (over labels y)
        """
        target = x.flatten(1).unsqueeze(1).expand(-1, self.nclasses, -1)
        bce = nn.BCEWithLogitsLoss(reduction='none',)(input=mus_xs_z,
                target=target)
        return bce

