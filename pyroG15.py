# import gdown
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import os
import pandas as pd

import pyro
import pyro.distributions as pdist
import scanpy as sc
import seaborn as sns

# import time
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
import skimage as skim
from abc import abstractmethod
from anndata.experimental.pytorch import AnnLoader
from importlib import reload
from math import pi, sin, cos, sqrt, log

from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam

import sklearn
from sklearn import datasets as skds
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import mixture
from toolz import partial, curry
from torch import nn, optim, distributions, Tensor
from torch.nn.functional import one_hot
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO
from torch.utils.data import WeightedRandomSampler

# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights, buildNetwork
from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import scsimDataset
import my_torch_utils as ut
from importlib import reload
from torch.nn import functional as F
import gmmvae03 as M3
import gmmvae04 as M4
import gmmvae05 as M5
import gmmvae06 as M6
import gmmvae07 as M7
import gmmvae08 as M8
import gmmvae09 as M9
import gmmvae10 as M10
import gmmvae11 as M11
import gmmvae12 as M12
import gmmvae13 as M13
import gmmvae14 as M14
import gmmvae15 as M15
import gmmTraining as Train

print(torch.cuda.is_available())

plt.ion()
sc.settings.verbosity = 3
sc.logging.print_header()
# sc.settings.set_figure_params(dpi=120, facecolor='white', )
# sc.settings.set_figure_params(figsize=(8,8), dpi=80, facecolor='white', )
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
)

enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc_ct = LabelEncoder()


#@curry
#def binarize(x: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
#    ret = (x > threshold).float()
#    return ret.bool()
@curry
def binarize(x, threshold: float = 0.25):
    ret = (x > threshold).float()
    return ret


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        binarize(threshold=0.18),
    ]
)
train_dataset = datasets.MNIST(
    "data/",
    train=True,
    download=True,
    transform=transform,
)
test_dataset = datasets.MNIST(
    "data/",
    train=False,
    download=True,
    transform=transform,
)

train_data = train_dataset.data.float() / 255
test_data = test_dataset.data.float() / 255

train_labels = F.one_hot(
    train_dataset.targets.long(),
    num_classes=10,
).float()
test_labels = F.one_hot(
    test_dataset.targets.long(),
    num_classes=10,
).float()

data_loader = torch.utils.data.DataLoader(
    dataset=ut.SynteticDataSetV2(
        dati=[
            binarize(train_data),
            train_labels,
        ],
    ),
    batch_size=128,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    dataset=ut.SynteticDataSetV2(
        dati=[
            binarize(test_data),
            test_labels,
        ],
    ),
    batch_size=128,
    shuffle=True,
)

adata = sc.AnnData(
    X=train_data.detach().flatten(1).numpy(),
)
adata.obs["labels"] = train_dataset.targets.numpy().astype(str)
bdata = sc.AnnData(
    X=test_data.detach().flatten(1).numpy(),
)
bdata.obs["labels"] = test_dataset.targets.numpy().astype(str)




#####

class pyroVAE(nn.Module):
    def __init__(self,
            nx=28**2,
            nh=1024,
            nz=64,
            nw=32,
            numhidden=2,
            dropout=0.2,
            bn=True,
            activation=nn.LeakyReLU(),
            ):
        super().__init__()
        self.nx = nx
        self.nz = nz
        self.nw = nw
        self.Px = ut.buildNetworkv5(
                [nz] + numhidden * [nh] + [nx],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        self.Px.add_module(
                "sigmoid", nn.Sigmoid(),)
        self.Pz = ut.buildNetworkv5(
                [nw] + numhidden * [nh] + [2*nz],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        self.Qw = ut.buildNetworkv5(
                [nx] + numhidden * [nh] + [nw*2],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
        self.Qz = ut.buildNetworkv5(
                [nx] + numhidden * [nh] + [nz*2],
                dropout=dropout, 
                #activation=nn.LeakyReLU(),
                #activation=nn.Sigmoid(),
                activation=activation,
                batchnorm=bn,
                )
    # define model P(x|z)P(z|w)P(w)
    def model(self, x):
        # register modules
        pyro.module("Px", self.Px)
        pyro.module("Pz", self.Pz)
        with pyro.plate("data", x.shape[0]):
            w_loc = torch.zeros(x.shape[0], self.nw, device=x.device)
            w_scale = torch.ones(x.shape[0], self.nw, device=x.device)
            w = pyro.sample("w_latent",
                    pdist.Normal(loc=w_loc,scale=w_scale).to_event(1))
            z_loc_scale = self.Pz(w)
            z_loc = z_loc_scale[:,:self.nz]
            z_scale = z_loc_scale[:,self.nz:].exp()
            z = pyro.sample("z_latent",
                    pdist.Normal(loc=z_loc,scale=z_scale).to_event(1))
            rec = self.Px(z)
            #rec = self.Px(z).sigmoid()
            pyro.sample("obs",
                    pdist.Bernoulli(probs=rec).to_event(1), obs=x.reshape(-1, self.nx))
    def guide(self, x):
        pyro.module("Qw", self.Qw)
        pyro.module("Qz", self.Qz)
        with pyro.plate("data", x.shape[0]):
            w_loc_scale = self.Qw(x.flatten(1))
            w_loc = w_loc_scale[:,:self.nw]
            w_scale = w_loc_scale[:,self.nw:].exp()
            w = pyro.sample("w_latent",
                    pdist.Normal(loc=w_loc,scale=w_scale).to_event(1))
            z_loc_scale = self.Qz(x.flatten(1))
            z_loc = z_loc_scale[:,:self.nz]
            z_scale = z_loc_scale[:,self.nz:].exp()
            z = pyro.sample("z_latent",
                    pdist.Normal(loc=z_loc,scale=z_scale).to_event(1))
    def reconstruct(self, x):
        z_loc_scale = self.Qz(x.flatten(1))
        z_loc = z_loc_scale[:,:self.nz]
        z_scale = z_loc_scale[:,self.nz:].exp()
        z = pdist.Normal(z_loc, z_scale).sample()
        rec = self.Px(z)
        #rec = self.Px(z).sigmoid()
        return rec
        





optimizer = Adam({"lr": 1.0e-3})

vae = pyroVAE()
svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())

def train(svi, train_loader, use_cuda=True):
    # initialize loss accumulator
    epoch_loss = 0.
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, _ in train_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)
    # return epoch loss
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def evaluate(svi, test_loader, use_cuda=True):
    # initialize loss accumulator
    test_loss = 0.
    # compute the loss over the entire test set
    for x, _ in test_loader:
        # if on GPU put mini-batch into CUDA memory
        if use_cuda:
            x = x.cuda()
        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(x)
    normalizer_test = len(test_loader.dataset)
    total_epoch_loss_test = test_loss / normalizer_test
    return total_epoch_loss_test


train_elbo = []
test_elbo = []
vae.cuda()
# training loop
for epoch in range(10):
    total_epoch_loss_train = train(svi, data_loader, use_cuda=True)
    train_elbo.append(-total_epoch_loss_train)
    print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))
    if epoch % 2 == 0:
        # report test diagnostics
        total_epoch_loss_test = evaluate(svi, test_loader, use_cuda=True)
        test_elbo.append(-total_epoch_loss_test)
        print("[epoch %03d] average test loss: %.4f" % (epoch, total_epoch_loss_test))




x,y = data_loader.__iter__().next()

for x, _ in data_loader:
    print(x.shape)
    break


vae.cpu()
vae.reconstruct(x)
