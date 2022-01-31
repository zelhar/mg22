# fun fun 2022-01-29
# https://github.com/eelxpeng/dec-pytorch/blob/master/lib/idec.py
# https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py
# https://github.com/orybkin/sigma-vae-pytorch
import argparse
from importlib import reload
import matplotlib.pyplot as plt
import my_torch_utils as ut
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
from math import pi, sin, cos, sqrt, log
from toolz import partial, curry
from torch import nn, optim, distributions
from torch.nn.functional import one_hot
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple,
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO

# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import fnorm, replicate, logNorm
from my_torch_utils import scsimDataset
import scsim.scsim as scsim

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam

from toolz import take, drop,
import opt_einsum


print(torch.cuda.is_available())

kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class Net(nn.Module):
    def __init__(
        self,
        nin: int = 28 ** 2,
        nh: int = 2*1024,
        nout: int = 20,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin
        self.nout = nout
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        z = self.net(x)
        return z

class AE(nn.Module):
    def __init__(
        self,
        nz: int = 20,
        nh: int = 2*1024,
        nin: int = 28 ** 2,
        imgsize: Optional[int] = 28,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin = imgsize ** 2
        self.nz = nz
        self.imgsize = imgsize
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nz),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nin),
            nn.Sigmoid(),
            #nn.Unflatten(1, (1, imgsize, imgsize)),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decode(z)
        return y, z


transform = transforms.Compose([
    transforms.ToTensor(),
    #normalize,
    ])
test_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=False,
       download=True,
       transform=transform,
       ),
   batch_size=128,
   shuffle=True,
)
train_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=True,
       download=True,
       transform=transform,
       ),
   batch_size=128,
   shuffle=True,
)

class IDEC(nn.Module):
    """
    Improved deep embedded clustering
    see Xifeng Guo. 2017.4.30
    """
    def __init__(
        self,
        nin: int = 28 ** 2,
        nh: int = 2*1024,
        nz: int = 30,
        nclusters : int = 20,
        alpha : float = 1.0,
        gamma : float = 0.1,
        encoder : Optional[Net] = None,
        decoder: Optional[Net] = None,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin
        self.nh = nh
        self.nz = nz
        self.nclusters = nclusters
        self.alpha = alpha
        self.gamma = gamma
        self.encoder = encoder
        self.decoder = decoder
        # the parameter mu designates the cluster centers
        self.mu = nn.Parameter(torch.Tensor(nclusters, nz))
        if not self.encoder:
            self.encoder = Net(nin, nh, nz)
        if not self.decoder:
            self.decoder = Net(nz, nh, nin)

    def save_model(self, path : Union[str, IO]) -> None:
        torch.save(self.state_dict(), path)

    def load_mode(self, path : Union[str, IO]) -> None:
        loaded_dict = torch.load(path, )
        model_dict = self.state_dict()
        model_dict.update(loaded_dict)
        self.load_state_dict(model_dict)

    def soft_asign(self, z):
        """
        calculate the batch's students t distribution on the hidden 
        space
        """
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        q = self.soft_asign(z)
        return z, q, x_hat
        


test_data = test_loader.dataset.data.float()/255
test_labels = test_loader.dataset.targets

data = train_loader.dataset.data.float()/255


# trying to see how to cycle over data
x = torch.ones(100, 3)
for batch in range(0, 100, 7):
    y = x[batch : batch + 7, :]
    print(y)

w = drop(6, x)
list(w)

z = take( 7, drop (96, x))
torch.stack(list(z))

encoder = Net()
decoder = Net(20, 2*1024, 28**2)

model = IDEC(encoder=encoder)

def buildNetwork(
    layers: List[int],
    dropout=0,
    activation: Optional[nn.Module] = nn.ReLU(),
    #batchnorm: bool = True,
):
    #net = []
    net = nn.Sequential()
    for i in range(1, len(layers)):
        if dropout > 0:
            net.add_module("dropout", nn.Dropout(dropout))
        net.add_module('linear', nn.Linear(layers[i - 1], layers[i]))
        if activation:
            net.add_module("activation", activation)
    return nn.Sequential(*net)

foo = buildNetwork([10,5,3], dropout=0.2)

foo

x = torch.rand(3)
y = torch.rand(2)

foo = torch.einsum('i,j -> ij', x, y)
foo

torch.einsum('ij -> j', torch.ones(2,3))

torch.einsum('ij... -> j...', torch.ones(2,3,4,5,6))

x = torch.arange(24).reshape(4,3,2)

z = torch.arange(6).reshape(2,3)


def pwdist(x,y, reduce_dim=-1, p=1):
    """
    student distribution similarty
    """
    e = torch.ones((x.size(0),) + x.shape)
    xx = e * x
    yy = (e * y).swapdims(1,0)
    if reduce_dim:
        return (xx - yy).abs().pow(p).sum(reduce_dim)
    else:
        return (xx - yy).abs().pow(p)

def soft_assign(z, mu, alpha=1):
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - mu)**2, dim=2) / alpha)
    q = q**(alpha+1.0)/2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q




pdist = nn.PairwiseDistance(p=1, keepdim=True)

pdist(x, x)

pwdist(x, x)
pwdist(z, z)

# 3 2-d vectors
z = torch.arange(6).reshape(3,2)
z = z + 0.0
w = z+5
w

e = torch.ones(3)
e = e.reshape(3,1,1)
e
e * z
z * e

soft_assign(z,z)
