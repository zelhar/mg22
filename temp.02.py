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
from typing import List, Set, Dict, Tuple
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
from toolz import take, drop
import opt_einsum
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from anndata.experimental.pytorch import AnnLoader

import pickle

import scanpy as sc
import anndata as ad

import gdown

print(torch.cuda.is_available())

#%load_ext autoreload
#%autoreload 2

### working with ....py
#from IDEC_test00 import *
from .sigmavae01 import *


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

test_data = test_loader.dataset.data.float()/255
test_labels = test_loader.dataset.targets

train_data = train_loader.dataset.data.float()/255
train_labels = train_loader.dataset.targets


#model = SigmaVAE(nz=20, nh=2048, nin=28**2, imgsize=28)
model = SigmaVAE()

model.apply(init_weights)

model.fit(train_loader,)

x, labels = test_loader.__iter__().next()
x = x.flatten(1)

zmu, zlogvar, xmu = model(x)
model.loss_function(x, xmu, model.log_sigma, zmu, zlogvar)
kld(zmu, zlogvar)

plot_images(xmu.view(-1,1,28,28))

plot_images(x.view(-1,1,28,28))

model = SigmaVAE()
model.load_state_dict(torch.load('./data/temp_sigmavae.pt'))



model.init_kmeans(20, train_data)

cluster_centers = torch.tensor(model.kmeans.cluster_centers_)

cluster_images = model.decode(cluster_centers)

plot_images(cluster_images.view(-1,1,28,28))

lattent_data, _ = model.encode(train_data)


##### anndata test
cdata = ad.read("./data/limb_sce_alternative.h5ad",)

url = 'https://drive.google.com/uc?id=1ehxgfHTsMZXy6YzlFKGJOsBKQ5rrvMnd'
output = './data/pancreas.h5ad'
gdown.download(url, output, quiet=False)

adata = sc.read('./data/pancreas.h5ad')


