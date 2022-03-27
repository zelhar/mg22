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
from torch.nn.functional import one_hot
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
from importlib import reload
from sklearn import mixture
from torch.nn import functional as F

import gmmvae02 as M

print(torch.cuda.is_available())

transform = transforms.Compose([
    #transforms.Resize((32,32)),
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

x, y = test_loader.__iter__().__next__()

test_data = test_loader.dataset.data.float()/255
test_labels = test_loader.dataset.targets
train_data = train_loader.dataset.data.float()/255
train_labels = train_loader.dataset.targets

# AE
model = M.AE()
model.fit(train_loader)

rec = model(x)["xhat"].reshape(-1,1,28,28)
ut.plot_2images(x, rec)

# Categorical AAE tests
model = M.BaseGAN()
model.apply(init_weights)
model.fit(num_epochs=1)

x = model.target_x.sample((8,64))
z = model.prior_z.sample((8,64))
x = model.target_x.sample((8,))
z = model.prior_z.sample((8,))
xgen = model.generator(z)

predict_real = model.discriminator(x)
predict_fake = model.discriminator(xgen)
real = torch.ones_like(predict_real)
fake = torch.zeros_like(predict_real)

criterion = nn.BCELoss(reduction="none")

criterion(predict_fake, fake).mean()

criterion(predict_real, real).mean()

criterion(predict_fake, real).mean()


criterion(predict_fake, predict_fake)


### Cat AAE
model = M.AAE(nclasses=10)
model.apply(init_weights)
model.fit(train_loader)

output = model(x.flatten(1))
rec = model(x.flatten(1))["xhat"].reshape(-1,1,28,28)

ut.plot_2images(x, rec)

ut.plot_2images(x[y==1], rec[y==1])

c = output["c"]
c[y==0].max(-1)

c[y==1].max(-1)

c[y==2].max(-1)

c[y==3].max(-1)

c[y==4].max(-1)

c[y==5].max(-1)

c[y==6].max(-1)

c[y==7].max(-1)

c[y==8].max(-1)

c[y==9].max(-1)

c = output["c"]
c = torch.eye(10)
cz = model.clusterhead_embedding(c)
cx = model.decoder(cz)
ut.plot_images(cx.reshape(-1,1,28,28))

z = model.z_prior.sample((5,))
zs = z.unsqueeze(1) + cz
zs = zs.reshape(5*10, 16)
xs = model.decoder(zs)

zs = cz[8] + z
xs = model.decoder(zs)
ut.plot_images(xs.reshape(-1,1,28,28), nrow=5)

