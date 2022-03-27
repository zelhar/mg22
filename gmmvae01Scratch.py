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

import gmmvae01 as M

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


## Simple AE
model = M.AE(nh=1024*8)

model.fit(train_loader)

x, y = test_loader.__iter__().__next__()


output = model(x[y==0])

xhat = output['xhat'].reshape(-1,1,28,28)


plot_images(xhat)

z = torch.cat([x[i] for i in range(5, 50, 3)]).unsqueeze(1)
output = model(z)

w = torch.cat((x[20:], x[:20]))
plot_images(w)

output = model(w)
what = output['xhat'].reshape(-1,1,28,28)
plot_images(what)


model = M.Autoencoder()


model.cuda()
optimizer = torch.optim.Adam(model.parameters())
idx=0
for epoch in range(10):
    model.train()
    for dx, (data, labels) in enumerate(train_loader):
        #x = data.cuda().flatten(1)
        x = data.cuda()
        optimizer.zero_grad()
        # ===================forward=====================
        rec, mu, logvar = model(x)
        #loss = nn.MSELoss()(x, output['xhat'])
        # ===================backward====================
        BCE = F.binary_cross_entropy(rec, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = BCE + KLD
        loss.backward()
        optimizer.step()
        if idx % 300 == 0:
            print(
                "loss = ",
                loss.item(),
                    )
        idx +=1
    # ===================log========================
model.cpu()


model.fit(train_loader)

model = M.VAE()

xhat, _, _ = model(x[y==2])

xhat, _, _ = model(x[y==2])

plot_images(x)

plot_images(xhat.view(-1,1,28,28))

plot_images(xhat.reshape(-1,1,28,28))


output = model(x[y==1])
xhat = output['xhat']
plot_images(xhat.view(-1,1,28,28))

model.cuda()
optimizer = torch.optim.Adam(model.parameters())
idx=0
for epoch in range(10):
    model.train()
    for dx, (data, labels) in enumerate(train_loader):
        #x = data.cuda().flatten(1)
        x = data.cuda()
        optimizer.zero_grad()
        # ===================forward=====================
        output = model(x)
        rec = output['xhat']
        #loss = nn.MSELoss()(x, output['xhat'])
        # ===================backward====================
        BCE = F.binary_cross_entropy(rec, x.view(-1, 784), reduction='sum')
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #loss = BCE + KLD
        loss = BCE
        loss.backward()
        optimizer.step()
        if idx % 300 == 0:
            print(
                "loss = ",
                loss.item(),
                    )
        idx +=1
    # ===================log========================
model.cpu()

output = model(x)
xhat = output['xhat']
plot_images(xhat.view(-1,1,28,28))
output = model(x[y==1])
xhat = output['xhat']
plot_images(xhat.view(-1,1,28,28))

model = M.VAE2(nh=1024)
model.fit(train_loader)

model.eval()
output = model(x)
xhat = output['x_hat']
plot_images(x)
plot_images(xhat.view(-1,1,28,28))

output = model(x[y==6])
xhat = output['x_hat']
plot_images(xhat.view(-1,1,28,28))
