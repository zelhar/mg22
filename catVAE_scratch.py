import gdown
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyro
import pyro.distributions as dist
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

print(torch.cuda.is_available())

from importlib import reload
#from catVAE import *
import catVAE as M


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


test_data = test_loader.dataset.data.float()/255
test_labels = test_loader.dataset.targets
train_data = train_loader.dataset.data.float()/255
train_labels = train_loader.dataset.targets

model = M.VAE(28**2, 20, 1024)

model = M.VAE_MC(28**2, 10, 1024)


x, y = test_loader.__iter__().__next__()

x = x.to(0)
model.to(0)

model(x)

model.fit(train_loader, )

x,y = iter(test_loader).next()

xhat = model.generate(x)
xhat = xhat.reshape(x.shape)

plot_images(x)

plot_images(xhat)

fig, axs = plt.subplots(1,2)

grid_img1 = make_grid(x, nrow=16).permute(1, 2, 0)
grid_img2 = make_grid(xhat, nrow=16).permute(1, 2, 0)

axs[0].imshow(grid_img1)
axs[1].imshow(grid_img2)


z_locs, _  = model.encode(test_data)

ut.plot_tsne(z_locs, test_labels, "mnist_vae_tsne")

latent = z_locs.detach().cpu()

reducer = umap.UMAP(random_state=42)
reducer.fit(X=latent)
embedding = reducer.transform(latent)

plt.scatter(embedding[:,0], embedding[:,1], c=test_labels, cmap='Spectral', s=5)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24);





  
