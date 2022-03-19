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

x, y = test_loader.__iter__().__next__()

test_data = test_loader.dataset.data.float()/255
test_labels = test_loader.dataset.targets
train_data = train_loader.dataset.data.float()/255
train_labels = train_loader.dataset.targets

model = M.VAE(28**2, 20, 1024)

model = M.VAE_MC(28**2, 10, 1024)



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




m = M.distributions.RelaxedOneHotCategorical(temperature=torch.tensor(0.4),
        logits=torch.randn(3,5), )

  
model3 = M.VAE_MC_Gumbel(28**2, 10, 20, 1024, 0.3)

model3.fit(train_loader, )



print()
model = M.GumbelSSAE()
#mydata = ut.LabeledDataset(test_data, test_labels, 0.1)
model = M.GumbelSSAE(tau=torch.tensor(0.29))

model.cuda()

model.apply(init_weights)



#x_hat, z_q, y_q, z_q_dist, y_q_dist = model(x.cuda())

x,y = iter(test_loader).next()
x_hat, z_q, y_q, z_q_dist, y_q_dist = model(x)

recon = x_hat.reshape(-1,1,28,28)

fig, axs = plt.subplots(1,2)

grid_img1 = make_grid(x, nrow=16).permute(1, 2, 0)
grid_img2 = make_grid(recon, nrow=16).permute(1, 2, 0)

axs[0].imshow(grid_img1)
axs[1].imshow(grid_img2)

model.loss_MC(x.cuda(), x_hat, z_q, y_q, z_q_dist, y_q_dist, )

model.cpu()

model.fit(train_loader, num_epochs=13)

m = M.distributions.RelaxedOneHotCategorical(temperature=torch.tensor(0.3),
        logits=torch.ones(3),)

s = m.sample((5,))
s

m.log_prob(s)

tau = 0.5
r = 3e-5
r

#while True:
#    x,y = iter(test_loader).next()
#    x_hat, z_q, y_q, z_q_dist, y_q_dist = model(x)
#    bce, kl = model.loss_MC(x, x_hat, z_q, y_q, z_q_dist, y_q_dist, )
#    loss = bce + kl
#    loss.backward()

x_hat, z_q, y_q, z_q_dist, y_q_dist = model(test_data)

predictions = y_q_dist.probs.argmax(dim=-1)

test_labels[predictions == 0]


model = M.GumbelGMSSAE(tau=torch.tensor(0.29))
model.apply(init_weights)

model.fit(train_loader, num_epochs=8)

model.cpu()
x_hat, z_q, y_q, z_q_dist, y_q_dist, mu_prior, logvar_prior = model(test_data)

predicts = y_q_dist.probs.argmax(dim=-1)

for i in range(10):
    cluster = test_labels[predicts == i]
    print(torch.bincount(cluster))



