#import gdown
import matplotlib.pyplot as plt
import numpy as np
#import os
import pandas as pd
import pyro
import pyro.distributions as pyrodist
import scanpy as sc
import seaborn as sns
#import time
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

import gmmvae03 as M

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


test_data = test_loader.dataset.data.float()/255
test_labels = test_loader.dataset.targets
train_data = train_loader.dataset.data.float()/255
train_labels = train_loader.dataset.targets

## AE2
model = M.AE2(nz=100, nclasses=16)
model.apply(init_weights)
model.fit(train_loader, num_epochs=10, lr=1e-3)

x, y = test_loader.__iter__().__next__()
output = model(x)
#x, y = train_loader.__iter__().__next__()
rec = output["rec"].reshape(-1,1,28,28)
q_y = output["q_y"]

ut.plot_2images(x, rec)


output["q_y"].max(-1)
output["q_y"][y==0].max(-1)
output["q_y"][y==1].max(-1)
output["q_y"][y==2].max(-1)
output["q_y"][y==3].max(-1)
output["q_y"][y==4].max(-1)
output["q_y"][y==5].max(-1)
output["q_y"][y==6].max(-1)
output["q_y"][y==7].max(-1)
output["q_y"][y==8].max(-1)
output["q_y"][y==9].max(-1)

c = torch.eye(model.nclasses)
#c = c + torch.randn_like(c)
c = model.clusterhead_embedding(c)
#cx = model.Px(c+c).reshape(-1,1,28,28)
cx = model.Px(c).reshape(-1,1,28,28)
ut.plot_images(cx)

z = output["mu_z"]
model.assignCluster(c)
model.assignCluster(z)

#zs = distributions.Normal(loc=c, scale=1.0).sample((5,))
zs = distributions.Normal(loc=c, scale=0.5).sample((5,))
xs = model.Px(zs).reshape(-1,1,28,28)
#ut.plot_images(xs, 16)
ut.plot_images(xs, model.nclasses)


## AE3
#model = M.AE3(nz=20, nclasses=16)
model = M.AE3(nz=20, nclasses=10)
model.apply(init_weights)
#model.fit_v2(train_loader)
model.fit(train_loader)

x, y = test_loader.__iter__().__next__()
output = model(x)
rec = output["rec"].reshape(-1,1,28,28)
ut.plot_2images(x, rec)

mu, logvar = output["mu_z"], output["logvar_z"]

c = torch.eye(model.nclasses)
#c = c + torch.randn_like(c)
c = model.C.clusterhead_embedding(c)
#cx = model.Px(c+c).reshape(-1,1,28,28)
cx = model.P(c).reshape(-1,1,28,28)
ut.plot_images(cx)

z = output["mu_z"]
model.assignCluster(c)
model.assignCluster(z)

#zs = distributions.Normal(loc=c, scale=1.0).sample((5,))
zs = distributions.Normal(loc=c, scale=0.1).sample((5,))
xs = model.P(zs).reshape(-1,1,28,28)
#ut.plot_images(xs, 16)
ut.plot_images(xs, model.nclasses)


## Dilo3
model = M.VAE_Dilo3(nz=20, nw=25, nh=1024, nclasses=10)

model = M.VAE_Dilo3(nz=37, nw=47, nh=1024*2, nclasses=18)

model = M.VAE_Dilo3(nz=37, nw=47, nh=1024*2, nclasses=21)

model.apply(init_weights)
model.fit(train_loader, num_epochs=10)

x, y = test_loader.__iter__().__next__()
output = model(x)
Pz = output["Pz"]
Qz = output["Qz"]
z = output["z"]
w = output["w"]
q_y = output["q_y"]
mu_z = output["mu_z"]
logvar_z = output["logvar_z"]
losses = output["losses"]
losses
q_y.max(-1)

w = torch.zeros(2, model.nw)
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(2*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

#for k,v in output.items():
#    print(v.shape)

w = model.w_prior.sample((5, ))
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(5*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

model = M.VAE_Dilo3(nz=20, nw=25, nh=1024, nclasses=20)
model = M.VAE_Dilo3(nz=80, nw=95, nh=4024, nclasses=20)
model.apply(init_weights)
model.fit(train_loader, num_epochs=10)


## AE4
model = M.AE2(nh=1024, nz=20, nclasses=10)

model = M.AE4(nh=1024*2, nz=45, nclasses=23)

model.apply(init_weights)
model.fit(train_loader, num_epochs=10, lr=5e-4)

x, y = test_loader.__iter__().__next__()
output = model(x)
q_y = output["q_y"]
q_y

f = nn.Threshold(threshold=0.51, value=0.)
g = nn.Threshold(threshold=-0.1, value=1.)
f(q_y)
g(-f(q_y))


c = torch.eye(model.nclasses)
c = model.clusterhead_embedding(c)
cx = model.Px(c).reshape(-1,1,28,28)
ut.plot_images(cx)

c = torch.eye(model.nclasses)
c = model.clusterhead_embedding(c)
cx = model.Px(c).reshape(-1,1,28,28)
zs = distributions.Normal(loc=c, scale=0.5).sample((5,)).reshape((-1,model.nz))
xs = model.Px(zs).reshape(-1,1,28,28)
ut.plot_images(xs, model.nclasses)



### VAE_dirichlet
# best model so far:
model = M.VAE_Dilo3(nz=20, nw=25, nh=1024, nclasses=20)
model = M.VAE_Dilo3(nz=20, nw=25, nh=1024, nclasses=10)

model = M.VAE_Dirichlet(nz=20, nw=25, nh=2*1024, nclasses=20)

model = M.VAE_Dirichlet(nz=30, nw=35, nh=2*1024, nclasses=25)

model = M.VAE_Dirichlet(nz=20, nw=25, nh=2*1024, nclasses=10)

model.apply(init_weights)
model.fit(train_loader, num_epochs=8)


w = torch.zeros(3, model.nw)
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(3*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

#for k,v in output.items():
#    print(v.shape)

w = model.w_prior.sample((5, ))

w = distributions.Normal(loc=torch.zeros_like(w)*1.5e0,
        scale=torch.ones_like(w)).sample()
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(5*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

f = nn.Threshold(threshold=0.51, value=0.)
g = nn.Threshold(threshold=-0.1, value=1.)
y = g(-f(q_y.exp().exp().exp().softmax(-1)))
y

a = torch.ones(5)*1e-2
b = torch.ones(5)*1e-2
b[0] = 1

f=distributions.Dirichlet(a)
g=distributions.Dirichlet(b)
distributions.kl_divergence(f,g)

c = torch.randn(5)
h = distributions.RelaxedOneHotCategorical(temperature=0.02, logits=c)

mu = torch.randn(4,5)
mu2 = torch.randn(4,5)
lv = torch.randn(4,5)
lv2 = torch.randn(4,5)
p = distributions.Normal(loc=mu, scale=(0.5*lv).exp())
q = distributions.Normal(loc=mu2, scale=(0.5*lv2).exp())

distributions.kl_divergence(p,q)
ut.kld2normal(mu, lv, mu2, lv2)

### Dirichlet Bug
a = torch.ones(3)*1e-2
b = torch.ones(3)*1e-3
f=distributions.Dirichlet(a)
g=distributions.Dirichlet(b)
f.sample((10,)).max(-1)
g.sample((10,)).max(-1)


### scRNAseq data set
cdata = sc.read("./data/limb_sce_alternative.h5ad",)
n_conds = len(cdata.obs['subtissue'].cat.categories)
n_classes = len(cdata.obs['cell_ontology_class'].cat.categories)
n_dims = cdata.shape[1]
enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc.fit(cdata.obs['subtissue'].to_numpy()[:,None])
enc_ct = LabelEncoder()
enc_ct.fit(cdata.obs['cell_ontology_class'])
use_cuda = torch.cuda.is_available()
convert = {
        'obs' : {
            'subtissue' : lambda s: enc.transform(s.to_numpy()[:, None]),
            'cell_ontology_class': enc_ct.transform,
            }
        }

cdataloader = AnnLoader(cdata, batch_size=128, shuffle=True, convert=convert,
        use_cuda=use_cuda)

temp = cdataloader.dataset[:10]
temp.obs['subtissue']
temp.obs['cell_ontology_class']

foo = cdataloader.__iter__().next()
foo.layers['logcounts']

for idx, data in  enumerate(cdataloader):
    print(idx)
    break

model = M.VAE_Dilo3AnnData(nx = n_dims, nh=1024, nz=50, nw=50, nclasses=8)

model = M.VAE_Dilo3AnnData(nx = n_dims, nh=1024, nz=50, nw=50, nclasses=18)
model.apply(init_weights)
model.fit(cdataloader, num_epochs=100)

labels = cdataloader.dataset.obs['cell_ontology_class']
labels = enc_ct.transform(labels)



output = model(foo.layers['logcounts'].cpu().float())

output = model(foo.X.cpu().float())

y = foo.obs['cell_ontology_class']
y

output["q_y"].max(-1)
output["q_y"][y==0].max(-1)
output["q_y"][y==2].max(-1)
output["q_y"][y==3].max(-1)
output["q_y"][y==4].max(-1)
output["q_y"][y==5].max(-1)
output["q_y"][y==6].max(-1)
output["q_y"][y==7].max(-1)





