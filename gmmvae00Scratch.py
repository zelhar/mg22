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
import gmmvae00 as M

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

model = M.GMMClustering()
h = model.enc_z_x(x)
y = model.logits_y_x(x)
z = model.mu_z_x(h)
w = model.mus_xs_z(z)

mu_z_x, logvar_z_x, logits_y_x, q_z, z, mus_xs_z, q_y = model(x)

bce , kld_y, kld_z, target = model.loss_v1(x, mu_z_x, logvar_z_x, logits_y_x, q_y, q_z, z, mus_xs_z)

(bce.sum(-1) * q_y.probs).sum(-1)

pxhat = mus_xs_z.sigmoid()

bce2 = -target * pxhat.log() - (1 - target) * (1 - pxhat).log() 

bce2.sum() - bce.sum()

loss = bce.sum(-1)
loss.shape
loss = bce.sum(-1) * q_y.probs
loss.shape
loss.sum(-1).shape

loss = (bce.sum(-1) * q_y.probs).sum(-1) + kld_y.sum(-1) + kld_z.sum(-1)
loss.shape
loss.mean().shape

model = M.GMMClustering()
model.fit(train_loader, num_epochs=4)

z = torch.randn(128, 20)
xs = model.mus_xs_z(z).sigmoid()

x = xs[:,9,:].reshape(-1,1,28,28)
x.shape
plot_images(x)


#kl test
# both need to be normalized to get correct result
p = torch.tensor((9,12,4.0)) / 25
q = torch.tensor((1,1,1.0)) / 3
# kl(p || q) = sum(p * (log p - log q)) = 0.0852
nn.KLDivLoss(reduction='sum')(input=q.log(), target=p)
# kl(q || p) = 0.09745
nn.KLDivLoss(reduction='sum')(p.log(),q)


model = M.GMMKoolooloo(nz=20, nw=30, tau=torch.tensor(0.3))

model.y_prior.sample((128,)).shape
model.w_prior.sample((128,model.nw)).shape

mu_z, logvar_z, z, mu_w, logvar_w, w, y_logit = model.encode(x)

mus, logvars, p_zs, zs, x_logit, choice = model.decode(w, y_logit)


q_y = distributions.RelaxedOneHotCategorical(temperature=0.3, logits=y_logit)
q_y.probs
y_logit.softmax(-1)
q_y.logits

mus, logvars, p_zs, zs = model.decode_zs(w)

pz = pyrodist.Normal(loc=mus, scale=(0.5*logvars).exp()).to_event(1)

foo = nn.Linear(2,3)
bar = torch.rand((50,10,2))
foo(bar).shape

q_y.rsample()


pp = torch.distributions.Categorical(probs=torch.ones(3))
idx = pp.sample((5,))
idx.shape
u = torch.rand((5,3,10))
u

v = nn.functional.gumbel_softmax(logits=torch.randn((5,3)), tau=0.3, hard=True, )
v
v.shape
u.shape
w=v.unsqueeze(-1) * u

model = M.GMMKoolooloo(nz=20, nw=30, tau=torch.tensor(0.1))

q_z, q_w, z, w, q_y, mus_z, logvars_z, y, logit_x = model(x)
y.max(dim=-1)
v = nn.functional.gumbel_softmax(logits=q_y.logits, tau=0.9, hard=True, )

loss_rec = model.reconstruction_loss(logit_x, x, False)
loss_z = model.kld_z_loss(q_z, mus_z, logvars_z, q_y)
loss_w = model.kld_w_loss(q_w)

q = q_y.probs
p = torch.ones_like(q)/10
nn.KLDivLoss(reduction='sum')(input=q.log(), target=p)

model = M.GMMKoolooloo(nz=10, nw=10, tau=torch.tensor(0.1), nclasses=10)
model.fit(train_loader, num_epochs=8)

model.cpu()

#w = model.w_prior.rsample((128,model.nw))
w = model.w_prior.rsample((20,model.nw))
#w = torch.zeros_like(w) + 0.2
c = 7
mus, logvars = model.decode_zs_w(w)
mu_z = mus[:,c,:]
xhat = model.decode_x_z(mu_z).reshape(-1,1,28,28)
xhat = model.decode_x_z(mus).reshape(-1,1,28,28)
plot_images(xhat.sigmoid(), nrow=20)

xhat = xhat.reshape(-1, 20, 1,28,28).detach()

plot_images(xhat[4].sigmoid(), nrow=20)

z = torch.randn((40,model.nz))
xhat = model.decode_x_z(z).reshape(-1,1,28,28)
plot_images(xhat.sigmoid(), nrow=4)

model.cpu()
def plot_classes(model):
    #w = model.w_prior.rsample((2 * model.nclasses,model.nw))
    w = model.w_prior.rsample((2*model.nclasses,model.nw))
    mus, logvars = model.decode_zs_w(w)
    P_zs = pyrodist.Normal(loc=mus, scale=(0.5*logvars).exp()).to_event(1)
    zs = P_zs.sample()
    xhat = model.decode_x_z(zs).reshape(-1,1,28,28)
    plot_images(xhat.sigmoid(), nrow=2*model.nclasses)
plot_classes(model)

z = model.generate_class(c=8, batch=30)
xhat = model.decode_x_z(z).reshape(-1,1,28,28)
plot_images(xhat.sigmoid(), nrow=2*model.nclasses)
