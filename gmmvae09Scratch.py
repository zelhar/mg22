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
print(torch.cuda.is_available())

enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc_ct = LabelEncoder()

transform = transforms.Compose([
    transforms.ToTensor(),
    ])
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
train_data = train_dataset.data.float()/255
test_data = test_dataset.data.float()/255
train_labels = F.one_hot(train_dataset.targets.long(),
        num_classes=10,).float()
test_labels = F.one_hot(test_dataset.targets.long(),
        num_classes=10,).float()


#foo = ut.randomSubset(len(train_data), 0.15)
#train_data[foo == False]
#foo.sum()


data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=train_data,
            labels=train_labels,
            ),
        batch_size=128,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=test_data,
            labels=test_labels,
            ),
        batch_size=128,
        shuffle=True,
        )

adata = sc.AnnData(X=train_data.detach().flatten(1).numpy(),)
adata.obs["labels"] = train_dataset.targets.numpy().astype(str)
bdata = sc.AnnData(X=test_data.detach().flatten(1).numpy(),)
bdata.obs["labels"] = test_dataset.targets.numpy().astype(str)


enc_ct.fit(adata.obs["labels"])
#data = torch.FloatTensor(adata.X)

sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.leiden(adata, )
sc.tl.louvain(adata, )
sc.pl.umap(adata, color=["labels", "leiden", "louvain"],)

sc.pp.pca(bdata,)
sc.pp.neighbors(bdata, use_rep="X_pca", n_neighbors=10,)
sc.tl.umap(bdata,)
sc.tl.leiden(bdata, )
sc.tl.louvain(bdata, )
sc.pl.umap(bdata, color=["labels", "leiden", "louvain"],)

model = M9.AE_Primer_Type901(
        nx=28**2, nh=1024, nz=64, bn=True,
        dropout=0.2, numhidden=3,
        )
model.apply(init_weights)

model = M7.VAE_Dirichlet_Type705(
        nx=28**2,
        nh=1024,
        nz=64,
        nw=15,
        nclasses=16,
        )
model.apply(init_weights)

model = M8.VAE_Dirichlet_Type807(
        nx=28**2,
        nh=1024,
        nz=64,
        nw=15,
        nclasses=16,
        dirscale=1e0,
        concentration=5e-1,
        )
model.apply(init_weights)

model = M9.VAE_Dirichlet_Type912(
        nx=28**2,
        nh=1024,
        nz=64,
        nw=25,
        nclasses=16,
        dirscale=1e0,
        zscale=1e0,
        concentration=5e-0,
        numhidden=2,
        dropout=0.2,
        )

# this one was good but might NaN in training
model = M9.VAE_Stacked_Dilo_Anndata_Type902(
        nx=adata.n_vars,
        nh=1024,
        nz=64,
        nw=28,
        nclasses=16,
        #nclasses=adata.obs["louvain"].cat.categories.size,
        )

model.apply(init_weights)

M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=15,
    test_loader=None,
    lr=1e-3,
    wt=0.0,
    report_interval=3,
)
M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=15,
    test_loader=None,
    lr=3e-4,
    wt=0.0,
    report_interval=3,
)
M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=15,
    test_loader=None,
    lr=4e-5,
    wt=0.0,
    report_interval=3,
)

model.eval()
output = model(train_data)
adata.obsm["z"] = output["z"].detach().numpy()
del(output)

sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.louvain(adata, )
sc.pl.umap(adata, color=["labels",],)
sc.pl.umap(adata, color=["louvain",],)

new_train_labels = F.one_hot(
        torch.tensor(
            adata.obs["louvain"].to_numpy().astype('long')
            ),
        num_classes=adata.obs["louvain"].cat.categories.size,).float()

subset = ut.randomSubset(s=len(new_train_labels), r=0.08)


labeled_data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=train_data[subset],
            labels=new_train_labels[subset],
            ),
        batch_size=128,
        shuffle=True,
        )
unlabeled_data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=train_data[subset == False],
            labels=new_train_labels[subset == False],
            ),
        batch_size=128,
        shuffle=True,
        )
test_data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=test_data,
            labels=test_labels,
            ),
        batch_size=128,
        shuffle=True,
        )

model = M9.VAE_Stacked_Dilo_Anndata_Type902(
        nx=adata.n_vars,
        nh=1024,
        nz=17,
        nw=8,
        nclasses=adata.obs["louvain"].cat.categories.size,
        )

model = M9.VAE_Stacked_Dilo_Anndata_Type902(
        nx=adata.n_vars,
        nh=1024,
        nz=27,
        nw=8,
        nclasses=adata.obs["louvain"].cat.categories.size,
        )



model = M9.VAE_Dirichlet_Type910(
        nx=adata.n_vars,
        nh=1024,
        nz=17,
        nw=8,
        numhidden=2, 
        dirscale=1e-0,
        concentration=1e-2,
        nclasses=adata.obs["louvain"].cat.categories.size,
        )

model = M8.VAE_Dirichlet_Type810(
        nx=adata.n_vars, nh=1024,
        nz=10, nw=5,
        dirscale=1e0,
        concentration=5e-3,
        numhidden=2,
        nclasses=adata.obs["louvain"].cat.categories.size,
        )

model = M9.VAE_Dirichlet_Type911(
        nx=adata.n_vars,
        nh=1024,
        nz=64,
        nw=17,
        numhidden=2, 
        dirscale=1e-2,
        concentration=1e0,
        nclasses=adata.obs["louvain"].cat.categories.size,
        )

model.apply(init_weights)

M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=7,
    test_loader=None,
    lr=1e-3,
    wt=1e-3,
    report_interval=3,
)

M9.trainSemiSuper(model, 
        labeled_data_loader,
        unlabeled_data_loader,
        test_data_loader,
        num_epochs=30,
        lr=1e-3,
        wt=0e-3,
        do_eval=False,
        report_interval=4,
        )

model.eval()
output = model(train_data)
adata.obsm["z2"] = output["z"].detach().numpy()
adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
sc.pp.neighbors(adata, use_rep="z2", n_neighbors=10,)
sc.tl.umap(adata,)
del(output)

sc.pl.umap(adata, color=["labels"], size=5)
sc.pl.umap(adata, color=["predict", ],size=5)
sc.pl.umap(adata, color=["predict2", ],size=5)

sc.tl.louvain(adata, )
sc.pl.umap(adata, color=["louvain", ],)

sc.tl.leiden(adata, )
sc.pl.umap(adata, color=["leiden"],)

x,y = test_loader.__iter__().next()
output = model(x)
q_y = output["q_y"]
q_y.argmax(-1) - y
torch.threshold(q_y, 0.5, 0).sum(0)
torch.sum(
    torch.threshold(q_y, 0.5, 0).sum(0) > 0)



model.eval()
w = torch.zeros(1, model.nw)
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(1*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

w = model.w_prior.sample((5, ))
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(5*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

del(output)
#adata = sc.AnnData(X=train_data.detach().flatten(1).numpy(), )
#adata.obs["labels"] = train_loader.dataset.targets.numpy().astype(str)

#test_loader = torch.utils.data.DataLoader(
#   dataset=datasets.MNIST(
#       root='data/',
#       train=False,
#       download=True,
#       transform=transform,
#       ),
#   batch_size=128,
#   shuffle=True,
#)
#train_loader = torch.utils.data.DataLoader(
#   dataset=datasets.MNIST(
#       root='data/',
#       train=True,
#       download=True,
#       transform=transform,
#       ),
#   batch_size=128,
#   shuffle=True,
#)


#test_data = test_loader.dataset.data.float()/255
#test_labels = F.one_hot(test_loader.dataset.targets.long(),
#        num_classes=10,).float()
#train_data = train_loader.dataset.data.float()/255
#train_labels = F.one_hot(
#        train_loader.dataset.targets.long(),
#        num_classes=10,).float()

#dataset = ut.SynteticDataSet(
#        train_data, train_labels,)
#data_loader = torch.utils.data.DataLoader(
#        dataset=dataset,
#        batch_size=128,
#        shuffle=True,
#        )
#
#labeled_set = ut.SynteticDataSet(
#        train_data[:2000], train_labels[:2000],)
#labeled_loader = torch.utils.data.DataLoader(
#        dataset=labeled_set,
#        shuffle=True,
#        batch_size=128,
#        )
#unlabeled_set = ut.SynteticDataSet(
#        train_data[2000:], train_labels[2000:],)
#unlabeled_loader = torch.utils.data.DataLoader(
#        dataset=unlabeled_set,
#        shuffle=True,
#        batch_size=128,
#        )
#test_set = ut.SynteticDataSet(
#        test_data, test_labels,)
#testloader = torch.utils.data.DataLoader(
#        dataset=test_set,
#        shuffle=True,
#        batch_size=128,
#        )

model = M8.VAE_Dirichlet_Type806(
        nx=adata.n_vars, nh=1024*2,
        nz=30, nw=25,
        dirscale=1e0,
        concentration=1e0,
        nclasses=23,
        )
model.apply(init_weights)

M8.basicTrain(model, data_loader, test_loader,
              num_epochs=10, wt=0e-3, lr=1e-3, report_interval=7)
M8.basicTrain(model, data_loader, test_loader,
              num_epochs=10, wt=0e-3, lr=1e-4, report_interval=7)
