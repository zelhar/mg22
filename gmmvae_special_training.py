#import gdown
import matplotlib
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
print(torch.cuda.is_available())


#matplotlib.use("QtCairo")
plt.ion()
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=120, facecolor='white', )

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

sampledata = ut.randomSubset(s=train_data.size(0), r=0.05)
cdata = sc.AnnData(X=train_data.detach().flatten(1).numpy()[sampledata],)
cdata.obs["labels"] = train_dataset.targets.numpy()[sampledata].astype(str)


enc_ct.fit(adata.obs["labels"])
#data = torch.FloatTensor(adata.X)






# interesting
model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        #nh=1024,
        nh=2324,
        #nw=15,
        #nw=25,
        nw=75,
        #nz=64,
        nz=174,
        #nclasses=30,
        nclasses=10,
        dscale=1e0,
        zscale=1e0,
        wscale=1e0,
        concentration=1e-0,
        #numhidden=2,
        numhidden=16,
        #dropout=0.25,
        dropout=0.1,
        bn=True,
        reclosstype='Bernoulli',
        )

model.apply(init_weights)

M9.basicTrainLoop(
    model,
    train_loader=data_loader,
    num_epochs=24,
    test_loader=None,
    lrs=[1e-5, 1e-4, 3e-4, 1e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    wt=0.0,
    report_interval=3,
)

M9.basicTrainLoop(
    model,
    train_loader=data_loader,
    num_epochs=5,
    test_loader=None,
    lrs=[1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
    wt=0.0,
    report_interval=3,
)



torch.save(model.state_dict(),
        "./results/model_m10T1004_mnistspecial.state_dict.pt",
        )

model.load_state_dict(torch.load("./results/model_m10T1004_mnistspecial.state_dict.pt"))

model = M10.VAE_Dirichlet_Type1004C(
        nx=28**2,
        #nh=1024,
        nh=2024,
        #nw=15,
        #nw=25,
        nw=65,
        #nz=64,
        nz=154,
        #nclasses=30,
        nclasses=10,
        dscale=1e0,
        zscale=1e0,
        wscale=1e0,
        concentration=1e-0,
        #numhidden=2,
        numhidden=6,
        nf=4,
        #dropout=0.25,
        dropout=0.1,
        bn=True,
        reclosstype='Bernoulli',
        )

torch.save(model.state_dict(),
        "./results/model_m10T1004C_2024h_65w_154z_10c_6nh_nf4_01do_mnistspecial.state_dict.pt",
        )

model = M10.VAE_Dirichlet_Type1004A(
        nx=28**2,
        #nh=1024,
        nh=4024,
        #nw=15,
        #nw=25,
        nw=15,
        #nz=64,
        nz=55,
        #nclasses=30,
        nclasses=20,
        dscale=1e0,
        zscale=1e0,
        wscale=1e0,
        concentration=2e-0,
        #numhidden=2,
        numhidden=1,
        #dropout=0.25,
        dropout=0.15,
        bn=True,
        reclosstype='Bernoulli',
        )

torch.cuda.empty_cache()

M9.basicTrainLoop(
    model,
    train_loader=data_loader,
    num_epochs=6,
    test_loader=None,
    lrs=[1e-5,
        1e-4,
        #3e-4,
        # 1e-3, 1e-3, 
        #3e-4,
        1e-4,
        3e-5,
        1e-5],
    wt=0.01,
    report_interval=3,
)

model.eval()
w = torch.zeros(1, model.nw)
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(1*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
if model.reclosstype == "Bernoulli":
    rec = rec.sigmoid()
ut.plot_images(rec, model.nclasses)

model.eval()
w = model.w_prior.sample((5, ))
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(5*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
if model.reclosstype == "Bernoulli":
    rec = rec.sigmoid()
ut.plot_images(rec, model.nclasses)

model.eval()
output = model(train_data)
adata.obsm["z"] = output["z"].detach().numpy()
adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
del(output)

#sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.louvain(adata, )
#sc.tl.leiden(adata, )
#sc.pl.umap(adata, color=["leiden"], size=5,)
sc.pl.umap(adata, color=["labels"], size=5)
sc.pl.umap(adata, color=["louvain",], size=5,)
sc.pl.umap(adata, color=["predict", ],size=5)
sc.pl.umap(adata, color=["predict2", ],size=5)

model.eval()
output = model(train_data[sampledata])
cdata.obsm["z"] = output["z"].detach().numpy()
cdata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
cdata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
del(output)

#sc.pp.pca(cdata,)
sc.pp.neighbors(cdata, use_rep="z", n_neighbors=10,)
sc.tl.umap(cdata,)
sc.tl.louvain(cdata, )
#sc.tl.leiden(cdata, )
#sc.pl.umap(cdata, color=["leiden"], size=5,)
sc.pl.umap(cdata, color=["labels"], size=5)
sc.pl.umap(cdata, color=["louvain",], size=5,)
sc.pl.umap(cdata, color=["predict", ],size=5)
sc.pl.umap(cdata, color=["predict2", ],size=5)

