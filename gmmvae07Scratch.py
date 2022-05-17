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
print(torch.cuda.is_available())


sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=120, facecolor='white', )

enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc_ct = LabelEncoder()

adatac = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)

adata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)

sc.pp.highly_variable_genes(adata, n_top_genes=1000, inplace=True, subset=True,)


data = torch.FloatTensor(adata.X.toarray())
enc_ct.fit(adata.obs["Broad cell type"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["Broad cell type"]))
labels = F.one_hot(labels.long(), num_classes=44).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )



model = M6.AE_TypeA608(nx=1000, nh=1024, nz=64,)

model = M6.AE_TypeA609(nx=1000, nh=1024, nz=64, )

model = M7.VAE_Primer_Type700(nx=1000, nh=1024, nz=64, )

model = M7.AE_Primer_Type701(nx=1000, nh=1024, nz=64,)

# this one's predict was very close to cell type level 2
model = M7.VAE_Dirichlet_Type705(nx=1000, nh=1024, nz=64, nw=15, nclasses=13,)

model.apply(init_weights)

M6.basicTrain(model, data_loader, num_epochs=10, wt=0.0, )

model.eval()
output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()

adata.obsm["umap_old"] = pd.DataFrame(adata.obsm["X_umap"])

adata.obs["broad"] = adata.obs["Broad cell type"]
adatac.obs["broad"] = adatac.obs["Broad cell type"]

sc.pl.umap(adata, color=["batch", "broad",])

sc.pp.neighbors(adata, use_rep="z", n_neighbors=8,)
sc.tl.umap(adata,)

sc.pl.umap(adata, color=["batch", "broad",])

sc.pl.umap(adatac, color=["batch", "broad",])

sc.pl.umap(adata, color="broad")

sc.pl.umap(adata, color="batch")


adata = sc.read("./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad",)
adatac = sc.read("./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad",)

sc.pp.filter_cells(adata,min_genes=10)
#sc.pp.normalize_per_cell(adata,counts_per_cell_after=1e4)
#sc.pp.log1p(adata)
sc.pp.filter_genes(adata,min_cells=20)

sc.pp.highly_variable_genes(adata, n_top_genes=1000, inplace=True, subset=True,)

data = torch.FloatTensor(adata.X.toarray())
enc_ct.fit(adata.obs["broad"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["broad"]))
labels = F.one_hot(labels.long(), num_classes=44).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

model2 = M6.AE_TypeA608(nx=1000, nh=1024, nz=64,)
model2.apply(init_weights)

M6.basicTrain(model2, data_loader, num_epochs=40, wt=0.0, )

output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()

sc.pl.umap(adata, color=["batch", "broad", "annotation",])

sc.pl.umap(adatac, color=["batch", "broad","annotation",])

model3 = M6.VAE_Dilo_Type601(nx=1000, nh=1024, nz=32, nw=15, nclasses=14,)
model3 = M6.VAE_Dilo_Type601(nx=1000, nh=1024, nz=32, nw=15, nclasses=20,)
model3.apply(init_weights)

M6.basicTrain(model3, data_loader, num_epochs=40, wt=0.0, )

model3.eval()

output = model3(data)
adata.obsm["z"] = output["z"].detach().numpy()

adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)

sc.pl.umap(adata, color=["predict", "annotation",])

sc.pl.umap(adatac, color=["batch", "broad","annotation",])
sc.pl.umap(adatac, color=["batch", "broad","annotation",])

model4 = M7.VAE_Primer_Type700(nx=1000, nh=1024, nz=32, )
model4.apply(init_weights)

M6.basicTrain(model4, data_loader, num_epochs=40, wt=0.0, )

output = model4(data)
adata.obsm["z"] = output["z"].detach().numpy()

sc.pp.neighbors(adata, use_rep="z", n_neighbors=8,)
sc.tl.umap(adata,)

sc.tl.louvain(adata,)

sc.pl.umap(adata, color=["annotation", "broad", "louvain",])

data = torch.FloatTensor(adata.obsm["z"])
enc_ct.fit(adata.obs["louvain"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["louvain"]))
labels = F.one_hot(labels.long(), num_classes=44).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

model5 = M7.VAE_Stacked_Dilo_Anndata_Type701(nx=1000, nh=1024, nz=10, nw=5,
        nclasses=17,)

model5 = M7.VAE_Stacked_Dilo_Anndata_Type701(nx=1000, nh=1024, nz=32, nw=15,
        nclasses=27,)

model5 = M6.VAE_Dilo_Type601(nx=1000, nh=1024, nz=32, nw=15, nclasses=40,)

model5.apply(init_weights)

M6.basicTrain(model5, data_loader, num_epochs=30, wt=0.0, )

output = model5(data)

adata.obsm["z"] = output["mu_z"].detach().numpy()

adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adatac.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)

sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.louvain(adata,)

sc.pl.umap(adata, color=["annotation", "predict", "louvain",])

sc.pl.umap(adata, color=["Broad cell type", "predict", "louvain",])

sc.pl.umap(adatac, color=["annotation","predict",  "louvain",])

model = M7.VAE_Dirichlet_Type705(nx=1000, nh=1024, nz=32, nw=15, nclasses=10,)

model = M7.VAE_Dirichlet_Type705(nx=1000, nh=1024, nz=32, nw=15, nclasses=44,)

M6.basicTrain(model, data_loader, num_epochs=10, wt=0.0, )

output = model(data)

enc_ct.fit(adata.obs["prep"])
enc_ct.fit(adata.obs["prep"])
prep = torch.IntTensor(
        enc_ct.transform(adata.obs["prep"]))
prep = F.one_hot(prep.long(), num_classes=enc_ct.classes_.size).float()
