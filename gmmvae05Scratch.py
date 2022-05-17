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
print(torch.cuda.is_available())

sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white',)

enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc_ct = LabelEncoder()


### blob test
blobs = sc.datasets.blobs(
        n_variables=20, n_centers=5, cluster_std=1.0, n_observations=2500,)
data = torch.FloatTensor(blobs.X)
labels = torch.IntTensor(blobs.obs['blobs'].apply(int))
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        )

model = M4.AE_Type00(nx=20, nh=1024, nz=10,)
model = M5.AE_Type02(nx=20, nh=1024, nz=10, nclasses=5,)
model = M5.AE_Type03(nx=20, nh=1024, nz=10, nclasses=5,)
model = M5.AE_Type03(nx=20, nh=1024, nz=10, nclasses=8,)

model = M5.VAE_Dilo_Type04(nx=20, nh=1024, nz=20, nw=15, nclasses=5,)

model = M5.VAE_Dirichlet_Type05(nx=20, nh=1024, nz=20, nw=15, nclasses=8)

model = M5.VAE_Dirichlet_Type05(nx=20, nh=1024, nz=20, nw=15, nclasses=5,)

model.apply(init_weights)
model.fit(data_loader, 30, lr=1e-3,)

output = model(data)

output["q_y"].argmax(-1)[labels == 0]
output["q_y"].argmax(-1)[labels == 1]
output["q_y"].argmax(-1)[labels == 2]
output["q_y"].argmax(-1)[labels == 3]
output["q_y"][labels == 4].argmax(-1)

blobs.obs['predict'] = output["q_y"].argmax(-1)

sc.tl.pca(blobs, svd_solver='arpack',)

sc.pl.pca(blobs, color="blobs",)

sc.pl.pca(blobs, color=["blobs", "predict"],)


### paul15 data

pdata = sc.datasets.paul15()


sc.pl.highest_expr_genes(pdata, n_top=20,)

sc.pp.filter_cells(pdata, min_genes=200, inplace=True,)
#sc.pp.filter_cells(pdata, min_genes=200, inplace=False,)
sc.pp.filter_genes(pdata, min_cells=100, inplace=True)
sc.pp.normalize_total(pdata, target_sum=1e4, inplace=True)
sc.pp.log1p(pdata, )
sc.pp.highly_variable_genes(pdata, min_mean=0.0125, min_disp=0.5, max_mean=3.0,
        subset=True, inplace=True,)
sc.pp.scale(pdata,max_value=10,)
sc.tl.pca(pdata, svd_solver="arpack",)
sc.pl.pca(pdata, color="paul15_clusters",)
sc.pl.pca_variance_ratio(pdata, log=True,)

sc.pp.neighbors(pdata, n_neighbors=10, n_pcs=40,)

sc.tl.leiden(pdata, )


pdata.obs['celltype']=pdata.obs['paul15_clusters'].str.split("[0-9]{1,2}", n = 1, expand = True).values[:,1]
pdata.obs['celltype2']=pdata.obs['paul15_clusters']

#sc.pl.paga(adata, plot=True,)
#sc.tl.umap(adata, init_pos='paga')

sc.tl.umap(pdata, n_components=15, )
sc.pl.umap(pdata, color=["paul15_clusters", "leiden", "celltype", ],)


data = torch.FloatTensor(pdata.X)

data = torch.FloatTensor(pdata.obsm["X_umap"])

labels = torch.IntTensor(pdata.obs['leiden'].apply(int))

labels = F.one_hot(labels.long(), num_classes=10).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        )

model = M5.VAE_Dilo_Type04(nx=pdata.n_vars, nh=1024, nz=20, nw=15, nclasses=9,)

model = M5.VAE_Dilo_Type04(nx=15, nh=1024, nz=20, nw=15, nclasses=9,)

model = M5.VAE_Dilo_Type04(nx=15, nh=1024, nz=20, nw=15, nclasses=10,)
#model = M5.VAE_Dilo_Type04(nx=15, nh=1024, nz=20, nw=15, nclasses=20,)

model = M5.AE_Type03(nx=pdata.n_vars, nh=1024, nz=20, nclasses=9,)
model = M5.AE_Type03(nx=15, nh=1024, nz=20, nclasses=9,)

model = M5.VAE_Dirichlet_Type05(nx=pdata.n_vars, nh=1024, nz=25, nw=20, nclasses=9,)
model = M5.VAE_Dirichlet_Type05(nx=15, nh=1024, nz=25, nw=20, nclasses=9,)

model = M4.AE_Type00(nx=pdata.n_vars, nh=1024, nz=20,)

model = M5.AE_Type02(nx=pdata.n_vars, nh=1024, nz=20, nclasses=19,)

model = M5.AE_Type02(nx=pdata.n_vars, nh=1024, nz=20, nclasses=10,)

model = M5.AE_Type02(nx=15, nh=1024, nz=20, nclasses=10,)

model.apply(init_weights)

model.fit(data_loader, 3, lr=1e-3,)

model.fit(data_loader, 50, lr=1e-3,)

output = model(data)

pdata.obs['predict'] = [str(x.item()) for x in output["q_y"].argmax(-1)]


sc.pl.pca(pdata, color=["paul15_clusters", "leiden", "predict", "celltype" ],)

sc.pl.pca(pdata, color=["paul15_clusters", "celltype2"],)

sc.pl.umap(pdata, color=["paul15_clusters", "celltype", "leiden", "predict" ],)



## Gtex
adata = sc.read('./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad',)

sc.pl.umap(adata, color=["tissue", "leiden",])
sc.pl.umap(adata, color=["Tissue", "annotation",])

adata = adata[:, adata.var.highly_variable]

enc.fit(adata.obs['annotation'].to_numpy()[:,None])
enc_ct.fit(adata.obs['annotation'])

n_classes = len(enc_ct.classes_)

data = torch.FloatTensor(adata.X.toarray())

data = torch.FloatTensor(adata.obsm["X_umap"])

labels = adata.obs['annotation']
labels = enc_ct.transform(labels)
labels = torch.IntTensor(labels)

#labels = F.one_hot(labels.long(), num_classes=9).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        )


model = M5.VAE_Dilo_Type04(nx=adata.n_vars, nh=1024, nz=30, nw=25,
        nclasses=n_classes,)

model.apply(init_weights)

model.fit(data_loader, 3, lr=1e-3,)

model.fit(data_loader, 30, lr=1e-3,)

output = model(data)

adata.obs['predict'] = [str(x.item()) for x in output["q_y"].argmax(-1)]

sc.pl.umap(adata, color=["predict", "annotation",])


adata = sc.read('./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad',)
sc.pp.filter_genes(adata, min_cells=20, inplace=True,)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.calculate_qc_metrics(adata,inplace=True,) 
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color="annotation")
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', ],
             jitter=0.4, multi_panel=True)
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
sc.pp.normalize_total(adata, target_sum=1e4, )
sc.pp.log1p(adata,)
sc.pp.highly_variable_genes(adata, subset=True, n_top_genes=1000,)

model = M5.AE_Type03(nx=1000, nh=2*1024, nz=10, nclasses=n_classes)


sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40,)

sc.tl.leiden(adata, )
sc.tl.umap(adata, n_components=15, )

sc.pl.umap(adata, color=["annotation", "leiden" ],)

model = M5.AE_Type03(nx=15, nh=2*1024, nz=10, nclasses=n_classes)

model = M5.VAE_Type00(nx=adata.n_vars, nh=1024, nz=20,)

output = model(data)

adata.obsm['muz'] = output['mu'].detach().numpy()
sc.pp.neighbors(adata, use_rep='muz', )
sc.tl.umap(adata, n_components=2, )
sc.tl.leiden(adata, )


adata = sc.read('./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad',)

bdata = sc.AnnData(adata.layers["counts"], obs=adata.obs[["annotation", "Tissue", ]],
        var=adata.var[["gene_ids", "gene_name",]], )

sc.pp.filter_cells(bdata, min_genes=200, inplace=True,)
sc.pp.filter_genes(bdata, min_cells=20, inplace=True,)
sc.pp.calculate_qc_metrics(bdata,inplace=True,) 
bdata = bdata[bdata.obs.n_genes_by_counts < 2500, :]

sc.pp.normalize_total(bdata, target_sum=1e4)
sc.pp.log1p(bdata,)
sc.pp.highly_variable_genes(bdata, n_top_genes=1000, subset=True,)
sc.pp.scale(bdata, max_value=10,)

enc_ct.fit(bdata.obs['annotation'])
n_classes = len(enc_ct.classes_)
data = torch.FloatTensor(bdata.X)
labels = bdata.obs['annotation']
labels = enc_ct.transform(labels)
labels = torch.IntTensor(labels)

#labels = F.one_hot(labels.long(), num_classes=9).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        )

model = M5.VAE_Type00(nx=bdata.n_vars, nh=1024, nz=80,)

model = M5.VAE_Type00(nx=bdata.n_vars, nh=3024, nz=100,)

model = M4.AE_Type00(nx=bdata.n_vars, nz=60)

model.apply(init_weights)

model.fit(data_loader, 3, lr=1e-3,)

model.fit(data_loader, 30, lr=1e-3,)

model.fit(data_loader, 30, lr=5e-4,)
model.fit(data_loader, 30, lr=5e-5,)

output = model(data)

bdata.obsm["z"] = output["z"].detach().numpy()

sc.pp.neighbors(bdata, n_neighbors=10, use_rep="z")
sc.tl.leiden(bdata,)
sc.tl.paga(bdata, )
sc.pl.paga(bdata,)
sc.tl.umap(bdata, init_pos='paga',)
sc.tl.umap(bdata, )
sc.pl.umap(bdata, color = ["annotation", "leiden"])

##### more Gtex
adata = sc.read('./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad',)
adata = adata[:, adata.var.highly_variable]
adata

sc.tl.tsne(adata, )

sc.pl.tsne(adata, color=["annotation", "leiden"])


enc_ct.fit(adata.obs['annotation'])
n_classes = len(enc_ct.classes_)

data = torch.FloatTensor(adata.X.toarray())

labels = adata.obs['annotation']
labels = enc_ct.transform(labels)
labels = torch.IntTensor(labels)

labels = F.one_hot(labels.long(), num_classes=n_classes).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        )




output = model(data)
adata.obs['predict'] = [str(x.item()) for x in output["q_y"].argmax(-1)]

sc.pl.umap(adata, color=["predict", "annotation",])

sc.pl.tsne(adata, color=["annotation", "leiden", "predict"])

labeled_data_loader = torch.utils.data.DataLoader(
        dataset= ut.SynteticDataSet(data[:1900], labels[:1900]),
        batch_size=100,
        shuffle=True,
        )
unlabeled_data_loader = torch.utils.data.DataLoader(
        dataset= ut.SynteticDataSet(data[1900:-500], labels[1900:-500]),
        batch_size=100,
        shuffle=True,
        )
test_data_loader = torch.utils.data.DataLoader(
        dataset= ut.SynteticDataSet(data[-500:], labels[-500:]),
        batch_size=100,
        shuffle=True,
        )

model = M5.VAE_Dilo_Type04(nx=adata.n_vars, nclasses=n_classes)
M5.trainSemiSuper(model, labeled_data_loader, unlabeled_data_loader,
        test_data_loader, num_epochs=100, do_unlabeled=False)
