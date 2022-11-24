# import gdown
# from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
# from pyro.optim import Adam
# import os
# import pyro
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import re
import pyro.distributions as pyrodist
import scanpy as sc
import seaborn as sns
from datetime import datetime
import time
import toolz
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
import skimage as skim
from abc import abstractmethod
from anndata.experimental.pytorch import AnnLoader
from importlib import reload
from math import pi, sin, cos, sqrt, log
import sklearn
from sklearn import datasets as skds
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import mixture
from toolz import partial, curry
from toolz import groupby, count, reduce, reduceby, countby, identity
from torch import nn, optim, distributions, Tensor
from torch.nn.functional import one_hot
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO
from torch.utils.data import WeightedRandomSampler
from scipy import stats
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
import gmmvae11 as M11
import gmmvae12 as M12
import gmmvae13 as M13
import gmmvae14 as M14
import gmmvae15 as M15
import gmmvae16 as M16
import gmmvaeBeta00 as Mb0
import gmmTraining as Train

print(torch.cuda.is_available())

#plt.ion()
#sc.settings.verbosity = 3
#sc.logging.print_header()
# sc.settings.set_figure_params(dpi=120, facecolor='white', )
# sc.settings.set_figure_params(figsize=(8,8), dpi=80, facecolor='white', )
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
)

adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_study.h5ad")
bdata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_study.h5ad")
adata.X = adata.X.toarray()
bdata.X = bdata.X.toarray()

enc_labels = LabelEncoder()
labels = enc_labels.fit_transform( adata.obs["cell_type"],)
labels = F.one_hot(torch.tensor(labels)).float()
enc_conds = LabelEncoder()
conditions = enc_conds.fit_transform(adata.obs["condition"],)
conditions = F.one_hot(torch.tensor(conditions)).float()
data = torch.tensor(adata.X)

test_enc_labels = LabelEncoder()
test_labels = test_enc_labels.fit_transform( bdata.obs["cell_type"],)
test_labels = F.one_hot(torch.tensor(test_labels)).float()
test_enc_conds = LabelEncoder()
test_conditions = test_enc_conds.fit_transform(bdata.obs["condition"],)
test_conditions = F.one_hot(torch.tensor(test_conditions)).float()
test_data = torch.tensor(bdata.X)

adata.obs["label"] = enc_labels.inverse_transform(
    labels.argmax(-1).detach().numpy(),
)
bdata.obs["label"] = test_enc_labels.inverse_transform(
    test_labels.argmax(-1).detach().numpy(),
)
adata.obs

data_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                data,
                labels,
                conditions,
                ],),
            batch_size=128,
            shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                test_data,
                test_labels,
                test_conditions,
                ],),
            batch_size=128,
            shuffle=True,
)

# so far the best unsupervised acc>0.82
model = Mb0.VAE_Dirichlet_GMM_TypeB1602xzC(
    nx=adata.n_vars,
    nz=24,
    nw=24,
    nclasses=labels.shape[1]*3,
    concentration=1e0,
    dropout=0.15,
    bn=True,
    reclosstype="mse",
    restrict_w=True,
    restrict_z=True,
    nc1=conditions.shape[1],
    #learned_prior=False,
    learned_prior=True,
    #positive_rec=True,
)
model.apply(init_weights)
print(model.__class__)


# 0.79
model = Mb0.VAE_Dirichlet_GMM_TypeB1602zC2(
    nx=adata.n_vars,
    nz=24,
    nw=24,
    #nz=8,
    #nw=8,
    #nz=48,
    #nw=48,
    #nz=12,
    #nw=12,
    nclasses=labels.shape[1]*3,
    concentration=1e0,
    dropout=0.15,
    #dropout=0,
    bn=True,
    reclosstype="mse",
    restrict_w=True,
    restrict_z=True,
    nc1=conditions.shape[1],
    #learned_prior=False,
    learned_prior=True,
    #positive_rec=True,
)
model.apply(init_weights)
print(model.__class__)

model = Mb0.VAE_Dirichlet_GMM_TypeB1602z(
    nx=adata.n_vars,
    nz=24,
    nw=24,
    nclasses=labels.shape[1]*3,
    concentration=1e0,
    dropout=0.15,
    #dropout=0,
    bn=True,
    reclosstype="mse",
    restrict_w=True,
    restrict_z=True,
)
model.apply(init_weights)
print(model.__class__)

#0.82
model = Mb0.VAE_Dirichlet_GMM_TypeB1602zC2(
    nx=adata.n_vars,
    #nz=24,
    #nw=24,
    #nz=8,
    #nw=8,
    #nz=48,
    #nw=48,
    nz=12,
    nw=12,
    #nclasses=labels.shape[1]*3,
    nclasses=labels.shape[1]*4,
    concentration=1e0,
    #dropout=0.15,
    dropout=0.1,
    bn=True,
    #reclosstype="mse",
    reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    nc1=conditions.shape[1],
    #learned_prior=False,
    learned_prior=True,
    positive_rec=True,
)
model.apply(init_weights)
print(model.__class__)

model = Mb0.VAE_Dirichlet_GMM_TypeB1602zC(
    nx=adata.n_vars,
    nz=24,
    nw=24,
    #nz=8,
    #nw=8,
    #nz=48,
    #nw=48,
    #nz=12,
    #nw=12,
    #nclasses=labels.shape[1]*3,
    nclasses=labels.shape[1]*4,
    concentration=1e0,
    #dropout=0.15,
    dropout=0.1,
    bn=True,
    #reclosstype="mse",
    reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    nc1=conditions.shape[1],
    #learned_prior=False,
    learned_prior=True,
    positive_rec=True,
)
model.apply(init_weights)
print(model.__class__)

Train.basicTrainLoop(
    model,
    data_loader,
    test_loader,
    num_epochs=20,
    lrs = [
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    report_interval=0,
    wt=0,
)

Train.basicTrainLoopCond(
    model,
    data_loader,
    test_loader,
    num_epochs=50,
    lrs = [
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    report_interval=0,
    wt=0,
)

r,p,s = ut.estimateClusterImpurityLoop(model, test_data, test_labels, "cuda", )
print(p,r,s)
s = s[s>=0]
r = r[r>=0]
#print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

#sc.pp.neighbors(bdata, use_rep="mu_z",)
sc.pp.neighbors(bdata, )
sc.pp.pca(bdata,)
sc.tl.umap(bdata,)
sc.tl.louvain(bdata,)

sc.pp.neighbors(adata, )
sc.pp.pca(adata,)
sc.tl.umap(adata,)
sc.tl.louvain(adata,)

ut.saveModelParameters(
        model,
        "./results/temp_gmmvaezc2_kang_us_082" + ut.timeStamp() + ut.randomString() + "params.pt",
        method="json",
        )
torch.save(
        model.state_dict(),
        "./results/temp_gmmvaezc2_kang_us_082" + ut.timeStamp() + ut.randomString() + "state.pt",
        )


scanpy_marker_genes_dict = {
    'B-cell': ['CD79A', 'MS4A1'],
    'Dendritic': ['FCER1A', 'CST3'],
    'Monocytes': ['FCGR3A'],
    'NK': ['GNLY', 'NKG7'],
    'Other': ['IGLL1'],
    'Plasma': ['IGJ'],
    'T-cell': ['CD3D'],
}
seurat_marker_genes_dict = {
    'CD4+ T' : ['ILR7', 'CCR7'],
    'CD14+ M' : ['CD14', 'LYZ'],
    'B-cell': ['CD79A', 'MS4A1'],
    'CD8+ T' : ['CD8A',],
    'NK': ['GNLY', 'NKG7'],
    'FCGR3A+ M' : ['FCGR3A', 'MS4A4'],
    'Dendritic': ['FCER1A', 'CST3'],
    'Platelet' : ['PPBP',],
    'Monocytes': ['FCGR3A'],
    'Other': ['IGLL1'],
    'Plasma': ['IGJ'],
    'T-cell': ['CD3D'],
}

seurat_marker_genes_dict2 = {
    'CD4+ T' : ['CCR7'],
    'CD14+ M' : ['CD14', 'LYZ'],
    'B-cell': ['CD79A', 'MS4A1'],
    'CD8+ T' : ['CD8A',],
    'NK': ['GNLY', 'NKG7',],
    'FCGR3A+ M' : ['FCGR3A',],
    'Dendritic': ['FCER1A', 'CST3'],
    'Platelet' : ['PPBP',],
    'Monocytes': ['FCGR3A'],
    'Plasma': ['IGJ'],
    'T-cell': ['CD3D'],
}

sc.pl.umap(
        adata,
        color = [
            "cell_type",
            "louvain",
            ],
        save="tmp.png",
        show=False,
        )

sc.pl.dotplot(
        adata,
        seurat_marker_genes_dict2,
        groupby=['cell_type',],
        dendrogram=True,
        save="tmp.png",
        )

sc.pl.dotplot(
        adata,
        seurat_marker_genes_dict2,
        groupby=['louvain',],
        dendrogram=True,
        save="tmp.png",
        )


fdata = sc.datasets.pbmc3k_processed()
fdata = sc.datasets.pbmc68k_reduced()

fdata = sc.read_h5ad(
        "./data/scgen/scGen_datasets/train_zheng.h5ad",
        )

sc.pl.dotplot(
        fdata,
        seurat_marker_genes_dict2,
        groupby=['cell_type',],
        dendrogram=True,
        save="tmp.png",
        )

model.cpu()
model.eval()
output = model(test_data, )
bdata.obs["predict"] =  output["q_y"].argmax(-1).detach().numpy().astype(str)
#bdata.obs["predict"] = test_enc_labels.inverse_transform(
#    output["q_y"].argmax(-1).detach().numpy(),
#)
bdata.obsm["mu_z"] = output["mu_z"].detach().numpy()
bdata.obsm["z"] = output["z"].detach().numpy()
bdata.obsm["mu_w"] = output["mu_w"].detach().numpy()
bdata.obsm["w"] = output["w"].detach().numpy()
rec = output["rec"].detach()
del output
sc.pp.neighbors(bdata, use_rep="mu_z",)
sc.pp.pca(bdata,)
sc.tl.umap(bdata,)
sc.tl.louvain(bdata,)

#fig, ax = plt.subplots(1,1)
sc.pl.umap(bdata, ncols=2, 
           color=[
               #"cell_type",
               "condition",
               "label",
               "louvain",
               "predict",
           ],
           #title="PBMCs (Kang): UMAP (pca)", 
           save="_tmp.png",
           show=False,
           #show=True,
           #ax=ax,
          )

sc.pl.dotplot(
        bdata,
        seurat_marker_genes_dict2,
        #groupby=['louvain',],
        #groupby=['label',],
        groupby=['predict',],
        dendrogram=True,
        save="_tmp.png",
        )



####
adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_study.h5ad",) #Kang train
adata.X = adata.X.toarray()
bdata = sc.read_h5ad("./data/scgen/scGen_datasets/train_zheng.h5ad",) #Zh
bdata.X = bdata.X.toarray()
cdata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_study.h5ad",) #Kang test
cdata.X = cdata.X.toarray()
ddata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_pbmc.h5ad",)
ddata.X = ddata.X.toarray()

sc.pp.scale(bdata, max_value=10,)


xdata = ut.balanceAnnData(bdata, "label", 250, False, 1e)
