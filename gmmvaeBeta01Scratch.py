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
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
)

adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_zheng.h5ad",) #Kang train
adata.X = adata.X.toarray()
sc.pp.scale(adata, max_value=10,)

sc.tl.louvain(
        adata,
        )

enc_labels = LabelEncoder()
labels = enc_labels.fit_transform( adata.obs["cell_type"],)
labels = F.one_hot(torch.tensor(labels)).float()
enc_conds = LabelEncoder()
conditions = enc_conds.fit_transform(adata.obs["condition"],)
conditions = F.one_hot(torch.tensor(conditions)).float()
data = torch.tensor(adata.X)

adata.obs["label"] = enc_labels.inverse_transform(
    labels.argmax(-1).detach().numpy(),
)
adata.obs

bdata = ut.balanceAnnData(
        adata,
        #"cell_type",
        "label",
        numSamples=500,
        noreps=False,
        eps=5e-5,
        add_noise=True,
        )


louvain = [int(x) for x in adata.obs["louvain"]]
louvain = F.one_hot(torch.tensor(louvain)).float()

louvainb = [int(x) for x in bdata.obs["louvain"]]
louvainb = F.one_hot(torch.tensor(louvainb)).float()

sc.pl.umap(
        adata,
        show=False,
        save="_zheng.png",
        color=[
            "louvain",
            "cell_type",
            "label",
            ],
        add_outline=True,
        ncols=2,
        color_map="viridis",
        #legend_loc="left",
        #title="Zheng PBMC umap",
        )

r,p,s = ut.estimateClusterAccuracy(louvain, labels)
r = r[r>=0]
s = s[s>=0]
print(p,(r*s).sum().item() / s.sum().item(), r.mean().item())

sc.pp.neighbors(bdata, n_neighbors=15, )
sc.pp.pca(bdata,)
sc.tl.umap(bdata,)
sc.tl.louvain(bdata, )
sc.pl.umap(
        bdata,
        show=False,
        save="_rebalanced_zheng.png",
        color=[
            "louvain",
            "label",
            #"cell_type",
            ],
        #add_outline=True,
        ncols=2,
        color_map="viridis",
        #legend_loc="left",
        #title="Zheng PBMC umap",
        )

#labelsb = enc_labels.fit_transform( bdata.obs["label"],)
labelsb = enc_labels.transform(bdata.obs["label"],)
labelsb = F.one_hot(torch.tensor(labelsb)).float()
conditionsb = enc_conds.fit_transform(bdata.obs["condition"],)
conditionsb = F.one_hot(torch.tensor(conditionsb)).float()
datab = torch.tensor(bdata.X)

data_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                datab,
                labelsb,
                #conditions,
                ],),
            batch_size= 2048,
            shuffle=True,
)

model = Mb0.VAE_Dirichlet_GMM_TypeB1602z(
    nx=adata.n_vars,
    #nclasses=louvain.shape[1] + 2,
    nclasses=10,
    nh=1024,
    nhq=1024,
    nhp=1024,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=15e-2,
    bn=True,
    #reclosstype="Gauss",
    reclosstype="mse",
    #relax=False,
    #relax=True,
    restrict_w=True,
    restrict_z=True,
    nz=12,
    nw=12,
)
model.apply(init_weights)
model.__class__

Train.basicTrainLoop(
    model,
    data_loader,
    data_loader,
    num_epochs=50,
    lrs = [
        1e-5,
        1e-4,
        1e-4,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    report_interval=0,
    wt=0,
)
model.cpu()
model.eval()
r,p,s = ut.estimateClusterImpurityLoop(model, datab, labelsb, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print(p,(r*s).sum().item() / s.sum().item(), r.mean().item())
r,p,s = ut.estimateClusterImpurityLoop(model, data, labels, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print(p,(r*s).sum().item() / s.sum().item(), r.mean().item())

data_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                data,
                labels,
                #conditions,
                ],),
            batch_size= 2048,
            shuffle=True,
)
# and train again
model = Mb0.VAE_Dirichlet_GMM_TypeB1602z(
    nx=adata.n_vars,
    nclasses=labels.shape[1],
    #nclasses=louvain.shape[1],
    nh=1024,
    nhq=1024,
    nhp=1024,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=15e-2,
    bn=True,
    #reclosstype="Gauss",
    reclosstype="mse",
    #relax=False,
    #relax=True,
    restrict_w=True,
    restrict_z=True,
    nz=12,
    nw=12,
)
model.apply(init_weights)
model.__class__

data_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                data,
                labels,
                #louvain,
                #conditions,
                ],),
            batch_size= 2048,
            shuffle=True,
)
data_loaderb = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                datab,
                labelsb,
                #louvainb,
                #conditions,
                ],),
            batch_size= 2048,
            shuffle=True,
)

Train.trainSemiSuperLoop(
    model,
    data_loaderb,
    data_loaderb,
    data_loaderb,
    num_epochs=50,
    lrs = [
        1e-5,
        1e-4,
        1e-4,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    report_interval=0,
    wt=0,
    #do_unlabeled=False,
    do_validation=False,
)
model.cpu()
model.eval()
r,p,s = ut.estimateClusterImpurityLoop(model, datab, labelsb, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print(p,(r*s).sum().item() / s.sum().item(), r.mean().item())
r,p,s = ut.estimateClusterImpurityLoop(model, data, labels, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print(p,(r*s).sum().item() / s.sum().item(), r.mean().item())

output = model(data)
adata.obs["predict"] =  output["q_y"].argmax(-1).detach().numpy().astype(str)
adata.obsm["mu_z"] = output["mu_z"].detach().numpy()
adata.obsm["z"] = output["z"].detach().numpy()
adata.obsm["mu_w"] = output["mu_w"].detach().numpy()
adata.obsm["w"] = output["w"].detach().numpy()
rec = output["rec"].detach()
del output

sc.pp.neighbors(adata, use_rep="mu_z",)
sc.pp.pca(adata,)
sc.tl.umap(adata,)
sc.tl.louvain(adata, )
sc.pl.umap(
        adata,
        show=False,
        save="_pred_zheng.png",
        color=[
            "louvain",
            "predict",
            "label",
            #"cell_type",
            ],
        #add_outline=True,
        ncols=2,
        color_map="viridis",
        #legend_loc="left",
        #title="Zheng PBMC umap",
        )

cdata = sc.read_h5ad("./data/scgen/scGen_datasets/train_study.h5ad")
ddata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_study.h5ad")
cdata.X = cdata.X.toarray()
ddata.X = ddata.X.toarray()
cdata = cdata[cdata.obs["condition"] == "control"].copy()
ddata = ddata[ddata.obs["condition"] == "control"].copy()

sc.pp.scale(cdata, max_value=10,)
sc.pp.scale(ddata, max_value=10,)

cdata = ut.balanceAnnData(
        cdata,
        "cell_type",
        noreps=True,
        )
#cdata = ut.balanceAnnData(
#        cdata,
#        "cell_type",
#        numSamples=250,
#        noreps=False,
#        eps=5e-5,
#        add_noise=True,
#        )
#ddata = ut.balanceAnnData(
#        ddata,
#        "cell_type",
#        numSamples=100,
#        noreps=False,
#        eps=5e-5,
#        add_noise=True,
#        )

#sc.pp.scale(cdata, max_value=10,)
#sc.pp.scale(ddata, max_value=10,)


labelsc = enc_labels.transform( cdata.obs["cell_type"],)
labelsc = F.one_hot(torch.tensor(labelsc)).float()
enc_conds = LabelEncoder()
conditionsc = enc_conds.fit_transform(cdata.obs["condition"],)
conditionsc = F.one_hot(torch.tensor(conditionsc)).float()
datac = torch.tensor(cdata.X)

labelsd = enc_labels.transform( ddata.obs["cell_type"],)
labelsd = F.one_hot(torch.tensor(labelsd)).float()
conditionsd = enc_conds.transform(ddata.obs["condition"],)
conditionsd = F.one_hot(torch.tensor(conditionsd)).float()
datad = torch.tensor(ddata.X)

cdata.obs["label"] = enc_labels.inverse_transform(
    labelsc.argmax(-1).detach().numpy(),
)
ddata.obs["label"] = enc_labels.inverse_transform(
    labelsd.argmax(-1).detach().numpy(),
)


data_loaderc = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                datac,
                labelsc,
                #conditions,
                ],),
            batch_size= 2048,
            shuffle=True,
)

Train.basicTrainLoop(
    model,
    data_loaderc,
    data_loaderc,
    num_epochs=50,
    lrs = [
        1e-5,
        1e-4,
        1e-4,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    report_interval=0,
    wt=0,
)
model.cpu()
model.eval()
r,p,s = ut.estimateClusterImpurityLoop(model, datac, labelsc, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print(p,(r*s).sum().item() / s.sum().item(), r.mean().item())
r,p,s = ut.estimateClusterImpurityLoop(model, datad, labelsd, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print(p,(r*s).sum().item() / s.sum().item(), r.mean().item())



output = model(datad)
ddata.obs["predict"] =  output["q_y"].argmax(-1).detach().numpy().astype(str)
ddata.obsm["mu_z"] = output["mu_z"].detach().numpy()
ddata.obsm["z"] = output["z"].detach().numpy()
ddata.obsm["mu_w"] = output["mu_w"].detach().numpy()
ddata.obsm["w"] = output["w"].detach().numpy()
rec = output["rec"].detach()
del output

sc.pp.neighbors(ddata, use_rep="mu_z",)
sc.pp.pca(ddata,)
sc.tl.umap(ddata,)
sc.tl.louvain(ddata, )
sc.pl.umap(
        ddata,
        show=False,
        save="_pred_kang.png",
        color=[
            "louvain",
            "predict",
            "label",
            #"cell_type",
            ],
        #add_outline=True,
        ncols=2,
        color_map="viridis",
        #legend_loc="left",
        #title="Zheng PBMC umap",
        )

output = model(datac)
cdata.obs["predict"] =  output["q_y"].argmax(-1).detach().numpy().astype(str)
cdata.obsm["mu_z"] = output["mu_z"].detach().numpy()
cdata.obsm["z"] = output["z"].detach().numpy()
cdata.obsm["mu_w"] = output["mu_w"].detach().numpy()
cdata.obsm["w"] = output["w"].detach().numpy()
rec = output["rec"].detach()
del output

sc.pp.neighbors(cdata, use_rep="mu_z",)
sc.pp.pca(cdata,)
sc.tl.umap(cdata,)
sc.tl.louvain(cdata, )
sc.pl.umap(
        cdata,
        show=False,
        save="_pred_kang.png",
        color=[
            "louvain",
            "predict",
            "label",
            #"cell_type",
            ],
        #add_outline=True,
        ncols=2,
        color_map="viridis",
        #legend_loc="left",
        #title="Zheng PBMC umap",
        )

