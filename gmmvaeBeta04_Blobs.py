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
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D
import plotly.io as pio
from dash import Dash, html, dcc
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
    dpi=200,
    facecolor="white",
)
pio.renderers.default = "png"
sns.set_palette(sns.color_palette("pastel"),)
sns.set(rc={"figure.dpi":200, 'savefig.dpi':100})
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['savefig.dpi'] = 100

# create conditional blobs dataset
adata = ut.blobs(ns=350, nc=2, ny=5, effect=1.7, nx=16)
df=adata.obs
df["cond_m"] = 350*5*["p"] + 350*5*["x"]
df["color"] = [int(x) for x in df["label"]]

cond_m = 350*5*["p"] + 350*5*["x"]
color = [int(x) for x in df["label"]]

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    data=df[df.cond == 'ctrl'],
    xs="x",
    ys="y",
    zs="z",
    c = "color",
    marker="p",
)
ax.scatter(
    data=df[df.cond == 'trtmnt'],
    xs="x",
    ys="y",
    zs="z",
    c = "color",
    marker="x",
)

plt.cla()
plt.clf()
plt.close()

labels_str = [str(x) for x in df["label"]]
labels = F.one_hot(torch.tensor([int(x) for x in labels_str])).float()
conditions_str = [str(x) for x in df["cond"]]
enc_conds = LabelEncoder()
conditions = F.one_hot(torch.tensor(enc_conds.fit_transform(conditions_str))).float()
data = torch.tensor(adata.X)

data_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [data,
                labels,
                conditions,
                ],),
            batch_size=2**11,
            shuffle=True,
            )

x,y,c = data_loader.__iter__().__next__()


model1 = Mb0.VAE_Dirichlet_GMM_TypeB1602zC2(
    nx=adata.n_vars,
    nz=3,
    nw=3,
    nclasses=labels.shape[1],
    nc1=conditions.shape[1],
    concentration=1.0e0,
    dropout=15e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    positive_rec=True,
    #nh=2**11,
    #nhp=2**11,
    #nhq=2**11,
    numhidden=1,
    numhiddenp=1,
    numhiddenq=1,
    learned_prior=True,
    #learned_prior=False,
)
model1.apply(init_weights)
print()


Train.basicTrainLoopCond(
    model1,
    data_loader,
    data_loader,
    num_epochs=20,
    lrs = [
        1e-5,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    test_accuracy=False,
    report_interval=0,
    wt=1e-4,
)
r,p,s = ut.estimateClusterImpurity(model1, data, labels, "cuda", conditions)
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print("acc= \n", (r*s).sum().item() / s.sum().item(), r.mean().item())


output = model1(data, cond1=conditions)
adata.obsm["mu_z"] = output["mu_z"].detach().numpy()
adata.obsm["z"] = output["z"].detach().numpy()
adata.obsm["mu_w"] = output["mu_w"].detach().numpy()
adata.obsm["w"] = output["w"].detach().numpy()
adata.obs["predict"] = output["q_y"].argmax(-1).detach().numpy().astype(str)
del output
df[["zx","zy","zz"]] = adata.obsm["mu_z"]
df[["wx","wy","wz"]] = adata.obsm["w"]
df["predict"] = adata.obs["predict"]
df["predict_int"] = [int(x) for x in adata.obs["predict"]]


fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(
    data=df[(df.cond == 'ctrl') & (df.label=='3')],
    xs="wx",
    ys="wy",
    zs="wz",
    c = "predict_int",
    marker="p",
)
ax.scatter(
    data=df[(df.cond == 'trtmnt') & (df.label=='3')],
    xs="wx",
    ys="wy",
    zs="wz",
    c = "predict_int",
    marker="x",
)

#### 
model2 = Mb0.VAE_Dirichlet_GMM_TypeB1602zC2(
    nx=adata.n_vars,
    nz=2,
    nw=2,
    nclasses=labels.shape[1],
    nc1=conditions.shape[1],
    concentration=1.0e0,
    dropout=15e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    positive_rec=True,
    #nh=2**11,
    #nhp=2**11,
    #nhq=2**11,
    numhidden=1,
    numhiddenp=1,
    numhiddenq=1,
    learned_prior=True,
    #learned_prior=False,
)
model2.apply(init_weights)
print()

Train.basicTrainLoopCond(
    model2,
    data_loader,
    data_loader,
    num_epochs=20,
    lrs = [
        1e-5,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    test_accuracy=False,
    report_interval=0,
    wt=1e-4,
)
r,p,s = ut.estimateClusterImpurity(model2, data, labels, "cuda", conditions)
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print("acc= \n", (r*s).sum().item() / s.sum().item(), r.mean().item())

output = model2(data, cond1=conditions)
adata.obsm["mu_z"] = output["mu_z"].detach().numpy()
adata.obsm["z"] = output["z"].detach().numpy()
adata.obsm["mu_w"] = output["mu_w"].detach().numpy()
adata.obsm["w"] = output["w"].detach().numpy()
adata.obs["predict"] = output["q_y"].argmax(-1).detach().numpy().astype(str)
del output
df[["zx","zy",]] = adata.obsm["z"]
df[["mu_zx","mu_mu_zy",]] = adata.obsm["mu_z"]
df[["wx","wy",]] = adata.obsm["w"]
df[["mu_wx","wy",]] = adata.obsm["mu_w"]
df["predict"] = adata.obs["predict"]
df["predict_int"] = [int(x) for x in adata.obs["predict"]]


ax = sns.relplot(
        df,
        x="x",
        y="y",
        #hue="label",
        hue="predict",
        kind="scatter",
        legend="brief",
        style="cond",
        palette=sns.color_palette(),
        )
sns.move_legend(ax, "upper right",)



######

model3 = Mb0.VAE_Dirichlet_GMM_TypeB1602zC2(
    nx=adata.n_vars,
    nz=2,
    nw=2,
    nclasses=labels.shape[1],
    nc1=conditions.shape[1],
    concentration=1.0e0,
    dropout=15e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    positive_rec=True,
    #nh=2**11,
    #nhp=2**11,
    #nhq=2**11,
    numhidden=1,
    numhiddenp=1,
    numhiddenq=1,
    #learned_prior=True,
    learned_prior=False,
)
model3.apply(init_weights)
print()

Train.basicTrainLoopCond(
    model3,
    data_loader,
    data_loader,
    num_epochs=20,
    lrs = [
        1e-5,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    test_accuracy=False,
    report_interval=0,
    wt=1e-4,
)
r,p,s = ut.estimateClusterImpurity(model3, data, labels, "cuda", conditions)
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print("acc= \n", (r*s).sum().item() / s.sum().item(), r.mean().item())

output = model3(data, cond1=conditions)
adata.obsm["mu_z"] = output["mu_z"].detach().numpy()
adata.obsm["z"] = output["z"].detach().numpy()
adata.obsm["mu_w"] = output["mu_w"].detach().numpy()
adata.obsm["w"] = output["w"].detach().numpy()
adata.obs["predict"] = output["q_y"].argmax(-1).detach().numpy().astype(str)
del output
df[["zx","zy",]] = adata.obsm["z"]
df[["mu_zx","mu_zy",]] = adata.obsm["mu_z"]
df[["wx","wy",]] = adata.obsm["w"]
df[["mu_wx","mu_wy",]] = adata.obsm["mu_w"]
df["predict"] = adata.obs["predict"]
df["predict_int"] = [int(x) for x in adata.obs["predict"]]


ax = sns.relplot(
        df,
        x="mu_zx",
        y="mu_zy",
        hue="label",
        kind="scatter",
        legend="brief",
        style="cond",
        palette=sns.color_palette(),
        )
sns.move_legend(ax, "upper right",)


#####

model4 = Mb0.VAE_TypeB1601C(
    nx=adata.n_vars,
    nz=2,
    nw=2,
    nc1=conditions.shape[1],
    dropout=15e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    positive_rec=True,
    #nh=2**11,
    #nhp=2**11,
    #nhq=2**11,
    numhidden=1,
    numhiddenp=1,
    numhiddenq=1,
    learned_prior=True,
    #learned_prior=False,
)
model4.apply(init_weights)
print()

Train.basicTrainLoopCond(
    model4,
    data_loader,
    data_loader,
    num_epochs=20,
    lrs = [
        1e-5,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    test_accuracy=False,
    report_interval=0,
    wt=1e-4,
)

output = model4(data, cond1=conditions)
adata.obsm["mu_z"] = output["mu_z"].detach().numpy()
adata.obsm["z"] = output["z"].detach().numpy()
adata.obsm["mu_w"] = output["mu_w"].detach().numpy()
adata.obsm["w"] = output["w"].detach().numpy()
del output
df[["zx","zy",]] = adata.obsm["z"]
df[["mu_zx","mu_zy",]] = adata.obsm["mu_z"]
df[["wx","wy",]] = adata.obsm["w"]
df[["mu_wx","mu_wy",]] = adata.obsm["mu_w"]


ax = sns.relplot(
        df,
        x="mu_zx",
        y="mu_zy",
        hue="label",
        kind="scatter",
        legend="brief",
        style="cond",
        palette=sns.color_palette(),
        )
sns.move_legend(ax, "upper right",)


#####

model5 = Mb0.VAE_TypeB1601C(
    nx=adata.n_vars,
    nz=2,
    nw=2,
    nc1=conditions.shape[1],
    dropout=15e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    positive_rec=True,
    #nh=2**11,
    #nhp=2**11,
    #nhq=2**11,
    numhidden=1,
    numhiddenp=1,
    numhiddenq=1,
    #learned_prior=True,
    learned_prior=False,
)
model5.apply(init_weights)
print()

Train.basicTrainLoopCond(
    model5,
    data_loader,
    data_loader,
    num_epochs=20,
    lrs = [
        1e-5,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    test_accuracy=False,
    report_interval=0,
    wt=1e-4,
)

output = model5(data, cond1=conditions)
adata.obsm["mu_z"] = output["mu_z"].detach().numpy()
adata.obsm["z"] = output["z"].detach().numpy()
adata.obsm["mu_w"] = output["mu_w"].detach().numpy()
adata.obsm["w"] = output["w"].detach().numpy()
del output
df[["zx","zy",]] = adata.obsm["z"]
df[["mu_zx","mu_zy",]] = adata.obsm["mu_z"]
df[["wx","wy",]] = adata.obsm["w"]
df[["mu_wx","mu_wy",]] = adata.obsm["mu_w"]


ax = sns.relplot(
        df,
        x="mu_zx",
        y="mu_zy",
        hue="label",
        kind="scatter",
        legend="brief",
        style="cond",
        palette=sns.color_palette(),
        )
sns.move_legend(ax, "upper right",)


#### 
model6 = Mb0.VAE_Dirichlet_GMM_TypeB1602zC(
    nx=adata.n_vars,
    nz=18,
    nw=18,
    nclasses=labels.shape[1],
    nc1=conditions.shape[1],
    concentration=1.0e0,
    dropout=15e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    positive_rec=True,
    #nh=2**11,
    #nhp=2**11,
    #nhq=2**11,
    numhidden=1,
    numhiddenp=1,
    numhiddenq=1,
    learned_prior=True,
    #learned_prior=False,
)
model6.apply(init_weights)
print()

Train.basicTrainLoopCond(
    model6,
    data_loader,
    data_loader,
    num_epochs=30,
    lrs = [
        1e-5,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    test_accuracy=False,
    report_interval=0,
    wt=1e-3,
)
r,p,s = ut.estimateClusterImpurity(model2, data, labels, "cuda", conditions)
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print("acc= \n", (r*s).sum().item() / s.sum().item(), r.mean().item())

output = model6(data, cond1=conditions)
adata.obsm["mu_z"] = output["mu_z"].detach().numpy()
adata.obsm["z"] = output["z"].detach().numpy()
adata.obsm["mu_w"] = output["mu_w"].detach().numpy()
adata.obsm["w"] = output["w"].detach().numpy()
adata.obs["predict"] = output["q_y"].argmax(-1).detach().numpy().astype(str)
del output
df[["zx","zy",]] = adata.obsm["z"]
df[["mu_zx","mu_mu_zy",]] = adata.obsm["mu_z"]
df[["wx","wy",]] = adata.obsm["w"]
df[["mu_wx","wy",]] = adata.obsm["mu_w"]
df["predict"] = adata.obs["predict"]
df["predict_int"] = [int(x) for x in adata.obs["predict"]]


ax = sns.relplot(
        df,
        x="x",
        y="y",
        hue="predict",
        kind="scatter",
        legend="brief",
        style="cond",
        palette=sns.color_palette(),
        )
sns.move_legend(ax, "upper right",)





plt.close()
