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
    dpi=100,
    facecolor="white",
)
sns.set_palette(sns.color_palette("pastel"),)
sns.set(rc={"figure.dpi":200, 'savefig.dpi':100})


## testing cGMVAE on toy blobs
n = 500 # number of observation per class
c = 2 # number of conditions
b = 7 #number of classes
d = 2 #dimensions
meansControl = (1 + torch.rand(b, d))*1e1
stdControl = torch.randn_like(meansControl).abs() * 1e-1
shiftStim = 5 + 1e0*torch.rand(b, d) + torch.randn(b,d)*2e0
meansStim = torch.randn_like(meansControl)*1e-4
meansStim = meansStim + meansControl + shiftStim
stdStim = 2e-1 * torch.randn_like(stdControl).abs() + stdControl
means = torch.concat(
        [
            meansControl,
            meansStim,
            ], dim=0,
        )
std = torch.concat(
        [
            stdControl,
            stdStim,
            ],
        )


gmm =    distributions.Normal(
        means,
        std,
    )
X = gmm.sample((n,)).reshape(-1,d).numpy() #~=meansControl

labels = list(toolz.concat(c*n * [np.arange(b)]) )
#conditions = b*n*["control"] + b*n*["stimulated"]
#conditions = toolz.concat(n*["control", "stimulated"])
conditions = n* (b*["control"] + b*["stimulated"])

df = pd.DataFrame()
df["label"] = labels
df["condition"] = conditions

df[["x","y"]] = X

plt.cla()
plt.clf()
plt.close()
ax = sns.relplot(
        df,
        x="x",
        y="y",
        hue="label",
        kind="scatter",
        legend="brief",
        style="condition",
        palette=sns.color_palette(),
        )
sns.move_legend(ax, "upper right",)
#ax.figure.savefig("foooo.png")


enc_labels = LabelEncoder()
enc_conds = LabelEncoder()
data = torch.tensor(X)
labelsd = enc_labels.fit_transform(labels)
labelsd = F.one_hot(torch.tensor(labelsd)).float()
conditionsd = enc_conds.fit_transform(conditions)
conditionsd = F.one_hot(torch.tensor(conditionsd)).float()

data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSetV2(
            [data, labelsd, conditionsd],
            ),
            batch_size=2**10,
            shuffle=True,
        )


model = Mb0.VAE_Dirichlet_GMM_TypeB1602zC2(
    nx=d,
    nz=2,
    nw=2,
    nclasses=b*3,
    nc1=c,
    concentration=1.0e0,
    dropout=1e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    positive_rec=True,
    #nh=2**11,
    #nhp=2**11,
    #nhq=2**11,
    numhidden=4,
    numhiddenp=4,
    numhiddenq=4,
    #learned_prior=True,
)
model.apply(init_weights)
print()

Train.basicTrainLoopCond(
    model,
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
        1e-4,
        1e-5,
    ],
    test_accuracy=False,
    report_interval=0,
    wt=1e-4,
)
model.train()
model.cpu()

r,p,s = ut.estimateClusterImpurityLoop(model, data, labelsd, device="cuda",
        cond1=conditionsd ) #broken need to fix this for conditional
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print("kt_acc= \n", (r*s).sum().item() / s.sum().item(), r.mean().item())

output = model(data, cond1=conditionsd)
df["predict"] =  output["q_y"].argmax(-1).detach().numpy().astype(str)
df[["zx","zy"]] = output["mu_z"].detach().numpy()
df[["wx","wy"]] = output["w"].detach().numpy()
del output
ax = sns.relplot(
        df,
        x="wx",
        y="wy",
        #hue="predict",
        hue="label",
        kind="scatter",
        legend="brief",
        style="condition",
        palette="pastell",
        )
sns.move_legend(ax, "upper right",)
ax.figure.savefig("foooo.png")
