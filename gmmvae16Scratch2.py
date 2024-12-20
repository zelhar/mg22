# import gdown
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import os
import pandas as pd

# import pyro
import pyro.distributions as pyrodist
import scanpy as sc
import seaborn as sns

from datetime import datetime
import time

# import time
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
import skimage as skim
from abc import abstractmethod
from anndata.experimental.pytorch import AnnLoader
from importlib import reload
from math import pi, sin, cos, sqrt, log

# from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
# from pyro.optim import Adam
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
import gmmvae11 as M11
import gmmvae12 as M12
import gmmvae13 as M13
import gmmvae14 as M14
import gmmvae15 as M15
import gmmvae16 as M16
import gmmTraining as Train

import re

print(torch.cuda.is_available())

plt.ion()
sc.settings.verbosity = 3
sc.logging.print_header()
# sc.settings.set_figure_params(dpi=120, facecolor='white', )
# sc.settings.set_figure_params(figsize=(8,8), dpi=80, facecolor='white', )
sc.settings.set_figure_params(
    dpi=80,
    facecolor="white",
)

enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc_ct = LabelEncoder()

#### PBMC
# adata = sc.read("./data/pbmc3k_processed.h5ad",)
#adata = sc.read(
#    "./data/pbmc3k_raw.h5ad",
#)

adata = sc.datasets.paul15()

adata.obs['type'] = adata.obs['paul15_clusters']

#sc.pp.filter_cells(adata, min_genes=200)
#sc.pp.filter_genes(adata, min_cells=3)

sc.pp.normalize_total(
    adata,
    target_sum=1e4,
)

sc.pp.log1p(adata,)

sc.pp.scale(adata, max_value=10)



data = torch.FloatTensor(adata.X)
#data = torch.FloatTensor(adata.X.round())
enc_ct.fit(adata.obs["paul15_clusters"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["paul15_clusters"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()

data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data, labels),
        batch_size=128,
        shuffle=True,
        )

subset = ut.randomSubset(s=len(labels), r=0.5)
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=data[subset],
            labels=labels[subset],
            ),
        batch_size=128,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=data[subset == False],
            labels=labels[subset == False],
            ),
        batch_size=128,
        shuffle=True,
        )

model = M16.VAE_Dirichlet_GMM_Type1602z(
    nx=adata.n_vars,
    #concentration=2e0,
    concentration=1e-0,
    nclasses=30,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=18,
    nw=18,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    #reclosstype="Bernoulli",
    reclosstype="mse",
    #relax=True,
    # use_resnet=True,
    restrict_w=True,
    restrict_z=True,
)
model.apply(init_weights)

model = M16.AE_Type1600N(
        nx=adata.n_vars,
        nz=16,
        dropout=0.2,
        bn=True,
        )
model.apply(init_weights)

model = M16.AE_Type1600G(
        nx=adata.n_vars,
        nz=16,
        dropout=0.2,
        bn=True,
        )
model.apply(init_weights)

model = M15.AE_Type1500(
        nx=adata.n_vars,
        nz=16,
        dropout=0.2,
        bn=True,
        reclosstype="mse",
        )
model.apply(init_weights)



Train.basicTrainLoop(
    model,
    data_loader,
    None,
    num_epochs=150,
    #lrs=[1e-5,1e-4,1e-3,1e-3,1e-4,1e-5],
    #lrs=[1e-4,1e-4,1e-4,1e-4],
    #lrs=[1e-5,1e-5,1e-5,1e-5],
    lrs=[1e-3,1e-3,1e-3,1e-4,1e-5],
    wt=1e-3,
    #report_interval=10,
    report_interval=30,
    #do_plot=True,
    #test_accuracy=True,
)

model.eval()

x,y = data_loader.__iter__().next()
output = model(x)
rec = output['rec']

ut.checkCosineDistance(x,model)

r,p,s = ut.estimateClusterImpurityLoop(model, x, y, "cuda", )

print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

##############################
P = distributions.NegativeBinomial(
    total_count=torch.rand(5) * 2,
    probs=torch.rand(5),
)
foo = torch.rand(5)
P.log_prob(foo.round())





#### Supervised training and generating test
model = M16.VAE_Dirichlet_GMM_Type1602xz(
    nx=adata.n_vars,
    #concentration=2e0,
    concentration=1e-0,
    nclasses=labels.shape[1],
    nh=2024,
    nhp=2024,
    nhq=2024,
    nz=18,
    nw=18,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.15,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    #reclosstype="Bernoulli",
    reclosstype="mse",
    #relax=True,
    # use_resnet=True,
    restrict_w=True,
    restrict_z=True,
)
model.apply(init_weights)

Train.trainSemiSuperLoop(model,
        labeled_loader,
        data_loader,
        #unlabeled_loader,
        unlabeled_loader,
        num_epochs=40,
        lrs = [1e-5,1e-4,1e-3,1e-3,1e-3,1e-4,1e-5],
        wt=1e-4,
        #do_unlabeled=True,
        do_unlabeled=False,
        do_validation=False,
        report_interval=10,
        test_accuracy=True,
        )

# generate synthetic data:
model.cpu()
model.eval()
w = torch.randn(150, model.nw).detach()
z = model.Pz(w)[:,:,:model.nz].reshape(150 * model.nclasses, -1).detach()
l = torch.eye(model.nclasses,).tile(150,1)
x = model.Px(z).detach()
xz = torch.cat([z,x], dim=1).detach()
y = model.Qy(xz).detach()
y.argmax(-1)
l.argmax(-1)

new_data = torch.cat([data, x], dim=0)
new_labels = torch.cat([labels, l], dim=0)

adata = sc.AnnData(X=new_data.numpy())
adata.obs['batch'] = data.shape[0] * ["original"] + x.shape[0] * ["generated"]
adata.obs["label"] = new_labels.argmax(-1).numpy().astype(str)

adata.obs["label"] = new_labels.argmax(-1).numpy().astype(str)

plt.cla()
plt.clf()
plt.close()

#sc.pp.pca(adata,)
sc.pp.neighbors(adata, n_neighbors=15,)
sc.tl.umap(adata, )

sc.pl.umap(adata, color=["batch", "label"])

#sc.pl.umap(adata, )

plt.savefig("./tmp.png")

plt.savefig("./results/paul15_generation.png")

ut.saveModelParameters(
        model,
        "./results/paul15_supervised" + str(datetime.timestamp(datetime.now())) + "model_params.pt",
        method="json",
        )
torch.save(
        model.state_dict(),
        "./results/paul15_supervised" + str(datetime.timestamp(datetime.now())) + "model_state.pt",
        )


adata.obsm["z"] = model(new_data)["mu_z"].detach().numpy()

sc.pp.neighbors(adata, n_neighbors=15, use_rep="z",)
sc.tl.umap(adata, )

sc.pl.umap(adata, color=["batch", "label"])

plt.savefig("./results/paul15_generation_zspace_umap.png")


d = np.arange(19)
d = zip(d, enc_ct.classes_.tolist())
#d = list(d)
d = dict(d)

celltypes = [d[int(x)] for x in adata.obs['label']]
adata.obs['celltypes'] = celltypes

sc.pl.umap(adata, color=["label", "celltypes",  ])

sc.pl.umap(adata, color=["celltypes", ], frameon=False, legend_loc='on data',
        title='Paul 15 cell types', legend_fontsize=9,)


plt.savefig("./results/paul15_generation_zspace_umap_pseuditime_names.png")

###
xdata = sc.datasets.paul15()

p = re.compile("[0-9]")

p.sub("", "12foo")

base_types = [p.sub("", s) for s in xdata.obs["paul15_clusters"]]
xdata.obs["base_type"] = base_types

sc.pp.normalize_total(
    xdata,
    target_sum=1e4,
)
sc.pp.log1p(xdata,)
sc.pp.scale(xdata, max_value=10)

data = torch.FloatTensor(xdata.X)
#data = torch.FloatTensor(adata.X.round())
#enc_ct.fit(xdata.obs["paul15_clusters"])
enc_ct.fit(xdata.obs["base_type"])

labels2 = torch.IntTensor(
        enc_ct.transform(xdata.obs["base_type"]))
labels2 = F.one_hot(labels2.long(), num_classes=enc_ct.classes_.size).float()

data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data, labels2),
        batch_size=128,
        shuffle=True,
        )
subset = ut.randomSubset(s=len(labels2), r=0.3)
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=data[subset],
            labels=labels2[subset],
            ),
        batch_size=128,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=data[subset == False],
            labels=labels2[subset == False],
            ),
        batch_size=128,
        shuffle=True,
        )

model2 = M16.VAE_Dirichlet_GMM_Type1602xz(
    nx=adata.n_vars,
    #concentration=2e0,
    concentration=1e-0,
    nclasses=labels2.shape[1]*3,
    nh=2024,
    nhp=2024,
    nhq=2024,
    nz=18,
    nw=18,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.15,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    #reclosstype="Bernoulli",
    reclosstype="mse",
    #relax=True,
    # use_resnet=True,
    restrict_w=True,
    restrict_z=True,
)
model2.apply(init_weights)

Train.trainSemiSuperLoop(
        model2,
        labeled_loader,
        unlabeled_loader,
        #data_loader,
        unlabeled_loader,
        num_epochs=40,
        lrs = [1e-5,1e-4,1e-3,1e-3,1e-3,1e-4,1e-5],
        wt=1e-4,
        do_unlabeled=True,
        #do_unlabeled=False,
        do_validation=False,
        report_interval=10,
        test_accuracy=True,
        )

Train.basicTrainLoop(
        model2,
        data_loader,
        None,
        num_epochs=50,
        lrs = [1e-5,1e-4,1e-3,1e-3,1e-4,1e-5,],
        wt=1e-4,
        report_interval=10,
        test_accuracy=True,
        )

model2.eval()
model2.cpu()
output = model2(data)

xdata.obsm['z'] = output["mu_z"].detach().numpy()
xdata.obs['predict'] = output["q_y"].detach().argmax(-1).numpy().astype(str)

sc.pp.neighbors(xdata, use_rep='z')
sc.tl.umap(xdata,)
sc.tl.louvain(xdata,)


sc.pl.umap(xdata, color=["base_type", "predict", "louvain",], frameon=False, legend_loc='on data',
        legend_fontsize=9,)

sc.pl.umap(xdata, color=["paul15_clusters", "predict", "louvain",],
        frameon=False, legend_loc='on data',
        legend_fontsize=9,)

plt.savefig("./tmp.png",)

#####

#xdata = sc.read("./data/pancreas.h5ad")
#xdata = sc.read("./data/pancreas_2.h5ad")
#xdata = sc.read("./data/pbmc3k_raw.h5ad")
#xdata = sc.read("./data/pbmc3k_processed.h5ad")
#xdata = sc.read_10x_mtx("./data/filtered_gene_bc_matrices/hg19/",  )
#xdata = sc.datasets.paul15()


