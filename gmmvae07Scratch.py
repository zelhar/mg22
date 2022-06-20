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




#### Semisuper tests
#adatac = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)
adata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)
sc.pp.highly_variable_genes(adata, n_top_genes=1000, inplace=True, subset=True,)


data = torch.FloatTensor(adata.X.toarray())
enc_ct.fit(adata.obs["Granular cell type"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["Granular cell type"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

#dataset = ut.SynteticDataSet(data, labels)
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[:5600], labels[:5600]),
        batch_size=256,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[5600:-5500], labels[5600:-5500]),
        batch_size=256,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[-5500:], labels[-5500:]),
        batch_size=256,
        shuffle=True,
        )


model_gmmdvae = M6.VAE_Dilo_Type601(nx=1000, nh=1064, nz=64,
        nclasses=74,)

model_gmmdvae = M7.VAE_Stacked_Dilo_Anndata_Type701(nx=1000, nh=1024,
        nz=64, nw=15, nclasses=74,)

model_gmmdvae = M7.VAE_Dirichlet_Type705(nx=1000, nh=1024, nz=64, nw=15,
        nclasses=74,)

model_gmmdvae.apply(init_weights)

M6.trainSemiSuper(model_gmmdvae, labeled_loader, unlabeled_loader, test_loader, 
        num_epochs=100, lr=1e-3, wt=0, do_unlabeled=True,)

model_gmmdvae.eval()
output = model_gmmdvae(data)
adata.obsm["z_gmmdvae"] = output["z"].detach().numpy()
adata.obs["predict_gmmdvae"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
sc.pp.neighbors(adata, use_rep="z_gmmdvae", n_neighbors=10,)
sc.tl.umap(adata,)

sc.pl.umap(adata, color=["predict_gmmdvae", "Granular cell type",])

sc.pl.umap(adata, color=["Cell types level 2", "Broad cell type",])


adata = sc.read("./data/gtex_v7_SMTS_PP_1k.h5ad",)
adata

data = torch.FloatTensor(adata.X)
enc_ct.fit(adata.obs["smts"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["smts"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

#dataset = ut.SynteticDataSet(data, labels)
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[:1600], labels[:1600]),
        batch_size=256,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[1600:-1500], labels[1600:-1500]),
        batch_size=256,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[-1500:], labels[-1500:]),
        batch_size=256,
        shuffle=True,
        )

model_gmmdvae = M7.VAE_Dirichlet_Type705(nx=1000, nh=1024, nz=64, nw=15,
        nclasses=enc_ct.classes_.size,)
model_gmmdvae.apply(init_weights)

M6.basicTrain(model_gmmdvae, unlabeled_loader, test_loader, num_epochs=20,
        wt=0, lr=1e-3,)

M6.trainSemiSuper(model_gmmdvae, labeled_loader, unlabeled_loader, test_loader, 
        num_epochs=100, lr=1e-3, wt=0, do_unlabeled=True,)

model_gmmdvae.eval()
output = model_gmmdvae(data)
adata.obsm["z_gmmdvae"] = output["z"].detach().numpy()
adata.obs["predict_gmmdvae"] = output["q_y"].detach().argmax(-1).numpy().astype(str)

sc.pl.umap(adata, color=["smts", "predict_gmmdvae"],)

sc.pp.neighbors(adata, use_rep="z_gmmdvae", n_neighbors=10,)
sc.tl.umap(adata,)

sc.pl.umap(adata, color=["smts", "predict_gmmdvae"],)

# so gtex v7 is sort of easy and unsup training produces nice latent and okay
# predicts. Semisup can perfectly learn the smts classes. On the other hand
# with gtex v9, much more complex scRNAseq, harder to learn. Semisup doesn't
# really work with this one.

#  trying unfilterred data set (all genes)
adata = sc.read("./data/gtex_v7_SMTS.h5ad")

sc.pp.filter_cells(adata, min_genes=200, inplace=True,)
sc.pp.filter_genes(adata, min_cells=20, inplace=True,)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True,)
sc.pp.log1p(adata, )

data = torch.FloatTensor(adata.X)
enc_ct.fit(adata.obs["smts"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["smts"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=10,)
sc.tl.umap(adata,)
sc.pl.umap(adata, color=["smts", "predict_gmmdvae"],)


#dataset = ut.SynteticDataSet(data, labels)
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[:1600], labels[:1600]),
        batch_size=256,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[1600:-1500], labels[1600:-1500]),
        batch_size=256,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[-1500:], labels[-1500:]),
        batch_size=256,
        shuffle=True,
        )

model_gmmdvae = M7.VAE_Dirichlet_Type705(nx=adata.n_vars, nh=1024, nz=64, nw=15,
        nclasses=enc_ct.classes_.size,)

model_gmmdvae = M7.VAE_Dirichlet_Type705(nx=adata.n_vars, nh=1024, nz=64, nw=15,
        nclasses=enc_ct.classes_.size*4,)

model_gmmdvae = M7.VAE_Stacked_Dilo_Anndata_Type701(nx=adata.n_vars, nh=1024,
        nz=64, nw=15, nclasses=enc_ct.classes_.size,)

model_gmmdvae.apply(init_weights)

M6.basicTrain(model_gmmdvae, unlabeled_loader, test_loader, num_epochs=20,
        wt=0, lr=1e-3,)

model_gmmdvae.eval()
output = model_gmmdvae(data)
adata.obsm["z_gmmdvae"] = output["z"].detach().numpy()
adata.obs["predict_gmmdvae"] = output["q_y"].detach().argmax(-1).numpy().astype(str)

sc.pp.neighbors(adata, use_rep="z_gmmdvae", n_neighbors=10,)
sc.tl.umap(adata,)
sc.pl.umap(adata, color=["smts", "predict_gmmdvae"],)


M6.trainSemiSuper(model_gmmdvae, labeled_loader, unlabeled_loader, test_loader, 
        num_epochs=100, lr=1e-3, wt=0, do_unlabeled=True,)

## gtexv9
adata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)


data = torch.FloatTensor(adata.X.toarray())
enc_ct.fit(adata.obs["Granular cell type"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["Granular cell type"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

#dataset = ut.SynteticDataSet(data, labels)
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[:1600], labels[:1600]),
        batch_size=256,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[1600:-1500], labels[1600:-1500]),
        batch_size=256,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[-1500:], labels[-1500:]),
        batch_size=256,
        shuffle=True,
        )

model = M7.VAE_Dirichlet_Type705(nx=adata.n_vars, nh=1024,
        nz=64, nw=15, nclasses=enc_ct.classes_.size,)
model.apply(init_weights)

M6.basicTrain(model, unlabeled_loader, test_loader, num_epochs=20,
        wt=0, lr=1e-3,)

model.eval()
output = model(data)
adata.obsm["z_gmmdvae"] = output["z"].detach().numpy()
adata.obs["predict_gmmdvae"] = output["q_y"].detach().argmax(-1).numpy().astype(str)

sc.pp.neighbors(adata, use_rep="z_gmmdvae", n_neighbors=10,)
sc.tl.umap(adata,)
sc.pl.umap(adata, color=["Granular cell type", "predict_gmmdvae"],)

M6.trainSemiSuper(model, labeled_loader, unlabeled_loader, test_loader, 
        num_epochs=100, lr=1e-3, wt=0, do_unlabeled=True,)

M6.trainSemiSuper(model, data_loader, unlabeled_loader, test_loader, 
        num_epochs=100, lr=1e-3, wt=0, do_unlabeled=True,)

## 2 moons
X,y = skds.make_moons(n_samples=20000, noise=3e-2, )
X = torch.FloatTensor(X)
y = torch.IntTensor(y)
adata = sc.AnnData(X=X.numpy(), )
adata = sc.AnnData(X=X.numpy(), )
adata.obs["y"] = y.numpy()

sc.tl.tsne(adata, )
sc.pl.tsne(adata, color="y")
sc.pp.neighbors(adata, n_neighbors=10, )
sc.tl.umap(adata, )
sc.pl.umap(adata, color="y")
sc.pl.scatter(adata, color="y", x='0',y='1')


dataset = ut.SynteticDataSet(X, y)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=256,
        )

model = M7.VAE_Dirichlet_Type705(nx=2, nh=1024, nz=32, nw=15, nclasses=2,)
model.apply(init_weights)
M6.basicTrain(model, data_loader , num_epochs=20,
        wt=0, lr=1e-3,)

model.eval()
output = model(X)
adata.obsm["z_gmmdvae"] = output["z"].detach().numpy()
adata.obs["predict_gmmdvae"] = output["q_y"].detach().argmax(-1).numpy().astype(str)

adata.obsm["rec"] = output["rec"].detach().numpy()

sc.pp.neighbors(adata, use_rep="z_gmmdvae", n_neighbors=10,)
sc.tl.umap(adata,)
sc.pl.umap(adata, color=["y", "predict_gmmdvae"],)

sc.pl.scatter(adata, color="predict_gmmdvae", x='0',y='1')

sns.scatterplot(data=pd.DataFrame(output['rec'].detach().numpy()), x=0, y=1, hue=y)

sc.tl.louvain(adata,)


## testing gmmvae08
adata = sc.read("./data/gtex_v7_SMTS.h5ad")
sc.pp.filter_cells(adata, min_genes=200, inplace=True,)
sc.pp.filter_genes(adata, min_cells=20, inplace=True,)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True,)
sc.pp.log1p(adata, )

data = torch.FloatTensor(adata.X)
enc_ct.fit(adata.obs["smts"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["smts"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.tsne(adata, )
sc.tl.leiden(adata, )
sc.tl.louvain(adata, )
sc.pl.umap(adata, color=["smts", "leiden", "louvain"],)
sc.pl.tsne(adata, color=["smts", "leiden", "louvain"],)



labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[:1600], labels[:1600]),
        batch_size=256,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[1600:-1500], labels[1600:-1500]),
        batch_size=256,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[-1500:], labels[-1500:]),
        batch_size=256,
        shuffle=True,
        )

model = M7.VAE_Dirichlet_Type705(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        nclasses=enc_ct.classes_.size,
        )

model = M7.VAE_Stacked_Dilo_Anndata_Type701(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        nclasses=enc_ct.classes_.size + 5,
        )


model = M8.VAE_Dirichlet_Type805(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        dirscale=1e0,
        concentration=1e-1,
        nclasses=enc_ct.classes_.size,
        )

model = M8.VAE_Dirichlet_Type805(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        #dirscale=1e2,
        dirscale=1e0,
        #concentration=1e-5,
        concentration=5e0,
        nclasses=enc_ct.classes_.size,
        )

model = M8.VAE_Dirichlet_Type806(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        dirscale=1e0,
        concentration=5e0,
        nclasses=enc_ct.classes_.size + 7,
        )

model.apply(init_weights)

M8.basicTrain(model, data_loader, test_loader,
        num_epochs=22, wt=0, lr=1e-3, report_interval=1)

M8.trainSemiSuper(model, labeled_loader, unlabeled_loader, test_loader, 
        num_epochs=63, lr=1e-3, wt=0, do_unlabeled=True, report_interval=5,)

output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()
adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.louvain(adata, )
sc.tl.leiden(adata, )
sc.pl.umap(adata, color=["smts", "predict", "predict2"],)

sc.pl.umap(adata, color=["smts", "leiden"],)
sc.pl.umap(adata, color=["smts", "louvain"],)

plt.close()

(output["q_y"].argmax(-1) - labels.argmax(-1)).float().mean()
