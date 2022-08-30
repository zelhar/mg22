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

#adata = sc.read_h5ad("./data/scgen/scGen_datasets/bbknn.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/cca.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/mnn.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/MouseAtlas.subset.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/pancreas.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/scanorama.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_salmonella.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_salmonella.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_hpoly.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_hpoly.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_species.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_species.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_study.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_study.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_zheng.h5ad")
#adata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_pbmc.h5ad")



#### PBMC Kang et. al
# This data contains immune cells of different types in two groups: control and
# simulated.

adata = sc.read_h5ad("./data/scgen/scGen_datasets/train_study.h5ad")
bdata = sc.read_h5ad("./data/scgen/scGen_datasets/valid_study.h5ad")

adata = ad.concat([adata, bdata], label="dataset",)

#sc.pp.filter_cells(adata, min_genes=200)
#sc.pp.filter_genes(adata, min_cells=5)
#sc.pp.normalize_total(adata, target_sum=1e4)
#sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True,)

#sc.pp.scale(adata, max_value=10)

enc_labels = LabelEncoder()
labels = enc_labels.fit_transform( adata.obs["cell_type"],)
labels = F.one_hot(torch.tensor(labels)).float()
enc_conds = LabelEncoder()
conditions = enc_conds.fit_transform(adata.obs["condition"],)
conditions = F.one_hot(torch.tensor(conditions)).float()
data = torch.tensor(adata.X.toarray())


test_enc_labels = LabelEncoder()
test_labels = test_enc_labels.fit_transform( bdata.obs["cell_type"],)
test_labels = F.one_hot(torch.tensor(test_labels)).float()
test_enc_conds = LabelEncoder()
test_conditions = test_enc_conds.fit_transform(bdata.obs["condition"],)
test_conditions = F.one_hot(torch.tensor(test_conditions)).float()
test_data = torch.tensor(bdata.X.toarray())

labeledSubset = ut.randomSubset(s=len(adata), r=0.15)


labeled_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [data[labeledSubset],
                labels[labeledSubset],
                conditions[labeledSubset],
                ],),
            batch_size=128,
            shuffle=True,
            )

unlabeled_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [data[~labeledSubset],
                labels[~labeledSubset],
                conditions[~labeledSubset],
                ],),
            batch_size=128,
            shuffle=True,
            )

test_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [test_data,
                test_labels,
                test_conditions,
                ],),
            batch_size=128,
            shuffle=True,
            )


train_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [data,
                labels,
                conditions,
                ],),
            batch_size=128,
            shuffle=True,
            )

x,y,c = train_loader.__iter__().next()

model = Mb0.VAE_Dirichlet_GMM_TypeB1602xzC(
        nx=adata.n_vars,
        nz=10,
        nw=10,
        nclasses=labels.shape[1],
        concentration=1e0,
        dropout=0.1,
        bn=True,
        reclosstype="mse",
        restrict_w=True,
        restrict_z=True,
        nc1=conditions.shape[1],
        learned_prior=False,
        )
model.apply(init_weights)



Train.trainSemiSuperLoopCond(
        model,
        labeled_loader,
        unlabeled_loader,
        test_loader,
        num_epochs=100,
        lrs = [1e-5,1e-4,1e-3,1e-3,1e-3,1e-4,1e-5,],
        wt=1e-4,
        do_unlabeled=True,
        do_validation=True,
        report_interval=10,
        test_accuracy=True,
        )


model.cpu()
model.eval()

r,p,s = ut.estimateClusterImpurityLoop(model, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

output = model(test_data)
bdata.obsm["z"] = output["mu_z"].detach().numpy()
bdata.obs["predict"] = test_enc_labels.inverse_transform(
    output["q_y"].detach().argmax(-1).numpy()
)

sc.pp.neighbors(bdata,use_rep="z")
sc.tl.umap(bdata, )
sc.tl.louvain(bdata,)

sc.pl.umap(bdata, color=["condition", "cell_type", "louvain", "predict"], 
        frameon=False,
        legend_loc='on data',
        legend_fontsize=9,
        ncols=2,
        color_map="viridis",
        )

plt.savefig("./tmp.png",)

plt.savefig("./results/Kang_ss_batch_reduction.",)

ut.saveModelParameters(
        model,
        "./results/Kang_ss_batch_reduction" + str(datetime.timestamp(datetime.now())) + "model_params.pt",
        method="json",
        )
torch.save(
        model.state_dict(),
        "./results/Kang_ss_batch_reduction" + str(datetime.timestamp(datetime.now())) + "model_state.pt",
        )


model = Mb0.VAE_Dirichlet_GMM_TypeB1602xzC(
        nx=adata.n_vars,
        nz=10,
        nw=10,
        nclasses=labels.shape[1],
        concentration=1e0,
        dropout=0.1,
        bn=True,
        reclosstype="mse",
        restrict_w=True,
        restrict_z=True,
        nc1=conditions.shape[1],
        learned_prior=True,
        )
model.apply(init_weights)



Train.trainSemiSuperLoopCond(
        model,
        labeled_loader,
        unlabeled_loader,
        test_loader,
        num_epochs=100,
        lrs = [1e-5,1e-4,1e-3,1e-3,1e-3,1e-4,1e-5,],
        wt=1e-4,
        do_unlabeled=True,
        do_validation=True,
        report_interval=10,
        test_accuracy=True,
        )


model.cpu()
model.eval()

# not quite correct b/c it doesn't uses the conditional info
r,p,s = ut.estimateClusterImpurityLoop(model, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())



output = model(test_data, cond1=test_conditions)
bdata.obsm["z"] = output["mu_z"].detach().numpy()
bdata.obsm["w"] = output["mu_w"].detach().numpy()
bdata.obs["predict"] = test_enc_labels.inverse_transform(
    output["q_y"].detach().argmax(-1).numpy()
)
bdata.obs["real_labels"] = test_enc_labels.inverse_transform(
        test_labels.argmax(-1))

print(
    torch.sum(output["q_y"].detach().argmax(-1) == test_labels.argmax(-1)) / len(bdata)
)

sc.pp.neighbors(bdata,use_rep="z")

sc.pp.neighbors(bdata,use_rep="w")
sc.tl.umap(bdata, )
sc.tl.louvain(bdata,)

sc.pl.umap(bdata, color=["condition", "real_labels", "louvain", "predict"], 
        frameon=False,
        legend_loc='on data',
        legend_fontsize=7,
        ncols=2,
        #color_map="magma",
        )

plt.savefig("./tmp.png",)


plt.savefig("./results/Kang_ss_conditiooned_prior.",)

ut.saveModelParameters(
        model,
        "./results/Kang_ss_conditiooned_prior" + str(datetime.timestamp(datetime.now())) + "model_params.pt",
        method="json",
        )

torch.save(
        model.state_dict(),
        "./results/Kang_ss_conditiooned_prior" + str(datetime.timestamp(datetime.now())) + "model_state.pt",
        )


bdata.obs["cond_label"] = (
        [x+"_"+y for (x,y) in zip( 
            list(bdata.obs["condition"]),
            list(bdata.obs["real_labels"]))
            ])


sc.pl.umap(bdata, color=[
    "cond_label", "real_labels", "louvain", "predict"], 
        frameon=False,
        legend_loc='on data',
        legend_fontsize=7,
        ncols=2,
        color_map="magma",
        )


output = model(test_data, cond1=1-test_conditions)
bdata.obsm["z_inv"] = output["mu_z"].detach().numpy()
bdata.obs["inverse_predict"] = test_enc_labels.inverse_transform(
    output["q_y"].detach().argmax(-1).numpy()
)

sc.pp.neighbors(bdata,use_rep="z_inv")
sc.tl.umap(bdata, )
sc.tl.louvain(bdata,)


sc.pl.umap(bdata, color=[
    "condition",
    #"cond_label",
    "real_labels",
    "inverse_predict",
    "predict",], 
        frameon=False,
        legend_loc='on data',
        legend_fontsize=7,
        ncols=2,
        color_map="magma",
        )






o1 = model(x, cond1=c)
o2 = model(x, cond1=1-c)

(o1["q_y"] - o2["q_y"]).abs().mean()

(o1["mu_z"] - o2["mu_z"]).abs().mean()
(o1["mu_w"] - o2["mu_w"]).abs().mean()

(o1["rec"] - o2["rec"]).abs().mean()

o1 = model(test_data, cond1=test_conditions)
o2 = model(test_data, cond1=1-test_conditions)



stats.linregress(x.mean(0), o1["rec"].detach().mean(0))

stats.linregress(x.mean(0), o2["rec"].detach().mean(0))



w = torch.randn(27, model.nw)
c = torch.zeros(27,2)
c[:,0] = 1
wc = torch.cat([w,c], dim=1)
wc.shape

wc2 = torch.cat([w,1-c], dim=1)

z = model.Pz(wc)[:,1,:model.nz] #cdt4 mono
z2 = model.Pz(wc2)[:,1,:model.nz]

zc = torch.cat([z,c], dim=1)
zc2 = torch.cat([z,1-c], dim=1)

rec = model.Px(zc)
rec2 = model.Px(zc2)

(rec - rec2).abs().max()

stats.linregress(rec.detach().mean(0), rec2.detach().mean(0))





model = Mb0.VAE_Dirichlet_GMM_TypeB1602xzCv2(
        nx=adata.n_vars,
        nz=10,
        nw=10,
        nclasses=labels.shape[1],
        concentration=1e0,
        dropout=0.1,
        bn=True,
        reclosstype="mse",
        restrict_w=True,
        restrict_z=True,
        nc1=conditions.shape[1],
        #learned_prior=False,
        learned_prior=True,
        )
model.apply(init_weights)



Train.trainSemiSuperLoopCond(
        model,
        labeled_loader,
        unlabeled_loader,
        test_loader,
        num_epochs=40,
        lrs = [1e-5,1e-4,1e-3,1e-3,1e-3,1e-4,1e-5,],
        wt=1e-4,
        do_unlabeled=True,
        do_validation=True,
        report_interval=15,
        test_accuracy=True,
        )


model.cpu()
model.eval()

# not quite correct b/c it doesn't uses the conditional info
r,p,s = ut.estimateClusterImpurityLoop(model, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

print(
    torch.sum(output["q_y"].detach().argmax(-1) == test_labels.argmax(-1)) / len(bdata)
)



output = model(test_data, cond1=test_conditions)
bdata.obsm["w"] = output["mu_w"].detach().numpy()
bdata.obsm["w"] = output["w"].detach().numpy()
bdata.obsm["z"] = output["mu_z"].detach().numpy()
bdata.obs["predict"] = test_enc_labels.inverse_transform(
    output["q_y"].detach().argmax(-1).numpy()
)
bdata.obs["real_labels"] = test_enc_labels.inverse_transform(
        test_labels.argmax(-1))

sc.pp.neighbors(bdata,use_rep="z")

sc.pp.neighbors(bdata,use_rep="w")

sc.tl.umap(bdata, )
sc.tl.louvain(bdata,)

sc.pl.umap(bdata, color=["condition", "real_labels", "louvain", "predict"], 
        frameon=False,
        legend_loc='on data',
        legend_fontsize=7,
        ncols=2,
        #color_map="magma",
        )

plt.savefig("./tmp.png",)




cd14_stimulated_marker = (
    (bdata.obs["cell_type"] == "CD14+ Monocytes") & (bdata.obs["condition"] == "stimulated")
)

cd14_control_markser = (
    (bdata.obs["cell_type"] == "CD14+ Monocytes") & (bdata.obs["condition"] == "control")
)

xdata = bdata[bdata.obs["cell_type"] == "CD14+ Monocytes"].copy()

sc.tl.rank_genes_groups(xdata, groupby="condition",method="wilcoxon",)

sc.pl.rank_genes_groups(xdata, n_genes=4,)

sc.pl.rank_genes_groups_dotplot(xdata, n_genes=4, cmap="viridis",)


plt.cla()
plt.clf()
plt.close()

fig, ax = plt.subplots(1,1)

sc.pl.rank_genes_groups_dotplot(xdata, n_genes=4, cmap="viridis", ax=ax)

plt.tight_layout()

plt.savefig("./tmp.png", layout="tight",)

fig.clf()

plt.cla()
plt.clf()
plt.close()

plt.savefig("./reg_mean.pdf", )


gene_list = [x for (x,y) in xdata.uns["rank_genes_groups"]["names"][:4] ]
gene_list += [y for (x,y) in xdata.uns["rank_genes_groups"]["names"][:4] ]

ut.reg_mean_plot(xdata, gene_list=gene_list )

# generate data (only CD14 + Monocytes)

c = torch.zeros(125,2)
# mu_c = model.Pw(c)[0,:model.nw]
w = torch.randn(125, model.nw)
c[:,0] = 1
wc = torch.cat([w,c], dim=1)
wc.shape
wc2 = torch.cat([w,1-c], dim=1)

z = model.Pz(wc)[:,1,:model.nz] #cdt4 mono
z2 = model.Pz(wc2)[:,1,:model.nz]

zc = torch.cat([z,c], dim=1)
zc2 = torch.cat([z,1-c], dim=1)

rec = model.Px(zc)
rec2 = model.Px(zc2)


xdata.X = xdata.X.toarray()

ydata = sc.AnnData(
        X = torch.cat([rec,rec2], dim=0).detach().numpy(),)

ydata.obs["condition"] = 125*["control"] +125*["stimulated"]
ydata.obs["real_labels"] = 250*["CD14 + Monocytes"]

ydata.var_names = xdata.var_names

merged_data = ad.concat([xdata, ydata], join="inner",)
merged_data

merged_data.obs["simcondition"] = list(xdata.obs["condition"]) + 125*["SIMcontrol"] +125*["SIMstimulated"]




fig, ax = plt.subplots(1,1)

merged_data.layers["non_negativeX"] = merged_data.X - merged_data.X.min() + 1e-3

sc.tl.rank_genes_groups(merged_data, groupby="condition",method="wilcoxon", layer="non_negativeX",)

sc.pl.rank_genes_groups_dotplot(merged_data, n_genes=4, cmap="viridis", ax=ax)

sc.tl.rank_genes_groups(merged_data, groupby="simcondition",method="wilcoxon", layer="non_negativeX",)

sc.pl.rank_genes_groups_dotplot(merged_data, n_genes=2, cmap="viridis", ax=ax,
        dendrogram=False,)

sc.pl.rank_genes_groups(merged_data, n_genes=2, cmap="viridis", ax=ax,)

plt.tight_layout()

plt.savefig("./tmp.png", )
#plt.savefig("./tmp.png", layout="tight",)

fig.clf()
plt.cla()
plt.clf()
plt.close()


###########################################################################################


cd14_stimulated_marker = (
    (bdata.obs["cell_type"] == "CD14+ Monocytes") & (bdata.obs["condition"] == "stimulated")
)

cd14_control_markser = (
    (bdata.obs["cell_type"] == "CD14+ Monocytes") & (bdata.obs["condition"] == "control")
)

x1 = test_data[cd14_stimulated_marker]
x2 = test_data[cd14_control_markser]

stats.linregress(x1.mean(0), x2.mean(0))

o1 = model(x1, cond1=test_conditions[cd14_stimulated_marker])
o1b = model(x1, cond1=1-test_conditions[cd14_stimulated_marker])

stats.linregress(x1.mean(0), o1["rec"].detach().mean(0))
stats.linregress(x1.mean(0), o1b["rec"].detach().mean(0))
stats.linregress(x2.mean(0), o1b["rec"].detach().mean(0))




















###########################################################################################

nk_stimulated_marker = (
    (adata.obs["cell_type"] == "NK cells") & (adata.obs["condition"] == "stimulated")
)

train_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [data[~nk_stimulated_marker],
                labels[~nk_stimulated_marker],
                conditions[~nk_stimulated_marker],
                ],),
            batch_size=128,
            shuffle=True,
            )

x,y,c = train_loader.__iter__().next()

model = Mb0.VAE_Dirichlet_GMM_TypeB1602xzC(
        nx=adata.n_vars,
        nz=10,
        nw=10,
        nclasses=labels.shape[1],
        concentration=1e0,
        dropout=0.1,
        bn=True,
        reclosstype="mse",
        restrict_w=True,
        restrict_z=True,
        nc1=conditions.shape[1],
        learned_prior=False,
        )
model.apply(init_weights)

Train.trainSemiSuperLoopCond(
        model,
        train_loader,
        train_loader,
        train_loader,
        num_epochs=50,
        lrs = [1e-5,1e-4,1e-3,1e-3,1e-3,1e-4,1e-5,],
        wt=1e-4,
        do_unlabeled=False,
        do_validation=False,
        report_interval=10,
        test_accuracy=True,
        )


model.cpu()
model.eval()
output = model(data[nk_stimulated_marker])

output["q_y"].argmax(-1)

#accuracy
torch.sum(output["q_y"].argmax(-1) == 6)/nk_stimulated_marker.sum()
output["q_y"][:,6].mean()



output = model(data[~nk_stimulated_marker])
xdata = adata[~nk_stimulated_marker].copy()

xdata.obsm["z"] = output["z"].detach().numpy()
xdata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy()

sc.pp.neighbors(xdata, use_rep="z")
sc.tl.umap(xdata,)

plt.cla()
plt.clf()
plt.close()

sc.pl.umap(xdata, color=["condition", "cell_type",], 
        frameon=False,
        legend_loc='on data',
        legend_fontsize=9,)

plt.savefig("./tmp.png",)

