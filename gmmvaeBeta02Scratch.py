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

## PBMC exmperiments with Kang and Zheng DB.

### Only with the control data.
# we'll check if learning supervised zheng helps learning unsup Kang.
# compare also to umap kang.

adataz = sc.read_h5ad("./data/scgen/scGen_datasets/train_zheng.h5ad",)
adatakt = sc.read_h5ad("data/scgen/scGen_datasets/train_study.h5ad",)
adatakv = sc.read_h5ad("data/scgen/scGen_datasets/valid_study.h5ad",)
adataz.X = adataz.X.toarray()
adatakt.X = adatakt.X.toarray()
adatakv.X = adatakv.X.toarray()
adatakt = adatakt[adatakt.obs["condition"]=="control"].copy()
adatakv = adatakv[adatakv.obs["condition"]=="control"].copy()

countby(identity, adataz.obs["cell_type"])
countby(identity, adatakt.obs["cell_type"])
countby(identity, adatakv.obs["cell_type"])

#adataz = ut.balanceAnnData(adataz, "cell_type", noreps=True, )
adatakt = ut.balanceAnnData(adatakt, "cell_type", noreps=True,)
#adatakv = ut.balanceAnnData(adatakv, "cell_type", noreps=True,)

adatazb = ut.balanceAnnData(
    adataz,
    "cell_type",
    eps=1e-6,
    numSamples=2000,
    augment_mode=True,
    add_noise=True,
)

adataz.X.mean(0)
adatazb.X.mean(0)

adataz.X.std(0)
adatazb.X.std(0)

#sc.pp.scale(adataz,)
#sc.pp.scale(adatakt,)
#sc.pp.scale(adatakv,)

#adataz = ut.balanceAnnData(adataz, "cell_type", numSamples=200, eps=1e-6, add_noise=True,)
#adatakt = ut.balanceAnnData(adatakt, "cell_type", numSamples=200, eps=1e-6, add_noise=True,)
#adatakv = ut.balanceAnnData(adatakv, "cell_type", numSamples=200, eps=1e-6, add_noise=True,)
#countby(identity, adataz.obs["cell_type"])
#countby(identity, adatakt.obs["cell_type"])
#countby(identity, adatakv.obs["cell_type"])
#sc.pp.scale(adataz,)
#sc.pp.scale(adatakt,)
#sc.pp.scale(adatakv,)

#adatakv.X.mean(0)
#adatakv.X.std(0)
#adatakv.X.var(0)

sc.pp.pca(adataz,)
sc.pp.pca(adatakt,)
sc.pp.pca(adatakv,)
sc.pp.neighbors(adataz,)
sc.pp.neighbors(adatakt,)
sc.pp.neighbors(adatakv,)
sc.tl.umap(adataz,)
sc.tl.umap(adatakt,)
sc.tl.umap(adatakv,)
sc.tl.louvain(adataz,)
sc.tl.louvain(adatakt,)
sc.tl.louvain(adatakv,)


sc.pl.umap(
        adataz,
        color = [
            "cell_type",
            "louvain",
            ],
            ncols=2,
            show=False,
            save="_temp.png",
            )

sc.pl.umap(
        adatakt,
        color = [
            "cell_type",
            "louvain",
            ],
            ncols=2,
            show=False,
            save="_temp.png",
            )

sc.pl.umap(
        adatakv,
        color = [
            "cell_type",
            "louvain",
            ],
            ncols=2,
            show=False,
            save="_temp.png",
            )

enc_labels = LabelEncoder()
labelskt = enc_labels.fit_transform(adatakt.obs["cell_type"],)
labelskt = F.one_hot(torch.tensor(labelskt)).float()
enc_conds = LabelEncoder()
conditionskt = enc_conds.fit_transform(adatakt.obs["condition"],)
conditionskt = F.one_hot(torch.tensor(conditionskt)).float()
datakt = torch.tensor(adatakt.X)
datakt.shape
adatakt.obs["label"] = enc_labels.inverse_transform(
    labelskt.argmax(-1).detach().numpy(),
)
adatakt.obs["label_num"] = labelskt.argmax(-1).detach().numpy().astype('str')
louvainkt = [int(x) for x in adatakt.obs["louvain"]]
louvainkt = F.one_hot(torch.tensor(louvainkt)).float()


labelskv = enc_labels.fit_transform(adatakv.obs["cell_type"],)
labelskv = F.one_hot(torch.tensor(labelskv)).float()
conditionskv = enc_conds.fit_transform(adatakv.obs["condition"],)
conditionskv = F.one_hot(torch.tensor(conditionskv)).float()
datakv = torch.tensor(adatakv.X)
datakv.shape
adatakv.obs["label"] = enc_labels.inverse_transform(
    labelskv.argmax(-1).detach().numpy(),
)
adatakv.obs["label_num"] = labelskv.argmax(-1).detach().numpy().astype('str')
louvainkv = [int(x) for x in adatakv.obs["louvain"]]
louvainkv = F.one_hot(torch.tensor(louvainkv)).float()



data_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                datakt,
                labelskt,
                #conditions,
                ],),
            batch_size=2**11,
            shuffle=True,
            )

model = Mb0.VAE_Dirichlet_GMM_TypeB1602z(
    nx=adatakt.n_vars,
    #nz=12,
    #nw=12,
    nz=18,
    nw=18,
    nclasses=labelskt.shape[1]*3,
    concentration=1e0,
    dropout=15e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    #positive_rec=True,
    numhidden=3,
    numhiddenp=3,
    numhiddenq=3,
)
model.apply(init_weights)
print(model.__class__)

Train.basicTrainLoop(
    model,
    data_loader,
    data_loader,
    num_epochs=100,
    lrs = [
        #1e-5,
        #1e-4,
        #1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    report_interval=0,
    wt=0,
)
r,p,s = ut.estimateClusterImpurityLoop(model, datakt, labelskt, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print((r*s).sum().item() / s.sum().item(), r.mean().item())


r,p,s = ut.estimateClusterAccuracy(louvainkt, labelskt)
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print((r*s).sum().item() / s.sum().item(), r.mean().item())

r,p,s = ut.estimateClusterImpurityLoop(model, datakv, labelskv, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print((r*s).sum().item() / s.sum().item(), r.mean().item())



output = model(datakv)
adatakv.obs["predict"] =  output["q_y"].argmax(-1).detach().numpy().astype(str)
adatakv.obsm["mu_z"] = output["mu_z"].detach().numpy()
adatakv.obsm["z"] = output["z"].detach().numpy()
adatakv.obsm["mu_w"] = output["mu_w"].detach().numpy()
adatakv.obsm["w"] = output["w"].detach().numpy()
rec = output["rec"].detach()
del output


sc.pp.pca(adatakv,)
sc.pp.neighbors(adatakv,)
#sc.pp.neighbors(adatakv, use_rep="mu_z",)
sc.tl.umap(adatakv,)
sc.tl.louvain(adatakv,)


sc.pl.umap(
        adatakv,
        color = [
            "louvain",
            "predict",
            "label",
            ],
            ncols=2,
            show=False,
            save="_temp.png",
            )


r,p,s = ut.estimateClusterAccuracy(louvainkv, labelskv)
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print((r*s).sum().item() / s.sum().item(), r.mean().item())


ut.saveModelParameters(
        model,
        "./results/temp_gmmvaez_kang_control_us" + ut.timeStamp() + ut.randomString() + "params.pt",
        method="json",
        )
torch.save(
        model.state_dict(),
        "./results/temp_gmmvaez_kang_control_us" + ut.timeStamp() + ut.randomString() + "state.pt",
        )

## supervised train on Zheng

adataz = sc.read_h5ad("./data/scgen/scGen_datasets/train_zheng.h5ad",)
adatakt = sc.read_h5ad("data/scgen/scGen_datasets/train_study.h5ad",)
adatakv = sc.read_h5ad("data/scgen/scGen_datasets/valid_study.h5ad",)
adataz.X = adataz.X.toarray()
adatakt.X = adatakt.X.toarray()
adatakv.X = adatakv.X.toarray()
adatakt = adatakt[adatakt.obs["condition"]=="control"].copy()
adatakv = adatakv[adatakv.obs["condition"]=="control"].copy()

sc.pp.scale(adataz,)
sc.pp.scale(adatakt,)
sc.pp.scale(adatakv,)

#adataz = ut.balanceAnnData(adataz, "cell_type", noreps=True, )
#adatakt = ut.balanceAnnData(adatakt, "cell_type", noreps=True,)
#adatakv = ut.balanceAnnData(adatakv, "cell_type", noreps=True,)
adataz = ut.balanceAnnData(adataz,
        "cell_type",
        numSamples=2000,
        #add_noise=True,
        eps=1e-6,
        )

#sc.pp.scale(adataz,)
#sc.pp.scale(adatakt,)
#sc.pp.scale(adatakv,)


countby(identity, adataz.obs["cell_type"])
countby(identity, adatakt.obs["cell_type"])
countby(identity, adatakv.obs["cell_type"])

sc.pp.pca(adataz,)
sc.pp.pca(adatakt,)
sc.pp.pca(adatakv,)
sc.pp.neighbors(adataz,)
sc.pp.neighbors(adatakt,)
sc.pp.neighbors(adatakv,)
sc.tl.umap(adataz,)
sc.tl.umap(adatakt,)
sc.tl.umap(adatakv,)
sc.tl.louvain(adataz,)
sc.tl.louvain(adatakt,)
sc.tl.louvain(adatakv,)


enc_labels = LabelEncoder()
labelskt = enc_labels.fit_transform(adatakt.obs["cell_type"],)
labelskt = F.one_hot(torch.tensor(labelskt)).float()
datakt = torch.tensor(adatakt.X)
datakt.shape
adatakt.obs["label"] = enc_labels.inverse_transform(
    labelskt.argmax(-1).detach().numpy(),
)
adatakt.obs["label_num"] = labelskt.argmax(-1).detach().numpy().astype('str')
louvainkt = [int(x) for x in adatakt.obs["louvain"]]
louvainkt = F.one_hot(torch.tensor(louvainkt)).float()


labelskv = enc_labels.fit_transform(adatakv.obs["cell_type"],)
labelskv = F.one_hot(torch.tensor(labelskv)).float()
datakv = torch.tensor(adatakv.X)
datakv.shape
adatakv.obs["label"] = enc_labels.inverse_transform(
    labelskv.argmax(-1).detach().numpy(),
)
adatakv.obs["label_num"] = labelskv.argmax(-1).detach().numpy().astype('str')
louvainkv = [int(x) for x in adatakv.obs["louvain"]]
louvainkv = F.one_hot(torch.tensor(louvainkv)).float()

labelsz = enc_labels.fit_transform(adataz.obs["cell_type"],)
labelsz = F.one_hot(torch.tensor(labelsz)).float()
dataz = torch.tensor(adataz.X)
dataz.shape
adataz.obs["label"] = enc_labels.inverse_transform(
    labelsz.argmax(-1).detach().numpy(),
)
adataz.obs["label_num"] = labelsz.argmax(-1).detach().numpy().astype('str')
louvainz = [int(x) for x in adataz.obs["louvain"]]
louvainz = F.one_hot(torch.tensor(louvainz)).float()



data_loader = torch.utils.data.DataLoader(
        dataset = ut.SynteticDataSetV2(
            [
                dataz,
                labelsz,
                #conditions,
                ],),
            batch_size=2**11,
            shuffle=True,
            )


model = Mb0.VAE_Dirichlet_GMM_TypeB1602z(
    nx=adatakt.n_vars,
    nz=12,
    nw=12,
    #nz=2,
    #nw=8,
    #nclasses=labelsz.shape[1]*2,
    nclasses=labelsz.shape[1],
    concentration=1e0,
    dropout=15e-2,
    bn=True,
    reclosstype="mse",
    #reclosstype="Gauss",
    restrict_w=True,
    restrict_z=True,
    #positive_rec=True,
    numhidden=3,
    numhiddenp=3,
    numhiddenq=3,
)
model.apply(init_weights)
print(model.__class__)



Train.basicTrainLoop(
    model,
    data_loader,
    data_loader,
    num_epochs=10,
    lrs = [
        1e-5,
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
r,p,s = ut.estimateClusterImpurityLoop(model, dataz, labelsz, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print((r*s).sum().item() / s.sum().item(), r.mean().item())

Train.trainSemiSuperLoop(
        model,
        data_loader,
        data_loader,
        data_loader,
    num_epochs=50,
    lrs = [
        1e-5,
        1e-4,
        1e-3,
        1e-3,
        1e-3,
        1e-4,
        1e-5,
    ],
    report_interval=0,
    do_unlabeled=True,
    #do_unlabeled=False,
    do_validation=False,
    wt=0,
)
r,p,s = ut.estimateClusterImpurityLoop(model, dataz, labelsz, "cuda", )
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print((r*s).sum().item() / s.sum().item(), r.mean().item())


output = model(dataz)
adataz.obs["predict"] =  output["q_y"].argmax(-1).detach().numpy().astype(str)
adataz.obsm["mu_z"] = output["mu_z"].detach().numpy()
adataz.obsm["z"] = output["z"].detach().numpy()
adataz.obsm["mu_w"] = output["mu_w"].detach().numpy()
adataz.obsm["w"] = output["w"].detach().numpy()
rec = output["rec"].detach()
del output

adataz.obs["x"] = np.log(np.log(7 + adataz.obsm["mu_z"][:,0]))
adataz.obs["y"] = np.log(np.log(7 + adataz.obsm["mu_z"][:,1]))


sc.pl.scatter(
        adataz,
        x="x",
        y="y",
        color="label",
        #color="predict",
        show=False,
        save="_tmp.png",
        )


#sns.scatterplot(
#        adataz.obs,
#        x="x",
#        y="y",
#        #hue="predict",
#        hue="label",
#        )
#plt.savefig("tmp2.png")


w = torch.randn(1000, model.nw)
z = model.Pz(w)[:,:,:model.nz].reshape(-1,model.nz)

label_g =torch.eye(model.nclasses,
        ).tile(1000,1).argmax(1).numpy().astype('str')
label_p = model.Qy(z).detach().argmax(1).numpy().astype('str')
adataz.obs["x"] = z[:,0].detach().numpy()
adataz.obs["y"] = z[:,1].detach().numpy()
adataz.obs["label_g"] = label_g
adataz.obs["label_p"] = label_p


sc.pl.scatter(
        adataz,
        x="x",
        y="y",
        #color="label",
        #color="predict",
        show=False,
        save="_tmp.png",
        #color="label_p",
        color="label_g",
        )



adatazz = sc.read_h5ad("./data/scgen/scGen_datasets/train_zheng.h5ad",)
adatazz.X = adatazz.X.toarray()
sc.pp.scale(adatazz,)
sc.pp.pca(adatazz,)
sc.pp.neighbors(adatazz,)
#sc.pp.neighbors(adatazz, use_rep="mu_z",)
sc.tl.umap(adatazz,)
sc.tl.louvain(adatazz,)
labelszz = enc_labels.fit_transform(adatazz.obs["cell_type"],)
labelszz = F.one_hot(torch.tensor(labelszz)).float()
datazz = torch.tensor(adatazz.X)
datazz.shape
adatazz.obs["label"] = enc_labels.inverse_transform(
    labelszz.argmax(-1).detach().numpy(),
)
adatazz.obs["label_num"] = labelszz.argmax(-1).detach().numpy().astype('str')
louvainzz = [int(x) for x in adatazz.obs["louvain"]]
louvainzz = F.one_hot(torch.tensor(louvainzz)).float()
output = model(datazz)
adatazz.obs["predict"] =  output["q_y"].argmax(-1).detach().numpy().astype(str)
adatazz.obsm["mu_z"] = output["mu_z"].detach().numpy()
adatazz.obsm["z"] = output["z"].detach().numpy()
adatazz.obsm["mu_w"] = output["mu_w"].detach().numpy()
adatazz.obsm["w"] = output["w"].detach().numpy()
rec = output["rec"].detach()
del output

sc.pp.neighbors(adatazz, use_rep="mu_z",)
sc.tl.umap(adatazz,)
sc.tl.louvain(adatazz,)

sc.pl.umap(
        adatazz,
        color = [
            "louvain",
            "predict",
            "label",
            "cell_type",
            ],
            ncols=2,
            show=False,
            save="_temp.png",
            )






















































