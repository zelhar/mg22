#import gdown
import matplotlib
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
import skimage as skim
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
import gmmvae09 as M9
import gmmvae10 as M10
print(torch.cuda.is_available())

import networkx as nwx

x = torch.randn(128,28**2)
y = torch.randn(128,5,28**2)
z = torch.randn_like(x)
w = torch.randn_like(y)

ut.kld2normal(mu=x.unsqueeze(1), 
        #logvar=z.unsqueeze(1),
        logvar=torch.tensor(-10),
        mu2=y,
        #logvar2=w,
        logvar2=torch.tensor(-10),
        )

#matplotlib.use("QtCairo")
plt.ion()
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=120, facecolor='white', )

enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc_ct = LabelEncoder()

transform = transforms.Compose([
    transforms.ToTensor(),
    ])
train_dataset = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transform,
        )
test_dataset = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transform,
        )
train_data = train_dataset.data.float()/255
test_data = test_dataset.data.float()/255
train_labels = F.one_hot(train_dataset.targets.long(),
        num_classes=10,).float()
test_labels = F.one_hot(test_dataset.targets.long(),
        num_classes=10,).float()


#foo = ut.randomSubset(len(train_data), 0.15)
#train_data[foo == False]
#foo.sum()

f = lambda x: 1 if x > 0.33 else 0
train_data.apply_(f)
test_data.apply_(f)


data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=train_data,
            labels=train_labels,
            ),
        batch_size=128,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=test_data,
            labels=test_labels,
            ),
        batch_size=128,
        shuffle=True,
        )

adata = sc.AnnData(X=train_data.detach().flatten(1).numpy(),)
adata.obs["labels"] = train_dataset.targets.numpy().astype(str)
bdata = sc.AnnData(X=test_data.detach().flatten(1).numpy(),)
bdata.obs["labels"] = test_dataset.targets.numpy().astype(str)

sampledata = ut.randomSubset(s=train_data.size(0), r=0.05)
cdata = sc.AnnData(X=train_data.detach().flatten(1).numpy()[sampledata],)
cdata.obs["labels"] = train_dataset.targets.numpy()[sampledata].astype(str)


enc_ct.fit(adata.obs["labels"])
#data = torch.FloatTensor(adata.X)

sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=10,)
sc.tl.umap(adata,)
#sc.tl.leiden(adata, )
sc.tl.louvain(adata, )
sc.pl.umap(adata, color=["labels", "louvain"], size=5,)

sc.pp.pca(bdata,)
sc.pp.neighbors(bdata, use_rep="X_pca", n_neighbors=10,)
sc.tl.umap(bdata,)
#sc.tl.leiden(bdata, )
sc.tl.louvain(bdata, )
sc.pl.umap(bdata, color=["labels", "louvain"], size=5,)

# VAE is useless for the purpose of dimensionality reduction
model = M10.VAE_Primer_Type1002(
        nx=28**2,
        nh=1024,
        nz=64,
        bn=True,
        dropout=0.25,
        numhidden=3,
        reclosstype='Bernoulli',
        )
model.apply(init_weights)

# AE test
model = M10.AE_Primer_Type1001(
        nx=28**2,
        nh=1024,
        nz=64,
        bn=True,
        dropout=0.25,
        numhidden=3,
        reclosstype='Gauss',
        #reclosstype='Bernoulli',
        )
model.apply(init_weights)

model = M10.AE_Primer_Type1001(
        nx=28**2,
        nh=1024,
        nz=64,
        bn=True,
        dropout=0.25,
        numhidden=1,
        reclosstype='Bernoulli',
        )
model.apply(init_weights)

M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=25,
    test_loader=None,
    lr=1e-3,
    wt=0.0,
    report_interval=3,
)



model.eval()
r,p = M10.estimateClusterImpurity(model, test_data, test_labels, )
print(p, "\n", r.mean(), "\n", r)

x,y = test_loader.__iter__().next()
output = model(x)
x.shape
output['rec'].shape
r,p = M10.estimateClusterImpurity(model, x, y)
print(p, "\n", r.mean(), "\n", r)


x,y = ut.plot_2images(
        x.reshape(-1,1,28,28),
        output['rec'].reshape(-1,1,28,28),
        )
del output



model.eval()
output = model(train_data)
adata.obsm["z"] = output["z"].detach().numpy()
del(output)

#sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.louvain(adata, )
#sc.tl.leiden(adata, )
#sc.pl.umap(adata, color=["leiden"], size=5,)
sc.pl.umap(adata, color=["labels", "louvain",], size=5)
#sc.pl.umap(adata, color=["labels"], size=5)
#sc.pl.umap(adata, color=["louvain",], size=5,)

sc.pp.neighbors(bdata, use_rep="z", n_neighbors=10,)
sc.tl.umap(bdata,)
sc.tl.louvain(bdata, )
#sc.tl.leiden(adata, )
#sc.pl.umap(adata, color=["leiden"], size=5,)
sc.pl.umap(bdata, color=["labels", "louvain",], size=5)

# model Bernoulli, was much better in terms of lattent space clusters.

# looking for best MNIST gmm model?
# looks like 

model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        #nh=1024,
        nh=124,
        #nw=15,
        nw=25,
        nz=62,
        #nclasses=30,
        nclasses=15,
        dscale=1e0,
        zscale=1e0,
        wscale=1e0,
        concentration=1e-0,
        numhidden=6,
        #dropout=0.25,
        dropout=0.2,
        bn=True,
        reclosstype='Bernoulli',
        )

#this one was quiet good.
model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        #nh=1024,
        nh=2024,
        #nw=15,
        #nw=25,
        nw=45,
        #nz=64,
        nz=104,
        #nclasses=30,
        nclasses=18,
        dscale=1e0,
        zscale=1e0,
        wscale=1e0,
        concentration=1e-0,
        #numhidden=2,
        numhidden=4,
        #dropout=0.25,
        dropout=0.2,
        bn=True,
        reclosstype='Bernoulli',
        )

#torch.save(model.state_dict(),
#        "./results/model_m10T1004_184x1024h104z45w_mnist.state_dict.pt",
#        )

# interesting
model = M10.VAE_Dirichlet_Type1004A(
        nx=28**2,
        #nh=1024,
        nh=2024,
        #nw=15,
        #nw=25,
        nw=65,
        #nz=64,
        nz=154,
        #nclasses=30,
        nclasses=10,
        dscale=1e0,
        zscale=1e0,
        wscale=1e0,
        concentration=1e-0,
        #numhidden=2,
        numhidden=6,
        #dropout=0.25,
        dropout=0.1,
        bn=True,
        reclosstype='Bernoulli',
        )

#torch.save(model.state_dict(),
#        "./results/model_m10T1004A_10c6x2024h154z65w_mnist.state_dict.pt",
#        )

model = M10.VAE_Stacked_Dilo_Anndata_Type1003(
        nx=28**2,
        nh=1024,
        nz=64,
        nw=25,
        nclasses=18,
        #bn=True,
        bn=False,
        dropout=0.2,
        #numhidden=3,
        numhidden=1,
        )

model = M10.VAE_Stacked_Dilo_Anndata_Type1003(
        nx=28**2,
        nh=2024,
        nz=64,
        nw=15,
        nclasses=30,
        bn=True,
        #bn=False,
        dropout=0.25,
        #numhidden=3,
        numhidden=3,
        )

# this one was pretty damn good
# Bernoulli is much better than gauss loss for reconstuction
model = M10.VAE_Stacked_Dilo_Anndata_Type1003(
        nx=28**2,
        nh=2024,
        nz=64,
        nw=25,
        nclasses=30,
        bn=True,
        dropout=0.25,
        numhidden=3,
        reclosstype='Bernoulli',
        loss_l_type = "heuristic1",
        )

model = M10.VAE_Dirichlet_Type1006(
        nx=28**2,
        nh=2024,
        nz=64,
        nw=25,
        #nclasses=30,
        nclasses=10,
        bn=True,
        dropout=0.25,
        numhidden=3,
        reclosstype='Bernoulli',
        #concentration=1e-4,
        concentration=5e0,
        )

#model = M10.VAE_Dirichlet_Type1004C(
model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        #nh=1024,
        nh=1024,
        #nw=15,
        #nw=25,
        nw=45*2,
        #nz=64,
        nz=64*2,
        #nclasses=30,
        #nclasses=18,
        nclasses=20,
        dscale=1e0,
        zscale=1e0,
        wscale=1e0,
        concentration=5e-1,
        #numhidden=2,
        #numhidden=3,
        numhidden=6,
        #dropout=0.25,
        #dropout=0.33,
        dropout=0.23,
        bn=True,
        reclosstype='Bernoulli',
        )

model.apply(init_weights)

M10.basicTrainLoop(
        model, 
        data_loader,
        num_epochs=5,
        lrs = [1e-5, 1e-4, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
        wt=0e0,
        report_interval=3,
        )


#torch.save(model.state_dict(),
#        "./results/model_M10T1004_mnist_2024h_25w_65z_30c_3nh_025dout.state.pt"
#        )

#torch.save(model.state_dict(),
#        "./results/model_M10T1003_mnist_2024h_25w_65z_30c_3nh.state.pt"
#        )
#torch.save(model.state_dict(),
#        "./results/model_M10T1003_mnist_2024h_25w_64z_30c_3nh.state.pt"
#        )
#torch.save(model.state_dict(),
#        "./results/model_M10T1003_mnist_2024h_25w_64z_30c_3nh.heuristic1.state.pt"
#        )

#model.load_state_dict(
#        torch.load("./results/model_M10T1003_mnist_2024h_25w_64z_30c_3nh.heuristic1.state.pt"),
#        )

model = M10.VAE_Stacked_Dilo_Anndata_Type1003(
        nx=28**2,
        nh=2024,
        nz=64,
        nw=25,
        nclasses=10,
        bn=True,
        dropout=0.25,
        numhidden=3,
        reclosstype='Bernoulli',
        loss_l_type = "heuristic1",
        )

model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        nh=2024,
        nw=25,
        nz=64,
        nclasses=30,
        dscale=1e0,
        zscale=1e0,
        concentration=1e-0,
        numhidden=3,
        dropout=0.25,
        bn=True,
        reclosstype='Bernoulli',
        )

#torch.save(model.state_dict(),
#        "./results/model_M10T1004_mnist_2024h_25w_65z_30c_3nh.state.pt"
#        )

model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        nh=2024,
        nw=20,
        nz=64,
        nclasses=30,
        dscale=1e0,
        zscale=1e0,
        concentration=1e-0,
        numhidden=4,
        dropout=0.15,
        bn=True,
        reclosstype='Bernoulli',
        )

#torch.save(model.state_dict(),
#        "./results/model_M10T1004_mnist_2024h_20w_64z_30c_4nh.state.pt"
#        )

# this one was very good, but training requires 
# starting with a small learning rate to avoid nanning
# tried also with concentration=5e-1(ok), 
#   1e-2(didn't work, got one cluster).
model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        nh=2024,
        nw=25,
        nz=64,
        nclasses=30,
        dscale=1e0,
        zscale=1e0,
        concentration=1e-0,
        numhidden=3,
        dropout=0.25,
        bn=True,
        reclosstype='Bernoulli',
        )

# this one was also good
# looks like larger concentration make 
# finding clusters faster.
model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        nh=1024,
        nw=15,
        nz=64,
        nclasses=30,
        dscale=1e0,
        zscale=1e0,
        concentration=5e-0,
        numhidden=3,
        dropout=0.2,
        bn=True,
        reclosstype='Bernoulli',
        )

model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        #nh=1024,
        nh=2024,
        #nw=15,
        nw=25,
        nz=64,
        #nclasses=30,
        nclasses=10,
        dscale=1e0,
        zscale=1e-0,
        #concentration=1e1,
        #concentration=1e-1,
        concentration=1e-0,
        numhidden=3,
        dropout=0.2,
        bn=True,
        reclosstype='Bernoulli',
        applytanh=True,
        )


model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        nh=1024,
        #nh=2024,
        nw=15,
        #nw=25,
        #nw=65,
        #nz=64,
        nz=85,
        #nclasses=30,
        nclasses=18,
        dscale=1e0,
        zscale=1e0,
        wscale=1e0,
        concentration=1.2e-0,
        #numhidden=2,
        numhidden=3,
        #dropout=0.25,
        dropout=0.1,
        bn=True,
        reclosstype='Bernoulli',
        #reclosstype='mse',
        )

#model = M10.VAE_Dirichlet_Type1003(
#        nx=28**2,
#        nh=1024,
#        nw=25,
#        nz=64,
#        nclasses=30,
#        dscale=1e0,
#        zscale=1e0,
#        concentration=5e-1,
#        numhidden=2,
#        dropout=0.25,
#        bn=True,
#        )

model.apply(init_weights)

M9.basicTrainLoop(
    model,
    train_loader=data_loader,
    num_epochs=5,
    test_loader=None,
    lrs=[1e-5, 1e-4, 3e-4, 1e-3, 1e-3, 1e-4, 1e-5],
    wt=0.0,
    report_interval=3,
)

M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=25,
    test_loader=None,
    lr=1e-3,
    wt=0.0,
    report_interval=3,
)

M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=10,
    test_loader=None,
    lr=3e-4,
    wt=0.0,
    report_interval=3,
)

M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=10,
    test_loader=None,
    lr=4e-5,
    wt=0.0,
    report_interval=3,
)

M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=10,
    test_loader=None,
    lr=1e-5,
    wt=0.0,
    report_interval=3,
)

M10.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=150,
    test_loader=None,
    lr=2.5e-4,
    wt=0.0,
    report_interval=3,
)

M10.preTrainAE(
        model,
        data_loader,
        10,
        1e-4,
        "cuda",
        wt=0,
        report_interval=3,
        )


model.eval()
w = torch.zeros(1, model.nw)
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(1*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
if model.reclosstype == "Bernoulli":
    rec = rec.sigmoid()
ut.plot_images(rec, model.nclasses)

model.eval()
w = model.w_prior.sample((5, ))
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(5*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
if model.reclosstype == "Bernoulli":
    rec = rec.sigmoid()
ut.plot_images(rec, model.nclasses)

model.eval()
output = model(train_data)
adata.obsm["z"] = output["z"].detach().numpy()
adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
del(output)

#sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.louvain(adata, )
#sc.tl.leiden(adata, )
#sc.pl.umap(adata, color=["leiden"], size=5,)
sc.pl.umap(adata, color=["labels"], size=5)
sc.pl.umap(adata, color=["louvain",], size=5,)
sc.pl.umap(adata, color=["predict", ],size=5)
sc.pl.umap(adata, color=["predict2", ],size=5)


# doing semisup tests
subset = ut.randomSubset(s=len(train_labels), r=0.1)
labeled_data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=train_data[subset],
            labels=train_labels[subset],
            ),
        batch_size=128,
        shuffle=True,
        )
unlabeled_data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=train_data[subset == False],
            labels=train_labels[subset == False],
            ),
        batch_size=128,
        shuffle=True,
        )

model = M10.VAE_Stacked_Dilo_Anndata_Type1003(
        nx=28**2,
        nh=2024,
        nz=64,
        nw=25,
        nclasses=10,
        bn=True,
        dropout=0.25,
        numhidden=3,
        reclosstype='Bernoulli',
        )
model.apply(init_weights)


model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        nh=2024,
        nw=25,
        nz=64,
        nclasses=10,
        dscale=1e0,
        zscale=1e0,
        concentration=1e-0,
        numhidden=3,
        dropout=0.25,
        bn=True,
        reclosstype='Bernoulli',
        )

model.apply(init_weights)


M9.trainSemiSuper(model, 
        labeled_data_loader,
        unlabeled_data_loader,
        test_loader,
        num_epochs=40,
        lr=1e-3,
        wt=0e-3,
        do_eval=False,
        report_interval=4,
        )

M9.trainSemiSuperLoop(
        model, 
        labeled_data_loader,
        unlabeled_data_loader,
        test_loader,
        num_epochs=10,
        lrs=[1e-5, 1e-4, 1e-3, 1e-4, 1e-5,],
        wt=0e-3,
        do_validation=False,
        report_interval=4,
        )

# further MNIST tests

model.eval()
output = model(test_data)
bdata.obsm["z"] = output["z"].detach().numpy()
bdata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
bdata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
del(output)

#sc.pp.pca(adata,)
sc.pp.neighbors(bdata, use_rep="z", n_neighbors=11,)
sc.tl.umap(bdata,)
sc.tl.louvain(bdata, )
#sc.tl.leiden(adata, )
#sc.pl.umap(adata, color=["leiden"], size=5,)
sc.pl.umap(bdata, color=["labels"], size=5)
sc.pl.umap(bdata, color=["louvain",], size=5,)
sc.pl.umap(bdata, color=["predict", ],size=5)
sc.pl.umap(bdata, color=["predict2", ],size=5)

model = M10.VAE_Dirichlet_Type1004C(
        nx=28**2,
        nw=25,
        nz=64,
        nclasses=30,
        dscale=1,
        wscale=1,
        zscale=1,
        yscale=1,
        #nh=1024,
        concentration=1e-0,
        #numhidden=1,
        dropout=0.2,
        #bn=True,
        reclosstype="Bernoulli",
        )
model.apply(init_weights)

M9.basicTrain(
    model,
    train_loader=data_loader,
    num_epochs=10,
    test_loader=None,
    lr=1e-3,
    wt=0.0,
    report_interval=1,
)

model = M10.VAE_Dirichlet_Type1004(
        nx=28**2,
        nh=2024,
        nw=25,
        nz=64,
        #nclasses=18,
        nclasses=16,
        dscale=5e1,
        zscale=1e0,
        #concentration=5e-1,
        concentration=5e-3,
        numhidden=3,
        dropout=0.25,
        bn=True,
        reclosstype='Bernoulli',
        )
model.apply(init_weights)

model = M10.VAE_Type1005(
        nx=28**2,
        nh=2024,
        #nh=1024,
        nz=64,
        #nclasses=16,
        nclasses=26,
        numhidden=3,
        #numhidden=2,
        dropout=0.25,
        bn=True,
        dscale=5e-1,
        concentration=1e-0,
        reclosstype='Bernoulli',
        )

model.apply(init_weights)

M9.basicTrainLoop(
    model,
    train_loader=data_loader,
    num_epochs=5,
    test_loader=None,
    lrs=[1e-5, 1e-4, 1e-3, 5e-4, 5e-5, 1e-5,],
    wt=0.0,
    report_interval=3,
)


c = torch.eye(model.nclasses)
cz = model.Ez(c)
cx = model.Px(cz).sigmoid()
cx.shape
ut.plot_images(cx.reshape(-1,1,28,28))




#####
x,y = test_loader.__iter__().next()
x = x.reshape(-1,28**2)

foo = nn.Sequential(
        nn.Linear(28**2, 2**11),
        nn.BatchNorm1d(2**11),
        nn.Unflatten(1, (1, 2**11),),
        nn.Conv1d(1, 5, 8, 4, 2, )
        )

foo = ut.buildCNetworkv2(nc=1, nin=28**2, nout=28**2)
foo(x).shape

z = torch.randn(128,64)
bar = ut.buildTCNetworkv1(nin=64, nout=28**2, nf=32,) 
bar(z).shape

model.eval()
output = model(train_data[sampledata])
cdata.obsm["z"] = output["z"].detach().numpy()
cdata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
#cdata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
del(output)

sc.pp.neighbors(cdata, use_rep="z", n_neighbors=10,)
sc.tl.umap(cdata,)
sc.tl.louvain(cdata, )

sc.pl.umap(cdata, color=["labels"], size=15)
sc.pl.umap(cdata, color=["louvain",], size=15,)
sc.pl.umap(cdata, color=["predict", ],size=15)
sc.pl.umap(cdata, color=["predict2", ],size=15)













######### linalg tessts
x = torch.randint(0,2, (10,10)).numpy()
x

k = ut.diffMatrix(x)

g = nwx.Graph()
g.add_nodes_from(np.arange(10))


g, cs = ut.diffCluster(x, 3)

cs
nwx.draw_spring(g, with_labels=True)


x = bdata.obsp["connectivities"].toarray()

g, cs = ut.diffCluster(x, 11)

g, cs = ut.diffCluster2(x, 3)
bdata.obs['cs'] = cs.astype(int).astype(str)
sc.pl.umap(bdata, color=["louvain", "cs"], size=15,)


##### cvae tests

model = M10.CVAE_Type1007(
        nx=28**2, nz=32, nh=1024, ny=10, dropout=0.25, 
        reclosstype="Bernoulli", )

x = torch.randn(128,1,28,28)
x = x.flatten(1)
x.shape
z = model.Qz(x)
z.shape
rec = model.Px(z[:,0,:64])
rec.shape

x,y = data_loader.__iter__().next()
output = model(x,y)
rec = output["rec"].reshape(-1,1,28,28).sigmoid()

f = lambda x: 1 if x > 0.5 else 0
x.apply_(f)

ut.plot_2images(x.reshape(-1,1,28,28), rec)

ut.plot_images(x.reshape(-1,1,28,28), model.ny)

ut.plot_images(rec, model.ny)

output["rec"].shape
output["z"].shape

M9.trainSemiSuper(model, data_loader, data_loader, test_loader, 
        num_epochs=51, lr=3e-4, wt=0, 
        do_unlabeled=False, do_eval=False, report_interval=1,)

model.eval()

model.eval()
z = torch.randn(5,model.nz)*1.01
rec = model.Px(z).reshape(-1,1,28,28)
if model.reclosstype == "Bernoulli":
    rec = rec.sigmoid()
ut.plot_images(rec, model.ny)

model = M10.CVAE_Type1008(
        nx=28**2, nz=32, nh=1024, ny=10, dropout=0.25, 
        reclosstype="Bernoulli", )

z = torch.randn(10,model.nz)*1.01
y = torch.eye(10)
zy = torch.cat([z,y], dim=-1)
rec = model.Px(zy).reshape(-1,1,28,28)
if model.reclosstype == "Bernoulli":
    rec = rec.sigmoid()
ut.plot_images(rec, model.ny)



### torchvision models test
resnet = models.resnet152()
