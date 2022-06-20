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

#matplotlib.use("QtCairo")
plt.ion()
sc.settings.verbosity = 3
sc.logging.print_header()
#sc.settings.set_figure_params(dpi=120, facecolor='white', )
#sc.settings.set_figure_params(figsize=(8,8), dpi=80, facecolor='white', )
sc.settings.set_figure_params(dpi=80, facecolor='white', )

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



#x,y = test_loader.__iter__().next()
#ut.plot_images(x.reshape(-1,1,28,28))




# pretty good (resnet18), saved it 0.84 acc
model = M10.VAE_Dirichlet_Type1004R(
        nx=28**2,
        nh=1024,
        nw=32,
        nz=64,
        nclasses=18,
        concentration=1e-0,
        numhidden=2,
        dropout=0.2,
        reclosstype="Bernoulli",
        bn=True,
        )

#torch.save(
#        model.state_dict(),
#        "./results/model_m10T004R_mnits_h1024_w32_z64_nh2_18C_B_v2.state.pt",
#        )

#best ever
model = M10.VAE_Dirichlet_Type1004R(
        nx=28**2,
        nh=1024,
        nw=32,
        nz=64,
        nclasses=10,
        #concentration=1.5e-0,
        concentration=1.0e-0,
        numhidden=2,
        dropout=0.3,
        reclosstype="Bernoulli",
        bn=True,
        )


#best ever
#torch.save(
#        model.state_dict(),
#        "./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.2.pt",
#        )

#torch.save(
#        model.state_dict(),
#        "./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.ss.4.pt",
#        )
#torch.save(
#        model.state_dict(),
#        "./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.ss.5.pt",
#        )

#model.load_state_dict(
#        torch.load(
#        "./results/model_m10T004R_mnits_h1024_w32_z64_nh2_B.state.pt",
#        ),)


#model.load_state_dict(
#    torch.load(
#        "./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.2.pt",
#    )
#)

model = M10.VAE_Dirichlet_Type1004R(
        nx=28**2,
        nh=1024,
        nw=55,
        nz=128,
        nclasses=10,
        concentration=1e-0,
        numhidden=1,
        dropout=0.15,
        reclosstype="Bernoulli",
        bn=True,
        )

model = M10.VAE_Dirichlet_Type1004S(
        nx=28**2,
        nh=1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=1e-0,
        numhidden=2,
        dropout=0.25,
        reclosstype="Bernoulli",
        bn=True,
        )

#torch.save(
#        model.state_dict(),
#        "./results/model_m10T004S_mnits_h1024_w32_z64_nh2_10C_B.state.3.pt",
#        )
#

M10.basicTrainLoop(
        model, 
        data_loader,
        num_epochs=7,
        lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
        wt=0e0,
        report_interval=3,
        do_plot=True,
        )

M10.basicTrainLoop(
        model, 
        data_loader,
        #num_epochs=107,
        num_epochs=10,
        lrs = [1e-5, 3e-5, 1e-5, 3e-5, 1e-5, 1e-5],
        wt=0e0,
        report_interval=3,
        do_plot=True,
        )

model.eval()
w = torch.zeros(1, model.nw)
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(1*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
if model.reclosstype == "Bernoulli":
    rec = rec.sigmoid()
ut.plot_images(rec, model.nclasses)

#plt.subplot(111)

plt.close()
plt.figure(figsize=(5,5), dpi=80)


plt.savefig("tmp.png")

#plt.savefig("model_mnist_10c_generation.png")
#plt.savefig("model_mnist_10c_umap.png")

model.eval()
#w = model.w_prior.sample((5, ))
w = torch.randn(15, model.nw) * 1.3
#w = -3 + 3*torch.rand(15,model.nw) 
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(15*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
if model.reclosstype == "Bernoulli":
    rec = rec.sigmoid()
ut.plot_images(rec, model.nclasses)

model.eval()
r,p,s = M10.estimateClusterImpurity(model, test_data, test_labels, "cpu", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

r,p,s = M10.estimateClusterImpurityLoop(model, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

x,y = test_loader.__iter__().next()
output = model(x)
x.shape
output['rec'].shape
r,p,s = M10.estimateClusterImpurity(model, x, y, "cuda",)
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

model.eval()
output = model(test_data)
bdata.obsm["z"] = output["z"].detach().numpy()
bdata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
bdata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
del(output)

#sc.pp.pca(adata,)
sc.pp.neighbors(bdata, use_rep="z", n_neighbors=10,)
sc.tl.umap(bdata,)
sc.tl.louvain(bdata, )

#sc.tl.leiden(adata, )
#sc.pl.umap(adata, color=["leiden"], size=5,)
sc.pl.umap(bdata, color=["labels"], size=5)
sc.pl.umap(bdata, color=["louvain",], size=5,)
sc.pl.umap(bdata, color=["predict", ],size=5)
sc.pl.umap(bdata, color=["predict2", ],size=5)

sc.pl.umap(bdata, color=["labels", "predict", "louvain"], size=5)

plt.figure(figsize=(4,3), dpi=80)

sc.pl.umap(bdata, color=["labels", "predict"], size=5)
plt.savefig("tmp2.png")



### semi sup improvement attempt
# repeated this step
#torch.save(
#        model.state_dict(),
#        "./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.ss.8.pt",
#        )
#

#torch.save(
#        model.state_dict(),
#        "./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.ss.9.pt",
#        )

#model.load_state_dict(
#    torch.load(
#        "./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.ss.6.pt",
#    )
#)

# best so far
model.load_state_dict(
    torch.load(
        "./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.ss.8.pt",
    )
)

#sampledata = ut.randomSubset(s=train_data.size(0), r=0.05)
#sampledata = ut.randomSubset(s=train_data.size(0), r=0.10)
#sampledata = ut.randomSubset(s=train_data.size(0), r=0.33)
#sampledata = ut.randomSubset(s=train_data.size(0), r=0.21)
sampledata = ut.randomSubset(s=train_data.size(0), r=0.15)
cdata = sc.AnnData(X=train_data.detach().flatten(1).numpy()[sampledata],)
cdata.obs["labels"] = train_dataset.targets.numpy()[sampledata].astype(str)


labeled_train_data = train_data[sampledata]
unlabeled_train_data = train_data[sampledata == False]

model.eval()
output = model(labeled_train_data)
train_p_labels = output["q_y"].detach().argmax(-1)
train_p_labels = F.one_hot(train_p_labels, num_classes=10,).float()
del output

train_no_labels = train_labels[sampledata == False]

labeled_data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=labeled_train_data,
            labels=train_p_labels,
            ),
        batch_size=128,
        shuffle=True,
        )

unlabeled_data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=unlabeled_train_data,
            labels=train_no_labels,
            ),
        batch_size=128,
        shuffle=True,
        )


M10.trainSemiSuperLoop(
        model,
        labeled_data_loader,
        unlabeled_data_loader,
        test_loader,
        #num_epochs=50,
        #num_epochs=20,
        num_epochs=150,
        #lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-4, 3e-5, 1e-5],
        lrs = [1e-5, 3e-5, 1e-5, 3e-6, 3e-6, 1e-6, 1e-6],
        do_unlabeled=True,
        #do_unlabeled=False,
        #do_validation=True,
        do_validation=False,
        report_interval=3,
        do_plot = True,
        )
