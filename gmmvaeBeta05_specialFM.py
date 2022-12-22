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


transform = transforms.Compose([
    transforms.ToTensor(),
    ])
train_dataset = datasets.FashionMNIST(
        "data/",
        train=True,
        download=True,
        transform=transform,
        )
test_dataset = datasets.FashionMNIST(
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
        batch_size=2**11,
        #batch_size=2**8,
        #batch_size=2**9,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(
            data=test_data,
            labels=test_labels,
            ),
        batch_size=2**11,
        #batch_size=2**8,
        #batch_size=2**9,
        shuffle=True,
        )

model = Mb0.VAE_Dirichlet_GMM_TypeB1602z(
        nx=28**2,
        nh=1024*2,
        nhq=1024*2,
        nhp=1024*2,
        concentration=1e0,
        nclasses=10,
        nz=32,
        nw=32,
        numhidden=4,
        dropout=0.2,
        reclosstype="Bernoulli",
        bn=True,
        restrict_w=True,
        restrict_z=True,
        yscale=5e0,
        )


model = Mb0.VAE_Dirichlet_GMM_TypeB1602xz(
        nx=28**2,
        nh=1024*2,
        nhq=1024*2,
        nhp=1024*2,
        concentration=1e0,
        nclasses=10,
        nz=32,
        nw=32,
        numhidden=4,
        dropout=0.2,
        reclosstype="Bernoulli",
        bn=True,
        restrict_w=True,
        restrict_z=True,
        yscale=5e0,
        )

model = Mb0.VAE_Dirichlet_GMM_TypeB1602xzR(
        nx=28**2,
        nh=1024*2,
        nhq=1024*2,
        nhp=1024*2,
        concentration=1e0,
        nclasses=10,
        nz=32,
        nw=32,
        numhidden=4,
        dropout=0.2,
        reclosstype="Bernoulli",
        bn=True,
        restrict_w=True,
        restrict_z=True,
        yscale=5e0,
        use_resnet=True,
        )
model.apply(init_weights)

model = M16.VAE_Dirichlet_GMM_Type1602xz(
        nx=28**2,
        nh=1024,
        nhq=1024,
        nhp=1024,
        concentration=1e0,
        nclasses=10,
        nz=32,
        nw=32,
        numhidden=4,
        dropout=0.2,
        reclosstype="Bernoulli",
        bn=True,
        restrict_w=True,
        restrict_z=True,
        yscale=5e0,
        use_resnet=True,
        )
model.apply(init_weights)

model = Mb0.VAE_Dirichlet_GMM_TypeB1602zR(
        nx=28**2,
        nh=1024,
        nhq=1024,
        nhp=1024,
        concentration=1e0,
        nclasses=10,
        nz=32,
        nw=32,
        numhidden=4,
        dropout=0.2,
        reclosstype="Bernoulli",
        bn=True,
        restrict_w=True,
        restrict_z=True,
        yscale=1e0,
        use_resnet=True,
        )
model.apply(init_weights)

model = Mb0.VAE_Dirichlet_GMM_TypeB1602xzR18(
        nx=28**2,
        nh=1024,
        nhq=1024,
        nhp=1024,
        concentration=1e0,
        nclasses=10,
        nz=32,
        nw=32,
        numhidden=4,
        dropout=0.2,
        reclosstype="Bernoulli",
        bn=True,
        restrict_w=True,
        restrict_z=True,
        yscale=1e0,
        use_resnet=True,
        )
model.apply(init_weights)

model = Mb0.VAE_Dirichlet_GMM_TypeB1602V2(
        nx=28**2,
        nh=1024,
        nhq=1024,
        nhp=1024,
        concentration=1e0,
        nclasses=10,
        nz=32,
        nw=32,
        numhidden=4,
        dropout=0.2,
        reclosstype="Bernoulli",
        bn=True,
        restrict_w=True,
        restrict_z=True,
        yscale=1e0,
        use_resnet=True,
        )
model.apply(init_weights)



Train.trainSemiSuperLoop(
        model,
        data_loader,
        data_loader,
        test_loader,
        num_epochs=50,
        lrs=[
            1e-6,
            1e-5,
            1e-4,
            1e-3,
            1e-3,
            1e-3,
            1e-4,
            1e-5,
            1e-6,
            ],device="cuda",
        wt=0e-3,
        do_unlabeled=False,
        do_validation=False,
        report_interval=10,
        do_plot=False,
        #test_accuracy=True,
        test_accuracy=False,
        )
print("done training")

Train.trainSemiSuperLoop(
        model,
        data_loader,
        data_loader,
        test_loader,
        num_epochs=1500,
        lrs=[
            1e-6,
            1e-5,
            1e-5,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            1e-5,
            1e-6,
            ],device="cuda",
        wt=0e-3,
        do_unlabeled=False,
        do_validation=False,
        report_interval=10,
        do_plot=False,
        #test_accuracy=True,
        test_accuracy=False,
        )

print("done training")
r,p,s = ut.estimateClusterImpurityLoop(model, test_data, test_labels, device="cuda",
        cond1=None) #broken need to fix this for conditional
print(p,r,s)
r = r[r>=0]
s = s[s>=0]
print("kt_acc= \n", (r*s).sum().item() / s.sum().item(), r.mean().item())

ut.saveModelParameters(
        model,
        "./results/fmnishtsupervised" + ut.timeStamp() + ut.randomString() + "params.pt",
        method="json",
        )
torch.save(
        model.state_dict(),
        "./results/fmnishtsupervised" + ut.timeStamp() + ut.randomString() + "state.pt",
        )
