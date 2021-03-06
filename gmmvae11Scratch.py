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
import gmmvae11 as M11
print(torch.cuda.is_available())

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

model = M11.VAE_Dirichlet_Type1101(
        nx=28**2,
        nh=1024,
        nhp=512,
        nhq=4*1024,
        nw=32,
        nz=64,
        nclasses=10,
        #concentration=1.0e-0,
        concentration=3*1.0e-0,
        #concentration=1.0e-1,
        #concentration=1.0e1,
        #concentration=1.0e2,
        numhidden=2,
        numhiddenp=1,
        numhiddenq=2,
        dropout=0.25,
        reclosstype="Bernoulli",
        bn=True,
        relax=True,
        )

model = M11.VAE_Dirichlet_Type1102(
        nx=28**2,
        nh=1024,
        nhp=512,
        nhq=4*1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=1,
        numhiddenq=2,
        dropout=0.25,
        reclosstype="Bernoulli",
        bn=True,
        )

model = M11.VAE_Dirichlet_Type1103(
        nx=28**2,
        nh=1024,
        nhp=512,
        nhq=4*1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=1,
        numhiddenq=2,
        dropout=0.25,
        reclosstype="Bernoulli",
        bn=True,
        )

model = M11.VAE_Dirichlet_Type1104(
        nx=28**2,
        nh=1024,
        nhp=512,
        nhq=4*1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=1,
        numhiddenq=2,
        dropout=0.25,
        reclosstype="Bernoulli",
        bn=True,
        )

model = M11.VAE_Dirichlet_Classifier_Type1105(
        nx=28**2,
        nh=1024,
        nhp=512,
        nhq=4*1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=5.0e-1,
        numhidden=2,
        numhiddenp=1,
        numhiddenq=2,
        dropout=0.25,
        reclosstype="Bernoulli",
        bn=True,
        )

M10.basicTrainLoop(
        model, 
        data_loader,
        #num_epochs=10,
        num_epochs=10,
        #lrs = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
        #lrs = [1e-5, 1e-4,1e-3, 1e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
        lrs = [1e-5, 1e-4,1e-3, 3e-4, 1e-4, 3e-5, 1e-5],
        wt=0e0,
        report_interval=3,
        #do_plot=True,
        do_plot=False,
        )

# semi sup
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



M10.trainSemiSuperLoop(
        model,
        labeled_data_loader,
        unlabeled_data_loader,
        test_loader,
        num_epochs=10,
        lrs = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3,1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
        wt=0,
        do_plot=True,
        do_unlabeled=True,
        do_validation=False,
        )


r,p,s = M10.estimateClusterImpurityLoop(model, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())


P0 = distributions.OneHotCategorical(probs=torch.ones(5))
P = distributions.RelaxedOneHotCategorical(
    temperature=torch.tensor([0.2]), probs=torch.ones(5)
)
Q0 = pyrodist.OneHotCategorical(probs=torch.ones(5))
Q = pyrodist.RelaxedOneHotCategoricalStraightThrough(
    temperature=torch.tensor([0.2]),
    probs=torch.ones(5),
)
Q1 = pyrodist.RelaxedOneHotCategorical(
    temperature=torch.tensor([0.2]),
    probs=torch.ones(5),
)

x = P0.sample((10,))
y = P.rsample((10,))
z = Q0.sample((10,))
w = Q.rsample((10,))
w1 = Q1.rsample((10,))


P = distributions.RelaxedOneHotCategorical(
    temperature=torch.tensor([0.15]), probs=torch.ones(5)
)
P.rsample((10,)).max(-1)

model1 = M11.VAE_Dirichlet_Type1101(
        nx=28**2,
        nh=1024,
        nhp=512,
        nhq=4*1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=1.0e-0,
        #concentration=3*1.0e-0,
        #concentration=1.0e-1,
        #concentration=1.0e1,
        #concentration=1.0e2,
        numhidden=2,
        numhiddenp=1,
        numhiddenq=2,
        dropout=0.25,
        reclosstype="Bernoulli",
        bn=True,
        relax=True,
        )
model2 = M11.VAE_Dirichlet_Type1101(
        nx=28**2,
        nh=1024,
        nhp=512,
        nhq=4*1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=1.0e-0,
        #concentration=3*1.0e-0,
        #concentration=1.0e-1,
        #concentration=1.0e1,
        #concentration=1.0e2,
        numhidden=2,
        numhiddenp=1,
        numhiddenq=2,
        dropout=0.25,
        reclosstype="Bernoulli",
        bn=True,
        relax=True,
        )

M11.tandemTrain(
        model1,
        model2,
        data_loader,
        num_epochs=15,
        lr=1e-3,
        wt=0,
        report_interval=3,
        do_plot=True,
        )

M11.tandemTrainV2(
        model1,
        model2,
        data_loader,
        num_epochs=250,
        #lr=1e-3,
        lr=5e-4,
        wt=0,
        report_interval=3,
        do_plot=True,
        )
