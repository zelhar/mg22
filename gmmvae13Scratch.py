#import gdown
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import os
import pandas as pd
#import pyro
#import pyro.distributions as pyrodist
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
#from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
#from pyro.optim import Adam
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
import gmmTraining as Train
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

fashion_dataset_train = datasets.FashionMNIST(
        "data/",
        train=True,
        download=True,
        transform=transform,
        )
fashion_dataset_test = datasets.FashionMNIST(
        "data/",
        train=True,
        download=True,
        transform=transform,
        )

a,b = fashion_dataset_test.__getitem__(1)


data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSetV2(
            dati=[train_data, train_labels, ],
            ),
        batch_size=128,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSetV2(
            dati=[test_data, test_labels, ],
            ),
        batch_size=128,
        shuffle=True,
        )

adata = sc.AnnData(X=train_data.detach().flatten(1).numpy(),)
adata.obs["labels"] = train_dataset.targets.numpy().astype(str)
bdata = sc.AnnData(X=test_data.detach().flatten(1).numpy(),)
bdata.obs["labels"] = test_dataset.targets.numpy().astype(str)



model = M13.CVAE_Dirichlet_Type1301(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        #nz=64,
        #nw=32,
        nw=3,
        nz=8,
        nc1=2,
        nclasses=2,
        #nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        reclosstype="Bernoulli",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        )
model.apply(init_weights)


M13.basicTrainLoopCond(
    model,
    data_loader,
    num_epochs=10,
    lrs=[
        1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        5e-4,
        1e-4,
        5e-5,
        1e-5,
    ],
    wt=1e-3,
    report_interval=3,
    do_plot=True,
    test_accuracy=False,
)

state = model.state_dict()

model = M13.VAE_GMM_Type1302(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        #nz=64,
        #nw=32,
        nw=3,
        nz=8,
        nclasses=2,
        #nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        reclosstype="Bernoulli",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        )
model.apply(init_weights)

state2 = model.state_dict()

model = M13.VAE_GMM_Type1302(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        nz=64,
        nw=32,
        #nw=3,
        #nz=8,
        nclasses=4,
        #nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        reclosstype="Bernoulli",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        )
model.apply(init_weights)

model = M13.VAE_MI_Type1304(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        nz=64,
        nw=32,
        #nw=3,
        #nz=8,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        mi_scale=1e2,
        reclosstype="Bernoulli",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        )
model.apply(init_weights)

model = M13.VAE_Dirichlet_Type1300(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        #nz=64,
        #nw=32,
        nw=3,
        nz=8,
        #nclasses=2,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        reclosstype="Bernoulli",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        restrict_w=True,
        )
model.apply(init_weights)

model = M13.VAE_Dirichlet_Type1300D(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        #nz=64,
        #nw=32,
        nw=3,
        nz=8,
        #nclasses=2,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        reclosstype="Bernoulli",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        )
model.apply(init_weights)

# interesting
model = M13.VAE_GMM_Type1302D(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        #nz=64,
        #nw=32,
        nw=3,
        nz=8,
        #nclasses=2,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        cc_scale=1e1,
        dropout=0.20,
        reclosstype="Bernoulli",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        )
model.apply(init_weights)

state3 = model.state_dict()

torch.save(
        model.state_dict(),
        "./results/model_M13_1302D_mnist_h1024hp3072w3z8c10nh2.state3.pt",
        )

model = M13.VAE_GMM_Type1302(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        #nz=64,
        #nw=32,
        #nw=3,
        #nz=8,
        nz=15,
        nw=8,
        nclasses=10,
        #nclasses=20,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        reclosstype="Bernoulli",
        bn=True,
        use_resnet=True,
        #softargmax=True,
        cc_scale=1e1,
        restrict_w=True,
        )
model.apply(init_weights)

M13.basicTrainLoop(
    model,
    data_loader,
    num_epochs=10,
    lrs=[
        #1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        5e-4,
        1e-4,
        5e-5,
        1e-5,
    ],
    wt=1e-3,
    report_interval=3,
    do_plot=True,
    #test_accuracy=False,
    test_accuracy=True,
)

M13.preTrainLoop(
    model,
    data_loader,
    wt=1e-3,
    num_epochs=10,
    report_interval=3,
    do_plot=True,
    test_accuracy=True,
    #lrs = [5e-5, 1e-4, 1e-3, 1e-4, 1e-5],
    lrs = [1e-5, 1e-4, 1e-3,],
        )

M13.advancedTrainLoop(
    model,
    data_loader,
    wt=1e-4,
    num_epochs=10,
    report_interval=3,
    do_plot=True,
    test_accuracy=True,
    lrs = [5e-5, 1e-4, 5e-4, 1e-3, 1e-4, 1e-5],
    advanced_semi=True,
    cc_extra_sclae = 1e1,
        )

r,p,s = M10.estimateClusterImpurityLoop(model, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

(x,y) = test_loader.__iter__().next()
q_y = model.justPredict(x)

q_y.argmax(-1)[y.argmax(-1) == 0]
q_y.argmax(-1)[y.argmax(-1) == 1]
q_y.argmax(-1)[y.argmax(-1) == 2]

train_conds = model.justPredict(train_data, ).detach()
test_conds = model.justPredict(test_data, ).detach()

train_conds = (1 + train_conds).exp().exp().softmax(-1)
test_conds = (1 + test_conds).exp().exp().softmax(-1)

data_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSetV2(
            dati=[train_data, train_labels, train_conds],
            ),
        batch_size=128,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSetV2(
            dati=[test_data, test_labels, test_conds],
            ),
        batch_size=128,
        shuffle=True,
        )

x,y,c = test_loader.__iter__().next()

model = M13.CVAE_GMM_Type1303(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        nz=64,
        nw=32,
        #nw=3,
        #nz=8,
        #nc1=2,
        nc1=4,
        #nclasses=2,
        #nclasses=10,
        nclasses=4,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        reclosstype="Bernoulli",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        )
model.apply(init_weights)

M13.basicTrainLoopCond(
    model,
    data_loader,
    num_epochs=10,
    lrs=[
        1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        5e-4,
        1e-4,
        5e-5,
        1e-5,
    ],
    wt=1e-3,
    report_interval=3,
    do_plot=True,
    test_accuracy=False,
)




#### fashionmnist

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
        dataset=ut.SynteticDataSetV2(
            dati=[train_data, train_labels, ],
            ),
        batch_size=128,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSetV2(
            dati=[test_data, test_labels, ],
            ),
        batch_size=128,
        shuffle=True,
        )




model = M13.VAE_Dirichlet_Type1300(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        nz=64,
        nw=32,
        #nw=13,
        #nz=28,
        #nclasses=2,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.20,
        reclosstype="Bernoulli",
        #reclosstype="mse",
        bn=True,
        #use_resnet=True,
        #softargmax=True,
        restrict_w=True,
        )
model.apply(init_weights)

model = M13.VAE_GMM_Type1302(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024*3,
        nz=64,
        nw=32,
        #nw=3,
        #nz=8,
        #nclasses=2,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.30,
        reclosstype="Bernoulli",
        #reclosstype="mse",
        bn=True,
        use_resnet=True,
        #softargmax=True,
        #restrict_w=True,
        )
model.apply(init_weights)

x,y = test_loader.__iter__().next()


M13.basicTrainLoop(
    model,
    data_loader,
    num_epochs=12,
    lrs=[
        1e-4,
        1e-3,
        1e-5,
        1e-5,
        5e-5,
        1e-4,
        1e-4,
        1e-4,
        5e-5,
        1e-5,
    ],
    wt=1e-5,
    report_interval=3,
    do_plot=True,
    #test_accuracy=False,
    test_accuracy=True,
)

M13.advancedTrainLoop(
    model,
    data_loader,
    wt=1e-4,
    num_epochs=10,
    report_interval=3,
    do_plot=True,
    test_accuracy=True,
    lrs = [5e-5, 1e-4, 5e-4, 1e-3, 1e-4, 1e-5],
    advanced_semi=True,
    cc_extra_sclae = 1e1,
        )

M13.trainSemiSuperLoop(
        model=model,
        train_loader_labeled=test_loader,
        train_loader_unlabeled=data_loader,
        num_epochs=15,
        lrs = [1e-4, 5e-4, 1e-3, 1e-4, 1e-5,],
        wt=1e-5,
        do_unlabeled=True,
        do_validation=False,
        test_accuracy=True,
        test_loader=test_loader,
        do_plot=True,
        )

