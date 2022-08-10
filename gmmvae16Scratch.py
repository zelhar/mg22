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


@curry
def binarize(x: torch.Tensor, threshold: float = 0.25) -> torch.Tensor:
    ret = (x > threshold).float()
    return ret


transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)
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
train_data = train_dataset.data.float() / 255
test_data = test_dataset.data.float() / 255
train_labels = F.one_hot(
    train_dataset.targets.long(),
    num_classes=10,
).float()
test_labels = F.one_hot(
    test_dataset.targets.long(),
    num_classes=10,
).float()

data_loader = torch.utils.data.DataLoader(
    dataset=ut.SynteticDataSetV2(
        dati=[
            train_data,
            train_labels,
        ],
    ),
    batch_size=128,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    dataset=ut.SynteticDataSetV2(
        dati=[
            test_data,
            test_labels,
        ],
    ),
    batch_size=128,
    shuffle=True,
)

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



model = M16.VAE_GMM_Type1603(
    nx=28**2,
    nclasses=10,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.3,
    bn=True,
    yscale=1e1,
    zscale=2e0,
    wscale=2e0,
    #reclosstype="Bernoulli",
    reclosstype="mse",
    relax=False,
    use_resnet=True,
    restrict_w=True,
    restrict_z=True,
    do_cc=True,
    cc_radius=0.01,
)
model.apply(init_weights)

model = M16.VAE_GMM_Type1603(
    nx=28**2,
    nclasses=10,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=16,
    nw=12,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.3,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    reclosstype="Bernoulli",
    #reclosstype="mse",
    relax=False,
    #use_resnet=True,
    #restrict_w=True,
    restrict_z=True,
    #do_cc=True,
    #cc_radius=0.03,
    #cc_scale=1e1,
)
model.apply(init_weights)


model.cc_scale = 2e1
model.cc_radius = 1e-1
model.do_cc = True

model.cc_scale = 2e1
model.cc_radius = 5e-1
model.do_cc = True

model.cc_scale = 3e1
model.cc_radius = 1e0
model.do_cc = True

model.cc_scale = 4e1
model.cc_radius = 2e0
model.yscale = 2e0
model.do_cc = True

model.cc_scale = 4e1
model.cc_radius = 5e0
model.yscale = 2e0
model.wscale = 2e0
model.do_cc = True

ut.is_jsonable(model.__dict__)
ut.is_pickleable(model.__dict__)
ut.is_serializeable(model.__dict__)

ut.saveModelParameters(
        model,
        "./results/" + str(datetime.timestamp(datetime.now())) + "model_params.pt",
        method="json",
        )
torch.save(
        model.state_dict(),
        "./results/" + str(datetime.timestamp(datetime.now())) + "model_state.pt",
        )

model.cc_scale = 5e1
model.cc_radius = 5e0
model.yscale = 2e0
model.wscale = 5e0
model.do_cc = True

Train.basicTrainLoop(
    model,
    data_loader,
    None,
    num_epochs=5,
    #lrs=[1e-5,1e-4,1e-3,1e-3,1e-4,1e-5],
    lrs=[1e-4,1e-4,1e-4,1e-4],
    wt=1e-3,
    #report_interval=10,
    report_interval=1,
    do_plot=True,
    test_accuracy=True,
)


r,p,s = ut.estimateClusterImpurityLoop(modela, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

model.eval()

x,y = test_loader.__iter__().next()


output = model(x)

qy = output['q_y'].detach().argmax(-1)
qy

model = M16.VAE_Dirichlet_GMM_Type1602Temp(
    nx=28**2,
    nclasses=10,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=16,
    nw=12,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.3,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    reclosstype="Bernoulli",
    concentration=2e0,
    #reclosstype="mse",
    relax=False,
    use_resnet=True,
    restrict_w=True,
    restrict_z=True,
    #do_cc=True,
    #cc_radius=0.03,
    #cc_scale=1e1,
)
model.apply(init_weights)

modela = M16.VAE_GMM_Type1603Temp(
    nx=28**2,
    nclasses=10,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=6,
    nw=6,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.3,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    reclosstype="Bernoulli",
    #reclosstype="mse",
    relax=False,
    use_resnet=True,
    restrict_w=True,
    restrict_z=True,
    #do_cc=True,
    #cc_radius=0.03,
    #cc_scale=1e1,
)
modela.apply(init_weights)

model.do_cc = True
model.cc_scale = 1
model.cc_radius = 0e-1



Train.basicTrainLoop(
    model,
    data_loader,
    None,
    num_epochs=50,
    #lrs=[1e-5,1e-4,1e-3,1e-3,1e-4,1e-5],
    #lrs=[1e-4,1e-4,1e-4,1e-4],
    lrs=[1e-5,1e-5,1e-5,1e-5],
    wt=1e-3,
    #report_interval=10,
    report_interval=1,
    do_plot=True,
    test_accuracy=True,
)

Train.basicTrainLoop(
    modela,
    data_loader,
    None,
    num_epochs=50,
    #lrs=[1e-5,1e-4,1e-3,1e-3,1e-4,1e-5],
    #lrs=[1e-4,1e-4,1e-4,1e-4],
    lrs=[1e-5,1e-5,1e-5,1e-5],
    wt=1e-3,
    #report_interval=10,
    report_interval=1,
    do_plot=True,
    test_accuracy=True,
)


adata = sc.AnnData(X = test_data.flatten(1).detach().numpy())
adata.obs['labels'] = test_labels.argmax(-1).numpy().astype(str)

output = modela(test_data, )
adata.obsm["z"] = output["mu_z"].detach().numpy()

adata.obs["pred"] = output["q_y"].detach().argmax(-1).numpy().astype(str)

sc.pp.neighbors(adata, n_neighbors=15, use_rep="z")
sc.tl.umap(adata,)

sc.settings.set_figure_params(
    #dpi=80,
    facecolor="white",
)

sc.pl.umap(adata, color = "labels")
sc.pl.umap(adata, color = "pred")

sc.pl.umap(adata, color = ["labels", "pred"],)

modela.load_state_dict(
        torch.load("./results/1660135620.387979modela_state.pt",))
ut.checkCosineDistance(x, modela) #0.9

plt.cla()
plt.clf()

plt.savefig("./temp_figure.png",)

plt.savefig("./tmp.png",)

plt.close()

d = ut.loadModelParameter("./results/1660132764.221625model_params.pt")

model1 = M16.VAE_GMM_Type1603Temp(
    nx=28**2,
    nclasses=10,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=6,
    nw=6,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.3,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    reclosstype="Bernoulli",
    #reclosstype="mse",
    relax=False,
    use_resnet=True,
    restrict_w=True,
    restrict_z=True,
    #do_cc=True,
    #cc_radius=0.03,
    #cc_scale=1e1,
)
model1.apply(init_weights)
model2 = M16.VAE_GMM_Type1603Temp(
    nx=28**2,
    nclasses=10,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=6,
    nw=6,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.3,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    reclosstype="Bernoulli",
    #reclosstype="mse",
    relax=False,
    use_resnet=True,
    restrict_w=True,
    restrict_z=True,
    #do_cc=True,
    #cc_radius=0.03,
    #cc_scale=1e1,
)
model2.apply(init_weights)

model = M16.VAE_GMM_Type1603Temp(
    nx=28**2,
    nclasses=10,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=6,
    nw=6,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.3,
    bn=True,
    yscale=1e0,
    zscale=1e0,
    wscale=1e0,
    reclosstype="Bernoulli",
    #reclosstype="mse",
    relax=False,
    use_resnet=True,
    restrict_w=True,
    restrict_z=True,
    do_cc=True,
    cc_radius=0.05,
    cc_scale=1e1,
)
model.apply(init_weights)

Train.basicDuoTrainLoop(
        model1,
        model2,
        data_loader,
        test_loader,
        num_epochs=50,
        lrs = [1e-4,1e-4,],
        wt=1e-3,
        report_interval=1,
        do_plot=True,
        test_accuracy=True,
        mi_scale=1e1,
        )

Train.basicTandemTrainLoop(
        model1,
        model2,
        data_loader,
        test_loader,
        num_epochs=50,
        lrs = [1e-3,1e-4,1e-5,],
        wt=1e-3,
        report_interval=1,
        do_plot=True,
        test_accuracy=True,
        mi_scale=1e1,
        )
