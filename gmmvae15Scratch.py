# import gdown
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# import os
import pandas as pd

# import pyro
# import pyro.distributions as pyrodist
import scanpy as sc
import seaborn as sns

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

adata = sc.AnnData(
    X=train_data.detach().flatten(1).numpy(),
)
adata.obs["labels"] = train_dataset.targets.numpy().astype(str)
bdata = sc.AnnData(
    X=test_data.detach().flatten(1).numpy(),
)
bdata.obs["labels"] = test_dataset.targets.numpy().astype(str)


#### models

model1500a = M15.AE_Type1500(
    nx=28 ** 2,
    nh=1024,
    nhq=1024,
    nhp=1024,
    nz=64,
    zscale=1,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    bn=True,
    reclosstype="Bernoulli",
    use_resnet=False,
    eps=1e-7,
)

model501a = M15.VAE_AE_Type1501(
    nx=28 ** 2,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    reclosstype="Bernoulli",
    restrict_w=False,
    use_resnet=False,
)

model1502a = M15.VAE_Dirichlet_GMM_Type1502(
    nx=28 ** 2,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    reclosstype="Bernoulli",
    #reclosstype="mse",
    #restrict_w=False,
    restrict_w=True,
    use_resnet=False,
    nclasses=10,
    #nclasses=30,
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    #concentration=1e0,
    concentration=2.5e0,
    temperature=0.1,
    #do_cc=True,
    softargmax=False,
    activation=nn.LeakyReLU(),
        )

model1502b = M15.VAE_Dirichlet_GMM_Type1502(
    nx=28 ** 2,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    reclosstype="Bernoulli",
    #reclosstype="mse",
    #restrict_w=False,
    restrict_w=True,
    use_resnet=False,
    #nclasses=10,
    nclasses=30,
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    concentration=1.5e0,
    #concentration=1e0,
    temperature=0.1,
    #do_cc=True,
    softargmax=False,
        )


model1503a = M15.VAE_GMM_Type1503(
    nx=28 ** 2,
    nh=1024,
    nhp=1024,
    nhq=2*1024,
    #nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    reclosstype="Bernoulli",
    #restrict_w=True,
    use_resnet=False,
    #nclasses=10,
    nclasses=30,
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    temperature=0.1,
    #do_cc=True,
    softargmax=False,
        )

model1503b = M15.VAE_GMM_Type1503(
    nx=28 ** 2,
    nh=1024,
    nhp=1024,
    nhq=2*1024,
    #nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    reclosstype="Bernoulli",
    #restrict_w=True,
    use_resnet=False,
    #nclasses=10,
    nclasses=30,
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    temperature=0.1,
    #do_cc=True,
    softargmax=False,
        )

model = M14.VAE_GMM_Type1402(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=30,
        wscale=1,
        yscale=1, #
        zscale=1,
        mi_scale=1,
        cc_scale=1e1,
        concentration=5e0,
        numhidden=2,
        numhiddenp=2,
        numhiddenq=2,
        dropout=0.3,bn=True,
        reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        restrict_w=False,
        )

model.apply(init_weights)


model = M15.VAE_GMM_Type1503(
    nx=28 ** 2,
    nh=3*1024,
    nhp=3*1024,
    nhq=3*1024,
    nz=20,
    nw=15,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.25,
    #reclosstype="Bernoulli",
    reclosstype="mse",
    restrict_w=True,
    use_resnet=False,
    #nclasses=10,
    nclasses=20,
    yscale=1e1,
    mi_scale=1e0,
    cc_scale=1e1,
    temperature=0.1,
    #do_cc=True,
    softargmax=False,
        )
model.apply(init_weights)

torch.save(
        model.state_dict(),
        "./results/model1503_20z_15w_20cl.state2.pt",
        )

Train.basicTrainLoop(
        #model1502a,
        #model1503a,
        model,
        data_loader,
        None,
        num_epochs=10,
        #lrs=[1e-4,1e-3,1e-4,1e-5,],
        lrs=[1e-4,1e-3,1e-3,1e-4,1e-5,],
        wt=0e-3,
        report_interval=4,
        do_plot=True,
        test_accuracy=True,
        )

model1503a.zsclae = 1.5e0
model1503a.cc_scale=1.5e1
model1503a.zsclae = 1.0e0
model1503a.cc_scale=0e1

model1503a.restrict_w=True

Train.basicTrainLoop(
        model1503a,
        data_loader,
        None,
        num_epochs=10,
        #lrs=[1e-3,1e-4,1e-5,],
        lrs=[1e-4,1e-5,],
        wt=0e-3,
        report_interval=4,
        do_plot=True,
        test_accuracy=True,
        )

ut.estimateClusterImpurityLoop(
        model1503a, test_data, test_labels, "cpu",
        )

torch.save(
        model1503a.state_dict(),
        "./results/model1503_a.state1.pt",
        )


#Train.basicDuoTrainLoop(
Train.basicTandemTrainLoop(
        #model1503a,
        #model1503b,
        model1502a,
        model1502b,
        data_loader,
        None,
        num_epochs=15,
        #lrs=[1e-4,1e-3,1e-5,],
        lrs=[1e-3,],
        wt=0e-3,
        report_interval=4,
        do_plot=True,
        test_accuracy=True,
        mi_scale=5e1,
        )

torch.save(
        model1503a.state_dict(),
        "./results/model1503_a.state2.pt",
        )

torch.save(
        model1502a.state_dict(),
        "./results/model1502_a_30classes.state2.pt",
        )

torch.save(
        model1502a.state_dict(),
        "./results/model1502_a_10classes_semisuper.state2.pt",
        )


r,p,s = ut.estimateClusterImpurityLoop(model1502a, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())



# semi supervised
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

model1502a.apply(init_weights)

Train.trainSemiSuperLoop(
        model1502a,
        labeled_data_loader,
        unlabeled_data_loader,
        test_loader,
        num_epochs=50,
        lrs = [1e-3,1e-4,1e-5,],
        wt=0,
        do_unlabeled=True,
        do_plot=True,
        do_validation=True,
        test_accuracy=True,
        report_interval=5,
        )


## fashiomnist

transform = transforms.Compose([
    transforms.ToTensor(),
    #binarize(threshold=0.08),
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

subset = ut.randomSubset(s=len(train_labels), r=0.051)
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

model1503c = M15.VAE_GMM_Type1503(
    nx=28 ** 2,
    nh=1024,
    nhp=1024,
    nhq=2*1024,
    #nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    reclosstype="Bernoulli",
    #restrict_w=True,
    use_resnet=False,
    nclasses=10,
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    temperature=0.1,
    #do_cc=True,
    softargmax=False,
    activation=nn.LeakyReLU(),
        )

model1503d = M15.VAE_GMM_Type1503(
    nx=28 ** 2,
    nh=1024,
    nhp=1024,
    nhq=2*1024,
    #nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    reclosstype="Bernoulli",
    #restrict_w=True,
    use_resnet=False,
    nclasses=10,
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    temperature=0.1,
    #do_cc=True,
    softargmax=False,
    activation=nn.LeakyReLU(),
        )

model1503c.apply(init_weights)

model1502a.apply(init_weights)

model1503d.apply(init_weights)

Train.trainSemiSuperLoop(
        model1503c,
        #model1502a,
        #model1503d,
        labeled_data_loader,
        unlabeled_data_loader,
        test_loader,
        num_epochs=50,
        lrs = [1e-3,1e-4,1e-5,],
        wt=0,
        do_unlabeled=True,
        do_plot=True,
        do_validation=True,
        test_accuracy=True,
        report_interval=5,
        )

torch.save(
        model1503c.state_dict(),
        "./results/model1503c_10classes_semisuper_fashiob.state0.pt",
        )

#r,p,s = ut.estimateClusterImpurityLoop(model1503c, test_data, test_labels, "cuda", )
r,p,s = ut.estimateClusterImpurityLoop(model1502a, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

torch.save(
        model1502a.state_dict(),
        "./results/model1502a_10classes_semisuper_fashiob.state0.pt",
        )



#################################################################################
### RNA seq
#################################################################################

adata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)

sc.pp.highly_variable_genes(adata, n_top_genes=1000, inplace=True, subset=True,)

adata.obs.columns
# "Granular cell type"
# "Broad cell type"
# "Broad cell type numbers"
# "Cell types level 2"
# "Cell types level 3"


data = torch.FloatTensor(adata.X.toarray())

#enc_ct.fit(adata.obs["Granular cell type"])
enc_ct.fit(adata.obs["Broad cell type"])
#labels = torch.IntTensor(
#        enc_ct.transform(adata.obs["Granular cell type"]))
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["Broad cell type"]))
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

model1500a = M15.AE_Type1500(
    nx=adata.n_vars,
    nh=1024,
    nhq=1024,
    nhp=1024,
    nz=64,
    zscale=1,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    bn=True,
    use_resnet=False,
    eps=1e-7,
)
model1500a.apply(init_weights)

model1501a = M15.VAE_AE_Type1501(
    nx=adata.n_vars,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    restrict_w=False,
    use_resnet=False,
)
model1501a.apply(init_weights)

model1502a = M15.VAE_Dirichlet_GMM_Type1502(
    nx=adata.n_vars,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    use_resnet=False,
    nclasses=labels.shape[1],
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    concentration=5e0,
    temperature=0.1,
    restrict_w=True,
    #do_cc=True,
    softargmax=False,
    activation=nn.LeakyReLU(),
        )
model1502a.apply(init_weights)

model1503a = M15.VAE_GMM_Type1503(
    nx=adata.n_vars,
    nh=1024,
    nhp=1024,
    nhq=1024,
    nz=64,
    nw=32,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=4,
    dropout=0.25,
    use_resnet=False,
    nclasses=labels.shape[1],
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    temperature=0.1,
    restrict_w=True,
    #do_cc=True,
    softargmax=False,
    activation=nn.LeakyReLU(),
        )
model1503a.apply(init_weights)


torch.save(
        model1503a.state_dict(),
        "./results/model1503a_44classes_rna.state0.pt",
        )

Train.basicTrainLoop(
        #model=model1500a,
        #model=model1501a,
        #model=model1502a,
        model=model1503a,
        train_loader=unlabeled_loader,
        test_loader=None,
        num_epochs=10,
        lrs=[1e-5, 1e-4, 1e-3, 1e-3, 1e-4, 1e-5],
        #lrs=[1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
        wt=0,
        report_interval=5,
        test_accuracy=True,
        )


model = M7.VAE_Dirichlet_Type705(nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        #nclasses=enc_ct.classes_.size,
        nclasses=labels.shape[1],
        )
model.apply(init_weights)

model = M15.VAE_GMM_Type1503(
    nx=adata.n_vars,
    nh=1024*3,
    nhp=1024*3,
    nhq=1024*3,
    nz=44,
    nw=22,
    numhidden=2,
    numhiddenp=2,
    numhiddenq=2,
    dropout=0.2,
    use_resnet=False,
    nclasses=labels.shape[1]*2,
    yscale=1e0,
    mi_scale=1e0,
    cc_scale=1e1,
    temperature=0.1,
    restrict_w=True,
    reclosstype="mse",
    #do_cc=True,
    softargmax=False,
    activation=nn.LeakyReLU(),
        )
model.apply(init_weights)

Train.basicTrainLoop(
        #model=model1500a,
        #model=model1501a,
        #model=model1502a,
        #model=model1503a,
        model=model,
        train_loader=unlabeled_loader,
        test_loader=None,
        num_epochs=30,
        lrs=[1e-5,1e-4,1e-3, 1e-4, 1e-5],
        #lrs=[1e-5, 1e-5, 1e-5, 1e-5, 1e-5],
        wt=0,
        report_interval=5,
        test_accuracy=True,
        )

torch.save(
        model.state_dict(),
        "./results/model7705_44classes_rna.state0.pt",
        )

torch.save(
        model.state_dict(),
        "./results/model15003_88classes_rna.state0.pt",
        )


model.eval()

output = model(data)

adata.obsm["z"] = output["z"].detach().numpy()

adata.obs["pred"] = output["q_y"].detach().argmax(-1).numpy().astype(str)

sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)

del output


sc.pl.umap(adata, color=["Broad cell type",],)

sc.pl.umap(adata, color=["pred",],)

sc.pl.umap(adata, color=["Broad cell type", "pred",],)

plt.savefig("./temp_pca_broad.png",)


def foo(x : str="x"):
    if x == "a":
        print("foo")
    else:
        pass
    print(x)

net1 = ut.buildNetworkv2(
        [500, 60, 60, 160],
        0.3,
        nn.ReLU(),
        True,
        )
net1

net2 = ut.buildNetworkv5(
        [500, 60, 60, 160,],
        0.3,
        nn.ReLU(),
        True,
        )
net2

#################################################################################
### RNA seq semi sup
#################################################################################

adata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)
#sc.pp.highly_variable_genes(adata, n_top_genes=1000, inplace=True, subset=True,)

adata.obs.columns
# "Granular cell type"
# "Broad cell type"
# "Broad cell type numbers"
# "Cell types level 2"
# "Cell types level 3"


data = torch.FloatTensor(adata.X.toarray())

enc_ct.fit(adata.obs["Granular cell type"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["Granular cell type"]))
#enc_ct.fit(adata.obs["Broad cell type"])
#labels = torch.IntTensor(
#        enc_ct.transform(adata.obs["Broad cell type"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

##dataset = ut.SynteticDataSet(data, labels)
#labeled_loader = torch.utils.data.DataLoader(
#        dataset=ut.SynteticDataSet(data[:1600], labels[:1600]),
#        batch_size=256,
#        shuffle=True,
#        )
#unlabeled_loader = torch.utils.data.DataLoader(
#        dataset=ut.SynteticDataSet(data[1600:-1500], labels[1600:-1500]),
#        batch_size=256,
#        shuffle=True,
#        )
#test_loader = torch.utils.data.DataLoader(
#        dataset=ut.SynteticDataSet(data[-1500:], labels[-1500:]),
#        batch_size=256,
#        shuffle=True,
#        )

subset = ut.randomSubset(s=len(labels), r=0.1)

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


#model = M15.VAE_Dirichlet_GMM_Type1502(
model = M15.VAE_GMM_Type1503(
        nx=adata.n_vars,
        nz=64,
        nw=32,
        nclasses=labels.shape[1],
        #concentration=1e0,
        dropout=0.2,
        relax=False,
        #restrict_w=True,
        #restrict_z=True,
        )
model.apply(init_weights)

Train.trainSemiSuperLoop(
        model,
        labeled_loader,
        unlabeled_loader,
        labeled_loader,
        num_epochs=50,
        lrs=[1e-4,1e-3,1e-3,],
        wt=0,
        do_unlabeled=True,
        do_plot=False,
        do_validation=False,
        test_accuracy=True,
        report_interval=25,
        )

Train.trainSemiSuperLoop(
        model,
        labeled_loader,
        unlabeled_loader,
        labeled_loader,
        num_epochs=5,
        lrs=[1e-3,1e-4,1e-5,],
        wt=0,
        do_unlabeled=True,
        do_plot=False,
        do_validation=False,
        test_accuracy=True,
        report_interval=2,
        )


model.eval()
output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()
adata.obs["pred"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
del output
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10)
sc.tl.umap(adata,)
sc.tl.louvain(adata,)

sc.pl.umap(adata, color=["pred", "Granular cell type",],)
plt.savefig("./temp_pred_granular.png",)

sc.pl.umap(adata, color=["pred", "louvain",],)
plt.savefig("./temp_pred_louvain.png",)

torch.save(
        model.state_dict(),
        "./results/semi_super_1503_granular.pt",
        )

ut.is_jsonable(model.__dict__)
ut.is_pickleable(model.__dict__)
ut.is_serializeable(model.__dict__)

ut.saveModelParameters(
        model,
        "./results/semi_super_1503_model_params_granular.json",
        method="json",
        )

ut.saveModelParameters(
        model,
        "./results/semi_super_1503_model_params_granular.pth",
        method="not json",
        )

ut.loadModelParameter(
        "./results/semi_super_1503_model_params_granular.json",
        method="json",
        )
