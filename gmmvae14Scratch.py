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
import gmmvae14 as M14
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

@curry
def binarize(x : torch.Tensor, threshold : float = 0.25) -> torch.Tensor:
    ret = (x > threshold).float()
    return ret

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

model = M14.VAE_Dirichlet_Type1400(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        #nclasses=20,
        nclasses=30,
        dscale=1,wscale=1,yscale=1,zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=5e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
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

model2 = M14.VAE_Dirichlet_Type1400(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        #nclasses=20,
        nclasses=30,
        dscale=1,wscale=1,yscale=1,zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=5e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        restrict_w=False,
        )
model2.apply(init_weights)

model = M14.VAE_GMM_Type1402D(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=20,
        wscale=1,
        yscale=0, #
        zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=5e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
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

torch.save(
        model.state_dict(),
        "./results/model_T1402D_nc20_yscale0.state1.pth",
        )

model2 = M14.VAE_GMM_Type1402D(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=20,
        wscale=1,
        yscale=0, #
        zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=5e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        restrict_w=False,
        )
model2.apply(init_weights)

torch.save(
        model2.state_dict(),
        "./results/model2_T1402D_nc20_yscale0.state1.pth",
        )

Train.basicTandemTrainLoop(
        model,
        model2,
        data_loader,
        None,
        num_epochs=20,
        lrs = [1e-4, 1e-3, 1e-4, 1e-5],
        wt=1e-5,
        report_interval=3,
        do_plot=True,
        test_accuracy=True,
        mi_scale=5e1,
        )

model.cc_scale=0
model2.cc_scale=0

model = M14.VAE_GMM_Type1402D_MNIST(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=16,
        nclasses=10,
        wscale=1,
        yscale=0, #
        zscale=1,
        mi_scale=1,
        cc_scale=1e1,
        concentration=5e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        #use_resnet=True,
        softargmax=False,
        eps=1e-9,
        restrict_w=True,
        )
model.apply(init_weights)


model2 = M14.VAE_GMM_Type1402D_MNIST(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=16,
        nclasses=10,
        wscale=1,
        yscale=0, #
        zscale=1,
        mi_scale=1,
        cc_scale=1e1,
        concentration=5e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        #use_resnet=True,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        restrict_w=True,
        )
model2.apply(init_weights)


x,y = test_loader.__iter__().next()
#model(x)

ut.plot_2images(
        binarize(x.reshape(-1,1,28,28), 0.08), 
        x.reshape(-1,1,28,28),
        nrow=16,
        #transform=binarize(threshold=0.1),
        )



ut.do_plot_helper(model2, device="cpu")
ut.test_accuracy_helper(model2, x, y, device="cpu")

Train.basicTandemTrainLoop(
        model,
        model2,
        data_loader,
        None,
        num_epochs=20,
        lrs = [1e-4, 5e-4, 1e-3, 1e-4, 1e-5],
        wt=1e-5,
        report_interval=3,
        do_plot=True,
        test_accuracy=True,
        mi_scale=5e1,
        )

model = M14.AE_Type1405(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=20,
        wscale=1,
        yscale=0, #
        zscale=1,
        #mi_scale=1,cc_scale=1e1,
        #concentration=5e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        reclosstype="Bernoulli",
        #temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        restrict_w=False,
        )
model.apply(init_weights)


Train.basicTrainLoop(
        model,
        data_loader,
        None,
        num_epochs=20,
        lrs = [1e-5, 1e-4, 1e-4, 1e-4, 1e-5],
        wt=1e-4,
        report_interval=3,
        do_plot=False,
        test_accuracy=False,
        )


bdata = sc.AnnData(X=test_data.detach().flatten(1).numpy(),)
bdata.obs["labels"] = test_dataset.targets.numpy().astype(str)

sc.tl.pca(bdata, )
sc.pp.neighbors(bdata,)
sc.tl.umap(bdata,)
sc.tl.louvain(bdata, )
sc.pl.umap(bdata, color=["labels"], size=25)
sc.pl.umap(bdata, color=["louvain",], size=25,)

model.cpu()
model.eval()
output = model(test_data, )
bdata.obsm["z"] = output["z"].detach().numpy()
del output

sc.pp.neighbors(bdata, use_rep="z", n_neighbors=7,)
sc.tl.umap(bdata,)
sc.tl.louvain(bdata, )
sc.pl.umap(bdata, color=["labels"], size=25)
sc.pl.umap(bdata, color=["louvain",], size=25,)



#### fashionmnist

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

model = M14.AE_Type1405(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=20,
        wscale=1,
        yscale=0, #
        zscale=1,
        #mi_scale=1,cc_scale=1e1,
        #concentration=5e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        #reclosstype="Bernoulli",
        #temperature=0.1,
        relax=False,
        use_resnet=True,
        softargmax=False,
        eps=1e-9,
        restrict_w=False,
        )
model.apply(init_weights)

model = M14.AE_Type1407(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        zscale=1,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        #reclosstype="Bernoulli",
        #temperature=0.1,
        use_resnet=True,
        eps=1e-9,
        )
model.apply(init_weights)

#model = M14.VAE_GMM_Type1402D(
model = M14.VAE_GMM_Type1402(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        #nclasses=20,
        nclasses=30,
        wscale=1,
        #yscale=0, #
        yscale=1, #
        zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=1e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        #reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        restrict_w=False,
        )
model.apply(init_weights)

torch.save(
        model.state_dict(),
        "./results/model_T1402_nc30_yscale1.state1.pth",
        )

model2 = M14.VAE_GMM_Type1402(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        #nclasses=20,
        nclasses=30,
        wscale=1,
        #yscale=0, #
        yscale=1, #
        zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=1e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        #reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        restrict_w=False,
        )
model2.apply(init_weights)


torch.save(
        model2.state_dict(),
        "./results/model2_T1402_nc30_yscale1.state1.pth",
        )

Train.basicTrainLoop(
        model,
        data_loader,
        None,
        num_epochs=20,
        lrs = [1e-5, 1e-4, 1e-3, 1e-4, 1e-5],
        wt=1e-4,
        report_interval=5,
        #do_plot=False,
        do_plot=True,
        #test_accuracy=False,
        test_accuracy=True,
        )

Train.basicTandemTrainLoop(
        model,
        model2,
        data_loader,
        None,
        #num_epochs=20,
        num_epochs=200,
        #lrs = [1e-4, 5e-4, 1e-3, 1e-4, 1e-5],
        lrs = [1e-4, 1e-4, 1e-4, 5e-5, 1e-5],
        wt=1e-5,
        report_interval=3,
        do_plot=True,
        test_accuracy=True,
        mi_scale=5e1,
        )

Train.basicDuoTrainLoop(
        model,
        model2,
        data_loader,
        None,
        num_epochs=10,
        #num_epochs=200,
        #lrs = [1e-4, 5e-4, 1e-3, 1e-4, 1e-5],
        lrs = [1e-4, 1e-4, 1e-4, 1e-4, 1e-5],
        #lrs = [1e-4, 1e-4, 1e-4, 5e-5, 1e-5],
        wt=1e-5,
        report_interval=3,
        do_plot=True,
        test_accuracy=True,
        mi_scale=5e1,
        )


r,p,s = ut.estimateClusterImpurityLoop(model, test_data, test_labels, "cuda", )
print(p, "\n", r.mean(), "\n", r)
print((r*s).sum() / s.sum())

modelT = M14.VAE_GMM_Type1408(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=20,
        #nclasses=30,
        wscale=1,
        #yscale=0, #
        yscale=1, #
        zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=1e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        #restrict_w=False,
        restrict_w=True,
        #mini_batch=16,
        mini_batch=8,
        )
model2.apply(init_weights)

Train.basicTrainLoop(
        modelT,
        data_loader,
        None,
        num_epochs=20,
        lrs = [1e-5, 1e-4, 1e-3, 1e-4, 1e-5],
        wt=1e-4,
        report_interval=5,
        #do_plot=False,
        #do_plot=True,
        #test_accuracy=False,
        test_accuracy=True,
        )


model1 = M14.VAE_GMM_Type1402(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=10,
        #nclasses=30,
        wscale=1,
        #yscale=0, #
        yscale=1, #
        zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=1e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        #reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        #restrict_w=False,
        restrict_w=True,
        #mini_batch=16,
        )
model1.apply(init_weights)
model2 = M14.VAE_GMM_Type1402(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=10,
        #nclasses=30,
        wscale=1,
        #yscale=0, #
        yscale=1, #
        zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=1e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        #reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        #restrict_w=False,
        restrict_w=True,
        #mini_batch=16,
        )
model2.apply(init_weights)
model3 = M14.VAE_GMM_Type1402(
        nx=28**2,
        nh=1024,
        nhq=1024*2,
        nhp=1024,
        nz=64,
        nw=32,
        nclasses=10,
        #nclasses=30,
        wscale=1,
        #yscale=0, #
        yscale=1, #
        zscale=1,
        mi_scale=1,cc_scale=1e1,
        concentration=1e0,
        numhidden=2,numhiddenp=2,numhiddenq=2,
        dropout=0.3,bn=True,
        reclosstype="Bernoulli",
        temperature=0.1,
        relax=False,
        use_resnet=False,
        softargmax=False,
        eps=1e-9,
        #restrict_w=False,
        restrict_w=True,
        #mini_batch=16,
        )
model3.apply(init_weights)

Train.basicTripleTrainLoop(
        model1, model2, model3,
        data_loader,
        None,
        num_epochs=20,
        #num_epochs=200,
        lrs = [1e-4, 5e-4, 1e-3, 1e-4, 1e-5],
        #lrs = [1e-4, 1e-4, 1e-4, 1e-4, 1e-5],
        #lrs = [1e-4, 1e-4, 1e-4, 5e-5, 1e-5],
        wt=1e-5,
        report_interval=3,
        do_plot=True,
        test_accuracy=True,
        mi_scale=1e2,
        )


#best ever
modelA = M10.VAE_Dirichlet_Type1004R(
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

modelB = M10.VAE_Dirichlet_Type1004R(
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

#modelB.load_state_dict(
#        torch.load("./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.2.pt"),
#        )

modelA.load_state_dict(
        torch.load("./results/model_m10T004R_mnist_h1024_w32_z64_nh2_10C_B.state.2.pt"),
        )


#modelA = M14.VAE_GMM_Type1402_MNIST(
modelA = M14.VAE_Dirichlet_Type1400_MNIST(
        nx=28**2,
        nh=1024, 
        nhp=1024,
        nhq=1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        dropout=0.3,
        reclosstype="Bernoulli",
        bn=True,
        use_resnet=True,
        )

modelB = M14.VAE_GMM_Type1402(
        nx=28**2,
        nh=1024,
        nhp=1024,
        nhq=1024,
        nw=32,
        nz=64,
        nclasses=10,
        concentration=1.0e-0,
        numhidden=2,
        dropout=0.3,
        reclosstype="Bernoulli",
        bn=True,
        use_resnet=True,
        )

Train.basicTandemTrainLoop(
        modelA, 
        modelB, 
        data_loader,
        #num_epochs=50,
        #lrs = [1e-5, 5e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 5e-5,
        #    1e-5, 1e-5],
        num_epochs=500,
        lrs = [5e-5, 1e-4, 5e-4, 1e-4, 1e-4, 5e-5, 5e-5, 1e-5, 1e-5],
        wt=1e-4,
        report_interval=5,
        mi_scale=1e1,
        do_plot=True,
        test_accuracy=True,
        )

Train.basicDuoTrainLoop(
        modelA, 
        modelB, 
        data_loader,
        #num_epochs=50,
        #lrs = [1e-5, 5e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 5e-5,
        #    1e-5, 1e-5],
        num_epochs=500,
        lrs = [5e-5, 1e-4, 5e-4, 1e-4, 1e-4, 5e-5, 5e-5, 1e-5, 1e-5],
        wt=5e-4,
        report_interval=5,
        mi_scale=5e1,
        do_plot=True,
        test_accuracy=True,
        )

ut.do_plot_helper(modelB, )

ut.do_plot_helper(modelA, )

modelA.apply(init_weights)

Train.basicTrainLoop(
        modelA,
        data_loader,
        #num_epochs=50,
        #lrs = [1e-5, 5e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 5e-5,
        #    1e-5, 1e-5],
        num_epochs=50,
        lrs = [5e-5, 1e-4, 5e-4, 1e-3, 1e-3, 5e-4, 1e-4, 1e-4, 5e-5, 5e-5, 1e-5, 1e-5],
        wt=5e-4,
        report_interval=5,
        do_plot=True,
        test_accuracy=True,
        )

M10.basicTrainLoop(
        #model, 
        modelA, 
        data_loader,
        #num_epochs=50,
        #lrs = [1e-5, 5e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 5e-5, 5e-5,
        #    1e-5, 1e-5],
        num_epochs=30,
        lrs = [1e-5, 1e-4, 1e-3, 1e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 1e-5],
        wt=0e0,
        report_interval=3,
        do_plot=True,
        test_accuracy=True,
        )
