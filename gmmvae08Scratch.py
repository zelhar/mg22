#import gdown
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
print(torch.cuda.is_available())


sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=120, facecolor='white', )

enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc_ct = LabelEncoder()

## testing gmmvae08
adata = sc.read("./data/gtex_v7_SMTS.h5ad")
sc.pp.filter_cells(adata, min_genes=200, inplace=True,)
sc.pp.filter_genes(adata, min_cells=20, inplace=True,)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True,)
sc.pp.log1p(adata, )

data = torch.FloatTensor(adata.X)
enc_ct.fit(adata.obs["smts"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["smts"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=256,
        shuffle=True,
        )

sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.tsne(adata, )
sc.tl.leiden(adata, )
sc.tl.louvain(adata, )

sc.pl.umap(adata, color=["smts", "leiden", "louvain"],)
sc.pl.tsne(adata, color=["smts", "leiden", "louvain"],)



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

model = M7.VAE_Dirichlet_Type705(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        nclasses=enc_ct.classes_.size,
        )

model = M7.VAE_Stacked_Dilo_Anndata_Type701(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        nclasses=enc_ct.classes_.size + 5,
        )


model = M8.VAE_Dirichlet_Type805(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        dirscale=1e0,
        concentration=1e-1,
        nclasses=enc_ct.classes_.size,
        )

model = M8.VAE_Dirichlet_Type805(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        #dirscale=1e2,
        dirscale=1e0,
        #concentration=1e-5,
        concentration=1e0,
        nclasses=enc_ct.classes_.size+3,
        )

model = M8.VAE_Dirichlet_Type806(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=25,
        dirscale=1e0,
        concentration=5e0,
        nclasses=enc_ct.classes_.size + 5,
        )

model.apply(init_weights)

M8.basicTrain(model, data_loader, test_loader,
        num_epochs=22, wt=1e-3, lr=1e-3, report_interval=1)

M8.trainSemiSuper(model, labeled_loader, unlabeled_loader, test_loader, 
        num_epochs=63, lr=1e-3, wt=0, do_unlabeled=True, report_interval=5,)

output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()
adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.louvain(adata, )
sc.tl.leiden(adata, )
sc.pl.umap(adata, color=["smts", "predict", "predict2"],)

sc.pl.umap(adata, color=["smts", "leiden"],)
sc.pl.umap(adata, color=["smts", "louvain"],)

plt.close()

(output["q_y"].argmax(-1) - labels.argmax(-1)).float().mean()

Dy = distributions.Dirichlet(output["d_logits"].exp().clamp(1e-7, 1e7))
qy = output["q_y"]

Qx = distributions.Normal(qy, 1)



### gtex 9
adata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad")

#sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)

data = torch.FloatTensor(adata.X.toarray())

enc_ct.fit(adata.obs["Granular cell type"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs["Granular cell type"]))
labels = F.one_hot(labels.long(), num_classes=enc_ct.classes_.size).float()
dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128*2,
        shuffle=True,
        )

sc.pl.umap(adata, color=["Cell types level 3",], size=5,)
sc.pl.umap(adata, color=["Cell types level 2",], size=5,)
sc.pl.umap(adata, color=["leiden", ], size=5,)
sc.pl.umap(adata, color=["Broad cell type numbers", ], size=5,)
sc.pl.umap(adata, color=["Granular cell type"], size=5)
#colorbar_loc='left', lgend_loc='best',)



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



model = M8.VAE_Dirichlet_Type806(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        dirscale=1e0,
        concentration=1e0,
        nclasses=enc_ct.classes_.size,
        )

# this one worked pretty ok
model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        dirscale=1e1,
        concentration=1e-2,
        nclasses=enc_ct.classes_.size,
        )

model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        dirscale=1e0,
        concentration=1e0,
        nclasses=6,
        #nclasses=enc_ct.classes_.size,
        )


M8.basicTrain(model, data_loader, test_loader,
        num_epochs=3, wt=0e-3, lr=1e-3, report_interval=1)

torch.cuda.empty_cache()

torch.save(model.state_dict(), "./results/tempgmm.pth")

model.load_state_dict(torch.load("./results/tempgmm.pth"))

model.eval()
output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()
adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
del(output)

sc.pl.umap(adata, color=["Cell types level 3",], size=5,)
sc.pl.umap(adata, color=["Cell types level 2",], size=5,)
sc.pl.umap(adata, color=["Broad cell type",], size=5,)
sc.pl.umap(adata, color=["Granular cell type"], size=5,)
sc.pl.umap(adata, color=["predict",], size=5,)
sc.pl.umap(adata, color=["predict2",], size=5,)

sc.tl.louvain(adata, )
sc.tl.leiden(adata, )
sc.pl.umap(adata, color=["leiden",], size=5,)
sc.pl.umap(adata, color=["louvain",], size=5,)


## MNIST semisuper tests

transform = transforms.Compose([
    transforms.ToTensor(),
    ])
test_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=False,
       download=True,
       transform=transform,
       ),
   batch_size=128,
   shuffle=True,
)
train_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=True,
       download=True,
       transform=transform,
       ),
   batch_size=128,
   shuffle=True,
)
test_data = test_loader.dataset.data.float()/255
test_labels = F.one_hot(test_loader.dataset.targets.long(),
        num_classes=10,).float()
train_data = train_loader.dataset.data.float()/255
train_labels = F.one_hot(
        train_loader.dataset.targets.long(),
        num_classes=10,).float()


dataset = ut.SynteticDataSet(
        train_data, train_labels,)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        )

labeled_set = ut.SynteticDataSet(
        train_data[:2000], train_labels[:2000],)
labeled_loader = torch.utils.data.DataLoader(
        dataset=labeled_set,
        shuffle=True,
        batch_size=128,
        )
unlabeled_set = ut.SynteticDataSet(
        train_data[2000:], train_labels[2000:],)
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_set,
        shuffle=True,
        batch_size=128,
        )
test_set = ut.SynteticDataSet(
        test_data, test_labels,)
testloader = torch.utils.data.DataLoader(
        dataset=test_set,
        shuffle=True,
        batch_size=128,
        )

adata = sc.AnnData(X=train_data.detach().flatten(1).numpy(), )
adata.obs["labels"] = train_loader.dataset.targets.numpy().astype(str)

sc.pp.pca(adata,)
sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=10,)
sc.tl.umap(adata,)
#sc.tl.tsne(adata, )
sc.tl.leiden(adata, )
sc.tl.louvain(adata, )
sc.pl.umap(adata, color=["labels", "leiden", "louvain"],)
#sc.pl.tsne(adata, color=["labels", "leiden", "louvain"],)


enc_ct.fit(adata.obs["labels"])
data = torch.FloatTensor(adata.X)

model = M8.VAE_Dirichlet_Type805(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        dirscale=1e0,
        concentration=1e-2,
        nclasses=enc_ct.classes_.size,
        )

model = M8.VAE_Dirichlet_Type806(
        nx=adata.n_vars, nh=2024,
        nz=24, nw=15,
        dirscale=1e1,
        concentration=1e-2,
        nclasses=enc_ct.classes_.size,
        )

model = M7.VAE_Dirichlet_Type705(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        nclasses=enc_ct.classes_.size,
        )

model = M6.VAE_Dilo_Type601(
        nx=28**2, nh=1024, nz=64, nw=15, nclasses=10,
        bn=True
        )

model = M6.VAE_Dirichlet_Type05(
        nx=28**2, nh=1524, nz=64, nw=25, nclasses=10,
        )

model = M7.AE_Primer_Type701()

model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=15,
        dirscale=1e2,
        concentration=1e-1,
        nclasses=2*enc_ct.classes_.size,
        )

model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=1024,
        nz=24, nw=15,
        dirscale=1e1,
        concentration=1e-3,
        nclasses=2*enc_ct.classes_.size,
        )

model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=2024,
        nz=14, nw=5,
        dirscale=1e1,
        concentration=1e-3,
        nclasses=16,
        )

# this one was not so bad
model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=2024,
        nz=11, nw=3,
        dirscale=1e1,
        concentration=1e0,
        nclasses=16,
        )

model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=3024,
        nz=5, nw=4,
        dirscale=1e2,
        concentration=1e0,
        nclasses=26,
        )

model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=3024,
        nz=18, nw=5,
        dirscale=1e0,
        concentration=5e-1,
        nclasses=10,
        )

# this one was not bad
model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=3024,
        nz=28, nw=9,
        dirscale=1e0,
        concentration=5e-1,
        nclasses=16,
        )

# okish
model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=3024,
        nz=28, nw=9,
        dirscale=1e0,
        concentration=5e-0,
        nclasses=18,
        )

model = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=3024,
        nz=48, nw=14,
        dirscale=1e1,
        concentration=1e-1,
        nclasses=18,
        zscale=1e-0,
        )

model = M8.VAE_Dirichlet_Type808(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=25,
        dirscale=1e0,
        concentration=1e-1,
        nclasses=18,
        zscale=1e-0,
        )

model = M8.VAE_Dirichlet_Type809(
        nx=adata.n_vars, nh=1024,
        nz=64, nw=25,
        nclasses=18,
        )

# not bad needs lots of training
model = M8.VAE_Dirichlet_Type810(
        nx=adata.n_vars, nh=1024,
        nz=150, nw=25,
        nclasses=18,
        dirscale=1e0,
        concentration=1e0,
        )

model = M8.VAE_Dirichlet_Type810(
        nx=adata.n_vars, nh=1024,
        nz=150, nw=45,
        nclasses=28,
        dirscale=1e0,
        concentration=1e-1,
        )

model = M8.VAE_Dirichlet_Type810(
        nx=adata.n_vars, nh=1024,
        nz=10, nw=5,
        nclasses=30,
        dirscale=1e1,
        concentration=5e-3,
        numhidden=2,
        )



M8.basicTrain(model, data_loader, test_loader,
        num_epochs=9, wt=0e-3, lr=1e-3, report_interval=3)

M8.basicTrain(model, data_loader, test_loader,
        num_epochs=20, wt=0e-3, lr=3e-4, report_interval=3)

M8.basicTrain(model, data_loader, test_loader,
        num_epochs=20, wt=0e-3, lr=1e-4, report_interval=3)

# okish after lots of training
M8.basicTrain(model, data_loader, test_loader,
        num_epochs=20, wt=0e-3, lr=3e-5, report_interval=3)

model.eval()

torch.cuda.empty_cache()

model.eval()
output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()
adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
del(output)

sc.pl.umap(adata, color=["labels"], size=5)
sc.pl.umap(adata, color=["predict", ],size=5)
sc.pl.umap(adata, color=["predict2", ],size=5)

sc.tl.louvain(adata, )
sc.tl.leiden(adata, )
sc.pl.umap(adata, color=["louvain", ],)
sc.pl.umap(adata, color=["leiden"],)

x,y = test_loader.__iter__().next()
output = model(x)
q_y = output["q_y"]
q_y.argmax(-1) - y
torch.threshold(q_y, 0.5, 0).sum(0)
torch.sum(
    torch.threshold(q_y, 0.5, 0).sum(0) > 0)



model.eval()
w = torch.zeros(1, model.nw)
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(1*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

w = model.w_prior.sample((5, ))
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(5*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

del(output)

model.fitAE(data_loader,)

model.eval()
adata.obsm["z"] = model.autoencode(data)[0].detach().numpy()
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)
sc.pl.umap(adata, color=["labels"], size=5)
sc.tl.louvain(adata, )
sc.tl.leiden(adata, )
sc.pl.umap(adata, color=["louvain", ], size=5,)
sc.pl.umap(adata, color=["leiden"], size=5,)


new_train_labels = F.one_hot(
        torch.tensor(
            adata.obs["louvain"].to_numpy().astype('long')
            ),
        num_classes=adata.obs["louvain"].cat.categories.size,).float()


new_train_labels = F.one_hot(
        torch.tensor(
            adata.obs["leiden"].to_numpy().astype('long')
            ),
        num_classes=adata.obs["leiden"].cat.categories.size,).float()

labeled_set = ut.SynteticDataSet(
        train_data[:2000], new_train_labels[:2000],)
labeled_loader = torch.utils.data.DataLoader(
        dataset=labeled_set,
        shuffle=True,
        batch_size=128,
        )
unlabeled_set = ut.SynteticDataSet(
        train_data[2000:], new_train_labels[2000:],)
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=unlabeled_set,
        shuffle=True,
        batch_size=128,
        )

model2 = M8.VAE_Dirichlet_Type807(
        nx=adata.n_vars, nh=3024,
        nz=28, nw=9,
        dirscale=1e0,
        concentration=5e-1,
        nclasses=23,
        )

model2 = M8.VAE_Dirichlet_Type810(
        nx=adata.n_vars, nh=1024,
        nz=28, nw=9,
        dirscale=1e0,
        concentration=5e-1,
        nclasses=adata.obs["louvain"].cat.categories.size,
        )

model2 = M8.VAE_Dirichlet_Type810(
        nx=adata.n_vars, nh=1024,
        nz=10, nw=5,
        nclasses=adata.obs["louvain"].cat.categories.size,
        dirscale=1e1,
        concentration=5e-3,
        numhidden=2,
        )


M8.trainSemiSuper(model2, labeled_loader, unlabeled_loader, test_loader, 
        num_epochs=53, lr=1e-3, wt=0, do_unlabeled=True, do_eval=False, report_interval=5,)

model2.eval()
output = model2(data)
adata.obsm["z"] = output["z"].detach().numpy()
adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy().astype(str)
adata.obs["predict2"] = output["d_logits"].detach().argmax(-1).numpy().astype(str)
sc.pp.neighbors(adata, use_rep="z", n_neighbors=10,)
sc.tl.umap(adata,)

sc.pl.umap(adata, color=["labels"], size=5)
sc.pl.umap(adata, color=["predict", ],size=5)
sc.pl.umap(adata, color=["predict2", ],size=5)

sc.pl.umap(adata, color=["leiden", ],size=5)

sc.tl.louvain(adata, copy=False)
sc.pl.umap(adata, color=["louvain", ],size=5)

model2.eval()
w = torch.zeros(1, model.nw)
z = model2.Pz(w)
mu = z[:,:,:model.nz].reshape(1*model.nclasses, model.nz)
rec = model2.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

