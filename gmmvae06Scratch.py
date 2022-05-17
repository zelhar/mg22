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
print(torch.cuda.is_available())


sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=120, facecolor='white', )

enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc_ct = LabelEncoder()

model = M6.AE_Type600(nx=28**2, nz=20, nclasses=10, nh=2000,)


X,y = skds.make_moons(n_samples=20000, noise=3e-2, )
X = torch.FloatTensor(X)
y = torch.IntTensor(y)

adata = sc.AnnData(X=X.numpy(), )
adata.obs["y"] = y.numpy()

sc.tl.tsne(adata, )
sc.pl.tsne(adata, color="y")
sc.pp.neighbors(adata, n_neighbors=10, )

sc.tl.umap(adata, )
sc.pl.umap(adata, color="y")

sc.pl.scatter(adata, color="y", x='0',y='1')

dataset = ut.SynteticDataSet(X, y)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=256,
        )

model = M6.AE_Type600(nx=2, nh=2000, nz=10, nclasses=2,)

M6.basicTrain(model, data_loader, num_epochs=3, )

M6.basicTrain(model, data_loader, num_epochs=100, )

model.cpu()
output = model(X)

adata.obsm["z"] = output["z"].detach().numpy()

Xd = pd.DataFrame(X.numpy())
rec = pd.DataFrame(output['rec'].detach().numpy())

sns.scatterplot(data=pd.DataFrame(output['rec'].detach().numpy()), x=0, y=1, hue=y)

sns.scatterplot(data=pd.DataFrame(output['rec'].detach().numpy()), x=0, y=1,
        hue=output["q_y"].argmax(-1))

cy = torch.eye(model.nclasses)
c = model.Px(model.Ez(cy)).detach().numpy()
c = pd.DataFrame(c)

m = model.Px(model.mz).detach().numpy()
m = pd.DataFrame(m)

sc.pp.neighbors(adata, n_neighbors=10, use_rep="z",)

sns.scatterplot(data=rec, x=0, y=1,
        hue=output["q_y"].argmax(-1))

sns.scatterplot(data=Xd, x=0, y=1,
        hue=output["q_y"].argmax(-1))

sns.scatterplot(data=m, x=0, y=1,
        hue=[2,3])

sns.scatterplot(data=c, x=0, y=1,
        hue=[4,5])

plt.cla()

m = torch.randn(10, 2)
z = torch.randn(128, 2)
def studentize(m, z):
    q = 1 + (m - z.unsqueeze(1)).pow(2).sum(-1)
    q = 1 / q
    s = q.sum(-1, keepdim=True,)
    q = q/s
    return q
q= studentize(m,z)
q.shape

adata = sc.datasets.blobs(1000, 5, n_observations=19000)
dataset = ut.SynteticDataSet(
        torch.tensor(adata.X).float(),
        torch.tensor(adata.obs["blobs"].apply(float)),
        )

data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=256,
        )

model = M6.AE_Type600(nx=1000, nh=3000, nz=100, nclasses=5,)

#model = M4.AE_Type00(nx=1000, nz=10)

M6.basicTrain(model, data_loader, num_epochs=30, )



output = model(dataset.data)

rec = output['rec'].detach().numpy()
z = output["z"].detach().numpy()
adata.obsm["z"] = z

sc.pp.neighbors(adata, n_neighbors=10, use_rep="z")
sc.tl.umap(adata, )

adata.obsm["q_y"] = output["q_y"].detach().numpy()

adata.obs["predict"] = output["q_y"].detach().argmax(-1).numpy()

sc.pl.umap(adata, color="predict")

cy = torch.eye(model.nclasses)

c = model.Px(model.Ez(cy)).detach().numpy()
c = pd.DataFrame(c)

m = model.Px(model.mz).detach().numpy()
m = pd.DataFrame(m)

m = model.Px(model.mz).detach()
model.Qy(m).argmax(-1)

c = model.Px(model.Ez(cy)).detach()
model.Qy(c).argmax(-1)


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

model = M6.VAE_Dilo_Type601(nx=28**2, nh=1200, nz=30, nw=25, nclasses=10,
        bn=True)

model = M6.VAE_Dirichlet_Type05(nx=28**2, nh=1024, nz=30, nw=25, nclasses=10,)


M6.trainSemiSuper(model, labeled_loader, unlabeled_loader, testloader, 
        num_epochs=100, lr=1e-3, wt=0, do_unlabeled=True,)

M6.basicTrain(model, unlabeled_loader, testloader,
        num_epochs=15, lr=1e-3, wt=1e-4, )

M6.basicTrain(model, unlabeled_loader, testloader,
        num_epochs=15, lr=1e-3, wt=0, )

model.fit(unlabeled_loader, num_epochs=13, lr=1e-3,)

model.eval()

x,y = test_loader.__iter__().next()
output = model(x)
q_y = output["q_y"]
q_y.argmax(-1) - y


w = torch.zeros(20, model.nw)
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(20*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

w = model.w_prior.sample((5, ))
z = model.Pz(w)
mu = z[:,:,:model.nz].reshape(5*model.nclasses, model.nz)
rec = model.Px(mu).reshape(-1,1,28,28)
ut.plot_images(rec, model.nclasses)

model = M6.AE_Type600(nx=28**2, nh=1024, nz=30, nclasses=20,)

model = M6.AE_Type603(nx=28**2, nh=1024, nz=30, nclasses=20)

model = M6.AE_Type603(nx=28**2, nh=1024, nz=30, nclasses=10)

model.eval()
c = torch.eye(model.nclasses,)
cz = model.Ez(c)
cx = model.Px(cz).reshape(-1,1,28,28)
ut.plot_images(cx, nrow=4,)



### Back to RNAseq

adata = sc.datasets.paul15()
adata.obs['celltype']=adata.obs['paul15_clusters'].str.split("[0-9]{1,2}", n = 1, expand = True).values[:,1]
adata.obs['celltype2']=adata.obs['paul15_clusters']

sc.pl.highest_expr_genes(adata, n_top=20,)

sc.pp.filter_cells(adata, min_genes=200, inplace=True,)
sc.pp.filter_genes(adata, min_cells=100, inplace=True)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True)
sc.pp.log1p(adata, )
sc.pp.highly_variable_genes(adata, min_mean=0.0125, min_disp=0.5, max_mean=3.0,
        subset=True, inplace=True, )
#sc.pp.highly_variable_genes(adata, min_mean=0.0125, min_disp=0.5, max_mean=3.0,
#        subset=True, inplace=True, n_top_genes=1500,)
sc.pp.scale(adata,max_value=10,)

sc.tl.pca(adata, svd_solver="arpack",)

sc.pl.pca(adata, color="paul15_clusters",)
plt.tight_layout()

sc.pl.pca(adata, color="celltype",)

sc.pl.pca_variance_ratio(adata, log=True, )

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, )
sc.tl.leiden(adata, )

sc.tl.umap(adata, n_components=2, )
sc.pl.umap(adata, color=["paul15_clusters", "leiden", "celltype", ],)

data = torch.FloatTensor(adata.X)

#data = torch.FloatTensor(adata.obsm["X_umap"])

enc_ct.fit(adata.obs["celltype"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs['celltype']))
#labels = torch.IntTensor(adata.obs['celltype'].apply(int))
labels = F.one_hot(labels.long(), num_classes=10).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        )
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[:600], labels[:600]),
        batch_size=128,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[600:-256], labels[600:-256]),
        batch_size=128,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[-256:], labels[-256:]),
        batch_size=128,
        shuffle=True,
        )


model = M6.AE_Type603(nx=adata.n_vars, nh=1024, nz=30, nclasses=10)

model = M6.AE_Type600(nx=adata.n_vars, nh=1024, nz=30, nclasses=10)

M6.trainSemiSuper(model, labeled_loader, unlabeled_loader, test_loader, 
        num_epochs=100, lr=1e-3, wt=0, do_unlabeled=True,)

M6.basicTrain(model, data_loader,
        num_epochs=1, lr=1e-3, wt=0, loss_type="rec",)
M6.basicTrain(model, data_loader,
        num_epochs=50, lr=1e-3, wt=0, loss_type="rec",)

model = M6.VAE_Dilo_Type601(nx=adata.n_vars, nh=2600, nz=80, nw=55, nclasses=10,
        bn=True)

model = M6.AE_TypeA608(nx=adata.n_vars, nh=2000, nz=30,bn=True, dropout=0.2,) 

model = M5.VAE_Dilo_Type04(nx=adata.n_vars, nh=1024, nz=20, nw=15, nclasses=10,)

model = M6.AE_TypeA609(nx=adata.n_vars, nh=1500, nz=30,bn=False, dropout=0,) 

M6.basicTrain(model, data_loader,
        num_epochs=1, lr=1e-3, wt=0, loss_type="total_loss", )

M6.basicTrain(model, data_loader,
        num_epochs=10, lr=1e-3, wt=1e-4, loss_type="total_loss", )

output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()

sc.pp.neighbors(adata, n_neighbors=1, use_rep="z", )
sc.tl.leiden(adata, )
sc.tl.umap(adata, n_components=2, )
sc.pl.umap(adata, color=["paul15_clusters", "leiden", "celltype", ],)



### Pancreas
# https://anndata-tutorials.readthedocs.io/en/latest/annloader.html
adata = sc.read("./data/pancreas.h5ad")

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        super().__init__()
        modules = []
        for in_size, out_size in zip([input_dim]+hidden_dims, hidden_dims):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.LayerNorm(out_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.05))
        modules.append(nn.Linear(hidden_dims[-1], out_dim))
        self.fc = nn.Sequential(*modules)
    def forward(self, *inputs):
        input_cat = torch.cat(inputs, dim=-1)
        return self.fc(input_cat)

class CVAE(nn.Module):
    # The code is based on the scarches trVAE model
    # https://github.com/theislab/scarches/blob/v0.3.5/scarches/models/trvae/trvae.py
    # and on the pyro.ai Variational Autoencoders tutorial
    # http://pyro.ai/examples/vae.html
    def __init__(self, input_dim, n_conds, n_classes, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = MLP(input_dim+n_conds, hidden_dims, 2*latent_dim) # output - mean and logvar of z
        self.decoder = MLP(latent_dim+n_conds, hidden_dims[::-1], input_dim)
        self.theta = nn.Linear(n_conds, input_dim, bias=False)
        self.classifier = nn.Linear(latent_dim, n_classes)
        self.latent_dim = latent_dim
    def model(self, x, batches, classes, size_factors):
        pyro.module("cvae", self)
        batch_size = x.shape[0]
        with pyro.plate("data", batch_size):
            z_loc = x.new_zeros((batch_size, self.latent_dim))
            z_scale = x.new_ones((batch_size, self.latent_dim))
            z = pyro.sample("latent", pyrodist.Normal(z_loc, z_scale).to_event(1))
            classes_probs = self.classifier(z).softmax(dim=-1)
            pyro.sample("class", pyrodist.Categorical(probs=classes_probs), obs=classes)
            dec_mu = self.decoder(z, batches).softmax(dim=-1) * size_factors[:, None]
            dec_theta = torch.exp(self.theta(batches))
            logits = (dec_mu + 1e-6).log() - (dec_theta + 1e-6).log()
            pyro.sample("obs", pyrodist.NegativeBinomial(total_count=dec_theta, logits=logits).to_event(1), obs=x.int())
    def guide(self, x, batches, classes, size_factors):
        batch_size = x.shape[0]
        with pyro.plate("data", batch_size):
            z_loc_scale = self.encoder(x, batches)
            z_mu = z_loc_scale[:, :self.latent_dim]
            z_var = torch.sqrt(torch.exp(z_loc_scale[:, self.latent_dim:]) + 1e-4)
            pyro.sample("latent", pyrodist.Normal(z_mu, z_var).to_event(1))

adata.X = adata.raw.X
adata.obs['size_factors'] = adata.X.sum(1)
encoder_study = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_study.fit(adata.obs['study'].to_numpy()[:, None])
encoder_celltype = LabelEncoder()
encoder_celltype.fit(adata.obs['cell_type'])
use_cuda = torch.cuda.is_available()

encoders = {
    'obs': {
        'study': lambda s: encoder_study.transform(s.to_numpy()[:, None]),
        'cell_type': encoder_celltype.transform
    }
}

#weights = np.ones(adata.n_obs)
#weights[adata.obs['cell_type'] == 'Pancreas Stellate'] = 2.
#sampler = WeightedRandomSampler(weights, adata.n_obs)
#dataloader = AnnLoader(adata, batch_size=128, sampler=sampler, convert=encoders, use_cuda=use_cuda)

dataloader = AnnLoader(adata, batch_size=128, shuffle=True, convert=encoders, use_cuda=use_cuda)

n_conds = len(adata.obs['study'].cat.categories)
n_classes = len(adata.obs['cell_type'].cat.categories)
cvae = CVAE(adata.n_vars, n_conds=n_conds, n_classes=n_classes, hidden_dims=[128, 128], latent_dim=10)

if use_cuda:
    cvae.cuda()


optimizer = pyro.optim.Adam({"lr": 1e-3})
svi = pyro.infer.SVI(cvae.model, cvae.guide, optimizer, loss=pyro.infer.TraceMeanField_ELBO())

def train(svi, train_loader):
    epoch_loss = 0.
    for batch in train_loader:
        epoch_loss += svi.step(batch.X, batch.obs['study'], batch.obs['cell_type'], batch.obs['size_factors'])

    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

NUM_EPOCHS = 210

for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, dataloader)
    if epoch % 40 == 0 or epoch == NUM_EPOCHS-1:
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))

full_data = dataloader.dataset[:]
means = cvae.encoder(full_data.X, full_data.obs['study'])[:, :10]
adata.obsm['X_cvae'] = means.data.cpu().numpy()
sc.pp.neighbors(adata, use_rep='X_cvae')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['study', 'cell_type'], wspace=0.35)

class CVAESimple(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.encoder = MLP(input_dim, hidden_dims, 2*latent_dim) # output - mean and logvar of z
        self.decoder = MLP(latent_dim, hidden_dims[::-1], input_dim)
        self.latent_dim = latent_dim
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        return
    def forward(self, input):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        z_loc_scale = self.encoder(x)
        z_mu = z_loc_scale[:, :self.latent_dim]
        z_logvar = z_loc_scale[:, self.latent_dim:]
        z_std = torch.sqrt(torch.exp(z_loc_scale[:, self.latent_dim:]) + 1e-4)
        z = z_mu + z_std * torch.rand_like(z_mu)
        rec = self.decoder(z)
        output["mu"] = z_mu
        output["std"] = z_std
        output["logvar"] = z_logvar
        output["z"]=z
        output["rec"]=rec
        loss_z = self.kld_unreduced(z_mu, z_logvar).sum(-1).mean()
        losses["z"] = loss_z
        loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        losses["rec"] = loss_rec
        total_loss = 1e0 * (
                loss_rec
                + 0e0
                + loss_z
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output
    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return
        

data = torch.FloatTensor(adata.X)
enc_ct.fit(adata.obs["cell_type"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs['cell_type']))
#labels = torch.IntTensor(adata.obs['celltype'].apply(int))
labels = F.one_hot(labels.long(), num_classes=8).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        )
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[:2600], labels[:2600]),
        batch_size=128,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[2600:-1000], labels[2600:-1000]),
        batch_size=128,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[-1000:], labels[-1000:]),
        batch_size=128,
        shuffle=True,
        )

model = CVAESimple(adata.n_vars, [128,128], 10)

model = M6.VAE_Dilo_Type601(nx=adata.n_vars, nh=1228, nz=30, nw=15, nclasses=8,)

M6.basicTrain(model, data_loader, num_epochs=10, wt=0,)

M6.trainSemiSuper(model, labeled_loader, unlabeled_loader, test_loader,
        num_epochs=20, wt=0,)

M6.trainSemiSuper(model, data_loader, unlabeled_loader, test_loader,
        num_epochs=20, wt=0, do_unlabeled=False,)

model.cpu()

full_data = dataloader.dataset[:]
means = model.encoder(full_data.X.cpu(), )

output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()

adata.obs["predict"] = output["q_y"].detach().argmax(-1).int().numpy()

adata.obsm['X_cvae'] = means.data.cpu().numpy()

sc.pp.neighbors(adata, use_rep='X_cvae')

sc.pp.neighbors(adata, use_rep='z')
sc.tl.umap(adata)

sc.pl.umap(adata, color=['study', 'cell_type', "predict",], wspace=0.35)

sc.pl.umap(adata, color=['cell_type', "predict",], wspace=0.35)


sc.pl.umap(adata, color=['cell_type', "predict", "leiden",], wspace=0.35)

sc.tl.leiden(adata,  )

sc.tl.louvain(adata, resolution=1.0, )

sc.pl.umap(adata, color=['cell_type', "predict", "leiden", "louvain",], wspace=0.35)


#### More RNAseq
adata = sc.read('./data/pbmc3k_raw.h5ad')

adata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad")

adata = sc.read("./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad")


adata = sc.read("./data/limb_sce_alternative.h5ad")
adata

sc.pp.neighbors(adata, n_neighbors=10,)

sc.tl.umap(adata, )

sc.tl.tsne(adata, )
sc.tl.louvain(adata, )
sc.tl.leiden(adata, )
sc.pl.tsne(adata, color=["cell_ontology_class", "louvain", "leiden"])

sc.pl.umap(adata, color=["cell_ontology_class", "louvain", "leiden"])

adata = sc.datasets.pbmc68k_reduced()

adata = sc.datasets.pbmc3k_processed()

sc.tl.leiden(adata,)

sc.pl.umap(adata, color=["bulk_labels", "louvain", "phase"])

sc.pl.umap(adata, color=["bulk_labels", "louvain", "leiden"])

adata = sc.read_csv(
        "./data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.edit.gct",
        delimiter="\t",)

adata = sc.read("./data/pancreas.h5ad")

sc.pp.scale(adata, max_value=6,)
sc.pp.pca(adata, )
sc.pp.neighbors(adata, n_neighbors=10,)
sc.tl.umap(adata,)
sc.tl.tsne(adata, )
sc.tl.leiden(adata, )
sc.tl.louvain(adata, )

sc.pl.tsne(adata, color=["louvain","leiden", "cell_type"])

sc.pl.umap(adata, color=["louvain","leiden", "cell_type"])


df = pd.read_csv("./data/GTEx_Analysis_2016-01-15_v7_RNASeQCv1.1.8_gene_reads.edit.csv", sep = "\t",)
obs = pd.read_csv("./data/GTEx_v7_Annotations_SampleAttributesDS.txt", sep="\t",)

X = df.iloc[:,2:].transpose()
X = X.sort_index()
X.to_numpy()
obs = obs[obs.SAMPID.isin(df.columns)]
adata = sc.AnnData(X=X, )
obs.index = X.index
adata.obs["smts"] = obs["SMTS"]
adata.write(filename="./data/gtex_v7_SMTS.h5ad",  )



adata = sc.read("./data/gtex_v7_SMTS.h5ad",)

adata.var_names_make_unique()

adata = sc.read("./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad", )

sc.pp.filter_cells(adata, min_genes=200, inplace=True,)
sc.pp.filter_genes(adata, min_cells=20, inplace=True,)
sc.pp.normalize_total(adata, target_sum=1e4, inplace=True,)
sc.pp.log1p(adata, )

sc.pp.highly_variable_genes(adata, subset=True, inplace=True, )

sc.pp.highly_variable_genes(adata, subset=True, inplace=True, n_top_genes=1000, )

sc.pp.scale(adata, max_value=10,)

sc.tl.pca(adata, )
sc.pp.neighbors(adata, )
sc.tl.tsne(adata,)
sc.tl.umap(adata,)
sc.tl.leiden(adata, )
sc.tl.louvain(adata,)


sc.pl.tsne(adata, color=["smts", "louvain", "leiden",])

sc.pl.umap(adata, color=["smts", "louvain", "leiden",])

sc.pl.umap(adata, color=["smts", "louvain",])

sc.pl.umap(adata, color=["tissue", "louvain", "leiden",])

adata.write(filename="./data/gtex_v7_SMTS_PP.h5ad",  )

adata.write(filename="./data/gtex_v7_SMTS_PP_1k.h5ad",  )


# worked pretty nicely with gtex ;)
model = M6.VAE_Dilo_Type601(nx=1000, nh=1024, nz=30, nw=15, nclasses=40,
        bn=True, dropout=0.2,)

model = M6.VAE_Dilo_Type601(nx=1000, nh=1024, nz=30, nw=15, nclasses=30,
        bn=True, dropout=0.2,)

data = torch.FloatTensor(adata.X)
enc_ct.fit(adata.obs["smts"])
labels = torch.IntTensor(
        enc_ct.transform(adata.obs['smts']))
labels = F.one_hot(labels.long(), num_classes=30).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        )
labeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[:2600], labels[:2600]),
        batch_size=128,
        shuffle=True,
        )
unlabeled_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[2600:-1000], labels[2600:-1000]),
        batch_size=128,
        shuffle=True,
        )
test_loader = torch.utils.data.DataLoader(
        dataset=ut.SynteticDataSet(data[-1000:], labels[-1000:]),
        batch_size=128,
        shuffle=True,
        )

M6.basicTrain(model, data_loader, num_epochs=30, wt=0,)

M6.trainSemiSuper(model, labeled_loader, unlabeled_loader, test_loader,
        num_epochs=20, wt=0,)

M6.trainSemiSuper(model, data_loader, unlabeled_loader, test_loader,
        num_epochs=20, wt=0, do_unlabeled=False,)

output = model(data)
adata.obsm["z"] = output["z"].detach().numpy()

adata.obs["predict"] = output["q_y"].detach().argmax(-1).int().numpy().astype(str)

sc.pl.umap(adata, color=["smts", "predict",])

sc.pl.umap(adata, color=["louvain", "predict",])



## more anndata tests
adata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)

adata = sc.read("./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad",)

bdata = sc.read("./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad",)

sc.pl.umap(adata, color=["Broad cell type", "batch", "leiden"])

sc.pl.umap(bdata, color=["annotation", "broad",])

sc.pp.neighbors(adata, use_rep="X_vae_mean")

sc.tl.tsne(adata, use_rep="X_vae_mean")

sc.tl.umap(adata,)

sc.tl.leiden(adata, )
sc.tl.louvain(adata,)

sc.pl.umap(bdata, color=["annotation", "broad",])

xdata = sc.AnnData(X=adata.X, obs=bdata.obs)

xdata = sc.AnnData(X=bdata.X, obs=bdata.obs)


sc.pp.highly_variable_genes(xdata, n_top_genes=2000, inplace=True, subset=True,)
sc.pp.scale(xdata,max_value=10,)


sc.tl.pca(xdata, )
sc.pp.neighbors(xdata, )
sc.tl.tsne(xdata,)
sc.tl.umap(xdata,)
sc.tl.leiden(xdata, )
sc.tl.louvain(xdata,)

sc.pl.umap(xdata, color=["annotation", "broad",])

encoder_sex = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_sex.fit(xdata.obs["Sex"].to_numpy()[:, None])
encoder_person = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_person.fit(xdata.obs["individual"].to_numpy()[:, None])
encoder_ischem = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_ischem.fit(xdata.obs["Sample Ischemic Time (mins)"].to_numpy()[:, None])
encoder_age = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_age.fit(xdata.obs["Age_bin"].to_numpy()[:, None])
encoder_batch = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_batch.fit(xdata.obs["batch"].to_numpy()[:, None])
encoder_prep = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_prep.fit(xdata.obs["prep"].to_numpy()[:, None])
encoder_annotation = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_annotation.fit(xdata.obs["annotation"].to_numpy()[:, None])
encoder_broad = OneHotEncoder(sparse=False, dtype=np.float32)
encoder_broad.fit(xdata.obs["broad"].to_numpy()[:, None])

#encoder_ = OneHotEncoder(sparse=False, dtype=np.float32)
#encoder_.fit(xdata.obs[""].to_numpy()[:, None])

encoders = {
    'obs': {
        'Sex': lambda s: torch.FloatTensor(
            encoder_sex.transform(s.to_numpy()[:, None])),
        'individual': lambda s: torch.FloatTensor(
            encoder_person.transform(s.to_numpy()[:, None])),
        'Sample Ischemic Time (mins)': lambda s: torch.FloatTensor(
            encoder_ischem.transform(s.to_numpy()[:, None])),
        'Age_bin': lambda s: torch.FloatTensor(
            encoder_age.transform(s.to_numpy()[:, None])),
        'batch': lambda s: torch.FloatTensor(
            encoder_batch.transform(s.to_numpy()[:, None])),
        'prep': lambda s: torch.FloatTensor(
            encoder_prep.transform(s.to_numpy()[:, None])),
        'annotation': lambda s: torch.FloatTensor(
            encoder_annotation.transform(s.to_numpy()[:, None])),
        'broad': lambda s: torch.FloatTensor(
            encoder_broad.transform(s.to_numpy()[:, None])),
    }
}

class AE_TypeB618(M6.Generic_Net):
    """
    Vanila Autoencoder.
    """
    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nz: int = 32,
        bn : bool = True,
        dropout : float = 0.2,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nz = nz
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.z_prior = distributions.Normal(
            loc=torch.zeros(nz),
            scale=torch.ones(nz),
        )
        self.encoder = models.resnet101()
        self.encoder.conv1 = nn.Conv2d(1, 64,(7,7),(2,2),(3,3),bias=False,)
        self.encoder.fc = nn.Linear(2048, nz)
        self.decoder = models.resnet101()
        self.decoder.conv1 = nn.Conv2d(1, 64,(7,7),(2,2),(3,3),bias=False,)
        self.decoder.fc = nn.Linear(2048, nx)
        self.fcz = nn.Sequential(
                nn.Linear(nz, 64**2),
                nn.Unflatten(-1, (1,64,64)),
                )
        self.fcx = nn.Sequential(
                nn.Linear(nx, 64**2),
                nn.Unflatten(-1, (1,64,64)),
                )
        #self.encoder = ut.buildNetworkv3([nx,nh,4*nh,nh,nz], 
        #        dropout=0.2, layernorm=True,)
        #self.decoder = ut.buildNetworkv3([nz,nh,4*nh,nh,nx], 
        #        dropout=0.2, layernorm=True,)
        return

    def studentize(self, m, z):
        q = 1 + (m - z.unsqueeze(1)).pow(2).sum(-1)
        q = 1 / q
        s = q.sum(-1, keepdim=True,)
        q = q/s
        return q

    def forward(self, input, y=None):
        x = nn.Flatten()(input)
        losses = {}
        output = {}
        xin = self.fcx(x)
        z = self.encoder(xin)
        #z = self.encoder(x)
        output["z"] = z
        zout = self.fcz(z)
        rec = self.decoder(zout)
        #rec = self.decoder(z)
        output["rec"]= rec
        logsigma_x = ut.softclip(self.logsigma_x, -7, 7)
        sigma_x = logsigma_x.exp()
        loss_rec = nn.MSELoss(reduction='none')(rec,x).mean()
        #loss_rec = nn.MSELoss(reduction='none')(rec,x).sum(-1).mean()
        #Qx = distributions.Normal(loc=rec, scale=sigma_x)
        #loss_rec = -Qx.log_prob(x).sum(-1).mean()
        losses["rec"] = loss_rec
        #losses["rec"] = loss_rec * 1e2
        total_loss = (
                loss_rec
                + 0e0
                )
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output


data_loader = AnnLoader(xdata,convert=encoders, batch_size=128, shuffle=True, use_cuda=False, )
foo =data_loader.__iter__().next()

model = M6.AE_TypeA609(nx=2000, nh=1024, nz=64, )




data = torch.FloatTensor(xdata.X)
enc_ct.fit(xdata.obs["annotation"])
labels = torch.IntTensor(
        enc_ct.transform(xdata.obs['annotation']))
labels = F.one_hot(labels.long(), num_classes=30).float()

data = torch.FloatTensor(bdata.X.toarray())
enc_ct.fit(bdata.obs["annotation"])
labels = torch.IntTensor(
        enc_ct.transform(bdata.obs['annotation']))
labels = F.one_hot(labels.long(), num_classes=30).float()

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True,
        )

model = AE_TypeB618(nx=xdata.n_vars, nh=1024, nz=32, )
model = AE_TypeB618(nx=bdata.n_vars, nh=1024, nz=32, )

model = M6.AE_TypeA608(nx=bdata.n_vars, nh=1000, nz=32,)

M6.basicTrain(model, data_loader, num_epochs=10,)

output = model(data)

xdata.obsm["z"] = output["z"].detach().numpy()


#sc.tl.pca(xdata, )
sc.pp.neighbors(xdata, use_rep="z" )
sc.tl.tsne(xdata, use_rep="z")
sc.tl.umap(xdata,)
sc.tl.leiden(xdata, )
sc.tl.louvain(xdata,)

sc.pl.umap(xdata, color=["annotation", "broad",])

sc.pl.umap(xdata, color=["individual", "batch"])

