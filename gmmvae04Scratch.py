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
# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights, buildNetwork
from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import scsimDataset
import my_torch_utils as ut
from importlib import reload
from torch.nn import functional as F
import gmmvae04 as M
print(torch.cuda.is_available())
#transform = transforms.Compose([
#    #transforms.Resize((32,32)),
#    transforms.ToTensor(),
#    #normalize,
#    ])
#test_loader = torch.utils.data.DataLoader(
#   dataset=datasets.MNIST(
#       root='data/',
#       train=False,
#       download=True,
#       transform=transform,
#       ),
#   batch_size=128,
#   shuffle=True,
#)
#train_loader = torch.utils.data.DataLoader(
#   dataset=datasets.MNIST(
#       root='data/',
#       train=True,
#       download=True,
#       transform=transform,
#       ),
#   batch_size=128,
#   shuffle=True,
#)
#test_data = test_loader.dataset.data.float()/255
#test_labels = test_loader.dataset.targets
#train_data = train_loader.dataset.data.float()/255
#train_labels = train_loader.dataset.targets


## Syntetic data
#x = torch.rand(100)
#y = torch.rand(100)
#plt.scatter(x, y, c=plt.cm.Set1(0))

sm = ut.SynteticSampler(
        means=torch.rand(6,2), 
        logscales=torch.randn(6) - 2e-1,
        noiseLevel=3e-2
        )

data = sm.sample((10000,))

#datan = data[0] + torch.randn_like(data[0]) * 1e-3
#plt.scatter(datan[:,0], datan[:,1])

sm.plotData()

#plt.scatter(data[0][:,0], data[0][:,1])

#sm.plotSampleData()


dataset = ut.SynteticDataSet(data[0], data[1])

data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=200,
        shuffle=True,
        )

model = M.VAE_Dilo3(nx=2, nh=1024, nz=20, nw=15, nclasses=4,)
model.apply(init_weights)
model.fit(data_loader)

output = model(data[0])
#
#output["q_y"][data[1] == 1][:100].max(-1)
#
rec = output["rec"].detach()
#
plt.scatter(rec[:,0], rec[:,1])

## AE_Type0
model = M.AE_Type00(nx=2)
model.apply(init_weights)
model.fit(data_loader)

batch = sm.sample((5000,))
output = model(batch[0])
rec = output["rec"].detach()
rec = output["rec"].detach()
plt.scatter(rec[:,0], rec[:,1])

#foo = ut.buildNetworkv2([2,1024,1024,2],)
#foo.add_module("softmax",nn.Softmax(dim=-1))
#foo(batch[0])

### AAE_Type01
model = M.AAE_Type01(nx=2, nclasses=4)
model.apply(init_weights)

model.fit(data_loader, num_epochs=8)

model.fitv2(data_loader, num_epochs=8)

#true_labels = torch.ones((50, 1), )
#false_labels = torch.zeros((50, 1), )
#
##loss_dz_true = bce(dz_true, true_labels).mean()
#
#y_sample = model.y_prior.sample((50,))
z_sample = model.z_prior.sample((50,))
#dy_true = model.Dy(y_sample)
dz_true = model.Dz(z_sample)
#loss_dy_true = nn.BCELoss(reduction="none")(dy_true, true_labels).mean()

blobs = ut.Blobs(means=torch.rand(6,2)*5, scales=torch.rand(6,2)*5e-1) 
blobs.sample((2,))
blobs.plotSample()

data = blobs.sample((10000,))
dataset = ut.SynteticDataSet(data[0], data[1])
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=200,
        shuffle=True,
        )

model = M.VAE_Dilo3(nx=2, nh=1024, nz=20, nw=15, nclasses=6,)

model = M.AE_Type00(nx=2,nh=1024,nz=10)

model = M.AAE_Type01(nx=2,nh=1024,nz=10,nw=15,nclasses=6)

model.apply(init_weights)

model.fit(data_loader)


model.fitv2(data_loader)

#output["q_y"][data[1] == 1][:100].max(-1)
output = model(data[0])
rec = output["rec"].detach()

plt.scatter(rec[:,0], rec[:,1])

fig, axs = plt.subplots(1,2)


axs[0].cla()
axs[1].cla()

sns.scatterplot(data[0][:,0], data[0][:,1], hue=output["q_y"].argmax(-1), ax=axs[1])

sns.scatterplot(data[0][:,0], data[0][:,1], hue=data[1], ax=axs[0])

axs[0].cla()

axs[0].plot([1,2])

sns.scatterplot(rec[:,0], rec[:,1], hue=data[1], ax=axs[1])

sns.scatterplot(rec[:,0], rec[:,1], hue=output["q_y"].argmax(-1), ax=axs[0])

sns.scatterplot(rec[:,0], rec[:,1], hue=output["y"].argmax(-1), ax=axs[0])

blobs = ut.Blobs(means=torch.rand(4,2)*3, scales=torch.rand(4,2)*3e-1) 
blobs.sample((2,))
blobs.plotSample()

plt.cla()

data = blobs.sample((10000,))
dataset = ut.SynteticDataSet(data[0], data[1])
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=200,
        shuffle=True,
        )

data = blobs.sample((1000,))

model = M.AAE_Type01(nx=2,nh=1024,nz=10,nw=15,nclasses=4)

model = M.AE_Type02(nx=2, nh=1024, nz=10, nclasses=4)

model = M.AE_Type02(nx=2, nh=1024, nz=10, nclasses=6)

model.apply(init_weights)
model.fit(data_loader)


gauss = ut.mixedGaussianCircular(rho=0.01, sigma=2.1, k=3, j=0)
data = gauss.sample((10000,)).flatten(0,1)
labels = torch.arange(3).tile((10000,1)).t().flatten()

gauss = ut.mixedGaussianCircular(rho=0.02, sigma=3.1, k=10, j=0)
samples = gauss.sample((2500,))
x = samples[:,:,0].flatten()
y = samples[:,:,1].flatten()
plt.scatter(x,y)

plt.scatter(data[:,0], data[:,1])

model = M.AE_Type02(nx=2, nh=1024, nz=20, nclasses=12)

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=200,
        shuffle=True,
        )

data = gauss.sample((2000,)).flatten(0,1)

output = model(data)
rec = output["rec"].detach()

fig, axs = plt.subplots(1,1)

sns.scatterplot(data[:,0], data[:,1], hue=labels, ax=axs,
        palette="viridis")

sns.scatterplot(rec[:,0], rec[:,1], hue=output["q_y"].argmax(-1), ax=axs,
        palette="viridis")

plt.cla()

x = torch.arange(4)
x.tile((4,1)).t().flatten()

theta = torch.rand((1000,)) * pi/2
x = theta.cos() * 1e-1
y = theta.sin() * 5e-1
plt.scatter(x,y)

sns.scatterplot(x,y)

sm = ut.SynteticSampler(
        means=torch.zeros(4,2), 
        logscales=torch.arange(1,5)*5e-1 - 2e-1,
        noiseLevel=3e-2
        )
sm.plotData()

model = M.AE_Type02(nx=2, nh=1024, nz=20, nclasses=4)



scurve = skds.make_s_curve(n_samples=10000, noise=0,)

sns.scatterplot(scurve[0][:,0], scurve[0][:,1],)

X, color = skds.make_s_curve(n_samples=10000, random_state=0)

sns.scatterplot(x=X[:,0], y=X[:,2], hue=color)

plt.scatter(X[:, 0], X[:, 2], c=color)

X = torch.FloatTensor(X)
color = torch.IntTensor(color)

dataset = ut.SynteticDataSet(X, color)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=200,
        shuffle=True,
        )

model = M.VAE_Dilo3(nx=3, nh=1024, nz=10, nw=5, nclasses=3,)

model = M.AE_Type02(nx=3, nh=1024, nz=10, nclasses=3)

model = M.AE_Type03(nx=3, nh=1024, nz=10, nclasses=3)

model.apply(init_weights)
model.fit(data_loader)

output = model(X)
rec = output["rec"].detach().numpy()
q_y = output["q_y"].detach().argmax(-1).numpy()

plt.cla()
plt.scatter(rec[:, 0], rec[:, 2], c=q_y)

plt.cla()
plt.scatter(X[:, 0], X[:, 2], c=q_y)

plt.cla()
plt.scatter(rec[:, 0], rec[:, 2], c=color)

model = M.VAE_dirichlet_type04()
model


X,y = skds.make_moons(n_samples=20000, noise=3e-2, )
X = torch.FloatTensor(X)
y = torch.IntTensor(y)

dataset = ut.SynteticDataSet(X, y)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=200,
        shuffle=True,
        )
plt.cla()
plt.scatter(X[:, 0], X[:, 1], c=y)


model = M.AE_Type02(nx=2,nh=1024,nz=10,nclasses=2)

model = M.AE_Type03(nx=2,nh=1024,nz=10,nclasses=4)

model = M.VAE_Dilo3(nx=2, nh=1024, nz=10, nw=5, nclasses=2,)

model.apply(init_weights)
model.fit(data_loader)

output = model(X)
q_y = output["q_y"].detach().argmax(-1).numpy()
rec = output["rec"].detach().numpy()

plt.cla()
plt.scatter(rec[:, 0], rec[:, 1], c=q_y)

plt.cla()
plt.scatter(X[:, 0], X[:, 1], c=q_y)

model = M.VAE_2moons_type005()
model

x = torch.rand(20,2)
y = torch.rand(20,2)
xy = torch.hstack([x,y])

model.Qz(xy).shape
model.Qy(x).shape
model.Px(model.Qz(xy)).shape

output = model(x)
output["xs"].shape

### LSTM sin wave
f = nn.LSTM(input_size=1, hidden_size=16, num_layers=2,
        batch_first=True,
        proj_size=1)


N = 100 # number of samples in a batch
L = 1000 # length of each sample
T = 1.9e1 # period factor
xs = torch.rand((N,L)) * torch.pi * 2 * 1e2
ys = (xs / T).sin()
plt.cla()
plt.scatter(xs[0], ys[0])

h0 = torch.zeros(2,N,1)
c0 = torch.zeros(2,N,16)
input = xs.unsqueeze(-1)

output, (hn, cn) = f(input, (h0,c0))

class LolSTM(nn.Module):
    def __init__(self, nx=1, nh=64, ny=1, L=250, N=20, T=1.9e1):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.nh = nh
        self.N = N
        self.L = L
        self.T = T
        self.lstm = nn.LSTM(
                input_size=1, 
                hidden_size=nh,
                num_layers=3,
                batch_first=True,
                )
        self.fc = nn.Linear(nh, 1)
        return
    def forward(self, input):
        batch_size = input.shape[0]
        h0 = torch.zeros(
                3, batch_size, self.nh
                ).to(input.device)
        c0 = torch.zeros(
                3, batch_size, self.nh
                ).to(input.device)
        output, (hn, cn) = self.lstm(input, (h0,c0))
        output = self.fc(output)
        return output
    def learnSin(self, device='cuda'):
        self.to(device)
        optimizer = optim.Adam(self.parameters(),)
        mse = nn.MSELoss(reduction='none')
        for i in range(10000):
            #xs = 2 * pi * torch.rand((self.N,self.L,1)).to(device)
            #ys = (xs * self.T).sin()
            xs = 2e2 * pi * torch.rand((self.N,self.L,1)).to(device)
            ys = (xs / self.T).sin()
            output = self.forward(xs)
            optimizer.zero_grad()
            loss = mse(output, ys).mean()
            loss.backward()
            optimizer.step()
            if i % 1000 == 0:
                print("loss = ", loss.item())
        print("done trianing " )
        model.cpu()
        return
    def plotPrediction(self,):
        #xs = 2 * pi * torch.rand((self.N,self.L,1))
        #ys = (xs * self.T).sin()
        xs = 2e2 * pi * torch.rand((self.N,self.L,1))
        ys = (xs / self.T).sin()
        pred = model(xs).detach()
        plt.scatter(xs.flatten().detach().numpy(), pred.flatten().detach().numpy())

model = LolSTM()

model.to('cuda')
x = torch.rand((N,L,1)).cuda()
model(x)

model.learnSin()

plt.cla()
model.plotPrediction()


xs = 2e2 * pi * torch.rand((model.N,model.L,1))
ys = (xs / model.T).sin()
pred = model(xs).detach()
plt.scatter(xs.flatten().detach().numpy(), pred.flatten().detach().numpy())

plt.scatter(xs.flatten().detach().numpy(), ys.flatten().detach().numpy())

plt.cla()


f = nn.Transformer(d_model=24, batch_first=True)

src = torch.rand((100, 32, 24))
tgt = torch.rand((100,22,24))
f(src,tgt)


##### 3k PBMC
# https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
sc.settings.verbosity = 3
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white',)
results_file = './results/pbmc3k.h5ad'

adata = sc.read_10x_mtx('data/filtered_gene_bc_matrices/hg19/', 
        var_names='gene_symbols', cache=True,)
adata.var_names_make_unique()
adata

sc.pl.highest_expr_genes(adata, n_top=20,)

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

#cdata = sc.read("./data/limb_sce_alternative.h5ad",)
gdata = sc.read("./data/GTEx_8_tissues_snRNAseq_atlas_071421.public_obs.h5ad",)

fdata = sc.read("./data/GTEx_8_tissues_snRNAseq_immune_atlas_071421.public_obs.h5ad",)

adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)
adata.raw = adata
adata = adata[:, adata.var.highly_variable]

sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack',)
sc.pl.pca(adata, color="CST3",)
sc.pl.pca_variance_ratio(adata, log=True,)


sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40,)

sc.tl.leiden(adata, )

sc.tl.paga(adata,)

sc.pl.paga(adata, plot=True,)
sc.tl.umap(adata, init_pos='paga')

sc.tl.umap(adata, n_components=2, )
sc.pl.umap(adata, color=["CST3", "NKG7", "PPBP"],)

sc.pl.umap(adata, color=["leiden", "CST3", ], use_raw=False)

sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

adata.write(results_file)

adata = sc.read(results_file)

pd.DataFrame(adata.uns['rank_genes_groups']['names']).head()

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)

sc.tl.rank_genes_groups(adata, 'leiden', groups=['0'], reference='1', method='wilcoxon')
sc.pl.rank_genes_groups(adata, groups=['0'], n_genes=20)

sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)

adata = sc.read(results_file)
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8,)
sc.pl.violin(adata, ['CST3', 'NKG7', 'PPBP'], groupby='leiden')

new_cluster_names = [
    'CD4 T', 'CD14 Monocytes',
    'B', 'CD8 T',
    'NK', 'FCGR3A Monocytes',
    'Dendritic', 'Megakaryocytes']
adata.rename_categories('leiden', new_cluster_names)

sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='.pdf')

n_conds = len(adata.obs['leiden'].cat.categories)
n_classes = len(adata.obs['leiden'].cat.categories)
n_dims = adata.shape[1]
use_cuda = torch.cuda.is_available()
enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc.fit(adata.obs['leiden'].to_numpy()[:,None])
enc_ct = LabelEncoder()
enc_ct.fit(adata.obs['leiden'])
convert = {
        'obs' : {
            #'subtissue' : lambda s: enc.transform(s.to_numpy()[:, None]),
            #'cell_ontology_class': enc_ct.transform,
            'leiden': enc_ct.transform,
            }
        }
#dataloader = AnnLoader(adata, batch_size=128, shuffle=True, convert=convert,
#        use_cuda=use_cuda)

data = torch.tensor(adata.X, dtype=torch.float)
labels = enc_ct.transform(adata.obs['leiden'])
labels = torch.tensor(labels, dtype=torch.int)

dataset = ut.SynteticDataSet(data, labels)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=60,
        shuffle=True,
        )


model = M.AE_Type00(nx=n_dims, nh=1024, nz=20,)

model.Px = ut.buildNetworkv2(
        [50, 3000, 2000, 2000, n_dims], 
        dropout=0.2, activation=nn.LeakyReLU(), batchnorm=True,)

model.Qz = ut.buildNetworkv2(
        [n_dims, 3000, 2000, 2000, 50], 
        dropout=0.2, activation=nn.LeakyReLU(), batchnorm=True,)


model = M.AE_Type02(nx=n_dims, nclasses=n_classes, nh=2024, nz=60,)

model = M.AE_Type03(nx=n_dims,nz=80, nclasses=n_classes,)

model.apply(init_weights)
model.fit(data_loader, num_epochs=1)

output = model(data)

output["q_y"].argmax(-1)[labels == 1]

pdata = sc.datasets.paul15()

sc.pp.filter_cells(pdata, min_genes=200)
sc.pp.filter_genes(pdata, min_cells=3)
sc.pp.log1p(pdata)
sc.pp.highly_variable_genes(pdata, n_top_genes=1000, subset=True, inplace=True,)


### blob test
data = sc


### blob test
data = sc.datasets.blobs(n_variables=1000, n_centers=5, cluster_std=1,
        n_observations=2000,)
labels = torch.tensor(data.obs['blobs'].apply(int), dtype=torch.int)
dataset = ut.SynteticDataSet(
        torch.tensor(data.X, dtype=torch.float),
        torch.tensor(data.obs['blobs'].apply(int), dtype=torch.int),)
data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=100,
        shuffle=True,
        )

model = M.VAE_Dilo3(nx=1000, nz=20, nclasses=5,)
model = M.AE_Type02(nx=1000, nz=20, nclasses=5,)

model.fit(data_loader, num_epochs=10, lr=1e-2)
