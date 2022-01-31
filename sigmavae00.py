# fun fun 2022-01-29
# https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py
# https://github.com/orybkin/sigma-vae-pytorch
import argparse
from importlib import reload
import matplotlib.pyplot as plt
import my_torch_utils as ut
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
from math import pi, sin, cos, sqrt, log
from toolz import partial, curry
from torch import nn, optim, distributions
from torch.nn.functional import one_hot
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast

# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import fnorm, replicate, logNorm
from my_torch_utils import scsimDataset
import scsim.scsim as scsim

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam

from toolz import take, drop,

print(torch.cuda.is_available())

kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def softclip(tensor, min=-6, max=9):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + nn.functional.softplus(tensor - min)
    result_tensor = max - nn.functional.softplus(max - result_tensor)
    return result_tensor

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

def log_gaussian_prob(x : torch.Tensor, 
        mu : torch.Tensor = torch.tensor(0), 
        logvar : torch.Tensor = torch.tensor(0)
        ) -> torch.Tensor:
    """
    compute the log density function of a gaussian.
    user must make sure that the dimensions are aligned correctly.
    """
    return -0.5 * (
            log(2 * pi) + 
            logvar +
            (x - mu).pow(2) / logvar.exp()
            )

class SoftClip(nn.Module):
    def __init__(self, min=-6, max=6):
        super(SoftClip, self).__init__()
        self.min = min
        self.max = max
    def forward(self, input):
        return softclip(input, self.min, self.max)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def __init__(self, n_channels):
        super(UnFlatten, self).__init__()
        self.n_channels = n_channels
    
    def forward(self, input):
        size = int((input.size(1) // self.n_channels) ** 0.5)
        return input.view(input.size(0), self.n_channels, size, size)

class VAE(nn.Module):
    """
    VAE with gaussian encoder and decoder.
    """

    def __init__(self, nz: int = 10, nh: int = 1024,
            imgsize : int = 28, is_Bernouli : bool = False) -> None:
        super(VAE, self).__init__()
        self.nin = nin = imgsize**2
        self.nz = nz
        self.imgsize = imgsize
        self.is_Bernouli = is_Bernouli
        self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(nin, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.Tanh(),
                )
        self.decoder = nn.Sequential(
                nn.Linear(nz, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Tanh(),
                )
        self.xmu = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nin),
                )
        self.xlogsig = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nin),
                SoftClip(),
                )
        # or if we prefer Bernoulu decoder
        self.bernouli = nn.Sequential(
                nn.Linear(nh, nin),
                nn.Sigmoid(),
                )
        self.zmu = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                )
        self.zlogsig = nn.Sequential(
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                nn.Linear(nh, nz),
                #SoftClip(),
                )
        self.log_sigma = torch.nn.Parameter(torch.zeros(1)[0], requires_grad=True)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.zmu(h)
        logsig = self.zlogsig(h)
        return mu, logsig

    def reparameterize(self, mu, logsig, ):
        eps = torch.randn(mu.shape).to(mu.device)
        sigma = (logsig).exp()
        return mu + sigma * eps

    def decode_bernouli(self, z):
        h = self.decoder(z)
        p = self.bernouli(h)
        return p, torch.tensor(0)

    def decode(self, z):
        h = self.decoder(z)
        mu = self.xmu(h)
        logsig = self.xlogsig(h)
        #logsig = softclip(self.xlogsig(h))
        return mu, logsig

    def forward(self, x):
        zmu, zlogsig = self.encode(x)
        z = self.reparameterize(zmu, zlogsig)
        if self.is_Bernouli:
            xmu, xlogsig = self.decode_bernouli(z)
        else:
            xmu, xlogsig = self.decode(z)
        return zmu, zlogsig, xmu, xlogsig,

    def reconstruction_loss(self, xmu, xlogsig, x):
        result = gaussian_nll(xmu, xlogsig, x).sum()
        #result = -log_gaussian_prob(x, xmu, xlogsig*2).sum()
        return result
    
    def loss_function(self, x, xmu, xlogsig, zmu, zlogsig):
        rec = self.reconstruction_loss(xmu, xlogsig, x)
        zlogvar = 2*zlogsig
        kl = kld(zmu, zlogvar)
        return rec, kl

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
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

model = VAE(nz=20,)

imgs, labels = test_loader.__iter__().next()
zmu, zlogs = model.encode(imgs)
z = model.reparameterize(zmu, zlogs)
xmu, xlogs = model.decode(z)

gaussian_nll(xmu, xlogs, imgs.flatten(1))
log_gaussian_prob(imgs.flatten(1), xmu, 2*xlogs)
kld(zmu, zlogs)

x = imgs.flatten(1)
model.cpu()

model.loss_function(x, xmu, xlogs, zmu, zlogs)

model.cuda()
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = "cuda"

x = imgs.flatten(1).to(device)

zmu, zlogs, xmu, xlogs = model(x)

rec, kl = model.loss_function(x, xmu, xlogs, zmu, zlogs)

mse = nn.MSELoss(reduction='sum')
bce = nn.BCELoss(reduction='sum')
l1loss = nn.L1Loss()

def train(epoch, method='nll'):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        # Run VAE
        #recon_batch, mu, logsig = model(data)
        zmu, zlogs, xmu, xlogs = model(data)
        # Compute loss
        #rec, kl = model.loss_function(recon_batch, data, mu, logsig)
        x = data.flatten(1).to(device)
        kl = kld(zmu, zlogs)
        if method == 'nll':
            rec = -log_gaussian_prob(x, xmu, 2*xlogs).sum()
        elif method == 'sigma':
            log_sigma = softclip(model.log_sigma, -6, 9)
            rec = gaussian_nll(xmu, log_sigma, x).sum()
        else:
            rec = mse(x, xmu)
        loss = kl + rec
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print("epoch", epoch, "losses: ", rec.item(), kl.item())



for epoch in range(10):
    #train(epoch)
    train(epoch, 'sigma')

model.cpu()

imgs, labels = test_loader.__iter__().next()

plot_images(denorm(imgs))

zmu, zlogs, xmu, xlogs = model(imgs)

plot_images(denorm(xmu.view(-1,1,28,28)))


data = test_loader.dataset.data.float().reshape(-1,1,28,28)/255
zmu, zlogs, xmu, xlogs = model(data)
mnist_labels = test_loader.dataset.targets


def plot_tsne(z_loc, classes, name):
    import matplotlib
    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        #ind_class = classes[:, ic] == 1
        ind_class = classes == ic
        color = plt.cm.Set1(ic)
        #plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, cmap="viridis")
        plt.title(name + " Latent Variable T-SNE per Class")
        #fig.savefig("./results/" + str(name) + "_embedding_" + str(ic) + ".png")
    plt.legend("0123456789a")
    #fig.savefig("./results/" + str(name) + "_embedding.png")

plot_tsne(zmu, mnist_labels, "ooVAE")



# scrna
countspath = "./data/scrnasim/counts.npz"
idpath = "./data/scrnasim/cellparams.npz"
genepath = "./data/scrnasim/geneparams.npz"

dataSet = scsimDataset("data/scrnasim/counts.npz", "data/scrnasim/cellparams.npz", genepath)

dataSet.normalized_counts = (dataSet.counts - dataSet.counts.mean().mean()) / dataSet.counts.max().max()

trainD, testD = dataSet.__train_test_split__(8500)

trainD.normalized_counts = (trainD.counts - dataSet.counts.mean().mean()) / dataSet.counts.max().max()
testD.normalized_counts = (testD.counts - dataSet.counts.mean().mean()) / dataSet.counts.max().max()

trainLoader = torch.utils.data.DataLoader(trainD, batch_size=128, shuffle=True)
testLoader = torch.utils.data.DataLoader(testD, batch_size=128, shuffle=True)

model = VAE(nz=100,nh=2048, imgsize=100, is_Bernouli=False)

ncounts, counts, classes = iter(testLoader).next()

zmu, zlogs, xmu, xlogs = model(ncounts)


model.cuda()

model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
device = "cuda"

x = ncounts.cuda()
zmu, zlogs, xmu, xlogs = model(data)

for batch_idx, (data, _, _) in enumerate(trainLoader):
    data = data.to(device)
    optimizer.zero_grad()
    zmu, zlogs, xmu, xlogs = model(data)
    x = data.flatten(1).to(device)
    print(x.shape, data.shape)
    kl = kld(zmu, zlogs)
    log_sigma = softclip(model.log_sigma, -6, 9)
    rec = gaussian_nll(xmu, log_sigma, x).sum()
    loss = kl + rec
    loss.backward()
    optimizer.step()
    print("epoch", epoch, "losses: ", rec.item(), kl.item())
    break






def trainSC(epoch, method='sigma'):
    model.train()
    for batch_idx, (data, _, _) in enumerate(trainLoader):
        data = data.to(device)
        optimizer.zero_grad()
        # Run VAE
        #recon_batch, mu, logsig = model(data)
        zmu, zlogs, xmu, xlogs = model(data)
        # Compute loss
        #rec, kl = model.loss_function(recon_batch, data, mu, logsig)
        x = data.flatten(1).to(device)
        kl = kld(zmu, zlogs)
        if method == 'nll':
            rec = -log_gaussian_prob(x, xmu, 2*xlogs).sum()
        elif method == 'sigma':
            log_sigma = softclip(model.log_sigma, -6, 9)
            rec = gaussian_nll(xmu, log_sigma, x).sum()
        else:
            rec = mse(x, xmu)
        loss = kl + rec
        loss.backward()
        optimizer.step()
        if batch_idx % 200 == 0:
            print("epoch", epoch, "losses: ", rec.item(), kl.item())



for epoch in range(10):
    #train(epoch)
    trainSC(epoch, 'sigma')

fcounts = np.load(countspath, allow_pickle=True)
countsDF = pd.DataFrame(**fcounts)

(countsDF.sum(axis = 0) == 0).sum()
ind = countsDF.sum(axis = 0) == 0

ind = countsDF.var(axis = 0) >= 0.5

len(countsDF[ind])

countsTensor = torch.FloatTensor(countsDF.to_numpy())

ind = countsTensor.std(dim=0) > 0.15

countsTensor = countsTensor[ind]

#normalizedCounts = (countsTensor - countsTensor.mean(dim=0)) / countsTensor.std(dim=0)
normalizedCounts = countsTensor / countsTensor.max()
normalizedCounts = normalizedCounts - normalizedCounts.mean()
normalizedCounts /= normalizedCounts.std()

data = torch.tensor(testD.normalized_counts.to_numpy())

labels = testD.labels.to_numpy()
labels = labels[:,0].astype('int')
labels = torch.IntTensor(labels)

model.cpu()

zmu, zlogs, xmu, xlogs = model(data.float())


plot_tsne(zmu, labels, "roVAE")

plot_tsne(data, labels, "roVAE")

clusterable_embedding = umap.UMAP(
    n_neighbors=20,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(data)

plt.scatter(clusterable_embedding[:,0], clusterable_embedding[:,1], c =
        labels, s=1.4, cmap="viridis", )

plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
