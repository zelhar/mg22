# fun fun 2022-01-29
# https://github.com/eelxpeng/dec-pytorch/blob/master/lib/idec.py
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
from typing import Union, Any, cast, IO, TextIO

# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
from my_torch_utils import scsimDataset
import scsim.scsim as scsim

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam

from toolz import take, drop
import opt_einsum

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans



print(torch.cuda.is_available())

# analytical kld between N(mu,logvar/2) and N(0,1)
kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def kl_discrete(p : torch.Tensor ,q : torch.Tensor):
    """mean over the batch dim...
        make sure that p,q are nowhere 0
    """
    return torch.mean(
            torch.sum(
                torch.log(p) - torch.log(q), dim=-1))

def soft_assign(z, mu, alpha=1):
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - mu)**2, dim=2) / alpha)
    q = q**(alpha+1.0)/2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q

class Net(nn.Module):
    def __init__(
        self,
        nin: int = 28 ** 2,
        nh: int = 2*1024,
        nout: int = 20,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin
        self.nout = nout
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        z = self.net(x)
        return z

class AE(nn.Module):
    def __init__(
        self,
        nz: int = 20,
        nh: int = 2*1024,
        nin: int = 28 ** 2,
        imgsize: Optional[int] = 28,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin
        self.nz = nz
        self.imgsize = imgsize
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nz),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nin),
            nn.Sigmoid(),
            #nn.Unflatten(1, (1, imgsize, imgsize)),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decode(z)
        return y, z


class SigmaVAE(nn.Module):
    def __init__(
        self,
        nz: int = 20,
        nh: int = 2*1024,
        nin: int = 28 ** 2,
        imgsize: Optional[int] = 28,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin
        self.nz = nz
        self.imgsize = imgsize
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
        )
        self.xmu = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nin),
                )
        self.zmu =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        self.zlogvar =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        #self.log_sigma = torch.nn.Parameter(torch.zeros(1)[0], requires_grad=True)
        # per pixel sigma:
        self.log_sigma = torch.nn.Parameter(torch.zeros(nin), requires_grad=True)

    def reparameterize(self, mu, logsig, ):
        eps = torch.randn(mu.shape).to(mu.device)
        sigma = (logsig).exp()
        return mu + sigma * eps

    def encode(self, x):
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder(z)
        mu = self.xmu(h)
        return mu

    def forward(self, x):
        zmu, zlogvar = self.encode(x)
        z = self.reparameterize(zmu, 0.5*zlogvar)
        xmu = self.decode(z)
        return zmu, zlogvar, xmu

    def reconstruction_loss(self, x, xmu, log_sigma):
        # log_sigma is the parameter for 'global' variance on x
        #result = gaussian_nll(xmu, xlogsig, x).sum()
        result = -log_gaussian_prob(x, xmu, log_sigma).sum()
        return result
    
    def loss_function(self, x, xmu, log_sigma, zmu, zlogvar):
        batch_size = x.size(0)
        rec = self.reconstruction_loss(x, xmu, log_sigma) / batch_size
        kl = kld(zmu, zlogvar) / batch_size
        return rec, kl

    def fit(self, train_loader, num_epochs=10, lr=1e-3,
            optimizer = None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        if not optimizer:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                log_sigma = ut.softclip(self.log_sigma, -2, 2)
                #self.log_sigma.fill_(log_sigma)
                x = data.flatten(1).to(device)
                zmu, zlogvar, xmu = self.forward(x)
                rec, kl = self.loss_function(x, xmu, log_sigma, zmu, zlogvar)
                loss = rec + kl
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print("loss = ",
                            loss.item(),
                            kl.item(),
                            rec.item(),
                            )
        self.cpu()
        optimizer = None
        print('done training')
        return None


transform = transforms.Compose([
    transforms.ToTensor(),
    #normalize,
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

class IDEC(nn.Module):
    """
    Improved deep embedded clustering
    see Xifeng Guo. 2017.4.30
    """
    def __init__(
        self,
        nin: int = 28 ** 2,
        nh: int = 2*1024,
        nz: int = 30,
        nclusters : int = 20,
        alpha : float = 1.0,
        gamma : float = 0.1,
        encoder : Optional[Net] = None,
        decoder: Optional[Net] = None,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nin = nin
        self.nh = nh
        self.nz = nz
        self.nclusters = nclusters
        self.alpha = alpha
        self.gamma = gamma
        self.encoder = encoder
        self.decoder = decoder
        # the parameter mu designates the cluster centers
        self.mu = nn.Parameter(torch.Tensor(nclusters, nz))
        if not self.encoder:
            self.encoder = Net(nin, nh, nz)
        if not self.decoder:
            self.decoder = Net(nz, nh, nin)

    def save_model(self, path : Union[str, IO]) -> None:
        torch.save(self.state_dict(), path)

    def load_model(self, path : Union[str, IO]) -> None:
        pretrained_dict = torch.load(path, map_location=lambda storage, loc: storage)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        self.load_state_dict(model_dict)
        #loaded_dict = torch.load(path, )
        #model_dict = self.state_dict()
        #model_dict.update(loaded_dict)
        #self.load_state_dict(model_dict)

    def soft_assign(self, z):
        """
        calculate the batch's students t distribution on the hidden 
        space
        """
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
        q = q**(self.alpha+1.0)/2.0
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    def target_distribution(self, q):
        p = q**2 / torch.sum(q, dim=0)
        p = p / torch.sum(p, dim=1, keepdim=True)
        return p

    def loss_function(self, x, x_hat, p, q):
        """loss function for the clustering phase."""
        kld_loss = kl_discrete(p,q)
        rec_loss = nn.MSELoss(reduction='mean')(x, x_hat)
        loss = rec_loss + kld_loss * self.gamma
        return loss

    def encode(self, x):
        return self.encoder(x)

    def decode(self,z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        q = self.soft_assign(z)
        return z, q, x_hat

    def trainAE(self, train_loader, lr=1e-3):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                batch_size = data.shape[0]
                x = data.flatten(1).to(device)
                z = self.encode(x)
                x_hat = self.decode(z)
                rec_loss = nn.MSELoss(reduction="mean")(x_hat, x)
                rec_loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print("rec_loss = ",
                            rec_loss.item(),
                            )
        self.cpu()
        optimizer = None
        print('done training')
        return None

    def init_kmeans(self, data):
        """
        initiate the kmeans cluster heads
        """
        self.cpu()
        lattent_data = self.encode(data)
        kmeans = KMeans(self.nclusters, n_init=20)
        y_pred = kmeans.fit_predict(lattent_data.detach().numpy())
        self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        self.y_pred = y_pred
        self.q = q = self.soft_assign(lattent_data)
        self.p = p = self.target_distribution(q)


    def fit(self, train_loader, num_epochs=10, lr=1e-3, tol=1e-3,
            update_interval = 1, noreconst=False,
            optimizer = None):
        data = train_loader.dataset.data.flatten(1).float()/255
        self.init_kmeans(data)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        if not optimizer:
            optimizer = optim.Adam(self.parameters(), lr=lr)
        y_pred_last = self.y_pred
        for epoch in range(num_epochs):
            if epoch % update_interval == 0:
                lattent_data = self.encode(data.to(device))
                q = self.q = self.soft_assign(lattent_data)
                p = self.p = self.target_distribution(q).data
                self.y_pred = y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / len(data)
                y_pred_last = y_pred
                if epoch > 0 and delta_label > tol:
                    print('reached tolerance threshold')
                    break
            self.to(device)
            step = 256
            idx=0
            for batch in range(0, len(data), step): 
                idx += batch
                #for idx, (data, labels) in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                x = data[batch : batch + step].to(device)
                pbatch = self.p[batch : batch + step].to(device)
                z, qbatch, x_hat = self(x)
                #loss = self.loss_function(x, x_hat, pbatch, qbatch)
                loss_kl = kl_discrete(pbatch, qbatch)
                loss_rec = nn.MSELoss(reduction="mean")(x, x_hat)
                loss = loss_rec + self.gamma * loss_kl
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print("loss = ",
                            loss_kl.item(),
                            loss_rec.item(),
                            loss.item(),
                            )
        self.cpu()
        optimizer = None
        print('done fitting')
        return None
        






def fit(model, train_loader, num_epochs=10, lr=1e-3, tol=1e-3,
        update_interval = 1, noreconst=False):
    data = train_loader.dataset.data.flatten(1).float()/255
    model.init_kmeans(data)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    y_pred_last = model.y_pred
    for epoch in range(num_epochs):
        if epoch % update_interval == 0:
            lattent_data = model.encode(data.to(device))
            q = model.q = model.soft_assign(lattent_data)
            p = model.p = model.target_distribution(q).data
            model.y_pred = y_pred = torch.argmax(q, dim=1).data.cpu().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / len(data)
            y_pred_last = y_pred
            if epoch > 0 and delta_label > tol:
                print('reached tolerance threshold')
                break
        model.to(device)
        step = 256
        idx=0
        for batch in range(0, len(data), step): 
            idx += batch
            #for idx, (data, labels) in enumerate(train_loader):
            model.train()
            model.requires_grad_(True)
            optimizer.zero_grad()
            x = data[batch : batch + step].to(device)
            pbatch = model.p[batch : batch + step].to(device)
            z, qbatch, x_hat = model(x)
            loss = model.loss_function(x, x_hat, pbatch, qbatch)
            loss.backward()
            optimizer.step()
            if idx % 300 == 0:
                print("loss = ",
                        loss.item(),
                        )
    model.cpu()
    print('done fitting')
    optimizer = None
    return None
    
print('foooooo')


