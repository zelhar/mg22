# fun fun 2022-01-29
# https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py
# https://anndata-tutorials.readthedocs.io/en/latest/annloader.html
# https://github.com/EricElmoznino/pyro_gmmvae
import gdown
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyro
import pyro.distributions as dist
import scanpy as sc
import seaborn as sns
import time
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
from anndata.experimental.pytorch import AnnLoader
from importlib import reload
from math import pi, sin, cos, sqrt, log
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
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
from my_torch_utils import fclayer, init_weights, buildNetwork
from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import scsimDataset
import my_torch_utils as ut

print(torch.cuda.is_available())


#class Encoder(nn.Module):
#    """
#    Gaussian Encoder module for VAE.
#    """
#
#    def __init__(
#        self,
#        nz: int = 10,
#        nh: int = 1024,
#        imgsize: int = 28,
#    ) -> None:
#        super(Encoder, self).__init__()
#        self.nin = nin = imgsize ** 2
#        self.nz = nz
#        self.imgsize = imgsize
#        self.encoder = nn.Sequential(
#            nn.Flatten(),
#            nn.Linear(nin, nh),
#            nn.LeakyReLU(),
#            nn.Linear(nh, nh),
#            nn.LeakyReLU(),
#            #nn.Tanh(),
#        )
#        self.zmu = nn.Sequential(
#            nn.Linear(nh, nz),
#        )
#        self.zlogvar = nn.Sequential(
#            nn.Linear(nh, nh),
#            nn.LeakyReLU(),
#            nn.Linear(nh, nz),
#            # nn.Softplus(),
#        )
#
#    def forward(self, x):
#        h = self.encoder(x)
#        mu = self.zmu(h)
#        logvar = torch.exp(self.zlogvar(h))
#        return mu, logvar
#
#
#class Decoder(nn.Module):
#    """
#    Bernoulli decoder module for VAE
#    """
#
#    def __init__(
#        self,
#        nz: int = 20,
#        nh: int = 1024,
#        imgsize: int = 28,
#        is_Bernoulli: bool = True,
#    ) -> None:
#        super(Decoder, self).__init__()
#        self.out = nout = imgsize ** 2
#        self.nz = nz
#        self.imgsize = imgsize
#        self.decoder = nn.Sequential(
#            nn.Linear(nz, nh),
#            nn.LeakyReLU(),
#            nn.Linear(nh, nh),
#            nn.LeakyReLU(),
#            #nn.Tanh(),
#        )
#        self.xmu = nn.Sequential(
#            nn.Linear(nh, nh),
#            nn.LeakyReLU(),
#            nn.Linear(nh, nout),
#        )
#
#    def forward(self, z):
#        h = self.decoder(z)
#        mu = self.xmu(h)
#        return mu
#
#
#class VAE(nn.Module):
#    """
#    VAE class for use with pyro!
#    note that we use
#    """
#
#    def __init__(
#        self,
#        nz: int = 10,
#        nh: int = 1024,
#        imgsize: int = 28,
#        is_Bernoulli: bool = True,
#    ) -> None:
#        super(VAE, self).__init__()
#        self.nin = nin = imgsize ** 2
#        self.out = nout = imgsize ** 2
#        self.nz = nz
#        self.imgsize = imgsize
#        self.encoder = Encoder(nz, nh, imgsize)
#        self.decoder = Decoder(nz, nh, imgsize)
#
#    # model describes p(x|z)p(z)
#    def model(self, x):
#        # register decoder
#        pyro.module("decoder", self.decoder)
#        with pyro.plate("data", x.shape[0]):
#            # setup hyperparameters for prior p(z)
#            z_loc = torch.zeros(x.shape[0], self.nz, dtype=x.dtype, device=x.device)
#            z_scale = torch.ones(x.shape[0], self.nz, dtype=x.dtype, device=x.device)
#            # sample from prior (value will be sampled by guide when computing the ELBO)
#            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#            # decode the latent code z
#            loc_img = self.decoder.forward(z)
#            # score against actual images (with relaxed Bernoulli values)
#            pyro.sample(
#                "obs",
#                #dist.Normal(loc_img, scale_img, validate_args=False).to_event(1),
#                dist.Bernoulli(loc_img, validate_args=False).to_event(1),
#                obs=x.reshape(-1, 784),
#            )
#            # return the loc so we can visualize it later
#            return loc_img
#
#    # define the guide (i.e. variational distribution) q(z|x)
#    def guide(self, x):
#        # register PyTorch module `encoder` with Pyro
#        pyro.module("encoder", self.encoder)
#        with pyro.plate("data", x.shape[0]):
#            # use the encoder to get the parameters used to define q(z|x)
#            z_loc, z_logvar = self.encoder.forward(x)
#            #z_scale = z_logvar.exp()  # it's technically logsigma
#            z_scale = z_logvar.clip(0,4)  # it's technically logsigma
#            # sample the latent code z
#            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
#
#    # define a helper function for reconstructing images
#    def reconstruct_img(self, x):
#        # encode image x
#        z_loc, z_logvar = self.encoder(x)
#        #z_scale = z_logvar.exp()
#        z_scale = z_logvar.clip(0,4)
#        # sample in latent space
#        z = dist.Normal(z_loc, z_scale).sample()
#        # decode the image (note we don't sample in image space)
#        loc_img = self.decoder(z)
#        return loc_img


class Encoder(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=1024):
        super().__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim=20, hidden_dim=1024):
        super().__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, 784)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=1024, use_cuda=True):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Bernoulli(loc_img, validate_args=False).to_event(1),
                obs=x.reshape(-1, 784),
            )
            # return the loc so we can visualize it later
            return loc_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        return loc_img

    # define a helper function to sample from p(z) and output p(x|z)
    def sample_img(self, num_samples, return_z=False):
        # sample from p(z)
        z = self.z_prior(num_samples).sample()
        loc_img = self.decoder.forward(z)
        if return_z:
            return loc_img, z
        else:
            return loc_img

    def z_prior(self, num_samples):
        # sample from p(z)
        z_loc = torch.zeros(num_samples, self.z_dim)
        z_scale = torch.ones(num_samples, self.z_dim)
        z = dist.Normal(z_loc, z_scale)
        return z

class GMMVAE(VAE):
    def __init__(
        self,
        n_cats,
        loc_sep=3.0,
        z_dim=50,
        hidden_dim=400,
        use_cuda=torch.cuda.is_available(),
    ):
        super().__init__(z_dim=z_dim, hidden_dim=hidden_dim, use_cuda=use_cuda)
        self.n_cats = n_cats
        self.component_locs = torch.zeros(self.n_cats, self.z_dim)
        for i in range(n_cats):
            self.component_locs[i, i] = 1 * loc_sep

    def z_prior(self, num_samples):
        cats = dist.Categorical(
            torch.ones(num_samples, self.n_cats) * 1 / self.n_cats
        ).sample()
        z_loc = self.component_locs[cats]
        z_scale = torch.ones(num_samples, self.z_dim)
        z = dist.Normal(z_loc, z_scale)
        return z


class Arguments:
    cuda: bool = True
    learning_rate: float = 1e-3
    jit: bool = False
    visdom_flag: bool = False
    num_epochs: int = 51
    test_frequency: int = 5
    tsne_iter: int = 10
    nz : int = 50
    is_Bernoulli : bool = False 

def train_epoch(svi, train_loader, device="cuda"):
    epoch_loss = 0
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        x = x.flatten(1)
        epoch_loss += svi.step(x)
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def test_epoch(svi, test_loader, device="cuda"):
    epoch_loss = 0
    for idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        x = x.flatten(1)
        epoch_loss += svi.evaluate_loss(x)
    normalizer_train = len(test_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

def create_svi(model, lr=1e-3, elbo=Trace_ELBO()):
    adam_args = {
        "lr": lr,
    }
    optimizer = Adam(adam_args)
    svi = SVI(model.model, model.guide, optimizer, loss=elbo)
    return svi

def train_loop(svi, model, train_loader, test_loader, num_epochs=10, device="cuda",
        test_frequency=5,):
    pyro.clear_param_store()
    model.to(device)
    train_elbo = []
    test_elbo = []
    for epoch in range(num_epochs):
        epoch_loss = train_epoch(svi, train_loader, device)
        print(
            "[epoch %03d] average training loss: %.4f"
            % (epoch, epoch_loss)
        )
        train_elbo.append(epoch_loss)
        if epoch % test_frequency == 0:
            test_epoch_loss = test_epoch(svi, test_loader, device)
            test_elbo.append(test_epoch_loss)
            print(
                "[epoch %03d] average testing loss: %.4f"
                % (epoch, test_epoch_loss)
            )


def train(args: Arguments, train_loader, test_loader) -> VAE:
    pyro.clear_param_store()
    # setup the VAE
    #vae = VAE(nz=args.nz, is_Bernoulli=args.is_Bernoulli)
    vae = VAE(50, 1024, True)
    # setup the optimizer
    adam_args = {
        "lr": args.learning_rate,
    }
    optimizer = Adam(adam_args)
    # setup the inference algorithm
    #elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    #elbo = TraceGraph_ELBO()
    elbo = Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    train_elbo = []
    test_elbo = []
    if args.cuda:
        vae.cuda()
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            x = x.flatten(1)
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)
        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )
        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.0
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(test_loader):
                x = x.flatten(1)
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)
            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print(
                "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )
        # if epoch == args.tsne_iter:
        # mnist_test_tsne(vae=vae, test_loader=test_loader)
        # plot_llk(np.array(train_elbo), np.array(test_elbo))
    return vae


