# useful function to be re-used in throught the tests in this project.
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple, ClassVar
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, NewType
import torch
from torch import nn, optim, distributions
import torch.utils.data
import torchvision.utils as vutils
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from torch import Tensor
from math import pi, sin, cos, sqrt, log
import json
import pickle
import anndata as ad
import scanpy as sc

import networkx as nx

from datetime import datetime

import toolz
from toolz import partial, curry
from toolz import groupby, count, reduce, reduceby, countby

import operator
from operator import add, mul

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

#kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def kld2normal(
    mu: Tensor,
    logvar: Tensor,
    mu2: Tensor,
    logvar2: Tensor,
):
    """
    unreduced KLD KLD(p||q) for two diagonal normal distributions.
    https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    result = 0.5 * (
        -1
        + (logvar.exp() + (mu - mu2).pow(2)) / logvar2.exp()
        + logvar2
        - logvar
    )
    return result

def soft_assign(z, mu, alpha=1):
    """
    Returns a nearly one-hot vector that indicates the nearest centroid
    to z.
    """
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - mu)**2, dim=2) / alpha)
    q = q**(alpha+1.0)/2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q

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

def softclip(tensor, min=-6.0, max=9.0):
    """
    softly clips the tensor values at the minimum/maximimum value.
    """
    result_tensor = min + nn.functional.softplus(tensor - min)
    result_tensor = max - nn.functional.softplus(max - result_tensor)
    return result_tensor


class SoftClip(nn.Module):
    """
    object oriented version of softclip
    """
    def __init__(self, min=-6.0, max=6.0):
        super(SoftClip, self).__init__()
        self.min = min
        self.max = max
    def forward(self, input):
        return softclip(input, self.min, self.max)

@curry
def fnorm(
    x: Tensor,
    mu: Union[float, Tensor] = 1.0,
    sigma: Union[float, Tensor] = 1.0,
    #reduction: str = "sum",
    reduction: Optional[str] = "None",
) -> Tensor:
    """
    normal distribution density (elementwise) with optional reduction (logsum/sum/mean)
    """
    x = -0.5 * ((x - mu) / (sigma)).pow(2)
    x = x.exp()
    x = x / (sqrt(2 * pi) * sigma)
    if reduction == "sum":
        return x.sum()
    elif reduction == "logsum":
        return x.log().sum()
    elif reduction == "mean":
        return x.mean()
    else:
        return x


# This one was tested and seems correct ;)
@curry
def logNorm(
        x : Tensor,
        mu : Tensor,
        logvar : Tensor,) -> Tensor:
    """
    gives log of the density of the normal distribution.
    element-wise, no reductions.
    """
    y = -0.5 * (log(2 * pi) + logvar + (x - mu).pow(2) / logvar.exp() )
    return y


def init_weights(m: torch.nn.Module) -> None:
    """
    Initiate weights with random values, depending
    on layer type.
    In place, use the apply method.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

@curry
def replicate(x : Tensor, expand=(1,)):
    """
    Replicate a tensor in a new dimension (the new 0 dimension),
    creating n=ncopies identical copies.
    """
    y = torch.ones(expand + x.shape).to(x.device)
    return y * x



@curry
def normalize(
    x: Tensor,
    mu: Union[float, Tensor] = 0.5,
    sigma: Union[float, Tensor] = 0.5,
    clamp: bool = False,
) -> Tensor:
    """
    x <- (x - mu) / sigma
    """
    y = (x - mu) / sigma
    if clamp:
        y = y.clamp(0, 1)
    return y


@curry
def denorm(
    x: Tensor,
    mu: Union[float, Tensor] = 0.5,
    sigma: Union[float, Tensor] = 0.5,
    clamp: bool = True,
) -> Tensor:
    """
    inverse of normalize
    x <- sigma * x + mu
    """
    y = sigma * x + mu
    if clamp:
        y = y.clamp(0, 1)
    return y


def save_reconstructs(encoder,
                      decoder,
                      x,
                      epoch,
                      nz=20,
                      device="cuda",
                      normalized=False):
    """
    saves reconstruction output of the decoder as .png immage,
    not a very useful function, just here for legacy.
    """
    with torch.no_grad():
        x = x.to(device)
        y = encoder(x)
        sample = decoder(y).cpu()
        if normalized:
            sample = denorm(sample)
            x = denorm(x)
        # save_image(x.view(x.shape[0], 3, 28, 28),
        save_image(x, "results/originals_" + str(epoch) + ".png")
        save_image(sample, "results/reconstructs_" + str(epoch) + ".png")


def save_random_reconstructs(model,
                             nz,
                             epoch,
                             device="cuda",
                             normalized=False):
    """
    Legacy function.
    """
    with torch.no_grad():
        # sample = torch.randn(64, nz).to(device)
        sample = torch.randn(64, nz).to(device)
        sample = model(sample).cpu()
        if normalized:
            sample = denorm(sample)
        save_image(sample, "results/sample_" + str(epoch) + ".png")


def plot_images(imgs, nrow=16, transform=nn.Identity(), out=plt):
    """
    plots input immages in a grid.
    imgs: tensor of images with dimensions (batch, channell, height, widt)
    nrow: number of rows in the grid
    out: matplotlib plot object
    outputs grid_imgs: the image grid ready to be ploted.
    also plots the result with 'out'.
    """
    imgs = transform(imgs)
    grid_imgs = make_grid(imgs, nrow=nrow).permute(1, 2, 0)
    #plt.imshow(grid_imgs)
    out.imshow(grid_imgs)
    out.grid(False)
    out.axis("off")
    plt.pause(0.05)
    return grid_imgs

def plot_2images(img1, img2, nrow=16, transform=nn.Identity(), ):
    """
    just like plot_images but takes two image sets and creates two image grids,
    which are also plotted side by side.
    """
    img1 = transform(img1)
    img2 = transform(img2)
    grid_img1 = make_grid(img1, nrow=nrow).permute(1, 2, 0)
    grid_img2 = make_grid(img2, nrow=nrow).permute(1, 2, 0)
    fig, axs = plt.subplots(2,1)
    axs[0].imshow(grid_img1)
    axs[1].imshow(grid_img2)
    axs[0].grid(False)
    axs[0].axis("off")
    axs[1].grid(False)
    axs[1].axis("off")
    plt.pause(0.05)
    return grid_img1, grid_img2

def plot_tsne(z_loc, classes, name):
    """
    Legacy function. just use your favorite plotting tool.
    """
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
    plt.legend([str(i) for i in classes])
    #fig.savefig("./results/" + str(name) + "_embedding.png")

def fclayer(
        nin: int,
        nout: int,
        batchnorm: bool = True,
        dropout: float = 0.2,
        activation: nn.Module = nn.ReLU(),
) -> nn.Module:
    """
    define one fully connected later where nin, nout are the dimensions
    of the input and output.
    Perform dropout on the input (a value between 0 and 1, 0 means no dropout)
    and optional batchnormalization before the activation.
    Can also provide the activation function (ReLU is th edefault)
    """
    fc = nn.Sequential()
    if 0 < dropout < 1:
        fc.add_module("dropout", nn.Dropout(p=dropout))
    fc.add_module("linear", nn.Linear(nin, nout))
    if batchnorm:
        fc.add_module("batchnorm", nn.BatchNorm1d(num_features=nout))
    fc.add_module("activation", activation)
    return fc

def buildNetwork(
    layers: List[int],
    dropout: int = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
        if activation:
            net.add_module("activation" + str(i), activation)
    return net
    #return nn.Sequential(*net)

def buildNetworkv2(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

def buildNetworkv3(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    layernorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if layernorm:
            #net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
            net.add_module("layernotm" + str(i), nn.LayerNorm(layers[i],))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

def buildNetworkv4(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            #net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
            net.add_module("layernotm" + str(i), nn.LayerNorm(layers[i],))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

def buildNetworkv5(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
            #net.add_module("layernotm" + str(i), nn.LayerNorm(layers[i],))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

def buildNetworkv6(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    # just one dropout layer
    for i in range(1, len(layers)-1):
        if dropout > 0 and i==1:
            net.add_module("dropout" + str(i-1), nn.Dropout(dropout))
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if batchnorm:
            net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
            #net.add_module("layernotm" + str(i), nn.LayerNorm(layers[i],))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net


def buildCNetworkv1(
    nc : int = 1,
    nin : int = 28**2,
    nf : int = 32,
    nout : int = 2**12,
    dropout: float = 0,
    #activation: Optional[nn.Module] = nn.LeakyReLU(),
    #batchnorm: bool = False,
):
    """
    build a 1d CNN.
    maps 1d input into initial size 2**12,
    each conv reduces size by a factor of 4.
    rturns output 1d of size nout.
    The output layer is always linear
    """
    net = nn.Sequential(
            nn.Linear(nin, 2**12,),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(num_features=2**12,),
            nn.LeakyReLU(),
            nn.Unflatten(1, (nc,2**12)),
            nn.Conv1d(nc, nf, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf,),
            nn.LeakyReLU(),
            nn.Conv1d(nf, nf*2, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*2,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*2, nf*4, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*4,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*4, nf*8, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*8,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*8, nf*16, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*16,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*16, nf*32, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*32,),
            nn.LeakyReLU(),
            nn.Flatten(1),
            nn.Linear(nf*32, nout),
            )
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    return net

def buildCNetworkv2(
    nc : int = 1,
    nin : int = 28**2,
    nf : int = 32,
    nout : int = 2**10,
    dropout: float = 0,
    #activation: Optional[nn.Module] = nn.LeakyReLU(),
    #batchnorm: bool = False,
):
    """
    build a 1d CNN.
    maps 1d input into initial size 2**10,
    each conv reduces size by a factor of 4.
    rturns output 1d of size nout.
    The output layer is always linear
    """
    net = nn.Sequential(
            nn.Linear(nin, 2**10,),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(num_features=2**10,),
            nn.LeakyReLU(),
            nn.Unflatten(1, (nc,2**10)),
            nn.Conv1d(nc, nf, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf,),
            nn.LeakyReLU(),
            nn.Conv1d(nf, nf*2, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*2,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*2, nf*4, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*4,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*4, nf*8, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*8,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*8, nf*16, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*16,),
            nn.LeakyReLU(),
            nn.Flatten(1),
            nn.Linear(nf*16, nout),
            )
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    return net

def buildTCNetworkv1(
    # nc : int = 1,
    nin: int = 28 ** 2,
    nf: int = 32,
    nout: int = 2 ** 12,
    dropout: float = 0,
    # activation: Optional[nn.Module] = nn.LeakyReLU(),
    # batchnorm: bool = False,
):
    """
    conv1t network, each step doubles or quadrupels input size.
    """
    net = nn.Sequential(
        nn.Unflatten(1, (1, nin)),
        nn.ConvTranspose1d(
            1,
            nf * 32,
            kernel_size=8,
            stride=4,
            padding=2,
        ),
        nn.BatchNorm1d(
            num_features=nf * 32,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 32,
            nf * 16,
            kernel_size=8,
            stride=4,
            padding=2,
        ),
        nn.BatchNorm1d(
            num_features=nf * 16,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 16,
            nf * 8,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm1d(
            num_features=nf * 8,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 8,
            1,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm1d(
            num_features=1,
        ),
        nn.LeakyReLU(),
        nn.Flatten(1),
        nn.Linear(nin*4*4*2*2, nout),
    )
    return net

def buildTCNetworkv2(
    # nc : int = 1,
    nin: int = 28 ** 2,
    nf: int = 32,
    nout: int = 2 ** 12,
    dropout: float = 0,
    # activation: Optional[nn.Module] = nn.LeakyReLU(),
    # batchnorm: bool = False,
):
    """
    conv1t network, each step doubles or quadrupels input size.
    """
    net = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Unflatten(1, (1, nin)),
        nn.ConvTranspose1d(
            1,
            nf * 32,
            kernel_size=8,
            stride=4,
            padding=2,
        ),
        nn.BatchNorm1d(
            num_features=nf * 32,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 32,
            nf * 16,
            kernel_size=8,
            stride=4,
            padding=2,
        ),
        nn.BatchNorm1d(
            num_features=nf * 16,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 16,
            1,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm1d(
            num_features=1,
        ),
        nn.LeakyReLU(),
        nn.Flatten(1),
        nn.Linear(nin*4*4*2, nout),
    )
    return net



@curry
def mixedGaussianCircular(k=10, sigma=0.025, rho=3.5, j=0):
    """
    Sample from a mixture of k 2d-gaussians. All have equal variance (sigma) and
    correlation coefficient (rho), with the means equally distributed on the
    unit circle.
    example:
    gauss = mixedGaussianCircular(rho=0.01, sigma=0.5, k=10, j=0)
    mix = distributions.Categorical(torch.ones(10,))
    comp = distributions.Independent(gauss, 0)
    gmm = distributions.MixtureSameFamily(mix, comp)
    """
    # cat = distributions.Categorical(torch.ones(k))
    # i = cat.sample().item()
    # theta = 2 * torch.pi * i / k
    theta = 2 * torch.pi / k
    v = torch.Tensor((1, 0))
    T = torch.Tensor([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    S = torch.stack([T.matrix_power(i) for i in range(k)])
    mu = S @ v
    # cov = sigma ** 2 * ( torch.eye(2) + rho * (torch.ones(2, 2) - torch.eye(2)))
    # cov = cov @ S
    cov = torch.eye(2) * sigma**2
    cov[1, 1] = sigma**2 * rho
    cov = torch.stack([
        T.matrix_power(i + j) @ cov @ T.matrix_power(-i - j) for i in range(k)
    ])
    gauss = distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
    return gauss

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, data : Tensor, labels : Tensor, labeled_portion : float):
        super().__init__()
        self.data = data
        self.labels = labels
        self.markedlabels = distributions.Bernoulli(labeled_portion).sample(labels.shape).long()
        return
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx], self.markedlabels[idx]

class scsimDataset(torch.utils.data.Dataset):
    def __init__(self, countspath : str, idspath : str, genepath : Optional[str] = None,):
        super(scsimDataset, self).__init__()
        fcounts = np.load(countspath, allow_pickle=True)
        flabels = np.load(idspath, allow_pickle=True)
        self.counts = pd.DataFrame(**fcounts)
        self.labels = pd.DataFrame(**flabels)
        flabels.close()
        fcounts.close()
        if genepath:
            fgeneparams = np.load(genepath, allow_pickle=True)
            self.geneparams = pd.DataFrame(**fgeneparams) 
            fgeneparams.close()

        self._cpath = countspath
        self._lpath = idspath
        self._gpath = genepath

        # notmalized counts:
        self.normalized_counts = (self.counts - self.counts.mean(axis=0)) / self.counts.std(ddof=0, axis=0)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx : int):
        ncounts = self.normalized_counts.iloc[idx].astype('float')
        counts = self.counts.iloc[idx].astype('float')
        label = self.labels.iloc[idx, 0]
        return torch.FloatTensor(ncounts), torch.FloatTensor(counts), label

    def __train_test_split__(self, n : int) -> Union[Tuple[Any,Any], None]:
        if n >= self.__len__() - 1:
            print("error, cannot split because n exceeds data length")
            return None
        trainD = scsimDataset(self._cpath, self._lpath)
        trainD.counts = self.counts[0:n]
        trainD.normalized_counts = self.normalized_counts[0:n]
        trainD.labels = self.labels[0:n]
        testD = scsimDataset(self._cpath, self._lpath)
        testD.counts = self.counts[n:]
        testD.normalized_counts = self.normalized_counts[n:]
        testD.labels = self.labels[n:]
        return trainD, testD

def randomSubset(
        s : int,
        r: float,
        ):
    """
    returns a numpy boolean 1d array of size size,
    with approximately ratio of r*s True values.
    s must be positive integer
    r must be in the range [0,1]
    """
    x = np.random.rand(s)
    x = x <= r
    return x


class Blobs:
    """
    samples gaussian blobs.
    """
    def __init__(
            self,
            means: Tensor = torch.rand(5,2) * 5e0,
            scales: Tensor = torch.rand(5,2) * 2e-1,
            ) -> None:
        self.means = means
        self.scales = scales
        self.nclasses = means.shape[0]
        self.ndim = means.shape[1]
        self.comp = distributions.Normal(means, scales)
        self.mix = distributions.Categorical(torch.ones((self.nclasses,)))
        return
    def sample(self, batch_size=(100,)):
        l = self.mix.sample(batch_size)
        x = self.comp.sample(batch_size)
        s = torch.vstack(
                [x[i,l[i]] for i in range(len(x))]
                )
        return s, l, x
    def plotSample(self, batch_size=(300,)):
        s, l, x = self.sample(batch_size)
        sns.scatterplot(x=s[:,0], y=s[:,1], hue=l,)
        return


class SynteticSampler:
    """
    An object which samples from a mixture distribution.
    Should contain methods that return batched samples.
    """

    def __init__(
        self,
        means: torch.Tensor = torch.rand(5, 2),
        logscales: torch.Tensor = torch.randn(5) * 5e-1,
        noiseLevel: float = 5e-2,
    ) -> None:
        self.means = means
        self.logscales = logscales
        self.scales = logscales.exp()
        self.n_dim = means.shape[1]
        self.n_classes = means.shape[0]
        #self.m = m = distributions.Normal(
        #    loc=means,
        #    scale=logscales.exp(),
        #)
        self.noiseLevel = noiseLevel
        return

    def sample(self, batch_size=(100,)):
        m = distributions.Categorical(probs=torch.ones(self.n_classes))
        labels = m.sample(batch_size)
        locs = torch.stack(
                [self.means[labels[i]] for i in range(len(labels))], dim=0)
        scales = torch.stack(
                [self.scales[labels[i]] for i in range(len(labels))],
                dim=0).unsqueeze(1)
        noise = torch.randn_like(locs) * self.noiseLevel
        theta = torch.rand_like(labels*1e-1) * pi * 2
        data = torch.zeros_like(locs)
        data[:,0] = theta.cos()
        data[:,1] = theta.sin()
        data = data * scales + locs + noise
        return data, labels, locs, scales

    def plotData(
        self,
    ) -> None:
        # data = self.m.sample((1000,))
        data = torch.rand((300, self.n_classes)) * pi * 2
        fig = plt.figure()
        for idx in range(self.n_classes):
            color = plt.cm.Set1(idx)
            # x = data[:,idx,0]
            # y = data[:,idx,1]
            x = (
                data[:, idx].cos()
                * self.logscales[idx].exp()
                + self.means[idx, 0]
                + torch.randn(300) * self.noiseLevel
            )
            y = (
                data[:, idx].sin()
                * self.logscales[idx].exp()
                + self.means[idx, 1]
                + torch.randn(300) * self.noiseLevel
            )
            plt.scatter(
                x,
                y,
                color=color,
                s=10,
                cmap="viridis",
            )
        plt.legend([str(i) for i in range(self.n_classes)])
        plt.title("data plot")

    def plotSampleData(
        self,
    ) -> None:
        data, labels, _, _  = self.sample((1000,))
        fig = plt.figure()
        for idx in range(self.n_classes):
            color = plt.cm.Set1(idx)
            # x = data[:,idx,0]
            # y = data[:,idx,1]
            x = data[labels == idx][:, 0]
            y = data[labels == idx][:, 1]
            plt.scatter(
                x,
                y,
                color=color,
                s=10,
                cmap="viridis",
            )
        plt.legend([str(i) for i in range(self.n_classes)])
        plt.title("data plot")

class SynteticDataSet(torch.utils.data.Dataset):
    def __init__(self, data : Tensor, labels : Tensor,):
        super().__init__()
        self.data = data
        self.labels = labels
        return
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]
    def __len__(self):
        return len(self.labels)

class SynteticDataSetV2(torch.utils.data.Dataset):
    """
    with arbitrary number of variables.
    """
    def __init__(self, dati : List[Tensor], ):
        super().__init__()
        self.dati = dati
        self.numvars = len(dati)
        return
    def __getitem__(self, idx : int):
        #ret = [x[idx] for x in self.dati]
        #ret = tuple(ret)
        #return ret
        #return tuple([x[idx] for x in self.dati])
        return [x[idx] for x in self.dati]
    def __len__(self):
        return len(self.dati[0])

def diffMatrix(A : np.ndarray, alpha : float = 0.25):
    """
    Returns the diffusion Kernel K for a given adjacency matrix 
    A, and restart pobability alpha.
    K = α[I - (1 - α)AD^-1]^-1
    """
    #D = np.diag(A.sum(0))
    T = A / A.sum(0)
    I = np.eye(A.shape[0])
    K = alpha * np.linalg.inv(I - (1 - alpha)*T)
    return K

def diffCluster(
        A : np.ndarray,
        num_clusters : int = 3,
        alpha : float=0.25, ):
    """
    A : adj matrix
    alpha: restart prob
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(A)))
    K = diffMatrix(A, alpha)
    # pageranking (increasing)
    pr = np.argsort(
            K @ np.ones(len(A)))
    n = len(K)
    nodes = list(np.arange(n))
    i = 0
    K = K.T
    #while n > num_clusters:
    while nx.number_connected_components(G) > num_clusters:
        s = pr[i]
        nodes.remove(s)
        t = K[s,nodes].argmax()
        x = nodes[t]
        G.add_edge(s, x)
        i = i+1
    clusters = np.zeros(len(A))
    i = 0
    for c in nx.connected_components(G):
        for j in c:
            clusters[j] = i
        i = i+1
    return G, clusters

def diffCluster2(
        A : np.ndarray,
        num_neighbors: int = 3,
        alpha : float=0.25, ):
    """
    A : adj matrix
    alpha: restart prob
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(A)))
    K = diffMatrix(A, alpha)
    # pageranking (increasing)
    n = len(K)
    nodes = list(np.arange(n))
    K = K.T
    i = 0
    for i in range(n):
       l = np.argpartition(-K[i], num_neighbors)[:num_neighbors]
       for j in l:
           G.add_edge(i,j)
    clusters = np.zeros(len(A))
    i = 0
    for c in nx.connected_components(G):
        for j in c:
            clusters[j] = i
        i = i+1
    return G, clusters

def softArgMaxOneHot(
        x : torch.Tensor,
        factor : float = 1.2e2,
        a : float = 4,
        #one_hot : bool = True,
        ) -> torch.Tensor:
    """
    x: 1d float tensor or batch of vectors.
    factor: larger factor will make the returned result
    more similar to pure argmax (less smooth).
    returns a nearly one-hot vector indicating
    the maximal value of x. 
    Possible not currently implemented feature:
    if one_hot==False,
    returns apprixmately the argmax index itselg.
    """
    #z = 1.2e1 * x / x.norm(1)
    #z = z.exp().softmax(-1)
    #z = factor * x / x.norm(1, dim=-1).unsqueeze(-1)
    z = factor * (1 + x / x.norm(1, dim=-1).unsqueeze(-1))
    z = z.pow(a).softmax(-1)
    return z

class SoftArgMaxOneHot(nn.Module):
    """
    class version of the eponymous function.
    """
    def __init__(self, factor=1.2e2,):
        super().__init__()
        self.factor = factor
    def forward(self, input):
        return softArgMaxOneHot(input, self.factor)

def mutualInfo(p : torch.Tensor, q : torch.Tensor,):
    """
    p,q : n×y (batches of categorical distributions over y).
    returns their mutual information:
    I(p,q) = \int \log(P(x,y) - \logp(x) - \logq(y)dP(x,y)
    """
    batch_size, y = p.size()
    pp = p.reshape(batch_size, y, 1)
    qq = q.reshape(batch_size, 1, y)
    P = pp @ qq
    P = P.mean(0) # P(x,y)
    Pi = P.sum(1).reshape(y,1)
    Pj = P.sum(0).reshape(1,y)
    #Q = Pi @ Pj #P(x)P(y)
    #I = torch.sum(
    #        P * (P.log() - Q.log())
    #        )
    # alternatively:
    #I(X,Y) = H(X) + H(Y) - H(X,Y)
    HP = -torch.sum(
            P * P.log()
            )
    HPi = -torch.sum(
            Pi * Pi.log()
            )
    HPj = -torch.sum(
            Pj * Pj.log()
            )
    Ia = HPi + HPj - HP
    #return I, P, Q, Ia
    return Ia

def urEntropy(p : torch.Tensor,):
    """
    returns -p * log(p) (element-wise, so unreduced).
    """
    return -p * p.log()

def mutualInfo2(p : torch.Tensor, q : torch.Tensor,):
    """
    p,q : n×y (batches of categorical distributions over y).
    returns their mutual information:
    I(p,q) = \int \log(P(x,y) - \logp(x) - \logq(y)dP(x,y)
    """
    batch_size, n = p.size()
    #p(x,y):
    P = torch.einsum("...x,...y -> ...xy", p, q).mean(0)
    #p(x):
    #Px = torch.einsum("xy -> x", P)
    Px = P.sum(1,keepdim=True) #(n,1)
    #p(y):
    #Py = torch.einsum("xy -> y", P)
    Py = P.sum(0,keepdim=True) #(1,n)
    #p(x | y):
    Px_y = P / Py
    #p(y | x):
    #Py_x = P / Px
    #H(X)
    Hx = urEntropy(Px).sum()
    #H(X|Y)
    Hx_y = -(P * Px_y.log()).sum()
    return Hx - Hx_y

def mutualInfo3(p : torch.Tensor, q : torch.Tensor, r : torch.Tensor):
    """
    p,q,r : n×y (batches of categorical distributions over y).
    returns their mutual information:
    I(p,q,r) = I(p,q) - I(p,q|r)
    be warned it can be negative and is hard to interpret,
    """
    Ipq = mutualInfo2(p,q)
    # p(x,y,z):
    P = torch.einsum("...x,...y,...z -> ...xyz", p, q,r).mean(0)
    # p(x,z):
    Pxz = P.sum(1, keepdim=True)
    # p(y,z)
    Pyz = P.sum(0, keepdim=True)
    # p(z):
    Pz = P.sum((0,1), keepdim=True)
    # I(x,y | z):
    #Ipq_r = (P * (P * Pz / Pxz / Pyz).log()).sum()
    temp = (P.log() + Pz.log() - Pxz.log() - Pyz.log())
    temp = temp * P
    Ipq_r = temp.sum()
    return Ipq - Ipq_r

def totalCorrelation3(p,q,r):
    """
    returns 
    Dkl(p(x,y,z) || p(x)p(y)p(z))
    """
    #p(x,y,z):
    P = torch.einsum("...x,...y,...z -> ...xyz", p, q,r).mean(0)
    Px = P.sum((1,2), keepdim=True)
    Py = P.sum((0,2), keepdim=True)
    Pz = P.sum((0,1), keepdim=True)
    #p(x)p(y)p(z):
    Q = Px * Py * Pz
    tc = P * (P.log() - Q.log())
    return tc.sum()

@curry
def checkCosineDistance(
        x : torch.Tensor,
        model : nn.Module,
        ) -> torch.Tensor:
    """
    x : the input tensor
    model: the Autoencoder to feed x into.
    outputs mean cosDistance(x, y)
    where y = reconstruction of x by model.
    """
    #model.to(x.device)
    y = model(x.flatten(1),)['rec']
    cosD = torch.cosine_similarity(x.flatten(1), y, dim=-1).mean()
    return cosD




def estimateClusterImpurity(
        model,
        x,
        labels,
        device : str = "cpu",
        ):
    model.eval()
    model.to(device)
    output = model(x.to(device))
    model.cpu()
    y = output["q_y"].detach().to("cpu")
    del output
    n = y.shape[1] # number of clusters
    r = -np.ones(n) # homogeny index
    #r = np.zeros(n) # homogeny index
    #p = np.zeros(n) # label assignments to the clusters
    p = -np.ones(n) # label assignments to the clusters
    s = -np.ones(n) # label assignments to the clusters
    for i in range(n):
        c = labels[y.argmax(-1) == i]
        if c.shape[0] > 0:
            r[i] = c.sum(0).max().item() / c.shape[0] 
            p[i] = c.sum(0).argmax().item()
            s[i] = c.shape[0]
    return r, p, s

def estimateClusterImpurityHelper(
        model,
        x,
        labels,
        device : str = "cpu",
        ):
    model.eval()
    model.to(device)
    output = model(x.to(device))
    model.cpu()
    y = output["q_y"].detach().to("cpu")
    del output
    return y


def estimateClusterImpurityLoop(
        model,
        xs,
        labels,
        device : str = "cpu",
        #cond1 : Optional[torch.Tensor],
        ):
    y = []
    model.eval()
    model.to(device)
    data_loader = torch.utils.data.DataLoader(
            dataset=SynteticDataSet(
                data=xs,
                labels=labels,
                ),
            batch_size=128,
            shuffle=False,
            )
    for x, label in data_loader.__iter__():
        x.to(device)
        q_y = estimateClusterImpurityHelper(model, x, label, device,)
        y.append(q_y.cpu())
    y = torch.concat(y, dim=0)
    n = y.shape[1] # number of clusters
    r = -np.ones(n) # homogeny index
    p = -np.ones(n) # label assignments to the clusters
    s = -np.ones(n) # label assignments to the clusters
    for i in range(n):
        c = labels[y.argmax(-1) == i]
        if c.shape[0] > 0:
            r[i] = c.sum(0).max().item() / c.shape[0] 
            p[i] = c.sum(0).argmax().item()
            s[i] = c.shape[0]
    return r, p, s

def estimateClusterAccuracy(
        y : Tensor,
        labels : Tensor,
        ):
    """
    y : (relaxed) one_hot tensor (cluster indicator)
    labels: one_hot vector (ground truth class indicator)
    returns: r,p,s 
    """
    n = y.shape[1] # number of clusters
    r = -np.ones(n) # homogeny index
    p = -np.ones(n) # label assignments to the clusters
    s = -np.ones(n) # label assignments to the clusters
    for i in range(n):
        c = labels[y.argmax(-1) == i]
        if c.shape[0] > 0:
            r[i] = c.sum(0).max().item() / c.shape[0] 
            p[i] = c.sum(0).argmax().item()
            s[i] = c.shape[0]
    return r, p, s

def do_plot_helper(model, device : str = "cpu",):
    """
    ploting helper function for 
    training procedures
    for gmm model
    """
    model.cpu()
    model.eval()
    w = model.w_prior.sample((16,))
    z = model.Pz(torch.cat([w, ], dim=-1))
    mu = z[:, :, : model.nz].reshape(16 * model.nclasses, model.nz)
    rec = model.Px(torch.cat([mu, ],dim=-1)).reshape(-1, 1, 28, 28)
    if model.reclosstype == "Bernoulli":
        rec = rec.sigmoid()
    plot_images(rec, model.nclasses )
    plt.pause(0.05)
    plt.savefig("tmp.png")
    #model.train()
    #model.to(device)
    return

def do_plot_helper_cmm(model, device : str = "cpu",):
    """
    ploting helper function for 
    training procedures
    for cmm model
    """
    model.cpu()
    model.eval()
    w = model.w_prior.sample((16,))
    w = w.repeat_interleave(repeats=model.nclasses,dim=0)
    y = torch.eye(model.nclasses)
    y = y.repeat(16, 1)
    wy = torch.cat([w,y], dim=1)
    z = model.Pz(torch.cat([wy, ], dim=-1))
    mu = z[:, : model.nz]
    rec = model.Px(torch.cat([mu, ],dim=-1)).reshape(-1, 1, 28, 28)
    if model.reclosstype == "Bernoulli":
        rec = rec.sigmoid()
    plot_images(rec, model.nclasses )
    plt.pause(0.05)
    plt.savefig("tmp.png")
    #model.train()
    #model.to(device)
    return


def test_accuracy_helper(model, x, y, device : str = "cpu",):
    model.cpu()
    model.eval()
    #r, p, s = estimateClusterImpurityLoop(
    r, p, s = estimateClusterImpurity(
        model,
        x,
        y,
        device,
    )
    print(p, "\n", r.mean(), "\n", r)
    print(
        (r * s).sum() / s.sum(),
        "\n",
    )
    #model.train()
    #model.to(device)
    return

def is_jsonable(x) -> bool:
    try:
        json.dumps(x)
        return True
    except:
        return False

def is_pickleable(x) -> bool:
    try:
        pickle.dumps(x)
        return True
    except:
        return False

def is_serializeable(x, method="json",) -> bool:
    if method == "json":
        return is_jsonable(x)
    else:
        return is_pickleable(x)


def saveModelParameters(
    model: nn.Module,
    fpath: str,
    method: str = "json",
) -> Dict:
    d = {}
    d['myName'] = str(model.__class__)
    for k, v in model.__dict__.copy().items():
        if is_serializeable(v, method):
            d[k] = v
    if method == "json":
        f = open(fpath, "w")
        json.dump(d, f)
        f.close()
    else:  # use pickle
        f = open(fpath, "wb")
        pickle.dump(d, f)
        f.close()
    return d

def loadModelParameter(
        fpath : str,
        method : str = "json",
        ) -> Dict:
    if method == "json":
        f = open(fpath, "r")
        params = json.load(f)
        f.close()
    else: # use pickle
        f = open(fpath, "rb")
        params = pickle.load(f,)
        f.close()
    return params

def balanceAnnData(
    adata: ad._core.anndata.AnnData,
    catKey: str,
    numSamples: int = 2500,
    noreps: bool = False,
    eps : float = 1e-4,
    add_noise : bool = False,
    augment_mode: bool = False,
) -> ad._core.anndata.AnnData:
    """
    creates a balanced set with numSamples objects per each
    category in the selected catKey category class
    by random selection with repetitions.
    IF noreps == True, numSamples is ignored and instead
    from each group m samples without repetitions are choses,
    where m is the size of the smallest group.
    if augment_mode is True, the original data will be included together
    with the samples, so the result dataset will not be exactly balanced.
    """
    andata_list = []
    if augment_mode:
        andata_list = [adata,]
    cats = list(np.unique(adata.obs[catKey]))
    m = 0
    if noreps:
        m = np.min(list(countby(lambda x: x, adata.obs[catKey]).values()))
    for c in cats:
        marker = adata.obs[catKey] == c
        n = np.sum(marker)
        if not noreps:
            s = np.random.randint(0, n, numSamples)  # select with repetitions
        else:
            s = np.random.permutation(n)[:m]
        andata_list.append(
            adata[marker][s].copy(),
        )
    xdata = ad.concat(
        andata_list,
        join="outer",
        label="set",
    )
    xdata.obs_names_make_unique()
    if add_noise:
        #sc.pp.scale(xdata,)
        #xdata.obs_names_make_unique()
        #noise = eps * np.random.randn(*xdata.X.shape).astype("float32")
        #xdata.X = xdata.X + noise
        xdata.X += eps * (np.random.randn(*xdata.X.shape)).astype("float32")
        #sc.pp.scale(xdata,)
        xdata.X -= (adata.X.var(0) > 0) * xdata.X.mean(0)
        xdata.X /= xdata.X.std(0)
    return xdata
    

def randomString(
        n : int = 8,
        pad : str ="_",
        ) -> str:
    """
    generate a random ascii string of length n, 
    padded from both ends with pad.
    """
    ls = np.random.randint(ord("A"), ord("z")+1, n)
    ls = [chr(i) for i in ls]
    ls = reduce(add, ls)
    cs = toolz.concatv(
            np.arange(ord("A"), ord("Z")+1,1),
            np.arange(ord("a"), ord("z")+1,1),
            )
    cs = list(cs)
    ls = np.random.choice(cs, n)
    ls = reduce(
            add, toolz.map(chr, ls))
    ls = pad + ls + pad
    return ls


def timeStamp() -> str:
    """
    generates a timestap string.
    """
    return str(datetime.timestamp(datetime.now()))






# stolen from https://github.com/theislab/scgen/blob/master/scgen/_scgen.py
def reg_mean_plot(
    adata,
    axis_keys = {"x" : "control", "y" : "stimulated"},
    labels = {"x" : "control", "y" : "stimulated"},
    condition_key : str="condition",
    path_to_save="./reg_mean.pdf",
    save=True,
    gene_list=None,
    show=False,
    top_100_genes=None,
    verbose=False,
    legend=True,
    title=None,
    x_coeff=0.30,
    y_coeff=0.8,
    fontsize=14,
    **kwargs,
):
    """
    Plots mean matching figure for a set of specific genes.
    Parameters
    ----------
    adata: `~anndata.AnnData`
        AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
        AnnData object used to initialize the model. Must have been setup with `batch_key` and `labels_key`,
        corresponding to batch and cell type metadata, respectively.
    axis_keys: dict
        Dictionary of `adata.obs` keys that are used by the axes of the plot. Has to be in the following form:
         `{"x": "Key for x-axis", "y": "Key for y-axis"}`.
    labels: dict
        Dictionary of axes labels of the form `{"x": "x-axis-name", "y": "y-axis name"}`.
    path_to_save: basestring
        path to save the plot.
    save: boolean
        Specify if the plot should be saved or not.
    gene_list: list
        list of gene names to be plotted.
    show: bool
        if `True`: will show to the plot after saving it.
    Examples
    --------
    >>> import anndata
    >>> import scgen
    >>> import scanpy as sc
    >>> train = sc.read("./tests/data/train.h5ad", backup_url="https://goo.gl/33HtVh")
    >>> scgen.SCGEN.setup_anndata(train)
    >>> network = scgen.SCGEN(train)
    >>> network.train()
    >>> unperturbed_data = train[((train.obs["cell_type"] == "CD4T") & (train.obs["condition"] == "control"))]
    >>> pred, delta = network.predict(
    >>>     adata=train,
    >>>     adata_to_predict=unperturbed_data,
    >>>     ctrl_key="control",
    >>>     stim_key="stimulated"
    >>>)
    >>> pred_adata = anndata.AnnData(
    >>>     pred,
    >>>     obs={"condition": ["pred"] * len(pred)},
    >>>     var={"var_names": train.var_names},
    >>>)
    >>> CD4T = train[train.obs["cell_type"] == "CD4T"]
    >>> all_adata = CD4T.concatenate(pred_adata)
    >>> network.reg_mean_plot(
    >>>     all_adata,
    >>>     axis_keys={"x": "control", "y": "pred", "y1": "stimulated"},
    >>>     gene_list=["ISG15", "CD3D"],
    >>>     path_to_save="tests/reg_mean.pdf",
    >>>     show=False
    >>> )
    """
    plt.cla()
    plt.clf()
    plt.close()

    sns.set()
    sns.set(color_codes=True)

    diff_genes = top_100_genes
    stim = adata[adata.obs[condition_key] == axis_keys["y"]]
    ctrl = adata[adata.obs[condition_key] == axis_keys["x"]]
    if diff_genes is not None:
        if hasattr(diff_genes, "tolist"):
            diff_genes = diff_genes.tolist()
        adata_diff = adata[:, diff_genes]
        stim_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["y"]]
        ctrl_diff = adata_diff[adata_diff.obs[condition_key] == axis_keys["x"]]
        x_diff = np.asarray(np.mean(ctrl_diff.X, axis=0)).ravel()
        y_diff = np.asarray(np.mean(stim_diff.X, axis=0)).ravel()
        m, b, r_value_diff, p_value_diff, std_err_diff = stats.linregress(
            x_diff, y_diff
        )
        if verbose:
            print("top_100 DEGs mean: ", r_value_diff**2)
    x = np.asarray(np.mean(ctrl.X, axis=0)).ravel()
    y = np.asarray(np.mean(stim.X, axis=0)).ravel()
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    if verbose:
        print("All genes mean: ", r_value**2)
    df = pd.DataFrame({axis_keys["x"]: x, axis_keys["y"]: y})
    ax = sns.regplot(x=axis_keys["x"], y=axis_keys["y"], data=df)
    ax.tick_params(labelsize=fontsize)
    if "range" in kwargs:
        start, stop, step = kwargs.get("range")
        ax.set_xticks(np.arange(start, stop, step))
        ax.set_yticks(np.arange(start, stop, step))
    ax.set_xlabel(labels["x"], fontsize=fontsize)
    ax.set_ylabel(labels["y"], fontsize=fontsize)
    if gene_list is not None:
        texts = []
        for i in gene_list:
            j = adata.var_names.tolist().index(i)
            x_bar = x[j]
            y_bar = y[j]
            texts.append(plt.text(x_bar, y_bar, i, fontsize=11, color="black"))
            plt.plot(x_bar, y_bar, "o", color="red", markersize=5)
            # if "y1" in axis_keys.keys():
            # y1_bar = y1[j]
            # plt.text(x_bar, y1_bar, i, fontsize=11, color="black")
    #if gene_list is not None:
    #    adjust_text(
    #        texts,
    #        x=x,
    #        y=y,
    #        arrowprops=dict(arrowstyle="->", color="grey", lw=0.5),
    #        force_points=(0.0, 0.0),
    #    )
    if legend:
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if title is None:
        plt.title("", fontsize=fontsize)
    else:
        plt.title(title, fontsize=fontsize)
    ax.text(
        max(x) - max(x) * x_coeff,
        max(y) - y_coeff * max(y),
        r"$\mathrm{R^2_{\mathrm{\mathsf{all\ genes}}}}$= " + f"{r_value ** 2:.2f}",
        fontsize=kwargs.get("textsize", fontsize),
    )
    if diff_genes is not None:
        ax.text(
            max(x) - max(x) * x_coeff,
            max(y) - (y_coeff + 0.15) * max(y),
            r"$\mathrm{R^2_{\mathrm{\mathsf{top\ 100\ DEGs}}}}$= "
            + f"{r_value_diff ** 2:.2f}",
            fontsize=kwargs.get("textsize", fontsize),
        )
    if save:
        plt.savefig(f"{path_to_save}", bbox_inches="tight", dpi=100)
    if show:
        plt.show()
    plt.close()
    if diff_genes is not None:
        return r_value**2, r_value_diff**2
    else:
        return r_value**2
