# useful function to be re-used in throught the tests in this project.
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast
import torch
from torch import nn, optim, distributions
import torch.utils.data
import torchvision.utils as vutils
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
from math import pi, sin, cos, sqrt

from toolz import partial, curry


def init_weights(m : torch.nn.Module) -> None:
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
        torch.nn.init.xavier_uniform(m.weight)


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


def save_reconstructs(
    encoder, decoder, x, epoch, nz=20, device="cuda", normalized=False
):
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


def save_random_reconstructs(model, nz, epoch, device="cuda", normalized=False):
    with torch.no_grad():
        # sample = torch.randn(64, nz).to(device)
        sample = torch.randn(64, nz).to(device)
        sample = model(sample).cpu()
        if normalized:
            sample = denorm(sample)
        save_image(sample, "results/sample_" + str(epoch) + ".png")


def plot_images(imgs, nrow=16, transform=nn.Identity()):
    imgs = transform(imgs)
    grid_imgs = make_grid(imgs, nrow=nrow).permute(1, 2, 0)
    plt.imshow(grid_imgs)


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
    #cat = distributions.Categorical(torch.ones(k))
    #i = cat.sample().item()
    #theta = 2 * torch.pi * i / k
    theta = 2 * torch.pi / k
    v = torch.Tensor((1, 0))
    T = torch.Tensor([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    S = torch.stack([T.matrix_power(i) for i in range(k)])
    mu = S @ v
    #cov = sigma ** 2 * ( torch.eye(2) + rho * (torch.ones(2, 2) - torch.eye(2)))
    #cov = cov @ S
    cov = torch.eye(2) * sigma ** 2
    cov[1,1] = sigma ** 2 * rho
    cov = torch.stack(
            [T.matrix_power(i+j) @ cov @ T.matrix_power(-i-j) for i in range(k)])
    gauss = distributions.MultivariateNormal(loc = mu, covariance_matrix= cov)
    return gauss

#transform = transforms.Compose(
#    [
#        transforms.ToTensor(),
#        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#    ]
#)
#
#dataset = datasets.CIFAR10(
#        root='data/',
#        train=True,
#        download=True,
#        transform=transform,
#        )
#
#train_loader = torch.utils.data.DataLoader(
#    dataset=dataset,
#    batch_size=128,
#    shuffle=True, 
#)
#test_loader = torch.utils.data.DataLoader(
#    dataset=datasets.CIFAR10(
#        root='data/',
#        train=False,
#        download=True,
#        transform=transform,
#        ),
#    batch_size=128,
#    shuffle=True,
#)

#imgs, labels = train_loader.__iter__().next()
#plot_images(imgs, nrow=16, transform=denorm)
#x = torch.linspace(0, 1, 16)
#x=x.reshape((1,1,1,4,4))
#f = transforms.Normalize(mean=0.5, std=0.5)
#g = normalize(clamp=False)
##g = partial(normalize, clamp=False)
#h = partial(denorm, clamp=False)
#
#@curry
#def mul(x,y):
#    return x*y
#ll = fclayer(10, 1, False, 0.0, nn.Sigmoid())

#gauss = mixedGaussianCircular(rho=0.01, sigma=0.6, k=10, j=0)
#mix = distributions.Categorical(torch.ones(10,))
#comp = distributions.Independent(gauss, 0)
#gmm = distributions.MixtureSameFamily(mix, comp)
#samples = gmm.sample((10000,))
#samples.shape
#samples = samples.cpu().numpy()
#x = samples[:,0]
#y = samples[:,1]
#plt.scatter(x,y)
#plt.legend(['star'])

