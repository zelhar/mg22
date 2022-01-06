import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision.utils as vutils
from math import pi, sin, cos, sqrt, log
from toolz import partial, curry
from torch import Tensor
from torch import nn, optim, distributions
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

device = "cuda" if torch.cuda.is_available() else "cpu"

class Encoder(nn.Module):
    """
    a gauusian encoder. using reparametrization trick.
    """
    def __init__(self, imgsize : int = 28, nz : int =2, nh : int = 1024) -> None:
        super(Encoder, self).__init__()
        self.nz = nz
        self.imgsize = imgsize
        self.nin = imgsize**2
        nin = imgsize**2

        self.encoder = nn.Sequential(
                nn.Flatten(),
                fclayer(nin, nh, False, 0, nn.LeakyReLU()),
                fclayer(nin=nh, nout=nh, batchnorm=True, dropout=0.2,
                    activation=nn.LeakyReLU()),
        )

        self.mu = nn.Linear(nh, nz)
        self.logvar = nn.Linear(nh, nz)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

class Decoder(nn.Module):
    """
    a deterministic decoder.
    """
    def __init__(self, imgsize : int = 28, nz : int =2, nh : int = 1024) -> None:
        super(Decoder, self).__init__()
        self.nz = nz
        self.imgsize = imgsize
        self.nin = imgsize**2
        nin = imgsize**2

        self.decoder = nn.Sequential(
                fclayer(nin=nz, nout=nh, batchnorm=False, dropout=0,
                    activation=nn.LeakyReLU()),
                fclayer(nin=nh, nout=nh, batchnorm=True, dropout=0.2,
                    activation=nn.LeakyReLU()),
                fclayer(nin=nh, nout=nin, batchnorm=False, dropout=0,
                    activation=nn.Tanh()),
                nn.Unflatten(1, (1, imgsize, imgsize)),
        )

    def forward(self, z):
        imgs = self.decoder(z)
        return imgs


class Discriminator(nn.Module):
    pass

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

imgs, labels = test_loader.__iter__().next()
imgs

encoder = Encoder(28, 2, 1024)

encoder(imgs).shape

encoder.encode(imgs)

decoder = Decoder(28, 2, 1024)

decoder(encoder(imgs)).shape
