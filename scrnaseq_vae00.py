import argparse
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import torchvision.utils as vutils
import pandas as pd
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
from my_torch_utils import scsimDataset

import scsim.scsim as scsim


device = "cuda" if torch.cuda.is_available() else "cpu"
dataSet = scsimDataset("data/scrnasim/counts.npz",
        "data/scrnasim/cellparams.npz")
trainD, testD = dataSet.__train_test_split__(8500)

trainLoader = torch.utils.data.DataLoader(trainD, batch_size=128, shuffle=True)
testLoader = torch.utils.data.DataLoader(testD, batch_size=128, shuffle=True)

class VAE(nn.Module):
    """
    basic VAE model
    """
    def __init__(self, nin : int = 10**4, nz : int = 2, nh : int = 1024,
            nclasses : int = 10) -> None:
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
                fclayer(nin, nh, True, 0.2, nn.LeakyReLU()),
                fclayer(nh,nh, True, 0.2, nn.LeakyReLU()),
                nn.Linear(nh, nh),
                nn.LeakyReLU(),
                )

        self.zmu = nn.Linear(nh, nz)
        self.zlogvar = nn.Linear(nh,nz)

        self.decoder = nn.Sequential(
                fclayer(nz, nh, False, 0, nn.LeakyReLU()),
                fclayer(nh, nh, True, 0.2, nn.LeakyReLU()),
                nn.Linear(nh, nin),
                )
    
    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        sigma = (0.5 *logvar).exp()
        return mu + eps * sigma

    def encode(self, x : Tensor):
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return mu, logvar

    def decode(self, z):
        xhat = self.decoder(z)
        return xhat

    def forward(self, x):
        zmu, zlogvar = self.encode(x)
        z = self.reparameterize(zmu, zlogvar)
        xhat = self.decode(z)
        return zmu, zlogvar, xhat


mse = nn.MSELoss(reduction="sum")
bce = nn.BCELoss(reduction = "sum")
kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


model = VAE().to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())


#xs, ls = trainLoader.__iter__().next()
for epoch in range(15):
    for idx, (data, labels) in enumerate(trainLoader):
        x = data.to(device)
        batch_size = data.shape[0]
        model.train()
        model.zero_grad()
        mu, logvar, recon = model(x)
        loss_recon = mse(recon, x)
        loss_kld = 100*kld(mu, logvar)
        loss = loss_kld + loss_recon
        loss.backward()
        optimizer.step()
        if idx % 300 == 0:
            print(
                "losses:\n",
                "reconstruction loss:",
                loss_recon.item(),
                "kld:",
                loss_kld.item(),
            )


# clustering
fig, ax = plt.subplots()
model.cpu()

for batch in range(100):
    xs , labels = testLoader.__iter__().next()
    mu, logvar, recons = model(xs)
    zs = model.reparameterize(mu, logvar)
    z = zs.detach().numpy()
    x = z[:,0]
    y = z[:,1]
    for i in range(10):
        ax.scatter(x[labels == i], y[labels == i], label=str(i))
#ax.scatter(x,y, c=labels, label=labels)
ax.legend("0123456789")

plt.cla()

# unrelated sh*t
theta = torch.rand((100,)) * pi

