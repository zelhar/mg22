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
from my_torch_utils import fnorm

device = "cuda" if torch.cuda.is_available() else "cpu"

class VAEBernoulli(nn.Module):
    """
    a Bernoulli decoder.
    """

    def __init__(self, nin: int, nz: int, nh: int,
            imgsize=28) -> None:
        super(VAEBernoulli, self).__init__()
        self.nin = nin
        self.nz = nz
        self.imgsize = imgsize

        self.encoder = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nin),
            nn.Sigmoid(),
        )

        self.mumap = nn.Linear(nh, nz)
        self.logvarmap = nn.Linear(nh, nz)


    def encode(self, x):
        h = self.encoder(x)
        mu = self.mumap(h)
        logvar = self.logvarmap(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        y = self.decoder(z)
        return y

    def forward(self, x):
        x = x.view(-1, self.nin)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        y = self.decode(z)
        return y, mu, logvar

class VAEGauss(nn.Module):
    """
    a Gauss VAE with fixed variance=1.
    use MSE (with reduction=sum) as reconstruction loss.
    """

    def __init__(self, nin: int, nz: int, nh: int, 
            imgsize=28) -> None:
        super(VAEGauss, self).__init__()
        self.nin = nin
        self.nz = nz
        self.imgsize = imgsize

        self.encoder = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
        )

        self.mumap = nn.Linear(nh, nz)
        self.logvarmap = nn.Linear(nh, nz)

        self.dmu = nn.Linear(nh, nin)
        self.dlv = nn.Linear(nh, nin)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mumap(h)
        logvar = self.logvarmap(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        h = nn.ReLU()(h)
        mu = self.dmu(h)
        logvar = self.dlv(h)
        return mu, logvar

    def forward(self, x):
        x = x.view(-1, self.nin)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        dmu, dlogvar = self.decode(z)
        recon = self.reparameterize(dmu, dlogvar)
        recon = recon.view(-1, 1, self.imgsize, self.imgsize)
        return dmu, dlogvar, mu, logvar, recon


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


a,b,c,d,r = model(imgs)
r.shape

bce = nn.BCELoss(reduction="sum")
mse = nn.MSELoss(reduction="sum")
kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

#### train Bernouli VAE
model = VAEBernoulli(28 ** 2, 2, 1000).to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
nin = 28**2

for epoch in range(10):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        x = data.view(-1, nin).to(device)
        #x = data.to(device)
        model.train()
        model.requires_grad_(True)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss_recon = bce(recon, x)
        loss_kld = kld(mu, logvar)
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


# compare orignal/recons from test set
imgs, labels = test_loader.__iter__().next()
plot_images(imgs)
x = imgs.view(-1, nin).to(device)
y, mu, lv = model(x)
y = y.view(-1, 1, 28, 28).cpu()
plot_images(y)

# construct images from random samples of the latent space 
model.to(device)
sample = torch.randn(64, 2).to(device)
sample = model.decode(sample).cpu()
sample = sample.view(-1, 1, 28, 28)
plot_images(sample)

# clustering
fig, ax = plt.subplots()

model.cpu()
xs , labels = test_loader.__iter__().next()
recons , mu, logvar = model(xs)
recons = recons.view(-1,1,28,28)
zs = model.reparameterize(mu, logvar)
z = zs.detach().numpy()
x = z[:,0]
y = z[:,1]
for i in range(10):
    ax.scatter(x[labels == i], y[labels == i], label=str(i))
#ax.scatter(x,y, c=labels, label=labels)

ax.legend("0123456789")

plt.cla()




#### Gauss VAE
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

model = VAEGauss(28**2, 2, 1024, 28).to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
nin = 28**2

for epoch in range(10):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        x = data.view(-1, nin).to(device)
        #x = data.to(device)
        model.train()
        model.requires_grad_(True)
        optimizer.zero_grad()
        dm, dlv, mu, logvar, r = model(x)
        recon = dm
        loss_recon = mse(recon, x)
        loss_kld = kld(mu, logvar)
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

# compare orignal/recons from test set
imgs, labels = test_loader.__iter__().next()

plot_images(imgs)

x = imgs.view(-1, nin).to(device)
y, _, mu, lv, r = model(x)
y = y.view(-1, 1, 28, 28).cpu()

plot_images(denorm(y))

# construct images from random samples of the latent space 
model.to(device)
sample = torch.randn(64, 2).to(device)
sample, _ = model.decode(sample)
sample=sample.cpu()
sample = sample.view(-1, 1, 28, 28)
plot_images(denorm(sample))

# clustering
fig, ax = plt.subplots()

model.cpu()
xs , labels = test_loader.__iter__().next()
recons , _, mu, logvar, r = model(xs)
recons = recons.view(-1,1,28,28)
zs = model.reparameterize(mu, logvar)
z = zs.detach().numpy()
x = z[:,0]
y = z[:,1]
for i in range(10):
    ax.scatter(x[labels == i], y[labels == i], label=str(i))
#ax.scatter(x,y, c=labels, label=labels)

ax.legend("0123456789")

plt.cla()


## Now trying GaussVAE but use also the varaince of the encoder.
model = VAEGauss(28**2, 2, 1024, 28).to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
nin = 28**2

for epoch in range(10):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        x = data.view(-1, nin).to(device)
        model.train()
        model.requires_grad_(True)
        optimizer.zero_grad()
        dm, dlv, mu, logvar, r = model(x)

        log_p_given_z = -torch.ones_like(x).sum(1) * log(2 * pi) / 2
        log_p_given_z -= 

        loss_recon = fnorm(x, dm, dlv.exp(), reduction="logsum")
        loss_kld = kld(mu, logvar)
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


x , labels = test_loader.__iter__().next()
x = x.view(-1,28**2)
z = torch.randn((x.shape[0], 28**2, 100))
z.shape
mu = torch.rand((x.shape[0], 28**2))
logvar = torch.rand((x.shape[0], 28**2))


a = Tensor([ [[1]], [[2]] ])
a.shape

b = 2*torch.ones((2, 1, 3))
b.shape
a*b


log_p_x_given_z = lambda x, logvar, mu : (
    -torch.ones_like(x).sum(1) * np.log(2 * pi) / 2
    - logvar.sum(1) / 2
    - ((x - mu) * (x - mu) / (2 * torch.exp(logvar))).sum(1)
)

log_p_x_given_z = lambda x, logvar, mu : (
    -torch.ones_like(x).sum(1) * np.log(2 * pi) / 2
    - logvar.sum(1) / 2
    - ((x - mu).pow(2)  / (2 * torch.exp(logvar))).sum(1)
)


x = torch.rand((10,1))
mu = torch.rand((10,1))
logvar = torch.rand((10,1)).log()
sigma = (0.5 * logvar).exp()



log_p_x_given_z(x, mu, logvar)

fnorm(x, mu, (0.5 * logvar).exp(), reduction="no").log()

(0.5 * logvar).exp()

@curry
def pxz(
    x: Tensor,
    mu: Union[float, Tensor] = 1.0,
    #logvar : Tensor,
    sigma: Union[float, Tensor] = 1.0,
    reduction: str = "none",
) -> Tensor:
    """
    normal distribution density (elementwise) with optional reduction (logsum/sum/mean)
    """
    #sigma = torch.exp(0.5 * logvar)
    #logvar = 2 * sigma.log()
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

gauss = distributions.Normal(loc=0, scale=1)
gauss.cdf(x)

gauss.log_prob(x)

pxz(x, mu=0, sigma=1).log()

log_p_x_given_z(x, torch.zeros_like(x), torch.zeros_like(x))

w = torch.ones((3,4,5))
w.sum(0)
w.sum(1)
w.sum(2)
w.sum((1,2))
