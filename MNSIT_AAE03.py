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
        return

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
        return

    def forward(self, z):
        imgs = self.decoder(z)
        return imgs


class Discriminator(nn.Module):
    def __init__(self, nin : int = 2, nh : int = 1024) -> None:
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
                fclayer(nin=nin, nout=nh, batchnorm=False, dropout=0,
                    activation=nn.LeakyReLU()),
                fclayer(nin=nh, nout=nh, batchnorm=True, dropout=0.2,
                    activation=nn.LeakyReLU()),
                fclayer(nin=nh, nout=1, batchnorm=False, dropout=0,
                    activation=nn.Sigmoid()),
                )
        return

    def forward(self, z):
        p = self.main(z)
        return p





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
discriminator = Discriminator(2, 1024)

discriminator(encoder(imgs)).shape


encoder = Encoder(28, 2, 1024).to(device)
encoder.apply(init_weights)
optimEnc = optim.Adam(encoder.parameters(), lr=5e-4)
decoder = Decoder(28, 2, 1024).to(device)
decoder.apply(init_weights)
optimDec = optim.Adam(decoder.parameters(), lr=5e-4)
discriminator = Discriminator(2, 1024).to(device)
discriminator.apply(init_weights)
optimDis = optim.Adam(discriminator.parameters(), lr=5e-4)
optimGen = optim.Adam(encoder.parameters(), lr=5e-4)

bce = nn.BCELoss(reduction="mean")
mse = nn.MSELoss(reduction="mean")

# let the training begin
for epoch in range(15):
    for idx, (data, labels) in enumerate(train_loader):
        x = data.to(device)
        batch_size = x.shape[0]
        # train for reconstruction
        encoder.train()
        encoder.zero_grad()
        decoder.train()
        decoder.zero_grad()
        z = encoder(x)
        recon = decoder(z)
        loss_recon = mse(recon, x)
        loss_recon.backward()
        optimDec.step()
        optimEnc.step()
        # train discriminator
        encoder.eval()
        discriminator.zero_grad()
        z = encoder(x)
        z_gauss = torch.randn_like(z).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros_like(real_labels).to(device)
        pred_real = discriminator(z_gauss)
        pred_fake = discriminator(z)
        loss_dis = bce(pred_real, real_labels) + bce(pred_fake, fake_labels)
        loss_dis.backward()
        optimDis.step()
        # train encoder as generator
        encoder.train()
        encoder.zero_grad()
        discriminator.eval()
        z = encoder(x)
        real_labels = torch.ones(batch_size, 1).to(device)
        pred_real = discriminator(z)
        loss_gen = bce(pred_real, real_labels)
        loss_gen.backward()
        optimGen.step()
        if idx % 100 == 0:
            print(
                "losses:\n",
                "reconstruction loss:",
                loss_recon.item(),
                "discriminator loss: ",
                loss_dis.item(),
                "loss_gen: ",
                loss_gen.item(),
            )


# compare orignal/recons from test set
imgs, labels = test_loader.__iter__().next()
z = encoder(imgs.cuda())
ximgs = decoder(z).cpu()

def plot_images(imgs, nrow=16, transform=nn.Identity(), out=plt):
    imgs = transform(imgs)
    grid_imgs = make_grid(imgs, nrow=nrow).permute(1, 2, 0)
    out.imshow(grid_imgs)

fig, ax = plt.subplots(2,1)

grid_imgs = make_grid(imgs, nrow=16).permute(1, 2, 0)
grid_ximgs = make_grid(ximgs, nrow=16).permute(1, 2, 0)

ax[0].imshow(denorm(grid_imgs))
ax[1].imshow(denorm(grid_ximgs))

# clustering
fig, ax = plt.subplots()
encoder.cpu()
decoder.cpu()
discriminator.cpu()

xs , labels = test_loader.__iter__().next()
zs = encoder(xs)
z = zs.detach().numpy()
x = z[:,0]
y = z[:,1]
for i in range(10):
    ax.scatter(x[labels == i], y[labels == i], label=str(i))
#ax.scatter(x,y, c=labels, label=labels)

ax.legend("0123456789")

plt.cla()

class Encoder(nn.Module):
    """
    a deterministic encoder.
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
                fclayer(nin=nh, nout=nz, batchnorm=False, dropout=0,
                    activation=nn.Tanh()),
        )

        return

    def forward(self, x):
        z = self.encoder(x)
        return z

gauss = mixedGaussianCircular(rho=0.01, sigma=0.6, k=10, j=0)
mix = distributions.Categorical(torch.ones(10,))
comp = distributions.Independent(gauss, 0)
gmm = distributions.MixtureSameFamily(mix, comp)
samples = gmm.sample((10000,))
samples = gmm.sample((10000,)) * 0.2
samples.shape
samples = samples.cpu().numpy()
x = samples[:,0]
y = samples[:,1]
plt.scatter(x,y)
plt.legend(['star'])


encoder = Encoder(28, 2, 1024).to(device)
encoder.apply(init_weights)
optimEnc = optim.Adam(encoder.parameters(), lr=5e-4)
decoder = Decoder(28, 2, 1024).to(device)
decoder.apply(init_weights)
optimDec = optim.Adam(decoder.parameters(), lr=5e-4)
discriminator = Discriminator(2, 1024).to(device)
discriminator.apply(init_weights)
optimDis = optim.Adam(discriminator.parameters(), lr=5e-4)
optimGen = optim.Adam(encoder.parameters(), lr=5e-4)

z_gauss = torch.randn((128,2))
z_star = gmm.sample((128,))


# let the training begin
for epoch in range(15):
    for idx, (data, labels) in enumerate(train_loader):
        x = data.to(device)
        batch_size = x.shape[0]
        # train for reconstruction
        encoder.train()
        encoder.zero_grad()
        decoder.train()
        decoder.zero_grad()
        z = encoder(x)
        recon = decoder(z)
        loss_recon = mse(recon, x)
        loss_recon.backward()
        optimDec.step()
        optimEnc.step()
        # train discriminator
        encoder.eval()
        discriminator.zero_grad()
        z = encoder(x)
        #z_gauss = torch.randn_like(z).to(device)
        #z_star = gmm.sample((batch_size,)).to(device)
        z_star = 0.2 * gmm.sample((batch_size,)).to(device)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros_like(real_labels).to(device)
        #pred_real = discriminator(z_gauss)
        pred_real = discriminator(z_star)
        pred_fake = discriminator(z)
        loss_dis = bce(pred_real, real_labels) + bce(pred_fake, fake_labels)
        loss_dis.backward()
        optimDis.step()
        # train encoder as generator
        encoder.train()
        encoder.zero_grad()
        discriminator.eval()
        z = encoder(x)
        real_labels = torch.ones(batch_size, 1).to(device)
        pred_real = discriminator(z)
        loss_gen = bce(pred_real, real_labels)
        loss_gen.backward()
        optimGen.step()
        if idx % 100 == 0:
            print(
                "losses:\n",
                "reconstruction loss:",
                loss_recon.item(),
                "discriminator loss: ",
                loss_dis.item(),
                "loss_gen: ",
                loss_gen.item(),
            )

# clustering
fig, ax = plt.subplots()

encoder.cpu()
decoder.cpu()
discriminator.cpu()

xs , labels = test_loader.__iter__().next()
zs = encoder(xs)
z = zs.detach().numpy()
x = z[:,0]
y = z[:,1]
for i in range(10):
    ax.scatter(x[labels == i], y[labels == i], label=str(i))
#ax.scatter(x,y, c=labels, label=labels)

ax.legend("0123456789")

plt.cla()
