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
from torch.nn.functional import one_hot

# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import fnorm, replicate, logNorm

device = "cuda" if torch.cuda.is_available() else "cpu"


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

class Encoder(nn.Module):
    def __init__(self, imgsize=28, nh=1024, nz=10):
        super(Encoder, self).__init__()
        self.nin = nin = imgsize**2
        self.nz = nz
        self.imgsize = imgsize

        self.encoder = nn.Sequential(
                nn.Flatten(),
                fclayer(nin, nh, False, 0, nn.LeakyReLU()),
                fclayer(nin=nh, nout=nh, batchnorm=True, dropout=0.2,
                    activation=nn.LeakyReLU()),
                fclayer(nh, nh, True, 0.2, nn.LeakyReLU()),
                fclayer(nh, nh, True, 0.2, nn.LeakyReLU()),
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
    def __init__(self, imgsize : int = 28, nz : int = 10, nh : int = 1024) -> None:
        super(Decoder, self).__init__()
        self.nz = nz
        self.imgsize = imgsize
        self.nin = nin = imgsize**2

        self.decoder = nn.Sequential(
                fclayer(nin=nz, nout=nh, batchnorm=False, dropout=0,
                    activation=nn.LeakyReLU()),
                fclayer(nin=nh, nout=nh, batchnorm=True, dropout=0.2,
                    activation=nn.LeakyReLU()),
                fclayer(nh, nh, True, 0.2, nn.LeakyReLU()),
                fclayer(nin=nh, nout=nin, batchnorm=False, dropout=0,
                    activation=nn.Sigmoid()),
                nn.Unflatten(1, (1, imgsize, imgsize)),
        )
        return

    def forward(self, z):
        imgs = self.decoder(z)
        return imgs

class Discriminator(nn.Module):
    def __init__(self, nin : int = 10, nh : int = 1024) -> None:
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
                fclayer(nin=nin, nout=nh, batchnorm=False, dropout=0,
                    activation=nn.LeakyReLU()),
                fclayer(nin=nh, nout=nh, batchnorm=True, dropout=0.2,
                    activation=nn.LeakyReLU()),
                fclayer(nh, nh, True, 0.2, nn.LeakyReLU()),
                fclayer(nin=nh, nout=1, batchnorm=False, dropout=0,
                    activation=nn.Sigmoid()),
                )
        return

    def forward(self, z):
        p = self.main(z)
        return p

class Critique(nn.Module):
    def __init__(self, nin : int = 10, nh : int = 1024) -> None:
        super(Critique, self).__init__()

        self.main = nn.Sequential(
                fclayer(nin=nin, nout=nh, batchnorm=False, dropout=0,
                    activation=nn.LeakyReLU()),
                fclayer(nin=nh, nout=nh, batchnorm=True, dropout=0.2,
                    activation=nn.LeakyReLU()),
                fclayer(nh, nh, True, 0.2, nn.LeakyReLU()),
                nn.Linear(nh, 1),
                )
        return

    def forward(self, z):
        p = self.main(z)
        return p

encoder = Encoder(28, 1024, 10)
decoder = Decoder(28, 10, 1024)
discriminator = Discriminator(10, 1024)

bce = nn.BCELoss(reduction="mean")
mse = nn.MSELoss(reduction="mean")
kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

encoder.to(device)
encoder.apply(init_weights)
optimEnc = optim.Adam(encoder.parameters(), lr=5e-4)

decoder.to(device)
decoder.apply(init_weights)
optimDec = optim.Adam(decoder.parameters(), lr=5e-4)

discriminator.to(device)
discriminator.apply(init_weights)
optimDis = optim.Adam(discriminator.parameters(), lr=5e-4)

optimGen = optim.Adam(encoder.parameters(), lr=5e-4)


# let the training begin
for epoch in range(45):
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

encoder.cpu()
decoder.cpu()
discriminator.cpu()

imgs, labels = test_loader.__iter__().next()

fig, ax = plt.subplots(2,2)

grid_imgs = make_grid(imgs, nrow=16).permute(1,2,0)
ax[0,0].imshow(grid_imgs)

z = encoder(imgs)
y = decoder(z)

grid_y = make_grid(y, nrow=16).permute(1,2,0)
ax[0,1].imshow(grid_y)

tiny = 5e-1
w = z + tiny * torch.rand_like(z) - tiny/2
w = decoder(w)
grid_w = make_grid(w, nrow=16).permute(1,2,0)
ax[1,1].imshow(grid_w)


plt.close()

import torch.linalg as lalg

from scipy import linalg as LA

from torch.linalg import eigh

cov = z @ z.t()

eigvals, eigvects = LA.eigh(cov.detach().numpy())


idx = np.argsort(eigvals)[::-1]
evecs = eigvects[:,idx]
evals = eigvals[idx]

evecs.shape

x = z.detach().numpy()

a = evecs @ x

xs = a[:,0]
ys = a[:,1]

plt.scatter(xs, ys)

for i in range(10):
    plt.scatter(xs[labels == i], ys[labels == i], label=str(i))


plt.cla()

m = distributions.OneHotCategorical(torch.ones(10))
m.sample((3,))

one_hot(torch.LongTensor([1,1,0,2]))
x=torch.LongTensor([1,1,0,2])

import umap

fit = umap.UMAP(n_components=2)


encoder = Encoder(28, 1024, 2)
decoder = Decoder(28, 2, 1024)
discriminator = Critique(2, 1024)


encoder.to(device)
encoder.apply(init_weights)
optimEnc = optim.Adam(encoder.parameters(), lr=5e-4)
decoder.to(device)
decoder.apply(init_weights)
optimDec = optim.Adam(decoder.parameters(), lr=5e-4)
optimGen = optim.Adam(encoder.parameters(), lr=5e-4)
discriminator.to(device)
discriminator.apply(init_weights)
optimDis = optim.Adam(discriminator.parameters(), lr=5e-4)

# let the training begin
for epoch in range(10):
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
        lossD_real = -discriminator(z_gauss).mean()
        lossD_fake = discriminator(z).mean()
        loss_dis = (lossD_fake + lossD_real)/2
        loss_dis.backward()
        optimDis.step()
        # clip discriminator's weights
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)
        # train encoder as generator
        if idx % 5 == 0:
            encoder.train()
            encoder.zero_grad()
            discriminator.eval()
            z = encoder(x)
            loss_gen = -discriminator(z).mean()
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


imgs, labels = test_loader.__iter__().next()
z = encoder(imgs)
x = z.detach().numpy()
xs = x[:,0]
ys = x[:,1]
for i in range(10):
    plt.scatter(xs[labels == i], ys[labels == i], label=str(i))

plt.close()

plt.scatter(xs, ys, c=labels)

plt.legend("0123456789")

