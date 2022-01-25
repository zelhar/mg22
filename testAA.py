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
from my_torch_utils import scsimDataset

import scsim.scsim as scsim

device = "cuda" if torch.cuda.is_available() else "cpu"

class AE(nn.Module):

    def __init__(self, nz: int = 10, nh: int = 1024,
            imgsize : int = 28) -> None:
        super(AE, self).__init__()
        self.nin = nin = imgsize**2
        self.nz = nz
        self.imgsize = imgsize

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nz),
            #nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nin),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, imgsize, imgsize)),
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encoder(x)
        y = self.decode(z)
        return y, z



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


model = AE(nz=2)

imgs, labels = test_loader.__iter__().next()
y,z = model(imgs)

model.to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

mse = nn.MSELoss()
bce = nn.BCELoss()

for epoch in range(10):
    for idx, (data, _) in enumerate(train_loader):
        model.train()
        model.requires_grad_(True)
        optimizer.zero_grad()
        batch_size = data.shape[0]
        x = data.to(device)
        xnoisy = x + torch.rand_like(x) * 5e-2
        y, z = model(x)
        ynoisy, znoisy = model(xnoisy)
        loss = mse(y,x) + mse(ynoisy, xnoisy) + mse(z, znoisy)
        loss.backward()
        optimizer.step()
        if idx % 300 == 0:
            print(
                "losses:\n",
                loss.item(),
            )

# clustering
model.cpu()

fig, ax = plt.subplots()

imgs, labels = test_loader.__iter__().next()
y, z = model(imgs)
x = z.detach().numpy()
xs = x[:,0]
ys = x[:,1]
for i in range(10):
    ax.scatter(xs[labels == i], ys[labels == i], label=str(i))

ax.legend("0123456789")

plt.cla()

plt.close()


l1norm = nn.L1Loss()

xold, _ = train_loader.__iter__().next()
xold = xold.to(device)

#### next try
model.to(device)
for epoch in range(10):
    for idx, (data, _) in enumerate(train_loader):
        model.train()
        model.requires_grad_(True)
        optimizer.zero_grad()
        batch_size = data.shape[0]
        x = data.to(device)
        xnoisy = x + torch.rand_like(x) * 5e-2
        y, z = model(x)
        ynoisy, znoisy = model(xnoisy)
        loss = mse(y,x) + mse(ynoisy, xnoisy) + mse(z, znoisy)
        loss.backward()
        optimizer.step()
        #xold = xold[:batch_size]
        if xold.shape == x.shape and l1norm(x, xold) >= 1e-1:
            optimizer.zero_grad()
            x = data.to(device)
            y, z = model(x)
            w, v = model(xold.detach())
            # distance 0.2 to 0.5 if they are disimilar
            loss2 = nn.ReLU()(
                    0.5 - l1norm(z,v)
                    ) + mse(y, x) + mse(w, xold)
            #loss2 += mse(y,x) + mse(w, xold)
            loss2.backward()
            optimizer.step()
        xold = data.to(device)
        if idx % 300 == 0:
            print(
                "losses:\n",
                loss.item(),
                loss2.item(),
            )

# clustering
model.cpu()

fig, ax = plt.subplots()

for idx, (imgs, labels) in enumerate(test_loader):
    y, z = model(imgs)
    x = z.detach().numpy()
    xs = x[:,0]
    ys = x[:,1]
    for i in range(10):
        ax.scatter(xs[labels == i], ys[labels == i], label=str(i))

ax.legend("0123456789")

plt.cla()

plt.close()

plt.savefig('./results/clustering_aae_tanh_close_and_far_training(testAA).png')

plt.savefig('./results/clustering_aae_linear_(testAA).png')

fig, ax = plt.subplots(2,1)

imgs, labels = test_loader.__iter__().next()
x, z = model(imgs)
grid_imgs = make_grid(imgs, nrow=16).permute(1, 2, 0)
grid_x = make_grid(x, nrow=16).permute(1, 2, 0)
ax[0].imshow(grid_imgs)
ax[1].imshow(grid_x)

plt.savefig('./results/reconstruction_aae_linear_(testAA).png')

zz = torch.randn_like(z)
xx = model.decode(zz)
grid_xx = make_grid(xx, nrow=16).permute(1, 2, 0)

ax[0].imshow(grid_xx)

plt.savefig('./results/random_normal_reconstruction(top)_aae_linear_(testAA).png')

plt.close()


### next test: rnaseq
dataSet = scsimDataset("data/scrnasim/counts.npz",
        "data/scrnasim/cellparams.npz")
trainD, testD = dataSet.__train_test_split__(8500)
trainLoader = torch.utils.data.DataLoader(trainD, batch_size=128, shuffle=True)
testLoader = torch.utils.data.DataLoader(testD, batch_size=128, shuffle=True)

class AE2(nn.Module):
    def __init__(self, nin : int = 10**4, nz : int = 2, nh : int = 1024,
            nclusters : int = 20) -> None:
        super(AE2, self).__init__()

        self.encoder = nn.Sequential(
                fclayer(nin, nh, True, 0.15, nn.LeakyReLU()),
                fclayer(nh,nh, True, 0.1, nn.LeakyReLU()),
                fclayer(nh,nh, True, 0.1, nn.LeakyReLU()),
                nn.Linear(nh, nz),
                nn.Tanh()
                )

        self.decoder = nn.Sequential(
                fclayer(nz, nh, False, 0, nn.LeakyReLU()),
                fclayer(nh, nh, True, 0.2, nn.LeakyReLU()),
                fclayer(nh, nh, True, 0.2, nn.LeakyReLU()),
                nn.Linear(nh, nin),
                )

        self.clusterheads = torch.zeros(nclusters,nz) 
        self.clusterheads[:,0] = 10*torch.sin(2*pi/nclusters * torch.arange(nclusters))
        self.clusterheads[:,1] = 10*torch.cos(2*pi/nclusters * torch.arange(nclusters))

        self.categorize = nn.Sequential(
                fclayer(nin, nh, True, 0.15, nn.LeakyReLU()),
                fclayer(nh,nh, True, 0.1, nn.LeakyReLU()),
                nn.Linear(nh, nclusters),
                nn.LogSoftmax(dim=1),
                ) #shape should be (batch_size, nclusters)

    def encode(self, x : Tensor) -> Tensor:
        v = self.clusterheads.to(x.device)
        w = self.categorize(x).exp()
        loc = w @ v
        z = self.encoder(x) + loc
        return z
        #return self.encoder(x)

    def decode(self, z : Tensor) -> Tensor:
        return self.decoder(z)

    def forward(self, x : Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encode(x)
        y = self.decode(z)
        return (y,z)


model = AE2()

data , labels = testLoader.__iter__().next()
data.shape
labels.shape

y,z = model(data)
y.shape
z.shape


model.to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters())
xold, _ = trainLoader.__iter__().next()
xold = xold.to(device)

for epoch in range(15):
    for idx, (data, labels) in enumerate(trainLoader):
        model.train()
        model.requires_grad_(True)
        optimizer.zero_grad()
        x = data.to(device)
        batch_size = data.shape[0]
        noise = xold[:batch_size] * 5e-2
        #xnoisy = x + torch.rand_like(x) * 5e2
        xnoisy = x + noise
        y, z = model(x)
        ynoisy, znoisy = model(xnoisy)
        #loss = mse(y,x) + mse(ynoisy, xnoisy) + mse(z, znoisy)
        loss = mse(y,x)
        loss.backward()
        optimizer.step()
        # categorical loss
        optimizer.zero_grad()
        x = data.to(device)
        logprobs = model.categorize(x)
        probs = logprobs.exp()
        loss_cat = (-probs * logprobs).sum()
        loss_cat.backward()
        optimizer.step()
        if idx % 300 == 0:
            print(
                "losses:\n",
                loss.item(),
                loss_cat.item()
            )
        xold[:batch_size] = data.to(device)

        #if xold.shape == x.shape and l1norm(x, xold) >= 1e-1:
        #if xold.shape == x.shape and mse(x, xold) >= 1e1:
        if xold.shape == x.shape and l1norm(x, xold) >= 2e-1:
            optimizer.zero_grad()
            x = data.to(device)
            y, z = model(x)
            w, v = model(xold.detach())
            # distance 0.2 to 0.5 if they are disimilar
            loss2 = nn.ReLU()(
                    50 - l1norm(z,v)
                    ) + mse(y, x) + mse(w, xold)
            #loss2 += mse(y,x) + mse(w, xold)
            loss2.backward()
            optimizer.step()
        xold = data.to(device)
        if idx % 300 == 0:
            print(
                "losses:\n",
                loss.item(),
                loss2.item(),
            )

# clustering
model.cpu()

fig, ax = plt.subplots()

for idx, (imgs, labels) in enumerate(testLoader):
    y, z = model(imgs)
    x = z.detach().numpy()
    xs = x[:,0]
    ys = x[:,1]
    for i in range(1,11):
        ax.scatter(xs[labels == i], ys[labels == i], label=str(i))


plt.cla()

x, _ = trainLoader.__iter__().next()
y, _ = trainLoader.__iter__().next()
z, _ = trainLoader.__iter__().next()


l1norm(x,x)
l1norm(x,y)

clusterheads = torch.zeros(2,20)
clusterheads[0] = torch.sin(2*pi/20 * torch.arange(20))
clusterheads[1] = torch.cos(2*pi/20 * torch.arange(20))

plt.cla()
xs = clusterheads[0].numpy()
ys = clusterheads[1].numpy()
ax.scatter(10*xs, 10*ys)
