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

model.to(device)
model.apply(init_weights)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

mse = nn.MSELoss()
bce = nn.BCELoss()
l1loss = nn.L1Loss()

xold, _ = train_loader.__iter__().next()
xold = xold.to(device)

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
        #loss = mse(y,x) + mse(ynoisy, xnoisy) + mse(z, znoisy)
        loss = mse(y,x)
        loss.backward()
        optimizer.step()
        ## and now train far inputs to be far in the lattent space
        #x = data.to(device)
        #xfar = xold[:batch_size].to(device)
        #farinds = (x - xfar).abs().mean(axis=(1,2,3)) > 1e-1
        #optimizer.zero_grad()
        #x = x[farinds]
        #xfar = xfar[farinds]
        #y, z = model(x)
        #w, v = model(xfar)
        #loss2 = nn.ReLU()(
        #        2.5 - l1loss(z,v)) + mse(y, x) + mse(w, xfar)
        #loss2.backward()
        #optimizer.step()
        if idx % 300 == 0:
            print(
                "losses:\n",
                loss.item(),
                #loss2.item(),
            )
        xold[:batch_size] = data.to(device)

fig, ax = plt.subplots()

model.cpu()
for idx, (imgs, labels) in enumerate(test_loader):
    y, z = model(imgs)
    x = z.detach().numpy()
    xs = x[:,0]
    ys = x[:,1]
    for i in range(0,10):
        ax.scatter(xs[labels == i], ys[labels == i], label=str(i))

ax.legend("0123456789")

plt.cla()

plt.close()
