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
