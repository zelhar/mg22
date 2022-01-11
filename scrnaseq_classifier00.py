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
from my_torch_utils import scsimDataset


device = "cuda" if torch.cuda.is_available() else "cpu"
dataSet = scsimDataset("data/scrnasim/counts.npz",
        "data/scrnasim/cellparams.npz")
trainD, testD = dataSet.__train_test_split__(8500)

trainLoader = torch.utils.data.DataLoader(trainD, batch_size=128, shuffle=True)
testLoader = torch.utils.data.DataLoader(testD, batch_size=128, shuffle=True)

xs, ls = trainLoader.__iter__().next()


class Classifier(nn.Module):
    def __init__(self, nin=10**4, nh=10**3, nclasses=10):
        super(Classifier, self).__init__()
        self.nclasses = nclasses
        self.nin = nin
        self.main = nn.Sequential(
                fclayer(nin=nin, nout=nh, batchnorm=True, dropout=0.2,
                    activation=nn.LeakyReLU(),),
                nn.Linear(nh, nclasses),
                nn.LogSoftmax(dim=1),
                )
        return

    def forward(self,x):
        logp = self.main(x)
        return logp

model = Classifier()

model(xs).shape

ys = model(xs)

criterion = nn.NLLLoss(reduction="mean")

criterion(ys, ls-1)


model.to(device)

model.apply(init_weights)

optimizer = optim.Adam(model.parameters())

for epoch in range(15):
    for idx, (data, labels) in enumerate(trainLoader):
        x = data.to(device)
        target = (labels - 1).to(device)
        model.train()
        model.zero_grad()
        logprob = model(x)
        loss = criterion(logprob, target)
        loss.backward()
        optimizer.step()
        if idx % 100 == 0:
            print(
                "loss:\n",
                loss.item(),
                )

# testing
xs, ls = testLoader.__iter__().next()

# convert the labels to 0-indexing
ls -= 1

model.to("cpu")

ys = model(xs)

probs = ys.exp()

criterion(ys, ls)

predicts = probs.argmax(axis=1)

predicts
ls

