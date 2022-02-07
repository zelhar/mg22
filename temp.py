# fun fun 2022-01-29
# https://github.com/eelxpeng/dec-pytorch/blob/master/lib/idec.py
# https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py
# https://github.com/orybkin/sigma-vae-pytorch
import argparse
from importlib import reload
import matplotlib.pyplot as plt
import my_torch_utils as ut
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
from math import pi, sin, cos, sqrt, log
from toolz import partial, curry
from torch import nn, optim, distributions
from torch.nn.functional import one_hot
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO
# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import fnorm, replicate, logNorm
from my_torch_utils import scsimDataset
import scsim.scsim as scsim
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from toolz import take, drop
import opt_einsum
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans



print(torch.cuda.is_available())

#%load_ext autoreload
#%autoreload 2

### working with IDEC_test00.py
from IDEC_test00 import *

test_data = test_loader.dataset.data.float()/255
test_labels = test_loader.dataset.targets

data = train_loader.dataset.data.float()/255

model = IDEC()
model.apply(init_weights)
model.load_model('./data/temp.pt')
model.fit(train_loader)
model.save_model('./data/temp.pt')

# trying to see how to cycle over data


model = IDEC()

model.apply(init_weights)

x, labels = test_loader.__iter__().next()
x = x.flatten(1)

model(x)

model.trainAE(train_loader)

model.cpu()
model.save_model('./data/temp.pt')


torch.cuda.empty_cache()


model.fit(train_loader)
model.load_model('./data/temp.pt')

z,q,y = model(x)

plot_images(y.reshape(-1,1,28,28))

plot_images(x.reshape(-1,1,28,28))

model.save_model('./data/idec.pt')

fit(model, train_loader)

latent = model.encode(data)
targets = train_loader.dataset.targets
pred = torch.tensor(model.y_pred)


ut.plot_tsne(latent[:5000], pred[:5000], "ooVAE")

ut.plot_tsne(latent[:5000], targets[:5000], "ooVAE")

model = IDEC()
model.load_model('./data/idec.pt')


model.init_kmeans(data)

model.fit(train_loader)

model.mu

for i in range(20):
    pred_i = pred == i
    #uniques = torch.unique(targets[pred_i])
    #print(uniques)
    print(targets[pred_i][:100])
    

model = IDEC(nclusters=10)
model.trainAE(train_loader)

model.fit(train_loader)


pred = torch.tensor(model.y_pred)

model.save_model('./data/idec10.pt')




# sigma-vae
model = SigmaVAE()


x, labels = test_loader.__iter__().next()
x = x.flatten(1)

zmu, zlogvar, xmu = model(x)
model.loss_function(x, xmu, model.log_sigma, zmu, zlogvar)
kld(zmu, zlogvar)
model.apply(init_weights)
model.fit(train_loader,)

plot_images(xmu.view(-1,1,28,28))

plot_images(x.view(-1,1,28,28))

torch.save(model.state_dict(), './data/temp_sigmavae.pt')


