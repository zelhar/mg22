from tensorfun03 import *

import numpy as np
import torch
import torch.nn as nn
import visdom
from utils.mnist_cached import MNISTCached as MNIST
from utils.mnist_cached import setup_data_loaders
from utils.vae_plots import mnist_test_tsne, plot_llk, plot_vae_samples

import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

print("gibt cuda?", torch.cuda.is_available())

model = train(args)

train_loader, test_loader = setup_data_loaders(
    MNIST, use_cuda=False, batch_size=256
)


model.cpu()

imgs, labels = iter(test_loader).next()

plot_images(imgs.view(-1,1,28,28))
plt.savefig('./vae_results/original.png')

rec = model.reconstruct_img(imgs)
plot_images(rec.view(-1,1,28,28))
plt.savefig('./vae_results/rec.png')

zmu, zsigma = model.encoder(imgs)



# different settings
args.visdom_flag=True
args.z_dim = 10

model2 = train(args)

model2.cpu()

plot_images(imgs.view(-1,1,28,28))
plt.savefig('./vae_results/original2.png')

rec = model.reconstruct_img(imgs)
plot_images(rec.view(-1,1,28,28))
plt.savefig('./vae_results/rec2.png')

zmu, zsigma = model.encoder(imgs)

vis = visdom.Visdom()

plot_vae_samples(model2, vis)
