# fun fun 2022-01-29
# https://github.com/pyro-ppl/pyro/blob/dev/examples/vae/vae.py
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
from typing import Union, Any, cast

# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import fnorm, replicate, logNorm
from my_torch_utils import scsimDataset
import scsim.scsim as scsim

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam


print(torch.cuda.is_available())

class Encoder(nn.Module):
    """
    Gaussian Encoder module for VAE.
    """

    def __init__(
        self,
        nz: int = 10,
        nh: int = 1024,
        imgsize: int = 28,
    ) -> None:
        super(Encoder, self).__init__()
        self.nin = nin = imgsize ** 2
        self.nz = nz
        self.imgsize = imgsize
        self.encoder = nn.Sequential(
            nn.Flatten(),
            fclayer(nin, nh, False, 0.0, nn.LeakyReLU()),
            fclayer(nh, nh, False, 0.0, nn.LeakyReLU()),
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
        )
        self.zmu = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz),
        )
        self.zlogvar = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nz),
            # nn.Softplus(),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """
    Gaussian / Bernoulli decoder module for VAE
    """

    def __init__(
        self,
        nz: int = 10,
        nh: int = 1024,
        imgsize: int = 28,
        is_Bernoulli: bool = True,
    ) -> None:
        super(Decoder, self).__init__()
        self.out = nout = imgsize ** 2
        self.nz = nz
        self.imgsize = imgsize
        self.is_Bernoulli = is_Bernoulli
        self.decoder = nn.Sequential(
            fclayer(nz, nh, False, 0.0, nn.LeakyReLU()),
            fclayer(nh, nh, False, 0.0, nn.LeakyReLU()),
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
        )
        self.xmu = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nout),
        )
        # or if we prefer Bernoulu decoder
        self.bernoulli = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nout),
            nn.Sigmoid(),
        )
        self.xlogvar = nn.Sequential(
            nn.Linear(nh, nh),
            nn.LeakyReLU(),
            nn.Linear(nh, nout),
            nn.Softplus(),
        )

    def forward(self, z):
        h = self.decoder(z)
        if self.is_Bernoulli:
            mu = self.bernoulli(h)
            return mu, torch.tensor(0)
        else:
            mu = self.xmu(h)
            logvar = self.xlogvar(h)
            return mu, logvar


class VAE(nn.Module):
    """
    VAE class for use with pyro!
    note that we use
    """

    def __init__(
        self,
        nz: int = 10,
        nh: int = 1024,
        imgsize: int = 28,
        is_Bernoulli: bool = True,
    ) -> None:
        super(VAE, self).__init__()
        self.nin = nin = imgsize ** 2
        self.out = nout = imgsize ** 2
        self.nz = nz
        self.imgsize = imgsize
        self.is_Bernoulli = is_Bernoulli
        self.encoder = Encoder(nz, nh, imgsize)
        self.decoder = Decoder(nz, nh, imgsize, is_Bernoulli)

    # model describes p(x|z)p(z)
    def model(self, x):
        # register decoder
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.nz, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.nz, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img, logvar_img = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Bernoulli(loc_img, validate_args=False).to_event(1),
                obs=x.reshape(-1, 784),
            )
            # return the loc so we can visualize it later
            return loc_img, logvar_img

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_logvar = self.encoder.forward(x)
            z_scale = z_logvar.exp()  # it's technically logsigma
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_logvar = self.encoder(x)
        z_scale = z_logvar.exp()
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img, logvar_img = self.decoder(z)
        return loc_img, logvar_img

class Arguments:
    cuda: bool = True
    learning_rate: float = 1e-3
    jit: bool = False
    visdom_flag: bool = False
    num_epochs: int = 21
    test_frequency: int = 5
    tsne_iter: int = 10
    nz : int = 50
    is_Bernoulli : bool = True

foo = {
        "s" : 0,
        "gg" : 1,
        }

args = Arguments()

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
   batch_size=256,
   shuffle=True,
)
train_loader = torch.utils.data.DataLoader(
   dataset=datasets.MNIST(
       root='data/',
       train=True,
       download=True,
       transform=transform,
       ),
   batch_size=256,
   shuffle=True,
)


def train(args: Arguments) -> VAE:
    pyro.clear_param_store()
    # setup the VAE
    vae = VAE(nz=args.nz)
    # setup the optimizer
    adam_args = {
        "lr": args.learning_rate,
    }
    optimizer = Adam(adam_args)
    # setup the inference algorithm
    elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)
    train_elbo = []
    test_elbo = []
    if args.cuda:
        vae.cuda()
    # training loop
    for epoch in range(args.num_epochs):
        # initialize loss accumulator
        epoch_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        for x, _ in train_loader:
            # if on GPU put mini-batch into CUDA memory
            x = x.flatten(1)
            if args.cuda:
                x = x.cuda()
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x)
        # report training diagnostics
        normalizer_train = len(train_loader.dataset)
        total_epoch_loss_train = epoch_loss / normalizer_train
        train_elbo.append(total_epoch_loss_train)
        print(
            "[epoch %03d]  average training loss: %.4f"
            % (epoch, total_epoch_loss_train)
        )
        if epoch % args.test_frequency == 0:
            # initialize loss accumulator
            test_loss = 0.0
            # compute the loss over the entire test set
            for i, (x, _) in enumerate(test_loader):
                x = x.flatten(1)
                # if on GPU put mini-batch into CUDA memory
                if args.cuda:
                    x = x.cuda()
                # compute ELBO estimate and accumulate loss
                test_loss += svi.evaluate_loss(x)
            # report test diagnostics
            normalizer_test = len(test_loader.dataset)
            total_epoch_loss_test = test_loss / normalizer_test
            test_elbo.append(total_epoch_loss_test)
            print(
                "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
            )
        # if epoch == args.tsne_iter:
        # mnist_test_tsne(vae=vae, test_loader=test_loader)
        # plot_llk(np.array(train_elbo), np.array(test_elbo))
    return vae




model = train(args)

model.cpu()
imgs, labels = test_loader.__iter__().next()

loc_im, logvar_img = model.reconstruct_img(imgs)

fig, ax = plt.subplots(2,1)

plot_images(imgs.view(-1,1,28,28))

plot_images(loc_im.view(-1,1,28,28))

plt.close()

mu, logvar = model.encoder(imgs)

z = distributions.Normal(mu, logvar.exp()).sample()

loc_im, loc_lv = model.decoder(z)

def save_plot_tsne(z_loc, classes, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        #ind_class = classes[:, ic] == 1
        ind_class = classes == ic
        color = plt.cm.Set1(ic)
        #plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, cmap="viridis")
        plt.title("Latent Variable T-SNE per Class")
        fig.savefig("./results/" + str(name) + "_embedding_" + str(ic) + ".png")
    fig.savefig("./results/" + str(name) + "_embedding.png")

def mnist_test_tsne(vae=None, test_loader=None, name="pyroXXXVAE"):
    """
    This is used to generate a t-sne embedding of the vae
    """
    #name = "pyroVAE"
    data = test_loader.dataset.test_data.float()
    mnist_labels = test_loader.dataset.test_labels
    z_loc, z_scale = vae.encoder(data)
    save_plot_tsne(z_loc, mnist_labels, name)

mnist_test_tsne(model, test_loader)


def plot_tsne(z_loc, classes, name):
    import matplotlib
    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        #ind_class = classes[:, ic] == 1
        ind_class = classes == ic
        color = plt.cm.Set1(ic)
        #plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, cmap="viridis")
        plt.title(name + " Latent Variable T-SNE per Class")
        #fig.savefig("./results/" + str(name) + "_embedding_" + str(ic) + ".png")
    plt.legend("0123456789a")
    #fig.savefig("./results/" + str(name) + "_embedding.png")

data = test_loader.dataset.data.float().reshape(-1,1,28,28)/255

plot_images(data[:24])

z_loc, z_scale = model.encoder(data)

mnist_labels = test_loader.dataset.targets

plot_tsne(z_loc, mnist_labels, "foooVAE")

plt.savefig('./results/fooVAE_tse.png')

plt.close()

x = test_loader.dataset.data.float().flatten(1)/255

plot_tsne(x, mnist_labels, "foooVAE")

plt.close()

# umap
z_data = z_loc.detach().numpy()

clusterable_embedding = umap.UMAP(
    n_neighbors=20,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(z_data)

plt.scatter(clusterable_embedding[:,0], clusterable_embedding[:,1], c =
        mnist_labels, s=1.4, cmap="viridis", )

plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))


plt.legend("0123456789a")

plt.cla()

# scrnaseq
countspath = "./data/scrnasim/counts.npz"
idpath = "./data/scrnasim/cellparams.npz"
genepath = "./data/scrnasim/geneparams.npz"


dataSet = scsimDataset("data/scrnasim/counts.npz",
        "data/scrnasim/cellparams.npz", genepath)

trainD, testD = dataSet.__train_test_split__(8500)

trainLoader = torch.utils.data.DataLoader(trainD, batch_size=128, shuffle=True)
testLoader = torch.utils.data.DataLoader(testD, batch_size=128, shuffle=True)

nxs, xs, ls = trainLoader.__iter__().next()

