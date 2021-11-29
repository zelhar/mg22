# based on examples from https://github.com/pytorch/examples
# https://github.com/L1aoXingyu/pytorch-beginner/https://github.com/L1aoXingyu/pytorch-beginner/
# https://github.com/bfarzin/pytorch_aae/blob/master/main_aae.py
# https://github.com/artemsavkin/aae/blob/master/aae.ipynb
import argparse
import os
import torch
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

import numpy as np

def save_reconstructs(model, z, epoch):
        with torch.no_grad():
            sample = model(z).cpu()
            save_image(sample.view(z.shape[0], 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

def save_random_reconstructs(model, nz, epoch):
        with torch.no_grad():
            sample = torch.randn(64, nz).to(device)
            sample = model(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

class Generator(nn.Module):
    """
    Generates fake image from random normal noise 
    """

    def __init__(self, nin, nout, nh1, nh2):
        """ """
        super(Generator, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.ReLU(),
            nn.Linear(nh2, nout),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """
    Discriminates between fake (generated) images and real ones
    """

    def __init__(self, nin, nh1, nh2):
        """dimensions of the input layer, the 1st and 2nd hidden layers."""
        super(Discriminator, self).__init__()
        self.nin = nin
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.LeakyReLU(0.2),
            nn.Linear(nh1, nh2),
            nn.LeakyReLU(0.2),
            nn.Linear(nh2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(-1, self.nin)
        return self.main(x)


# parameters
xdim = 28*28
zdim = 100
h1dim = 256
h2dim = 256
batchSize = 128
epochs = 3
device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data", train=True, download=True, transform=transforms.ToTensor()
    ),
    batch_size=batchSize,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=False, transform=transforms.ToTensor()),
    batch_size=batchSize,
    shuffle=True,
)


gen = Generator(zdim, xdim, h2dim, h1dim).to(device)
dis = Discriminator(xdim, h1dim, h2dim).to(device)

optim_dis = optim.Adam(dis.parameters(), lr=3e-4)
optim_gen = optim.Adam(gen.parameters(), lr=3e-4)

gen_losses = []
dis_losses = []

bce = nn.BCELoss(reduction="mean")
mse = nn.MSELoss(reduction="mean")
l1 = nn.L1Loss(reduction="mean")

################################################

(xs, _) = iter(train_loader).next()
xs = xs.to(device)
ys = dis(xs)
xs.shape
ys.shape

zs = torch.randn(xs.shape[0], zdim).to(device)
zs.shape

ws = gen(zs)
ws.shape



########################################

## Training
epochs = 10
gen = Generator(zdim, xdim, 800, 2400).to(device)
dis = Discriminator(xdim, 2400, 800 ).to(device)
optim_dis = optim.Adam(dis.parameters(), lr=3e-4)
optim_gen = optim.Adam(gen.parameters(), lr=3e-4)
gen_losses = []
dis_losses = []
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        x = data.to(device)
        x = x.view(-1, xdim)

        # train dis
        gen.eval()
        gen.requires_grad_(False)
        dis.train()
        dis.requires_grad_(True)
        optim_dis.zero_grad()

        z = torch.randn(batch_size, zdim).to(device)
        y = gen(z)
        predict_fake = dis(y)
        loss_dis_fake = bce(predict_fake, labels_fake)
        loss_dis_fake.backward() # this gradient will be accumulated to the next

        predict_real = dis(x)
        loss_dis_real = bce(predict_real, labels_real)
        loss_dis_real.backward()

        loss_dis = loss_dis_real + loss_dis_fake
        #loss_dis.backward()
        optim_dis.step()

        #w = torch.cat((x, y), 0)
        #predicts = dis(w)
        #labels = torch.cat((labels_real, labels_fake), 0)
        #loss_dis = bce(predicts, labels)
        #loss_dis.backward()
        #optim_dis.step()

        # train gen
        gen.train()
        gen.requires_grad_(True)
        dis.eval()
        dis.requires_grad_(False)
        optim_gen.zero_grad()

        z = torch.randn(batch_size, zdim).to(device)
        y = gen(z)
        predict = dis(y)
        loss_gen = bce(predict, labels_real)
        loss_gen.backward()
        optim_gen.step()


        if idx % 50 == 0:
            print(epoch, idx, "losses:",
                    "dis loss", loss_dis.item(),
                    "gen loss", loss_gen.item(),
                    "loss_fake", predict_fake.mean().item(),
                    "loss real", predict_real.mean().item(),
                    )









save_random_reconstructs(gen, zdim, 666)




