import argparse
import numpy as np
import os
import time
from math import pi, sin, cos, sqrt
import torch
from torch import nn
from torch import distributions
from torch import optim 
#import torch.utils.data
#import torchvision.utils as vutils
#from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision import models
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


def mixedGaussianCircular(k=10, sigma=0.025, rho=3.5, j=0):
    """
    Sample from a mixture of k 2d-gaussians. All have equal variance (sigma) and
    correlation coefficient (rho), with the means equally distributed on the
    unit circle.
    """
    #cat = distributions.Categorical(torch.ones(k))
    #i = cat.sample().item()
    #theta = 2 * torch.pi * i / k
    theta = 2 * torch.pi / k
    v = torch.Tensor((1, 0))
    T = torch.Tensor([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    S = torch.stack([T.matrix_power(i) for i in range(k)])
    mu = S @ v
    #cov = sigma ** 2 * ( torch.eye(2) + rho * (torch.ones(2, 2) - torch.eye(2)))
    #cov = cov @ S
    cov = torch.eye(2) * sigma ** 2
    cov[1,1] = sigma ** 2 * rho
    cov = torch.stack(
            [T.matrix_power(i+j) @ cov @ T.matrix_power(-i-j) for i in range(k)])
    gauss = distributions.MultivariateNormal(loc = mu, covariance_matrix= cov)
    return gauss

def fclayer(nin, nout, batchnorm=True, dropout=0.2, activation=nn.ReLU()):
    """
    define one fully connected later where nin, nout are the dimensions
    of the input and output.
    Perform dropout on the input (a value between 0 and 1, 0 means no dropout)
    and optional batchnormalization before the activation.
    Can also provide the activation function (ReLU is th edefault)
    """
    fc = nn.Sequential()
    if 0 < dropout < 1:
        fc.add_module("dropout", nn.Dropout(p=dropout))
    fc.add_module("linear", nn.Linear(nin, nout))
    if batchnorm:
        fc.add_module("batchnorm", nn.BatchNorm1d(num_features=nout))
    fc.add_module("activation", activation)
    return fc


class VAE(nn.Module):
    def __init__(self, nin, nz, nh1, nh2, nh3, nh4):
        super(VAE, self).__init__()

        self.nin = nin
        self.nz = nz

        self.encoder = nn.Sequential(
                nn.Linear(nin, nh1), 
                nn.ReLU(),
                nn.Linear(nh1, nh2),
                nn.ReLU(),
                )

        self.decoder = nn.Sequential(
                nn.Linear(nz, nh3),
                nn.ReLU(),
                nn.Linear(nh2, nh4),
                #nn.ReLU(),
                nn.Tanh(),
                #nn.Linear(nh4, nin),
                #nn.Sigmoid(),
                )

        self.mumap = nn.Linear(nh2, nz)
        self.logvarmap = nn.Linear(nh2, nz)

        self.dmu = nn.Linear(nh4, nin)
        self.dlv = nn.Linear(nh4, nin)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.mumap(h)
        logvar = self.logvarmap(h)
        return mu, logvar
        #h1 = F.relu(self.fc1(x))
        #return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = self.decoder(z)
        mu = self.dmu(h)
        s = self.dlv(h)
        return h

    def forward(self, x):
        x = x.view(-1, self.nin)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        h = self.decoder(z)
        m = self.dmu(h)
        s = self.dlv(h)
        return m, s, mu, logvar


# Batch size during training
batch_size = 128
nz = 4
#nz = 20
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
device = "cuda" if torch.cuda.is_available() else "cpu"
#bce = nn.BCELoss()
mse = nn.MSELoss()
bce = nn.BCELoss(reduction="sum")
kld = lambda mu, logvar : -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

model = VAE(nin=2, nz=nz, nh1=2*1024, nh2=2*512, nh3=2*512, nh4=2*1024).to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)


def fnorm(x, mu=0, s=1):
    """
    normal distribution function.
    """
    x = (x - mu) / s
    x = x ** 2
    x = -0.5 * x
    x = x.exp()
    x = x / (s * 2 * sqrt(pi) )
    return torch.sum(torch.log(x))

for epoch in range(9000):
    model.zero_grad()
    x = torch.randn((128,2)).to(device)
    m,s , mu, logvar = model(x)
    recon = m + torch.randn_like(m) * (0.5 * s).exp()
    #loss_recon = bce(recon, x)
    loss_recon = -fnorm(x, m, (0.5 * s).exp())
    loss_kld = kld(mu, logvar)
    loss = loss_kld + loss_recon
    loss.backward()
    optimizer.step()
    if epoch % 250 == 0:
        print(
                "loss_kld = ", loss_kld.item(),
                "loss_recon = ",
                loss_recon.item(),
                )


x = torch.randn((3280,2)).to(device)
a,b,c,d = model(x)

z = a + torch.randn_like(a) * (0.5 * b).exp()

xs = x.detach().cpu().numpy()
zs = z.detach().cpu().numpy()
u = zs[:,0]
v = zs[:,1]
plt.scatter(u,v)

ux = xs[:,0]
vx = xs[:,1]
plt.scatter(ux,vx)

plt.cla()

