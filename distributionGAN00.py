#https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils

from math import pi, sin, cos
from torch import distributions
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision import models
from torchvision.utils import make_grid
from torchvision.utils import save_image

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

class LinDiscriminator(nn.Module):
    def __init__(self, nin, nh):
        super(LinDiscriminator, self).__init__()
        fc1 = fclayer(
            nin=nin,
            nout=nh,
            batchnorm=True,
            dropout=0.2,
            activation=nn.LeakyReLU(),
        )
        fc2 = fclayer(
            nin=nh,
            nout=nh,
            batchnorm=True,
            dropout=0.2,
            activation=nn.LeakyReLU(),
        )
        fc3 = fclayer(
            nin=nh,
            nout=nh,
            batchnorm=True,
            dropout=0.2,
            activation=nn.LeakyReLU(),
        )
        fc4 = fclayer(
            nin=nh,
            nout=1,
            batchnorm=False,
            dropout=0,
            activation=nn.Sigmoid(),
        )
        self.main = nn.Sequential(
            nn.Flatten(),
            fc1,
            fc2,
            fc3,
            fc4,
            )

    def forward(self, input):
        output = self.main(input)
        return output

class LinGenerator(nn.Module):
    def __init__(self, nin, nh=512, nout=3):
        super(LinGenerator, self).__init__()
        fc1 = fclayer(
            nin=nin,
            nout=nh,
            batchnorm=True,
            dropout=0.2,
            activation=nn.LeakyReLU(),
        )
        fc2 = fclayer(
            nin=nh,
            nout=nh,
            batchnorm=True,
            dropout=0.2,
            activation=nn.LeakyReLU(),
        )
        fc3 = fclayer(
            nin=nh,
            nout=nh,
            batchnorm=True,
            dropout=0.2,
            activation=nn.LeakyReLU(),
        )
        fc4 = fclayer(
            nin=nh,
            nout=nout,
            batchnorm=False,
            dropout=0,
            activation=nn.Tanh(),
        )
        self.main = nn.Sequential(
            nn.Flatten(),
            fc1,
            fc2,
            fc3,
            fc4,
            #nn.Unflatten(1, (nc, image_size, image_size)),
        )

    def forward(self, input):
        output = self.main(input)
        return output

# Batch size during training
batch_size = 128
nz = 2
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
device = "cuda" if torch.cuda.is_available() else "cpu"
bce = nn.BCELoss()
mse = nn.MSELoss()

netG = LinGenerator(nin=nz, nh=3000, nout=nz).to(device)
netD = LinDiscriminator(nin=nz, nh=3000).to(device)

netG.apply(init_weights)
netD.apply(init_weights)


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1,0.999) )
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1,0.999) )

gauss = mixedGaussianCircular(rho=0.01, sigma=0.5, k=10, j=0)
#gauss.sample()
mix = distributions.Categorical(torch.ones(10,))
comp = distributions.Independent(gauss, 0)
gmm = distributions.MixtureSameFamily(mix, comp)

z = gmm.sample((5000,)).clip(-1,1).numpy()
z.shape
x = z[:,0]
y = z[:,1]
plt.scatter(x,y)
#
#samples = gauss.sample((2500,))
#x = samples[:,:,0].flatten()
#y = samples[:,:,1].flatten()
#plt.scatter(x,y)

sampler = distributions.Normal(0, 1)

# sampler to generate input for the generator
sampler.sample((19,1))

# generate batch of 100 samples from this distribution like this:
foo = gmm.sample((100,))
foo.shape

netD(foo.cuda()).shape
netG(foo.cuda()).shape

fixed_noise = sampler.sample((batch_size,nz)).to(device)
real_label = 1
fake_label = 0

###################### Training
criterion = nn.BCELoss()
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 10
rounds = 1000
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i in range(rounds):
    #for i, data in enumerate(train_loader, 0):
        data = gmm.sample((batch_size,)).clip(-1,1)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data.to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,1), real_label, dtype=torch.float, device=device)
        label = torch.ones((b_size,1)).to(device)
        m = distributions.Uniform(0,1e-2)
        label = 1 - m.sample((b_size,1)).to(device)
        # Forward pass real batch through D
        #output = netD(real_cpu).view(-1)
        output = netD(real_cpu)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        #noise = torch.randn(b_size, nz, 1, 1, device=device)
        #noise = torch.randn(b_size, nz, device=device)
        noise = sampler.sample((batch_size, nz)).to(device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        label = torch.zeros((b_size,1)).to(device)
        label = 0 + m.sample((b_size,1)).to(device)
        # Classify all fake batch with D
        #output = netD(fake.detach()).view(-1)
        output = netD(fake.detach())
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        label = torch.ones((b_size,1)).to(device)
        label = 1 - m.sample((b_size,1)).to(device)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        #output = netD(fake).view(-1)
        output = netD(fake)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, rounds ,
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == rounds -1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


noise = sampler.sample((5000,2)).to(device)
z = netG(noise).detach().cpu().numpy()
z.shape
x = z[:,0]
y = z[:,1]
plt.scatter(x,y)
#

z = gmm.sample((batch_size,)).clip(-1,1).numpy()
z.shape
x = z[:,0]
y = z[:,1]
plt.scatter(x,y)

plt.cla()
