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

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def plot_images(imgs, nrow=4):
    grid_imgs = make_grid(imgs, nrow=nrow).permute(1, 2, 0)
    plt.imshow(grid_imgs)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


class Generator(nn.Module):
    def __init__(self, ngpu=1):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.Unflatten(1, (nz, 1, 1)),
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            #nn.Flatten(),
            #nn.Linear(64**2, 28**2),
            #nn.Sigmoid(),
            #nn.Unflatten(1, (1, 28, 28)),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ngpu=1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            #nn.Flatten(),
            #nn.Linear(28**2, 64**2),
            #nn.Unflatten(1, (1,64,64)),
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            nn.Flatten(),
        )

    def forward(self, input):
        return self.main(input)

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
    def __init__(self, nin, nh=512, image_size=64, nc=3):
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
            nout=nc * image_size ** 2,
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
            nn.Unflatten(1, (nc, image_size, image_size)),
        )

    def forward(self, input):
        output = self.main(input)
        return output

transform = transforms.Compose(
    [
        transforms.Resize(64),
#        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


# Root directory for dataset
#dataroot = "data/celeba"
# Number of workers for dataloader
workers = 2
# Batch size during training
batch_size = 128
# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64
# Number of channels in the training images. For color images this is 3
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Size of feature maps in generator
ngf = 64
# Size of feature maps in discriminator
ndf = 64
# Number of training epochs
num_epochs = 5
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
device = "cuda" if torch.cuda.is_available() else "cpu"
bce = nn.BCELoss()
mse = nn.MSELoss()

dataset = datasets.CIFAR10(
        root='data/',
        train=True,
        download=True,
        transform=transform,
        )

train_loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=128,
    shuffle=True, 
)
test_loader = torch.utils.data.DataLoader(
    dataset=datasets.CIFAR10(
        root='data/',
        train=False,
        download=True,
        transform=transform,
        ),
    batch_size=128,
    shuffle=True,
)

netG = Generator(ngpu).to(device)
netG.apply(init_weights)
netD = Discriminator(ngpu).to(device)
netD.apply(init_weights)

fixed_noise = torch.randn((batch_size,nz)).to(device)
real_label = 1
fake_label = 0

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1,0.999) )
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1,0.999) )



###################### Training
criterion = nn.BCELoss()
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 10
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(train_loader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,1), real_label, dtype=torch.float, device=device)
        label = torch.ones((b_size,1)).to(device)
        m = distributions.Uniform(0,1e-2)
        label = 1 + m.sample((b_size,1)).to(device)
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
        noise = torch.randn(b_size, nz, device=device)
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
        label = 1 + m.sample((b_size,1)).to(device)
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
                  % (epoch, num_epochs, i, len(train_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


fake = netG(fixed_noise).detach().cpu()

fake.shape

plot_images(denorm(fake), nrow=16)

plot_images(torch.cat(img_list, dim=2), nrow=128)

img_list[0].shape

torch.cat(img_list, dim=1).shape




################### trying same stuff but with fully connected NN
transform = transforms.Compose(
    [
#        transforms.Resize(64),
#        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

netG = LinGenerator(nin=nz, nh=3000, image_size=32, nc=3).to(device)
netD = LinDiscriminator(nin=32**2 * 3, nh=3000).to(device)

netG.apply(init_weights)
netD.apply(init_weights)

imgs, ls = train_loader.__iter__().next()
imgs.shape
netD(imgs.cuda()).shape
netG(fixed_noise).shape


optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1,0.999) )
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1,0.999) )

