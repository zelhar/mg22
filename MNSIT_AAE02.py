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

def mixedGaussianCircular(k=10, sigma=0.35, rho=0.75, j=0):
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

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)

def fclayer(nin, nout, batchnorm=True, dropout=0.2, activation=nn.ReLU()):
    """
    define one fully connected later where nin, nout are the dimensions
    of the input and output.
    Perform dropout on the input (a value between 0 and 1, 0 means no dropout)
    and optional batchnormalization after the activation.
    Can also provide the activation function (ReLU is th edefault)
    """
    fc = nn.Sequential()
    if 0 < dropout < 1:
        fc.add_module("dropout", nn.Dropout(p=dropout))
    fc.add_module("linear", nn.Linear(nin, nout))
    fc.add_module("activation", activation)
    if batchnorm:
        fc.add_module("batchnorm", nn.BatchNorm1d(num_features=nout))
    return fc

class Net(nn.Module):
    def __init__(
        self,
        nin,
        nout,
        nh=3000,
        dropout=0,
        activation=nn.Identity(),
        unflatten=False,
        image_size=28,
        batchnorm=True,
    ):
        """
        nin, nout, nh are (self explanatory) the dimensions of the input,
        output, and the hidden layers.
        dropout should be a real value between 0 and 1.
        activation is the the activation function used on the output layer.
        If batcnorm=True, batch normalization is applied to the hidden layers,
        after activation.
        """
        super(Net, self).__init__()
        self.nin = nin
        self.nz = nout
        self.nh = nh
        self.unflatten = unflatten
        self.image_size = image_size
        self.dropout = 1.0 * dropout
        self.batchnorm = nn.Identity()
        #self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorm = nn.BatchNorm1d(num_features=nh)
        fc1 = fclayer(self.nin, self.nh, batchnorm=batchnorm,
                dropout=self.dropout, activation=nn.ReLU())
        fc2 = fclayer(self.nh, self.nh, batchnorm=batchnorm,
                dropout=self.dropout, activation=nn.ReLU())
        fc3 = fclayer(self.nh, self.nz, batchnorm=False,
                dropout=self.dropout, activation=activation)

        self.main = nn.Sequential(
            nn.Flatten(),
            fc1,
            fc2,
            fc3,
                )
    def forward(self, input):
        output = self.main(input)
        if self.unflatten:
            output = nn.Unflatten(1, (1, self.image_size, self.image_size))(output)
        return output

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def save_reconstructs(
    encoder, decoder, x, epoch, nz=20, device="cuda", normalize=False
):
    with torch.no_grad():
        x = x.to(device)
        y = encoder(x)
        sample = decoder(y).cpu()
        if normalize:
            sample = denorm(sample)
            x = denorm(x)
        # save_image(x.view(x.shape[0], 3, 28, 28),
        save_image(x, "results/originals_" + str(epoch) + ".png")
        save_image(sample, "results/reconstructs_" + str(epoch) + ".png")


def save_random_reconstructs(model, nz, epoch, device="cuda", normalize=False,
        sampler=None):
    with torch.no_grad():
        if sampler == None:
            sample = torch.randn(64, nz).to(device)
        else:
            sample = sampler((64, )).to(device)
        sample = model(sample).cpu()
        if normalize:
            sample = denorm(sample)
        save_image(sample, "results/sample_" + str(epoch) + ".png")

def plot_images(imgs, nrow=4):
    grid_imgs = make_grid(imgs, nrow=nrow).permute(1,2,0)
    plt.imshow(grid_imgs)



transform = transforms.Compose(
    [
        transforms.ToTensor(),
        # transforms.Resize(64),
        # transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),  # 3 for RGB channels
        # transforms.Normalize(mean=(0.0,), std=(0.5,)),  # 3 for RGB channels
    ]
)


def trainAE(E, D, optE, optD, data, device, criterion=nn.BCELoss()):
    """
    Autoencoder training.
    Train encoder and decoder to reconstruct the input.
    """
    E.train()
    D.train()
    optE.zero_grad()
    optD.zero_grad()
    x = data.to(device)
    z = E(x)
    recon = D(z)
    lossED = criterion(recon, x)
    lossED.backward()
    optD.step()
    optE.step()
    return lossED

def trainG(G, Dz, optG, data, device, criterion=nn.BCELoss()):
    """
    train Generator to create 'fake' data and 
    fool the discriminator to label it True
    """
    batch_size = data.shape[0]
    Dz.eval()
    G.train()
    optG.zero_grad()
    labels_real = torch.ones(batch_size, 1).to(device)
    x = data.to(device)
    z = G(x)
    pred = Dz(z)
    lossG = criterion(pred, labels_real)
    lossG.backward()
    optG.step()
    return lossG

def trainDz(
    Dz,
    G,
    optDz,
    data,
    device,
    criterion=nn.BCELoss(),
    sampler=distributions.Normal(torch.zeros(2), torch.ones(2)),
):
    """
    Train the discriminator Dz to identify 'real' data (in this case,
    'real' means random vector samples from the desired distribution),
    and 'fake' data (generated by the encoder from the input)
    sampler should be a generator of the target distribution, i.e normal 
    or whatever.
    """
    # train Dz to disctiminate z
    batch_size = data.shape[0]
    G.eval()
    Dz.train()
    optDz.zero_grad()
    labels_real = torch.ones(batch_size, 1).to(device)
    labels_fake = torch.zeros(batch_size, 1).to(device)
    zreal = sampler.sample((batch_size,)).to(device)
    predZreal = Dz(zreal)
    lossZreal = criterion(predZreal, labels_real)
    #lossZreal.backward()
    x = data.to(device)
    zfake = G(x)
    predZfake = Dz(zfake)
    lossZfake = criterion(predZfake, labels_fake)
    #lossZfake.backward()
    lossZ = 0.5 * (lossZfake + lossZreal)
    lossZ.backward()
    optDz.step()
    return lossZ

# parameters
momentumAE = 0.9
momentumGD = 0.1
batch_size = 128
image_size = 28
lr1 = 1e-4
lr2 = 1e-5
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 2
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"
bce = nn.BCELoss()
mse = nn.MSELoss()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/", train=True, download=True, transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data/", train=False, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

#gmm = mixedGaussianCircular(rho=0.01, sigma=3.3, k=10, j=0)
gauss = mixedGaussianCircular(rho=0.01, sigma=3.4, k=10, j=0)
mix = distributions.Categorical(torch.ones(10,))
comp = distributions.Independent(gauss, 0)
gmm = distributions.MixtureSameFamily(mix, comp)


def training(epochstart, epochs, E, D, Dz, optE, optG, optD, optDz, 
        testname="gmm", device="cuda"):
    """
    a wrapper function to execute the training routines.
    """
    for epoch in range(epochstart, epochstart + epochs):
        for idx, (data, _) in enumerate(train_loader):
            # reconstruction phase
            lossED = trainAE(E, D, optE, optD, data, device, bce)
            # Dz discriminator
            lossDz = trainDz(Dz, E, optDz, data, device, bce)
            # E as generator
            lossG = trainG(E, Dz, optG, data, device, bce)
            if (epoch % 2 == 0) and (idx % 313 == 0):
                xreal = data.to(device)
                save_random_reconstructs(
                    D, nz, testname + str(epoch) + ":" + str(idx), sampler=gmm.sample
                )
                save_reconstructs(E, D, xreal, testname)
                print(
                    str(epoch) + ":" + str(idx) + ":\n",
                    lossED.mean().item(),
                    lossG.mean().item(),
                    lossDz.mean().item(),
                )

########################################################################
### Model and Ooptimizers Initiation
# Encoder
E = Net(
    image_size ** 2,
    nz,
    nh=1000,
    dropout=0.1,
    activation=nn.Identity(),
    batchnorm=True,
).to(device)
E.apply(init_weights)
# Decoder
D = Net(
    nz,
    image_size ** 2,
    nh=1000,
    dropout=0.1,
    activation=nn.Sigmoid(),
    unflatten=True,
    batchnorm=True,
).to(device)
D.apply(init_weights)
# discriminator for the latent space
Dz = Net(
    nz,
    1,
    nh=1000,
    dropout=0.2,
    activation=nn.Sigmoid(),
    batchnorm=True,
).to(device)
# Dz = Discriminator(nz=nz).to(device)
Dz.apply(init_weights)
# optDz = optim.Adam(Dz.parameters())

#optDz = optim.SGD(Dz.parameters(), lr=1e-2, momentum=0.9)
optE = optim.Adam(E.parameters(), )
optD = optim.Adam(D.parameters(), )
optG = optim.Adam(E.parameters())
optDz = optim.Adam(Dz.parameters())

#training(0, 5, E, D, Dz, optE, optG, optD, optDz, "gmm", device, )

start = 0
epochs = 15
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        # reconstruction phase
        lossED =  trainAE(E, D, optE, optD, data, device, bce)
        # Dz discriminator
        lossDz = trainDz(Dz, E, optDz, data, device, bce, sampler=gmm)
        # E as generator
        lossG = trainG(E, Dz, optG, data, device, bce)
        if (epoch % 2 == 0) and (idx % 313 == 0):
            xreal = data.to(device)
            save_random_reconstructs(D, nz, "gmm" + str(epoch) +":" + str(idx), sampler=gmm.sample)
            save_reconstructs(E, D, xreal, "gmm")
            print(
                    str(epoch) + ":" + str(idx) + ":\n",
                    lossED.mean().item(),
                    lossG.mean().item(),
                    lossDz.mean().item()
                    )


(data, labels) = iter(test_loader).next()

plot_images(data.cpu())
plot_images(D(E(data.cuda())).cpu())

z = E(data.cuda()).cpu().detach().numpy()
x = z[:,0]
y = z[:,1]

fig, ax = plt.subplots()
for i in range(10):
    ax.scatter(x[labels == i], y[labels == i], label=str(i))
ax.legend()

########################################################################
# GAN sanity test
ndf = 64
ngf = 64
nc = 1
nz = 2

class Discriminator(nn.Module):
    def __init__(self,):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28**2, 64**2),
            nn.Unflatten(1, (1,64,64)),
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

class Generator(nn.Module):
    def __init__(self, ):
        super(Generator, self).__init__()
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
            nn.Flatten(),
            nn.Linear(64**2, 28**2),
            nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



modelG = Net(
    nin=nz,
    nout=image_size ** 2,
    nh=1000,
    dropout=0,
    activation=nn.Sigmoid(),
    unflatten=True,
    image_size=28,
    batchnorm=False,
).to(device)

modelD = Net(
    nin=image_size ** 2,
    nout=1,
    nh=1000,
    dropout=0.2,
    activation=nn.Sigmoid(),
    unflatten=False,
    image_size=28,
    batchnorm=True,
).to(device)

modelG = Generator().to(device)
modelD = Discriminator().to(device)

data.shape
x = torch.rand((128,2))
x.shape

modelG(x.cuda()).shape
modelD(data.cuda()).shape




modelG = Generator().to(device)
modelD = Discriminator().to(device)
modelG.apply(init_weights)
modelD.apply(init_weights)
optG = optim.Adam(modelG.parameters(), )
optD = optim.Adam(modelD.parameters(), )

start = 0
epochs = 2
for epoch in range(start, start+epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        #z = gmm.sample((batch_size,)).to(device)
        z = torch.randn((batch_size,nz)).to(device)
        # D turns
        modelG.eval()
        modelD.train()
        optD.zero_grad()
        xreal = data.to(device)
        xfake  = modelG(z)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        pReal = modelD(xreal)
        pFake = modelD(xfake)
        errDreal = bce(pReal, labels_real)
        errDfake = bce(pFake, labels_fake)
        errD = 0.5 * (errDreal + errDfake)
        errD.backward()
        optD.step()
        #if (epoch % 2 == 0) and (idx % 313 == 0):
        if idx % 313 == 0:
            print(
                    errDfake.mean().item(),
                    errDreal.mean().item(),
                    )
        # G turn
        modelG.train()
        modelD.eval()
        optG.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        x  = modelG(z)
        p = modelD(x)
        errG = bce(p, labels_real)
        errG.backward()
        optG.step()
        if (epoch % 2 == 0) and (idx % 313 == 0):
            print(
                    errG.mean().item()
                    )



z = gmm.sample((batch_size,)).to(device)
x = modelG(z).detach().cpu()

plot_images(data, nrow=16)

plot_images(x, nrow=16)

netG = Generator().to(device)
netD = Discriminator().to(device)
modelG.apply(init_weights)
modelD.apply(init_weights)
optimizerD = optim.Adam(netD.parameters(), )
optimizerG = optim.Adam(netG.parameters(), )
real_label = 1
fake_label = 0

label = torch.full((128,1), real_label, dtype=torch.float, device=device)
label.shape

criterion = nn.BCELoss()
# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
#fixed_noise = torch.randn(64, nz, 1, 1, device=device)
fixed_noise = torch.randn((batch_size,nz)).to(device)

num_epochs = 2
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



dataset = datasets.CelebA('data/', split='train',
        download=True,
        transform=transforms.Compose(
    [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
    )

dataset = datasets.CIFAR10(
        root='data/',
        train=True,
        download=True,
        )

train_loader = torch.utils.data.DataLoader(
    dataset = datasets.CIFAR10(
        "data/", train=True, download=True, transform=transform
    ),
    batch_size=batch_size,
    shuffle=True,
)


(img, ls) = iter(train_loader).next()

img.shape

plot_images(img, nrow=16)
