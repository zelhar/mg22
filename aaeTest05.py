#43 based on examples from https://github.com/pytorch/examples
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
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torch.distributions as D

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def save_reconstructs(encoder, decoder, x, epoch):
        with torch.no_grad():
            x = x.to(device)
            sample = decoder(encoder(x))
            save_image(x.view(x.shape[0], 1, 28, 28),
                       'results/originals_' + str(epoch) + '.png')
            save_image(sample.view(x.shape[0], 1, 28, 28),
                       'results/reconstructs_' + str(epoch) + '.png')
            #save_image(denorm(x.view(x.shape[0], 1, 28, 28)),
            #           'results/originals_' + str(epoch) + '.png')
            #save_image(denorm(sample.view(x.shape[0], 1, 28, 28)),
            #           'results/reconstructs_' + str(epoch) + '.png')

def save_random_reconstructs(model, nz, epoch):
        with torch.no_grad():
            sample = torch.randn(64, nz).to(device)
            sample = model(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')
            #save_image(denorm(sample.view(64, 1, 28, 28)),
            #           'results/sample_' + str(epoch) + '.png')


# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,),   # 3 for RGB channels
                                     std=(0.5,))])                

class Discriminator(nn.Module):
    """
    Classifies real distribution and fake (generated) distribution.
    """
    def __init__(self, nin, nh1, nh2):
        super(Discriminator, self).__init__()
        self.nin = nin
        self.main = nn.Sequential(
                nn.Linear(nin, nh1),
                nn.ReLU(),
                nn.Linear(nh1, nh2),
                nn.ReLU(),
                nn.Linear(nh2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        return self.main(input)

class Encoder(nn.Module):
    """
    Encodes high dimensional data point into
    a low dimension latent space. 
    """

    def __init__(self, nin, nout, nh1, nh2):
        super(Encoder, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.ReLU(),
            nn.Linear(nh2, nout),
            #nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nin))



# the generator is the decoder
class Decoder(nn.Module):
    """
    Reconstructs a input from the latent, low dimensional space, into
    the original data space.
    """

    def __init__(self, nin, nout, nh1, nh2):
        super(Decoder, self).__init__()
        self.nin = nin
        self.nout = nout
        self.main = nn.Sequential(
            nn.Linear(nin, nh1),
            nn.ReLU(),
            nn.Linear(nh1, nh2),
            nn.ReLU(),
            nn.Linear(nh2, nout),
            nn.Sigmoid()
            #nn.Tanh(),
        )

    def forward(self, z):
        return self.main(z)

# parameters
nin = 28*28
nz = 2
batchSize = 256
epochs = 10
beta = 0.5
lr = 2e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data/", train=True, download=True, transform=transform
    ),
    batch_size=batchSize,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("data/", train=False, transform=transform),
    batch_size=batchSize,
    shuffle=True,
)

Dx = Discriminator(nin, 256, 256).to(device)
Dz = Discriminator(nz, 256, 256).to(device)
Gx = Decoder(nz, nin, 256, 256).to(device)
Gz = Encoder(nin, nz, 256, 256).to(device)
criterion = nn.BCELoss()
dx_optimizer = torch.optim.Adam(Dx.parameters(), lr=0.0002)
dz_optimizer = torch.optim.Adam(Dz.parameters(), lr=0.0002)
gx_optimizer = torch.optim.Adam(Gx.parameters(), lr=0.0002)
gz_optimizer = torch.optim.Adam(Gz.parameters(), lr=0.0002)



sample_dir = 'samples4'
epochs = 13
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        # discriminator Dx turn
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        dx_optimizer.zero_grad()
        gx_optimizer.zero_grad()
        xreal = data.view(-1, nin).to(device)
        predreal = Dx(xreal)
        lossDreal = criterion(predreal, labels_real)
        D_x = predreal.mean().item()
        z = torch.randn(batch_size, nz).to(device)
        xfake = Gx(z)
        predfake = Dx(xfake)
        lossDfake = criterion(predfake, labels_fake)
        lossD = lossDreal + lossDfake
        lossD.backward()
        dx_optimizer.step()
        D_z = predfake.mean().item()
        # generator Gx turn
        dx_optimizer.zero_grad()
        gx_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz).to(device)
        xfake = Gx(z)
        pred = Dx(xfake)
        lossG = criterion(pred, labels_real)
        lossG.backward()
        gx_optimizer.step()
        DG_z = pred.mean().item()
        # discriminator Dz turn
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        dz_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        x = data.view(-1, nin).to(device)
        zreal = torch.randn(batch_size, nz).to(device)
        predreal = Dz(zreal)
        lossDreal = criterion(predreal, labels_real)
        D_x = predreal.mean().item()
        zfake = Gz(x)
        predfake = Dz(zfake)
        lossDfake = criterion(predfake, labels_fake)
        lossD = lossDreal + lossDfake
        lossD.backward()
        dz_optimizer.step()
        D_z = predfake.mean().item()
        # generator Gz turn
        dz_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz).to(device)
        zfake = Gz(x)
        pred = Dz(zfake)
        lossG = criterion(pred, labels_real)
        lossG.backward()
        gx_optimizer.step()
        DG_z = pred.mean().item()
        if idx % 2000 == 0:
            print(D_x, D_z, DG_z, lossD.mean().item(), lossG.mean().item(), )
            fake_images = Gx(z)
            fake_images = fake_images.view(data.size(0), 1, 28, 28)
            #fake_images = denorm(fake_images)
            save_image(denorm(fake_images.data), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

save_random_reconstructs(Gx, nz, 336)


def train(epochstart, epochstop):
    for epoch in range(epochstart, epochstop):
        for idx, (data, _) in enumerate(train_loader):
            # discriminator Dx turn
            batch_size = data.shape[0]
            labels_real = torch.ones(batch_size, 1).to(device)
            labels_fake = torch.zeros(batch_size, 1).to(device)
            dx_optimizer.zero_grad()
            gx_optimizer.zero_grad()
            xreal = data.view(-1, nin).to(device)
            predreal = Dx(xreal)
            lossDreal = criterion(predreal, labels_real)
            D_x = predreal.mean().item()
            z = torch.randn(batch_size, nz).to(device)
            xfake = Gx(z)
            predfake = Dx(xfake)
            lossDfake = criterion(predfake, labels_fake)
            lossD = lossDreal + lossDfake
            lossD.backward()
            dx_optimizer.step()
            D_z = predfake.mean().item()
            # generator Gx turn
            dx_optimizer.zero_grad()
            gx_optimizer.zero_grad()
            labels_real = torch.ones(batch_size, 1).to(device)
            z = torch.randn(batch_size, nz).to(device)
            xfake = Gx(z)
            pred = Dx(xfake)
            lossG = criterion(pred, labels_real)
            lossG.backward()
            gx_optimizer.step()
            DG_z = pred.mean().item()
            # discriminator Dz turn
            batch_size = data.shape[0]
            labels_real = torch.ones(batch_size, 1).to(device)
            labels_fake = torch.zeros(batch_size, 1).to(device)
            dz_optimizer.zero_grad()
            gz_optimizer.zero_grad()
            x = data.view(-1, nin).to(device)
            zreal = torch.randn(batch_size, nz).to(device)
            predreal = Dz(zreal)
            lossDreal = criterion(predreal, labels_real)
            D_x = predreal.mean().item()
            zfake = Gz(x)
            predfake = Dz(zfake)
            lossDfake = criterion(predfake, labels_fake)
            lossD = lossDreal + lossDfake
            lossD.backward()
            dz_optimizer.step()
            D_z = predfake.mean().item()
            # generator Gz turn
            dz_optimizer.zero_grad()
            gz_optimizer.zero_grad()
            labels_real = torch.ones(batch_size, 1).to(device)
            z = torch.randn(batch_size, nz).to(device)
            zfake = Gz(x)
            pred = Dz(zfake)
            lossG = criterion(pred, labels_real)
            lossG.backward()
            gx_optimizer.step()
            DG_z = pred.mean().item()
            if idx % 2000 == 0:
                print(D_x, D_z, DG_z, lossD.mean().item(), lossG.mean().item(), )
                fake_images = Gx(z)
                fake_images = fake_images.view(data.size(0), 1, 28, 28)
                #fake_images = denorm(fake_images)
                save_image(denorm(fake_images.data), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))




Dx = Discriminator(nin, 128, 128).to(device)
Dz = Discriminator(nz, 128, 128).to(device)
Gx = Decoder(nz, nin, 4048, 2048).to(device)
Gz = Encoder(nin, nz, 4048 ,2048).to(device)
criterion = nn.BCELoss()
dx_optimizer = torch.optim.Adam(Dx.parameters(), )
dz_optimizer = torch.optim.Adam(Dz.parameters(), )
gx_optimizer = torch.optim.Adam(Gx.parameters(), )
gz_optimizer = torch.optim.Adam(Gz.parameters(), )
mse = nn.MSELoss()
bce = nn.BCELoss()

sample_dir = 'samples4'
epochs = 15
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]

        # generator Gx turn
        dx_optimizer.zero_grad()
        gx_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz).to(device)
        xfake = Gx(z)
        pred = Dx(xfake)
        lossGx = criterion(pred, labels_real)
        lossGx.backward()
        gx_optimizer.step()
        DG_z = pred.mean().item()

        # generator Gz turn
        dz_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz).to(device)
        x = data.view(-1, nin).to(device)
        zfake = Gz(x)
        pred = Dz(zfake)
        lossGz = criterion(pred, labels_real)
        lossGz.backward()
        gx_optimizer.step()
        DG_z = pred.mean().item()

        # discriminator Dx turn
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        dx_optimizer.zero_grad()
        gx_optimizer.zero_grad()
        xreal = data.view(-1, nin).to(device)
        predreal = Dx(xreal)
        lossDreal = criterion(predreal, labels_real)
        D_x = predreal.mean().item()
        z = torch.randn(batch_size, nz).to(device)
        xfake = Gx(z)
        predfake = Dx(xfake)
        lossDfake = criterion(predfake, labels_fake)
        lossDx = lossDreal + lossDfake
        lossDx.backward()
        dx_optimizer.step()
        D_z = predfake.mean().item()

        # discriminator Dz turn
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        dz_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        x = data.view(-1, nin).to(device)
        zreal = torch.randn(batch_size, nz).to(device)
        predreal = Dz(zreal)
        lossDreal = criterion(predreal, labels_real)
        D_x = predreal.mean().item()
        zfake = Gz(x)
        predfake = Dz(zfake)
        lossDfake = criterion(predfake, labels_fake)
        lossDz = lossDreal + lossDfake
        lossDz.backward()
        dz_optimizer.step()
        D_z = predfake.mean().item()

        # reconstruction error
        gx_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        x = data.view(-1, nin).to(device)
        z = Gz(x)
        recon = Gx(z)
        lossR = mse(recon, x)
        lossR.backward()
        gx_optimizer.step()
        gz_optimizer.step()


        if idx % 3000 == 0:
            print(lossDx.mean().item(),
                    lossDz.mean().item(),
                    lossGx.mean().item(),
                    lossGz.mean().item(),
                    lossR.mean().item(),
                    )
            fake_images = Gx(z)
            fake_images = fake_images.view(data.size(0), 1, 28, 28)
            #fake_images = denorm(fake_images)
            save_image(fake_images.data, os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch+1)))

save_random_reconstructs(Gx, nz, 336)





# testing AE
bce =nn.BCELoss()
mse = nn.MSELoss()
sample_dir = 'samples4'
epochs = 15
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # reconstruction error
        gx_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        x = data.view(-1, nin).to(device)
        z = Gz(x)
        recon = Gx(z)
        #lossR = mse(recon, x)
        lossR = bce(recon, x)
        lossR.backward()
        gx_optimizer.step()
        gz_optimizer.step()

        # generator Gz turn
        dz_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        labels_real = torch.ones(batch_size, 1).to(device)
        z = torch.randn(batch_size, nz).to(device)
        x = data.view(-1, nin).to(device)
        zfake = Gz(x)
        pred = Dz(zfake)
        lossGz = criterion(pred, labels_real)
        lossGz.backward()
        gz_optimizer.step()
        DG_z = pred.mean().item()


        # discriminator Dz turn
        batch_size = data.shape[0]
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        dz_optimizer.zero_grad()
        gz_optimizer.zero_grad()
        x = data.view(-1, nin).to(device)
        zreal = torch.randn(batch_size, nz).to(device)
        predreal = Dz(zreal)
        lossDreal = criterion(predreal, labels_real)
        D_x = predreal.mean().item()
        zfake = Gz(x)
        predfake = Dz(zfake)
        lossDfake = criterion(predfake, labels_fake)
        lossDz = lossDreal + lossDfake
        lossDz.backward()
        dz_optimizer.step()
        D_z = predfake.mean().item()

        if idx % 3000 == 0:
            print(lossR.mean().item(),
                    lossGz.mean().item(),
                    lossDz.mean().item(),
                    )
            originals = x.view(data.size(0), 1, 28, 28)
            z = Gz(originals.view(-1, nin))
            rec_images = Gx(z)
            rec_images = rec_images.view(data.size(0), 1, 28, 28)
            #fake_images = denorm(fake_images)
            save_image(rec_images, os.path.join(sample_dir, 'rec_images-{}.png'.format(epoch+1)))
            save_image(originals, os.path.join(sample_dir, 'original_images-{}.png'.format(epoch+1)))

save_random_reconstructs(Gx, nz, 336)







