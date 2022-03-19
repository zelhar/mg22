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
import torch.distributions as D

def save_reconstructs(encoder, decoder, x, epoch):
        with torch.no_grad():
            x = x.to(device)
            sample = decoder(encoder(x))
            save_image(x.view(x.shape[0], 1, 28, 28),
                       'results/originals_' + str(epoch) + '.png')
            save_image(sample.view(x.shape[0], 1, 28, 28),
                       'results/reconstructs_' + str(epoch) + '.png')

def save_random_reconstructs(model, nz, epoch):
        with torch.no_grad():
            sample = torch.randn(64, nz).to(device)
            sample = model(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')


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
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nin))


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
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.main(z)

# parameters
nin = 28*28
nz = 20
batchSize = 256
epochs = 10

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

bce = nn.BCELoss(reduction="mean")
kld = lambda mu, logvar : -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
mse = nn.MSELoss()

encoder = Encoder(nin, nz, nin*4, 1024).to(device)
decoder = Decoder(nz, nin, 1024, nin*4).to(device)
# discriminator for gaussian on the latent space
disGauss = Discriminator(nz, 1024, 512).to(device)
# discriminator for the data
disData = Discriminator(nin, nin*4, 1024).to(device)

optimGauss = optim.Adam(disGauss.parameters(), lr=3e-4)
optimData = optim.Adam(disData.parameters(), lr=3e-4)
optimEnc = optim.Adam(encoder.parameters(), lr=3e-4)
optimDec = optim.Adam(decoder.parameters(), lr=3e-4)

epochs = 1
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train disData
        encoder.requires_grad_(False)
        decoder.requires_grad_(False)
        disGauss.requires_grad_(False)
        disData.requires_grad_(True)
        encoder.eval()
        decoder.eval()
        disGauss.eval()
        disData.train()
        optimData.zero_grad()
        xreal = data.view(-1, nin).to(device)
        xfake = decoder(encoder(xreal))
        yreal = disData(xreal)
        yfake = disData(xfake)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        loss_data_real = bce(yreal, labels_real)
        loss_data_fake = bce(yfake, labels_fake)
        loss_data = loss_data_real + loss_data_fake
        loss_data.backward()
        optimData.step()
        # train disGauss
        encoder.requires_grad_(False)
        decoder.requires_grad_(False)
        disGauss.requires_grad_(True)
        disData.requires_grad_(False)
        disGauss.train()
        disData.eval()
        optimGauss.zero_grad()
        x = data.view(-1, nin).to(device)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        zfake = encoder(x)
        zreal = torch.randn(batch_size, nz).to(device)
        yfake = disGauss(zfake)
        yreal = disGauss(zreal)
        loss_gauss_real = bce(yreal, labels_real)
        loss_gauss_fake = bce(yfake, labels_fake)
        loss_gauss = loss_gauss_fake + loss_gauss_real
        loss_gauss.backward()
        optimGauss.step()
        # train Encoder to generate gaussian and fool disGauss
        encoder.requires_grad_(True)
        decoder.requires_grad_(False)
        disGauss.requires_grad_(False)
        disData.requires_grad_(False)
        encoder.train()
        decoder.eval()
        disGauss.eval()
        disData.eval()
        optimEnc.zero_grad()
        x = data.view(-1, nin).to(device)
        z = encoder(x)
        labels_real = torch.ones(batch_size, 1).to(device)
        y = disGauss(z)
        loss_enc = bce(y, labels_real)
        loss_enc.backward()
        optimEnc.step()
        # train Decoder and Encoder to generate fake data
        encoder.requires_grad_(True)
        decoder.requires_grad_(True)
        disGauss.requires_grad_(False)
        disData.requires_grad_(False)
        disGauss.eval()
        disData.eval()
        encoder.train()
        decoder.train()
        optimEnc.zero_grad()
        optimDec.zero_grad()
        x = data.view(-1, nin).to(device)
        z = encoder(x)
        recon = decoder(z)
        predict = disData(recon)
        labels_real = torch.ones(batch_size, 1).to(device)
        loss_encdec = bce(predict, labels_real)
        loss_encdec.backward()
        optimDec.step()
        optimEnc.step()
        if idx % 100 == 0:
            print("loss data:", loss_data.item())
            print("loss gauss:", loss_gauss.item())
            print("loss enc:", loss_enc.item())
            print("loss encdec:", loss_encdec.item())
        #break
    #break





save_random_reconstructs(decoder, nz, 680)
xs, _ = iter(train_loader).next()
save_reconstructs(encoder, decoder, xs.to(device), 680)

torch.save(encoder, "results/aae_encoder_mnist.pth")
torch.save(decoder, "results/aae_decoder.pth")
torch.save(disData, "results/aae_disData.pth")
torch.save(disGauss, "results/aae_disGauss.pth")



epochs = 13
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train disData
        encoder.requires_grad_(False)
        decoder.requires_grad_(False)
        disGauss.requires_grad_(False)
        disData.requires_grad_(True)
        encoder.eval()
        decoder.eval()
        disGauss.eval()
        disData.train()
        optimData.zero_grad()
        xreal = data.view(-1, nin).to(device)
        xfake = decoder(encoder(xreal))
        yreal = disData(xreal)
        yfake = disData(xfake)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        loss_data_real = bce(yreal, labels_real)
        loss_data_fake = bce(yfake, labels_fake)
        loss_data = loss_data_real + loss_data_fake
        loss_data.backward()
        optimData.step()
        # train Decoder and Encoder 
        encoder.requires_grad_(True)
        decoder.requires_grad_(True)
        disGauss.requires_grad_(False)
        disData.requires_grad_(False)
        disGauss.eval()
        disData.eval()
        encoder.train()
        decoder.train()
        optimEnc.zero_grad()
        optimDec.zero_grad()
        x = data.view(-1, nin).to(device)
        z = encoder(x)
        recon = decoder(z)
        predict = disData(recon)
        labels_real = torch.ones(batch_size, 1).to(device)
        loss_encdec = bce(predict, labels_real)
        loss_encdec.backward()
        optimDec.step()
        optimEnc.step()
        if idx % 100 == 0:
            print("loss data:", loss_data.item())
            print("loss encdec:", loss_encdec.item())
        #break
    #break

# trying a vanila autoencoder with the enc/dec
epochs = 13
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train disData
        # train Decoder and Encoder 
        encoder.requires_grad_(True)
        decoder.requires_grad_(True)
        disGauss.requires_grad_(False)
        disData.requires_grad_(False)
        disGauss.eval()
        disData.eval()
        encoder.train()
        decoder.train()
        optimEnc.zero_grad()
        optimDec.zero_grad()
        x = data.view(-1, nin).to(device)
        z = encoder(x)
        recon = decoder(z)
        loss_encdec = bce(recon, x)
        loss_encdec.backward()
        optimDec.step()
        optimEnc.step()
        if idx % 100 == 0:
            print("loss encdec:", loss_encdec.item())
        #break
    #break

save_random_reconstructs(decoder, nz, 680)
xs, _ = iter(train_loader).next()
save_reconstructs(encoder, decoder, xs.to(device), 680)

# trying again to train as AAE
epochs = 3
for epoch in range(epochs):
    for idx, (data, _) in enumerate(train_loader):
        batch_size = data.shape[0]
        # train disData
        encoder.requires_grad_(False)
        decoder.requires_grad_(False)
        disGauss.requires_grad_(False)
        disData.requires_grad_(True)
        encoder.eval()
        decoder.eval()
        disGauss.eval()
        disData.train()
        optimData.zero_grad()
        xreal = data.view(-1, nin).to(device)
        xfake = decoder(encoder(xreal))
        yreal = disData(xreal)
        yfake = disData(xfake)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        loss_data_real = bce(yreal, labels_real)
        loss_data_fake = bce(yfake, labels_fake)
        loss_data = loss_data_real + loss_data_fake
        loss_data.backward()
        optimData.step()
        if idx % 100 == 0:
            print("predicts disData:",
                    yreal.sum().item()/batch_size,
                    yfake.sum().item()/batch_size
                    )
        # train disGauss
        encoder.requires_grad_(False)
        decoder.requires_grad_(False)
        disGauss.requires_grad_(True)
        disData.requires_grad_(False)
        disGauss.train()
        disData.eval()
        optimGauss.zero_grad()
        x = data.view(-1, nin).to(device)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        zfake = encoder(x)
        zreal = torch.randn(batch_size, nz).to(device)
        yfake = disGauss(zfake)
        yreal = disGauss(zreal)
        loss_gauss_real = bce(yreal, labels_real)
        loss_gauss_fake = bce(yfake, labels_fake)
        loss_gauss = loss_gauss_fake + loss_gauss_real
        loss_gauss.backward()
        optimGauss.step()
        if idx % 100 == 0:
            print("predicts disGauss:",
                    yreal.sum().item()/batch_size,
                    yfake.sum().item()/batch_size
                    )

        # train Encoder to generate gaussian and fool disGauss
        encoder.requires_grad_(True)
        decoder.requires_grad_(False)
        disGauss.requires_grad_(False)
        disData.requires_grad_(False)
        encoder.train()
        decoder.eval()
        disGauss.eval()
        disData.eval()
        optimEnc.zero_grad()
        x = data.view(-1, nin).to(device)
        z = encoder(x)
        labels_real = torch.ones(batch_size, 1).to(device)
        y = disGauss(z)
        loss_enc = bce(y, labels_real)
        loss_enc.backward()
        optimEnc.step()
        # train Decoder to generate fake data
        encoder.requires_grad_(True)
        decoder.requires_grad_(True)
        disGauss.requires_grad_(False)
        disData.requires_grad_(False)
        disGauss.eval()
        disData.eval()
        encoder.train()
        decoder.train()
        optimEnc.zero_grad()
        optimDec.zero_grad()
        x = data.view(-1, nin).to(device)
        z = encoder(x)
        recon = decoder(z)
        predict = disData(recon)
        labels_real = torch.ones(batch_size, 1).to(device)
        #loss_encdec = bce(predict, labels_real)
        loss_encdec = bce(recon, x)
        loss_encdec.backward()
        optimDec.step()
        #optimEnc.step()
        if idx % 100 == 0:
            print("loss data:", loss_data.item())
            print("loss gauss:", loss_gauss.item())
            print("loss enc:", loss_enc.item())
            print("loss encdec:", loss_encdec.item())
        #break
    #break


save_random_reconstructs(decoder, nz, 681)
xs, _ = iter(train_loader).next()
save_reconstructs(encoder, decoder, xs.to(device), 681)










