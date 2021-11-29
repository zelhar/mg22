# based on examples from https://github.com/pytorch/examples
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F

# dimension of a data point
nin = 28*28
# dimension of the latent space
nz = 20
# dimension of hidden layer 1 etc.
nh1 = 28*28*5
nh2 = 40
nh3 = 28*28*5
nh4 = 28*28
# other parameters
batchSize = 128
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

def trainOneEpochGAN(epoch, Gmodel, Dmodel, Goptim, Doptim):
    Gmodel.train()
    Dmodel.train()
    criterion = nn.BCELoss()
    tiny = 1e-9
    for batch_idx, (data, _) in enumerate(train_loader):
        tempBatchSize = data.shape[0]
        real_labels = torch.ones(tempBatchSize , 1).to(device)
        fake_labels = torch.zeros(tempBatchSize , 1).to(device)
        Dmodel.zero_grad()
        data = data.to(device)
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        predictsReal = Dmodel(data)
        DerrReal = criterion(predictsReal, real_labels)
        DerrReal.backward()
        # train with fake
        znoise = torch.randn(data.shape[0], nz).to(device)
        fakedata = Gmodel(znoise)
        predictsFake = Dmodel(fakedata.detach()) # reserve gradient or something...
        DerrFake = criterion(predictsFake, fake_labels)
        DerrFake.backward()
        Doptim.step()
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        Gmodel.zero_grad()
        predictsFake = Dmodel(fakedata)
        Gerr = criterion(predictsFake, real_labels)
        Gerr.backward()
        Goptim.step()




def trainOneEpoch(epoch, batchSize, nin, enc, dec, disc, optimizerEnc, optimizerDec, optimizerDisc):
    enc.train()
    dec.train()
    disc.train()
    #zrandom = torch.randn_like(....)
    train_loss = 0
    tiny = 1e-9
    for batch_idx, (data, _) in enumerate(train_loader):
        tempBatchSize = data.shape[0]
        real_labels = torch.ones(tempBatchSize , 1).to(device)
        fake_labels = torch.zeros(tempBatchSize , 1).to(device)
        # train steps ....
        optimizerEnc.zero_grad()
        optimizerDisc.zero_grad()
        optimizerDec.zero_grad()
        data = data.to(device)
        ## Train the Encoder and Decoder to minimize reconstruction loss ##
        ## on the real sample ##
        # encode batch of samples
        z_sample_batch = enc(data)
        # decode sample batch
        recon_sample_batch = dec(z_sample_batch)
        #recon_loss = nn.MSELoss(reduction="mean")(recon_sample_batch,
        #        data.view(-1,nin))
        recon_loss = nn.BCELoss(reduction="mean")(recon_sample_batch,
                data.view(-1,nin))
        # update encoder and decoder: 
        recon_loss.backward()
        optimizerEnc.step()
        optimizerDec.step()

        ## Train the discriminator on real and fake (==encoded sample) ##
        # train on real (real sample, is 'fake' for the disc)
        enc.eval() # deactivate dropout if we use it
        z_real = enc(data)
        output_labels = disc(z_real)
        disc_loss_sample = nn.BCELoss(reduction="mean")(output_labels + tiny,
                fake_labels)
        disc_loss_sample.backward()
        
        ## train on random generated sample
        z_random_batch = torch.randn_like(z_sample_batch)
        output_labels = disc(z_random_batch)
        disc_loss_random = nn.BCELoss(reduction="mean")(output_labels + tiny,
                real_labels)
        disc_loss_random.backward()
        optimizerDisc.step()

        ## Train encoder as a generator to fool the discriminator
        enc.train()
        z = enc(data)
        labels = disc(z)
        # we want to fool disc so it would return real (1)
        enc_gen_loss = nn.BCELoss(reduction="mean")(labels, real_labels)
        enc_gen_loss.backward()
        optimizerEnc.step()

        if batch_idx % 50 == 0:
            print("batch ", batch_idx, "losses", recon_loss.item(), disc_loss_random.item(),
                    disc_loss_random.item(), enc_gen_loss.item())
        ### (idea for later: train the discriminate on the samples space between
        ### fake and real samples)

class GeneratorX(nn.Module):
    def __init__(self, nin, nz, nh1, nh2, nh3, nh4):
        super(GeneratorX, self).__init__()
        self.encode = nn.Sequential(
                nn.Linear(nin, nh1), 
                nn.ReLU(),
                nn.Linear(nh1, nh2),
                nn.ReLU(),
                nn.Linear(nh2, nz),
                )

        self.decode = nn.Sequential(
                nn.Linear(nz, nh3),
                nn.ReLU(),
                nn.Linear(nh3,nh4),
                nn.ReLU(),
                nn.Linear(nh4, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        z = self.encode(input).view(-1, nin)
        output = self.decode(z)
        return output

class Generator(nn.Module):
    def __init__(self, nin, nout, nh1, nh2):
        super(Generator, self).__init__()

        self.decode = nn.Sequential(
                nn.Linear(nin, nh1),
                nn.ReLU(),
                nn.Linear(nh1,nh2),
                nn.ReLU(),
                nn.Linear(nh2, nout),
                nn.Sigmoid()
                )

    def forward(self, input):
        return self.decode(input)


class Encoder(nn.Module):
    # the encoder is the generator in the AAE case
    # this is a deterministic, non-variational encoder
    def __init__(self, nin, nz, nh1, nh2):
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(nin, nh1), 
                nn.ReLU(),
                nn.Linear(nh1, nh2),
                nn.ReLU(),
                nn.Linear(nh2, nz),
                )

    def forward(self, input):
        return self.main(input.view(-1, nin))

class Decoder(nn.Module):
    def __init__(self, nin, nz, nh3):
        super(Decoder, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(nz, nh3),
                nn.ReLU(),
                nn.Linear(nh3, nin),
                nn.Sigmoid()
                )

    def forward(self, z):
        return self.main(z)

class DiscriminatorG(nn.Module):
    def __init__(self, nin, nh1, nh2):
        super(DiscriminatorG, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(nin, nh1),
                nn.ReLU(),
                nn.Linear(nh1,nh2),
                nn.ReLU(),
                nn.Linear(nh2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        return self.main(input.view(-1, nin))

class Discriminator(nn.Module):
    def __init__(self, nin, nh1, nh2):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(nin, nh1),
                nn.ReLU(),
                nn.Linear(nh1,nh2),
                nn.ReLU(),
                nn.Linear(nh2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        return self.main(input.view(-1, nin))


def save_random_reconstructs(model, nz, epoch):
        with torch.no_grad():
            sample = torch.randn(64, nz).to(device)
            sample = model(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')

# realise models and run training
e = Encoder(nin, nz, nh1, nh2)
dec = Decoder(nin, nz, nh3)
disc = Discriminator(nz, nh1, nh2)

e.to(device)
optimizerEnc = optim.Adam(e.parameters())
dec.to(device)
optimizerDec = optim.Adam(dec.parameters())
disc.to(device)
optimizerDisc = optim.Adam(disc.parameters())

for epoch in range(epochs):
    trainOneEpoch(epoch, batchSize, nin, e, dec, disc, optimizerEnc, optimizerDec, optimizerDisc)
    save_random_reconstructs(dec, nz, epoch)



g = Generator(nz, nin, nh2, nh1)
g
g.to(device)
goptim = optim.Adam(g.parameters())

x,l = iter(train_loader).next()
x.shape
l.shape
znoise = torch.randn(x.shape[0], nz)
g(znoise).shape

d = DiscriminatorG(nin, nh1, nh2)
d
d.to(device)
doptim = optim.Adam(d.parameters())

for epoch in range(epochs):
    trainOneEpochGAN(epoch, g, d, goptim, doptim) 
    save_random_reconstructs(g, nz, epoch)




#def trainOneEpochGAN(epoch, Gmodel, Dmodel, Goptim, Doptim):
g.train()
d.train()
criterion = nn.BCELoss()
iny = 1e-9

x,l = iter(train_loader).next()

for foo, bar in enumerate(train_loader):
    print(bar[0].shape)
    break


#for batch_idx, (data, _) in enumerate(train_loader):
tempBatchSize = x.shape[0]
real_labels = torch.ones(tempBatchSize , 1).to(device)
fake_labels = torch.zeros(tempBatchSize , 1).to(device)
d.zero_grad()
x = x.to(device)

############################
# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
###########################
# train with real
predictsReal = d(x)
DerrReal = criterion(predictsReal, real_labels)
DerrReal.backward()

# train with fake
znoise = torch.randn(x.shape[0], nz).to(device)
fakedata = g(znoise)
predictsFake = d(fakedata)
DerrFake = criterion(predictsFake, fake_labels)
DerrFake.backward()
doptim.step()

############################
# (2) Update G network: maximize log(D(G(z)))
###########################
g.zero_grad()
znoise = torch.randn(x.shape[0], nz).to(device)
fakedata = g(znoise)
predictsFake = d(fakedata)
real_labels = torch.ones(tempBatchSize , 1).to(device)
Gerr = criterion(predictsFake, real_labels)

Gerr.backward()

goptim.step()


