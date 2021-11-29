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
from torch.nn import functional as F

# dimension of a data point
nin = 28*28
# dimension of the latent space
nz = 2
# dimension of hidden layer 1 etc.
nh1 = 28*28*5
nh2 = 40
nh3 = 28*28*5
nh4 = 28*28

batchSize = 100

epochs = 3

real_label = 1
fake_label = 0

device = "cuda" if torch.cuda.is_available() else "cpu"

x = torch.randint(0,2, (5,9), dtype=torch.float64)
y = torch.randint(0,2, (5,9), dtype=torch.float64)
f = nn.BCELoss(reduction='sum')
g = nn.BCELoss(reduction='mean')
x
y
f(x,y)
g(x,y)


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





def trainOneEpoch(epoch, batchSize, nin, enc, dec, disc, optimizerEnc, optimizerDec, optimizerDisc):
    enc.train()
    dec.train()
    disc.train()
    real_labels = torch.ones(batchSize, 1).to(device)
    fake_labels = torch.zeros(batchSize, 1).to(device)
    #zrandom = torch.randn_like(....)
    train_loss = 0
    tiny = 1e-9
    for batch_idx, (data, _) in enumerate(train_loader):
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
        recon_loss = nn.MSELoss(reduction="mean")(recon_sample_batch,
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





        ### (idea for later: train the discriminate on the samples space between
        ### fake and real samples)
        # for the discriminator the encoded sample is fake (label 0), while normal
        # randomly generated data is real (label 1)
        # create a batch of random sample from the latent space with the normal
        # (or whatever) distribution
        # set the encoder to evaluation mode to keep the grad unchanged
        # or ...
        # recon_loss = nn.BCELoss(reduction="sum")




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
                nn.Sigmoid()
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
                nn.Tanh()
                )

    def forward(self, z):
        return self.main(z)

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
        return self.main(input)


bceloss = nn.BCELoss()

e = Encoder(nin, nz, nh1, nh2)
dec = Decoder(nin, nz, nh3)
disc = Discriminator(nz, nh1, nh2)




xs = torch.randn((5,28,28))
ys = e.main(xs.view(-1, 28*28))
ys.shape
zs = dec.main(ys)
zs.shape
cs = disc.main(xs.view(-1, 28*28))
cs


e.to(device)
optimizerEnc = optim.Adam(e.parameters())
dec.to(device)
optimizerDec = optim.Adam(dec.parameters())
disc.to(device)
optimizerDisc = optim.Adam(disc.parameters())


trainOneEpoch(0, batchSize, nin, e, dec, disc, optimizerEnc, optimizerDec, optimizerDisc)



#trainOneEpoch(0, batchSize, nin, e, dec, disc, optimizerEnc, optimizerDec, optimizerDisc)
# testing one iteration of the train function
e.train()
dec.train()
disc.train()

real_labels = torch.ones(batchSize, 1).to(device)
fake_labels = torch.zeros(batchSize, 1).to(device)

train_loss = 0
tiny = 1e-9

data, tags = iter(train_loader).next()

optimizerEnc.zero_grad()
optimizerDisc.zero_grad()
optimizerDec.zero_grad()
data = data.to(device)

z_sample_batch = e(data)
recon_sample_batch = dec(z_sample_batch)
recon_loss = nn.MSELoss(reduction="mean")(recon_sample_batch,
        data.view(-1,nin))
# update encoder and decoder: 
recon_loss.backward()
optimizerEnc.step()
optimizerDec.step()

## Train the discriminator on real and fake (==encoded sample) ##
# train on real (real sample, is 'fake' for the disc)
e.eval() # deactivate dropout if we use it
z_real = e(data)
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
e.train()
z = e(data)
labels = disc(z)

# we want to fool disc so it would return real (1)
enc_gen_loss = nn.BCELoss(reduction="mean")(labels, real_labels)
enc_gen_loss.backward()
optimizerEnc.step()




