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
        optimizerDec.step()
        optimizerEnc.step()

        ## Train the discriminator on real and fake (==encoded sample) ##
        # train on fake (real sample, is 'fake' for the disc)
        enc.eval() # deactivate dropout if we use it
        z_fake = enc(data)
        z_real = torch.randn_like(z_sample_batch)
        output_labels_real = disc(z_real)
        output_labels_fake = disc(z_fake)
        disc_loss_fake = nn.BCELoss(reduction="mean")(output_labels_fake,
                fake_labels)
        disc_loss_real = nn.BCELoss(reduction="mean")(output_labels_real,
                real_labels)
        disc_loss = disc_loss_fake + disc_loss_real
        disc_loss.backward()
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
            print("batch ", batch_idx, "losses", recon_loss.item(), disc_loss.item(), enc_gen_loss.item())
        ### (idea for later: train the discriminate on the samples space between
        ### fake and real samples)



class Encoder(nn.Module):
    """Encodes high dimensional data point into
    a low dimension latent space. It is also considered as
    the generator in this AAE model so it is trained to fool the discriminator.
    """
    # the encoder is the generator in the AAE case
    # this is a deterministic, non-variational encoder
    def __init__(self, nin, nz, nh1, nh2):
        """dimensions of the input, the lattent space, and the two hidden
        layers.
        """
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


