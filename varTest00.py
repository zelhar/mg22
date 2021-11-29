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

class VAEOLD(nn.Module):
    def __init__(self):
        super(VAEOLD, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class VAE(nn.Module):
    def __init__(self, nin, nz, nh1, nh2, nh3):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.encoder = nn.Sequential(
                nn.Linear(nin, nh1), 
                nn.ReLU(),
                nn.Linear(nh1, nh2),
                nn.ReLU(),
                )

        self.decoder = nn.Sequential(
                nn.Linear(nz, nh2),
                nn.ReLU(),
                nn.Linear(nh2, nh3),
                nn.ReLU(),
                nn.Linear(nh3, nin),
                nn.Sigmoid()
                )

        self.mumap = nn.Linear(nh2, nz)
        self.logvarmap = nn.Linear(nh2, nz)

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
        x = self.decoder(z)
        return x
        #h3 = F.relu(self.fc3(z))
        #return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, nin):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, nin), reduction='sum')
    # bce with sum reduction heavily depends on the dimension of x, maybe try
    # mean reduction?
    # BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='mean')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def trainOneEpoch(model, optimizer):
    model.train()
    train_loss = 0
    tiny = 1e-9
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        tempBatchSize = data.shape[0]
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, nin)
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            print("batch ", batch_idx, "loss", loss.item())


def save_random_reconstructs(model, nz, epoch):
        with torch.no_grad():
            sample = torch.randn(64, nz).to(device)
            sample = model.decode(sample)
            sample = sample.cpu()
            save_image(sample.view(64, 1, 28, 28),
                       'results/sample_' + str(epoch) + '.png')


# dimension of a data point
nin = 28*28
# dimension of the latent space
nz = 20
# dimension of hidden layer 1 etc.
nh1 = 28*28*5
nh2 = 400
nh3 = 28*28*5
nh4 = 400
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

model = VAE(nin, nz, nh1, nh2, nh3).to(device)
print(device)
#model = VAEOLD().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

trainOneEpoch(model, optimizer)
print("first save")
save_random_reconstructs(model, nz, 0)

print("looping")
for epoch in range(epochs):
    trainOneEpoch(model, optimizer)
    save_random_reconstructs(model, nz, epoch)






