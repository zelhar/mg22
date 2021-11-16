import numpy as np
import torch

#import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image


device = 'cpu'
gen = torch.Generator(device=device)
print(gen.device)

gen.get_state()
gen.manual_seed(42)

mus = torch.rand(size=(3,2), generator=gen)
mus
sigmas = torch.rand(size=(3,2), generator=gen)
sigmas

ws = torch.rand((2,1), generator=gen)
ws = torch.nn.functional.normalize(ws, p=1, dim=0)
ws.sum()
ws
mus @ ws

x = torch.tensor([1,2,3,1.5])
x / torch.norm(x, 1)

x = torch.normal(mus, sigmas, generator=gen)
y = torch.randn((5,3,2))
y

one_sample = x @ ws
one_sample

z = None
if z== None:
    print("hi")
else:
    print("bye")

class DataGenerator:
    """
    A class that constructs synthetic tabular data of real numbers. 
    initialization parameters:
    param N: the dimension of a data point
    param K: The number of signatures. 
    param M: The number of samples
    param seed: default=42
    param fixed_proportion: default=False
    param device = 'cpu'
    
    The K N-dimensional signatures are similar to a
    base which would define a k-dimensional hyperplane, however we use random
    variables. Every signature v_i corresponds to a N-dimensional vector of
    means and std, mu_i and sigma_i.

    If fixed_proportion=True, every generated data point will be generated from
    a fixed convex combination of the signatures. This simulates data which was
    generated from a fixed proportion of base signals. Otherwise, every data
    point is a random convex combination of the signatures.

    computed outputs:
    output means a class variable which stores some value
    output data: a NxM tensor of the generated data.
    output mus: a NxK tensor of the signatures' means
    output sigmnas: a NxK tensor of the sgnatures' sigmas
    output weights: if fixed_proportion=True, a Kx1 non negative vector which
    sums to one. This is the fixed convex combination which generated every data
    point. If fixed_proportion=False, returns a KxM tensor, every column
    represents the convex combination used to generate the corresponding data
    point.
    """

    def __init__(self, N, K, M, seed=42, fixed_proportion=False, device='cpu'):
        # initiate a random generator
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        self.seed = seed
        self.gen = gen
        self.N = N 
        self.K = K
        self.M = M
        self.fixed_proportion = fixed_proportion
        self.device = device
        self.generateData()

    def generateData(self):
        # generate means and stds of the signatures
        self.mus = torch.rand(size=(self.N,self.K), generator=self.gen)
        self.sigmas = torch.rand(size=(self.N,self.K), generator=self.gen)
        if self.fixed_proportion:
            self.weights = torch.rand((self.K,1), generator=self.gen)
            # copy the same weights for all structures
            #self.weights = torch.ones(self.K,self.M) * self.weights
            self.weights = torch.ones(self.M,self.K,1) * self.weights
        else:
            #self.weights = torch.rand((self.K,self.M), generator=self.gen)
            self.weights = torch.rand((self.M,self.K,1), generator=self.gen)
        # normalize the weights column-wise
        self.weights = torch.nn.functional.normalize(self.weights, p=1, dim=1)
        # stretch the segmas and mus to fit M samples
        self.moos = torch.ones((self.M, self.N, self.K)) * self.mus
        self.soos = torch.ones((self.M, self.N, self.K)) * self.sigmas
        # M random samples of K signatures (M,K,N):
        self.randSigs = torch.normal(self.moos,self.soos,generator=self.gen)
        #self.weights = self.weights.reshape((self.M,self.K,1))
        # (M,N,K) x (M, K, 1) = (M, N, 1)
        self.data = self.randSigs @ self.weights
        #self.data = self.data.reshape((self.M,self.N)).T
        self.data = self.data.reshape((self.M,self.N)) # M samples, each sample is in R^N

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(400, 100)
        #self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(100, 20)
        self.fc22 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 400)
        #self.fc4 = nn.Linear(400, 784)
        self.fc4 = nn.Linear(400, 400)

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
        #mu, logvar = self.encode(x.view(-1, 784))
        mu, logvar = self.encode(x.view(-1, 400))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 400), reduction='sum')
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(epoch):
    model.train()
    train_loss = 0
    #for batch_idx, (data, _) in enumerate(train_loader):
    for batch_idx, (data, ) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

        
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        #for i, (data, _) in enumerate(test_loader):
        for i, (data, ) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(batch_size, 400)[:n]])
                                      #recon_batch.view(batch_size, 3, 32, 32)[:n]])
                                      #recon_batch.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def runit():
    for epoch in range(1, epochs + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            #save_image(sample.view(64, 3, 32, 32),
            ##save_image(sample.view(64, 1, 28, 28),
            #           'results/sample_' + str(epoch) + '.png')

# 5 dimensional data, 2 5-dimensional signatures, 8 data points.
x = DataGenerator(5, 2, 8)

x.weights
x.mus
x.sigmas

x.data


xs = DataGenerator(400, 20, 1000)

mydata = TensorDataset(xs.data)

test = mydata.__getitem__(0)
test[0].shape


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

output = model(test[0])
output[0].shape
output[1].shape
output[2].shape

tests = xs.data[:10,:]
tests.shape
outputs = model(tests)
outputs[0].shape
outputs[1].shape
outputs[2].shape


model(xs.data)

x = torch.rand((3,2))
y = torch.rand((2,5))
x
y
x@y

z = x * torch.ones((7,3,2))
z
z @ y

z = torch.rand((10,3,2))
z
z @ y


z = torch.rand((10,3,2))
w = torch.rand((10, 2, 5))
z @ w

x = torch.arange(6)
x
x = x.reshape((2,3))
x
x = x.reshape((3,2,1))
x


# create train/test data and train model 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xs = DataGenerator(400, 2, 10000)
mydata = TensorDataset(xs.data)

traindata = TensorDataset(xs.data[:9000,:])
testdata = TensorDataset(xs.data[9000:,:])

batch_size = 50
log_interval = 100
epochs = 150

train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(testdata, batch_size=batch_size, shuffle=True)

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)




runit()











