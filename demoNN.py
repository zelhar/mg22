import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt

import torch.distributions as D

import torch.nn.functional as F

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# testing synthetic dataset
x = torch.randn((100,3,28,28))

d = TensorDataset(x)

z = d.__getitem__(2) # retuns 1-tuple of tensor (no label) 
z[0].shape

# with labels
y = torch.randint(low=0, high=1, size=(100,))
d = TensorDataset(x,y)
z = d.__getitem__(2) # retuns 1-tuple of tensor (no label) 
z[0].shape
z[1].shape

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")



bce = nn.BCELoss(reduction="none")
x = torch.tensor(0.5)
y = torch.tensor(0.7)
bce(x,y)

f = lambda x, y: y * torch.log(x) + (1-y) * torch.log(1-x)
f(x,y)


torch.softmax(torch.tensor([1,2,3]), 0, torch.float64)

# generate mixed distributions
m = D.OneHotCategorical(torch.tensor([1,2,3,6]))
m.sample()
m.sample_n(10)
m.sample((3,4))

m = D.Normal(torch.tensor([0,10.0]), torch.tensor([1.0,2]))

m.sample((3,4))

# Example of target with class indices
loss = nn.CrossEntropyLoss()

input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()

input = torch.randn((3, 2), requires_grad=True)
target = torch.rand((3, 2), requires_grad=False)
loss = F.binary_cross_entropy(F.sigmoid(input), target)
loss.backward()

loss = nn.BCELoss(reduction="none")
x = torch.tensor([0,0.25,0.5,0.75,1])
F.binary_cross_entropy(x,x,reduction="none")
loss(x,x)

x = torch.tensor([0,25,0.5,0.75,1])
y = torch.tensor([0,0.25,0.5,0.75,1])
loss(x,y)
