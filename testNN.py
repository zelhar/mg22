import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

from skimage import io



transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=4, shuffle=True, num_workers=4)

# get some random training images
dataiter = iter(trainloader)

images, labels = dataiter.next()

def detensorizeImage(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    # convert from (c, h, w) pytorch format to standard (h,w,c)
    npimg = np.transpose(npimg, (1, 2, 0))
    return npimg
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #plt.show()

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        pass
    def forward(self, x):
        pass

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

myImage = 0

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

z = transforms.Resize((28,28))(images)
w = z.view(-1, )
