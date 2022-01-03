import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, nin, nh, nout):
        super(Net, self).__init__()

        self.main = nn.Sequential(
                nn.Linear(nin, nh), 
                nn.ReLU(),
                nn.Linear(nh, nh), 
                nn.ReLU(),
                nn.Linear(nh, nout), 
                )

    def forward(self, x):
        return self.main(x)

mse = nn.MSELoss()

device = "cuda"
net = Net(1, 20, 1).to(device)
optimizer = optim.Adam(net.parameters())

x = torch.rand((128,1)).to(device)
y = torch.sin(x)
x.shape
y.shape

net(x).shape


for epoch in range(3000):
    net.zero_grad()
    x = torch.rand((128,1)).to(device)
    x = 2 * torch.pi * x
    y = torch.sin(x)
    z = net(x)
    loss = mse(z,y)
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0:
        print("loss = ", loss.item())


x = torch.pi * torch.rand((1280,1)).to(device)

x = torch.linspace(0, torch.pi * 2, 1000).to(device)
x = x.reshape((1000,1))

y = torch.sin(x)
z = net(x)
mse(z, y)

x = x.flatten().detach().cpu().numpy()
y = y.flatten().detach().cpu().numpy()
z = z.flatten().detach().cpu().numpy()


plt.plot(x, y)

plt.plot(x, z)

plt.cla()


z = torch.randn((128,2))
x = z[:,0].numpy()
y = z[:,1].numpy()

plt.scatter(x,y)

class AE(nn.Module):
    def __init__(self, nin, nh, nz):
        super(AE, self).__init__()

        self.encode = nn.Sequential(
                nn.Linear(nin, nh), 
                nn.ReLU(),
                nn.Linear(nh, nh), 
                nn.ReLU(),
                nn.Linear(nh, nz), 
                )

        self.decode = nn.Sequential(
                nn.Linear(nz, nh*2), 
                nn.ReLU(),
                nn.Linear(nh*2, nh*2), 
                nn.ReLU(),
                nn.Linear(nh*2, nin), 
                #nn.Tanh(),
                nn.ReLU(),
                )

        self.fc1 = nn.Linear(nin, 1)
        self.fc2 = nn.Linear(nin, 1)

    def forward(self, x):
        h = self.encode(x)
        h = self.decode(h)
        x1 = self.fc1(h)
        x2 = self.fc2(h)
        #y = torch.cat((x1,x2)).view(-1, 2)
        #y = torch.zeros_like(x)
        #y[:,0] = x1
        #y[:,1] = x2
        y = torch.cat((x1,x2), dim=1)
        return y

model = AE(2, 1000, 4).to(device)
#model = AE(2, 1000, 1).to(device)
optimizer = optim.Adam(model.parameters())


x = torch.randn((128,2)).to(device)
model(x).shape

for epoch in range(3000):
    model.zero_grad()
    x = torch.rand((128,2)).to(device)
    #x = x / 10.0
    x = x 
    y = model(x)
    loss = mse(y,x)
    loss.backward()
    optimizer.step()
    if epoch % 25 == 0:
        print("loss = ", loss.item())

x = torch.randn((1280,2)).to(device)
xx = torch.randn((1280,2)).to(device)
y = model(x)
mse(x,y)

z = model(x).detach().cpu().numpy()

z = x.detach().cpu().numpy()
x = z[:,0]
y = z[:,1]
plt.scatter(x,y)

#x = torch.randn((1280,2)).to(device) / 10
x = torch.randn((1280,2)).to(device)

z = model(x).detach().cpu().numpy()

x = z[:,0]
y = z[:,1]
plt.scatter(x,y)

x = torch.randn((12,2)).to(device)
y = model(x)
mse(x,y)

plt.cla()


x = torch.randn((5,1))
x

x.shape

torch.cat((x,x)).shape
torch.cat((x,x)).view(-1, 2).shape

torch.stack((x,x))
