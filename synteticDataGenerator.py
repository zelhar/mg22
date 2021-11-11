import numpy as np
import torch

import pickle


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
        self.data = self.data.reshape((self.M,self.N)).T




        

# 5 dimensional data, 2 5-dimensional signatures, 8 data points.
x = DataGenerator(5, 2, 8)

x.weights
x.mus
x.sigmas

x.data





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






















