#43 based on examples from https://github.com/pytorch/examples
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
import torchvision.utils as vutils
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
import torch.distributions as D
from torchvision import models

def save_reconstructs(encoder, decoder, x, epoch, device="cuda"):
        with torch.no_grad():
            x = x.to(device)
            sample = decoder(encoder(x))
            save_image(x.view(x.shape[0], 1, 28, 28),
                       'results/originals_' + str(epoch) + '.png')
            save_image(sample.view(x.shape[0], 1, 28, 28),
                       'results/reconstructs_' + str(epoch) + '.png')

def save_random_reconstructs(model, nz, epoch, device="cuda"):
        with torch.no_grad():
            #sample = torch.randn(64, nz).to(device)
            sample = torch.randn(64, nz,1,1).to(device)
            sample = model(sample).cpu()
            sample = denorm(sample)
            sample = transforms.Resize(32)(sample)
            save_image(sample,
                       'results/sample_' + str(epoch) + '.png')

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
