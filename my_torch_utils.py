# useful function to be re-used in throught the tests in this project.
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple, ClassVar
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, NewType
import torch
from torch import nn, optim, distributions
import torch.utils.data
import torchvision.utils as vutils
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import Tensor
from math import pi, sin, cos, sqrt, log

import networkx as nx


from toolz import partial, curry

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

#kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

def kld2normal(
    mu: Tensor,
    logvar: Tensor,
    mu2: Tensor,
    logvar2: Tensor,
):
    """
    unreduced KLD KLD(p||q) for two diagonal normal distributions.
    https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    """
    result = 0.5 * (
        -1
        + (logvar.exp() + (mu - mu2).pow(2)) / logvar2.exp()
        + logvar2
        - logvar
    )
    return result

def soft_assign(z, mu, alpha=1):
    q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - mu)**2, dim=2) / alpha)
    q = q**(alpha+1.0)/2.0
    q = q / torch.sum(q, dim=1, keepdim=True)
    return q

def log_gaussian_prob(x : torch.Tensor, 
        mu : torch.Tensor = torch.tensor(0), 
        logvar : torch.Tensor = torch.tensor(0)
        ) -> torch.Tensor:
    """
    compute the log density function of a gaussian.
    user must make sure that the dimensions are aligned correctly.
    """
    return -0.5 * (
            log(2 * pi) + 
            logvar +
            (x - mu).pow(2) / logvar.exp()
            )

def softclip(tensor, min=-6.0, max=9.0):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + nn.functional.softplus(tensor - min)
    result_tensor = max - nn.functional.softplus(max - result_tensor)
    return result_tensor


class SoftClip(nn.Module):
    def __init__(self, min=-6.0, max=6.0):
        super(SoftClip, self).__init__()
        self.min = min
        self.max = max
    def forward(self, input):
        return softclip(input, self.min, self.max)

@curry
def fnorm(
    x: Tensor,
    mu: Union[float, Tensor] = 1.0,
    sigma: Union[float, Tensor] = 1.0,
    reduction: str = "sum",
) -> Tensor:
    """
    normal distribution density (elementwise) with optional reduction (logsum/sum/mean)
    """
    x = -0.5 * ((x - mu) / (sigma)).pow(2)
    x = x.exp()
    x = x / (sqrt(2 * pi) * sigma)
    if reduction == "sum":
        return x.sum()
    elif reduction == "logsum":
        return x.log().sum()
    elif reduction == "mean":
        return x.mean()
    else:
        return x


# This one was tested and seems correct ;)
@curry
def logNorm(
        x : Tensor,
        mu : Tensor,
        logvar : Tensor,) -> Tensor:
    """
    gives log of the density of the normal distribution.
    element-wise, no reductions.
    """
    y = -0.5 * (log(2 * pi) + logvar + (x - mu).pow(2) / logvar.exp() )
    return y


def init_weights(m: torch.nn.Module) -> None:
    """
    Initiate weights with random values, depending
    on layer type.
    In place, use the apply method.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

@curry
def replicate(x : Tensor, expand=(1,)):
    """
    Replicate a tensor in a new dimension (the new 0 dimension),
    creating n=ncopies identical copies.
    """
    y = torch.ones(expand + x.shape).to(x.device)
    return y * x



@curry
def normalize(
    x: Tensor,
    mu: Union[float, Tensor] = 0.5,
    sigma: Union[float, Tensor] = 0.5,
    clamp: bool = False,
) -> Tensor:
    """
    x <- (x - mu) / sigma
    """
    y = (x - mu) / sigma
    if clamp:
        y = y.clamp(0, 1)
    return y


@curry
def denorm(
    x: Tensor,
    mu: Union[float, Tensor] = 0.5,
    sigma: Union[float, Tensor] = 0.5,
    clamp: bool = True,
) -> Tensor:
    """
    inverse of normalize
    x <- sigma * x + mu
    """
    y = sigma * x + mu
    if clamp:
        y = y.clamp(0, 1)
    return y


def save_reconstructs(encoder,
                      decoder,
                      x,
                      epoch,
                      nz=20,
                      device="cuda",
                      normalized=False):
    with torch.no_grad():
        x = x.to(device)
        y = encoder(x)
        sample = decoder(y).cpu()
        if normalized:
            sample = denorm(sample)
            x = denorm(x)
        # save_image(x.view(x.shape[0], 3, 28, 28),
        save_image(x, "results/originals_" + str(epoch) + ".png")
        save_image(sample, "results/reconstructs_" + str(epoch) + ".png")


def save_random_reconstructs(model,
                             nz,
                             epoch,
                             device="cuda",
                             normalized=False):
    with torch.no_grad():
        # sample = torch.randn(64, nz).to(device)
        sample = torch.randn(64, nz).to(device)
        sample = model(sample).cpu()
        if normalized:
            sample = denorm(sample)
        save_image(sample, "results/sample_" + str(epoch) + ".png")


def plot_images(imgs, nrow=16, transform=nn.Identity(), out=plt):
    imgs = transform(imgs)
    grid_imgs = make_grid(imgs, nrow=nrow).permute(1, 2, 0)
    #plt.imshow(grid_imgs)
    out.imshow(grid_imgs)
    out.grid(False)
    out.axis("off")
    plt.pause(0.05)
    return grid_imgs

def plot_2images(img1, img2, nrow=16, transform=nn.Identity(), ):
    img1 = transform(img1)
    img2 = transform(img2)
    grid_img1 = make_grid(img1, nrow=nrow).permute(1, 2, 0)
    grid_img2 = make_grid(img2, nrow=nrow).permute(1, 2, 0)
    fig, axs = plt.subplots(2,1)
    axs[0].imshow(grid_img1)
    axs[1].imshow(grid_img2)
    axs[0].grid(False)
    axs[0].axis("off")
    axs[1].grid(False)
    axs[1].axis("off")
    plt.pause(0.05)
    return grid_img1, grid_img2

def plot_tsne(z_loc, classes, name):
    import matplotlib
    #matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    model_tsne = TSNE(n_components=2, random_state=0)
    z_states = z_loc.detach().cpu().numpy()
    z_embed = model_tsne.fit_transform(z_states)
    classes = classes.detach().cpu().numpy()
    fig = plt.figure()
    for ic in range(10):
        #ind_vec = np.zeros_like(classes)
        #ind_vec[:, ic] = 1
        #ind_class = classes[:, ic] == 1
        ind_class = classes == ic
        color = plt.cm.Set1(ic)
        #plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, color=color)
        plt.scatter(z_embed[ind_class, 0], z_embed[ind_class, 1], s=10, cmap="viridis")
        plt.title(name + " Latent Variable T-SNE per Class")
        #fig.savefig("./results/" + str(name) + "_embedding_" + str(ic) + ".png")
    plt.legend([str(i) for i in classes])
    #fig.savefig("./results/" + str(name) + "_embedding.png")

def fclayer(
        nin: int,
        nout: int,
        batchnorm: bool = True,
        dropout: float = 0.2,
        activation: nn.Module = nn.ReLU(),
) -> nn.Module:
    """
    define one fully connected later where nin, nout are the dimensions
    of the input and output.
    Perform dropout on the input (a value between 0 and 1, 0 means no dropout)
    and optional batchnormalization before the activation.
    Can also provide the activation function (ReLU is th edefault)
    """
    fc = nn.Sequential()
    if 0 < dropout < 1:
        fc.add_module("dropout", nn.Dropout(p=dropout))
    fc.add_module("linear", nn.Linear(nin, nout))
    if batchnorm:
        fc.add_module("batchnorm", nn.BatchNorm1d(num_features=nout))
    fc.add_module("activation", activation)
    return fc

def buildNetwork(
    layers: List[int],
    dropout: int = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
        if activation:
            net.add_module("activation" + str(i), activation)
    return net
    #return nn.Sequential(*net)

def buildNetworkv2(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

def buildNetworkv3(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    layernorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if layernorm:
            #net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
            net.add_module("layernotm" + str(i), nn.LayerNorm(layers[i],))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

def buildNetworkv4(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            #net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
            net.add_module("layernotm" + str(i), nn.LayerNorm(layers[i],))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

def buildNetworkv5(
    layers: List[int],
    dropout: float = 0,
    activation: Optional[nn.Module] = nn.ReLU(),
    batchnorm: bool = False,
):
    """
    build a fully connected multilayer NN.
    The output layer is always linear
    """
    net = nn.Sequential()
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    for i in range(1, len(layers)-1):
        net.add_module('linear' + str(i), nn.Linear(layers[i - 1], layers[i]))
        if dropout > 0:
            net.add_module("dropout" + str(i), nn.Dropout(dropout))
        if batchnorm:
            net.add_module("batchnorm" + str(i), nn.BatchNorm1d(num_features=layers[i]))
            #net.add_module("layernotm" + str(i), nn.LayerNorm(layers[i],))
        if activation:
            net.add_module("activation" + str(i), activation)
    n = len(layers) - 1
    net.add_module("output_layer", nn.Linear(layers[n-1], layers[n]))
    return net
    #return nn.Sequential(*net)

def buildCNetworkv1(
    nc : int = 1,
    nin : int = 28**2,
    nf : int = 32,
    nout : int = 2**12,
    dropout: float = 0,
    #activation: Optional[nn.Module] = nn.LeakyReLU(),
    #batchnorm: bool = False,
):
    """
    build a 1d CNN.
    maps 1d input into initial size 2**12,
    each conv reduces size by a factor of 4.
    rturns output 1d of size nout.
    The output layer is always linear
    """
    net = nn.Sequential(
            nn.Linear(nin, 2**12,),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(num_features=2**12,),
            nn.LeakyReLU(),
            nn.Unflatten(1, (nc,2**12)),
            nn.Conv1d(nc, nf, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf,),
            nn.LeakyReLU(),
            nn.Conv1d(nf, nf*2, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*2,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*2, nf*4, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*4,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*4, nf*8, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*8,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*8, nf*16, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*16,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*16, nf*32, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*32,),
            nn.LeakyReLU(),
            nn.Flatten(1),
            nn.Linear(nf*32, nout),
            )
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    return net

def buildCNetworkv2(
    nc : int = 1,
    nin : int = 28**2,
    nf : int = 32,
    nout : int = 2**10,
    dropout: float = 0,
    #activation: Optional[nn.Module] = nn.LeakyReLU(),
    #batchnorm: bool = False,
):
    """
    build a 1d CNN.
    maps 1d input into initial size 2**10,
    each conv reduces size by a factor of 4.
    rturns output 1d of size nout.
    The output layer is always linear
    """
    net = nn.Sequential(
            nn.Linear(nin, 2**10,),
            nn.Dropout(p=dropout),
            nn.BatchNorm1d(num_features=2**10,),
            nn.LeakyReLU(),
            nn.Unflatten(1, (nc,2**10)),
            nn.Conv1d(nc, nf, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf,),
            nn.LeakyReLU(),
            nn.Conv1d(nf, nf*2, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*2,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*2, nf*4, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*4,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*4, nf*8, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*8,),
            nn.LeakyReLU(),
            nn.Conv1d(nf*8, nf*16, kernel_size=8, stride=4, padding=2,),
            nn.BatchNorm1d(num_features=nf*16,),
            nn.LeakyReLU(),
            nn.Flatten(1),
            nn.Linear(nf*16, nout),
            )
    # linear > batchnorm > dropout > activation
    # or rather linear > dropout > act > batchnorm
    return net

def buildTCNetworkv1(
    # nc : int = 1,
    nin: int = 28 ** 2,
    nf: int = 32,
    nout: int = 2 ** 12,
    dropout: float = 0,
    # activation: Optional[nn.Module] = nn.LeakyReLU(),
    # batchnorm: bool = False,
):
    """
    conv1t network, each step doubles or quadrupels input size.
    """
    net = nn.Sequential(
        nn.Unflatten(1, (1, nin)),
        nn.ConvTranspose1d(
            1,
            nf * 32,
            kernel_size=8,
            stride=4,
            padding=2,
        ),
        nn.BatchNorm1d(
            num_features=nf * 32,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 32,
            nf * 16,
            kernel_size=8,
            stride=4,
            padding=2,
        ),
        nn.BatchNorm1d(
            num_features=nf * 16,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 16,
            nf * 8,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm1d(
            num_features=nf * 8,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 8,
            1,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm1d(
            num_features=1,
        ),
        nn.LeakyReLU(),
        nn.Flatten(1),
        nn.Linear(nin*4*4*2*2, nout),
    )
    return net

def buildTCNetworkv2(
    # nc : int = 1,
    nin: int = 28 ** 2,
    nf: int = 32,
    nout: int = 2 ** 12,
    dropout: float = 0,
    # activation: Optional[nn.Module] = nn.LeakyReLU(),
    # batchnorm: bool = False,
):
    """
    conv1t network, each step doubles or quadrupels input size.
    """
    net = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Unflatten(1, (1, nin)),
        nn.ConvTranspose1d(
            1,
            nf * 32,
            kernel_size=8,
            stride=4,
            padding=2,
        ),
        nn.BatchNorm1d(
            num_features=nf * 32,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 32,
            nf * 16,
            kernel_size=8,
            stride=4,
            padding=2,
        ),
        nn.BatchNorm1d(
            num_features=nf * 16,
        ),
        nn.LeakyReLU(),
        nn.ConvTranspose1d(
            nf * 16,
            1,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm1d(
            num_features=1,
        ),
        nn.LeakyReLU(),
        nn.Flatten(1),
        nn.Linear(nin*4*4*2, nout),
    )
    return net



@curry
def mixedGaussianCircular(k=10, sigma=0.025, rho=3.5, j=0):
    """
    Sample from a mixture of k 2d-gaussians. All have equal variance (sigma) and
    correlation coefficient (rho), with the means equally distributed on the
    unit circle.
    example:
    gauss = mixedGaussianCircular(rho=0.01, sigma=0.5, k=10, j=0)
    mix = distributions.Categorical(torch.ones(10,))
    comp = distributions.Independent(gauss, 0)
    gmm = distributions.MixtureSameFamily(mix, comp)
    """
    # cat = distributions.Categorical(torch.ones(k))
    # i = cat.sample().item()
    # theta = 2 * torch.pi * i / k
    theta = 2 * torch.pi / k
    v = torch.Tensor((1, 0))
    T = torch.Tensor([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    S = torch.stack([T.matrix_power(i) for i in range(k)])
    mu = S @ v
    # cov = sigma ** 2 * ( torch.eye(2) + rho * (torch.ones(2, 2) - torch.eye(2)))
    # cov = cov @ S
    cov = torch.eye(2) * sigma**2
    cov[1, 1] = sigma**2 * rho
    cov = torch.stack([
        T.matrix_power(i + j) @ cov @ T.matrix_power(-i - j) for i in range(k)
    ])
    gauss = distributions.MultivariateNormal(loc=mu, covariance_matrix=cov)
    return gauss

class LabeledDataset(torch.utils.data.Dataset):
    def __init__(self, data : Tensor, labels : Tensor, labeled_portion : float):
        super().__init__()
        self.data = data
        self.labels = labels
        self.markedlabels = distributions.Bernoulli(labeled_portion).sample(labels.shape).long()
        return
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx], self.markedlabels[idx]

class scsimDataset(torch.utils.data.Dataset):
    def __init__(self, countspath : str, idspath : str, genepath : Optional[str] = None,):
        super(scsimDataset, self).__init__()
        fcounts = np.load(countspath, allow_pickle=True)
        flabels = np.load(idspath, allow_pickle=True)
        self.counts = pd.DataFrame(**fcounts)
        self.labels = pd.DataFrame(**flabels)
        flabels.close()
        fcounts.close()
        if genepath:
            fgeneparams = np.load(genepath, allow_pickle=True)
            self.geneparams = pd.DataFrame(**fgeneparams) 
            fgeneparams.close()

        self._cpath = countspath
        self._lpath = idspath
        self._gpath = genepath

        # notmalized counts:
        self.normalized_counts = (self.counts - self.counts.mean(axis=0)) / self.counts.std(ddof=0, axis=0)
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx : int):
        ncounts = self.normalized_counts.iloc[idx].astype('float')
        counts = self.counts.iloc[idx].astype('float')
        label = self.labels.iloc[idx, 0]
        return torch.FloatTensor(ncounts), torch.FloatTensor(counts), label

    def __train_test_split__(self, n : int) -> Union[Tuple[Any,Any], None]:
        if n >= self.__len__() - 1:
            print("error, cannot split because n exceeds data length")
            return None
        trainD = scsimDataset(self._cpath, self._lpath)
        trainD.counts = self.counts[0:n]
        trainD.normalized_counts = self.normalized_counts[0:n]
        trainD.labels = self.labels[0:n]
        testD = scsimDataset(self._cpath, self._lpath)
        testD.counts = self.counts[n:]
        testD.normalized_counts = self.normalized_counts[n:]
        testD.labels = self.labels[n:]
        return trainD, testD

#cpath = "data/scrnasim/counts.npz"
#lpath = "data/scrnasim/cellparams.npz"
#
#ds = scsimDataset(cpath, lpath)
#ds.__len__()
#ds.counts
#ds.labels
#ds.__getitem__(5)
#dlen = ds.__len__()
#
#traind, testd = ds.__train_test_split__(7000)
#trainloader = torch.utils.data.DataLoader(traind, batch_size=128, shuffle=True)
#trainloader.__iter__().next()

# transform = transforms.Compose(
#    [
#        transforms.ToTensor(),
#        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#    ]
# )
#
# dataset = datasets.CIFAR10(
#        root='data/',
#        train=True,
#        download=True,
#        transform=transform,
#        )
#
# train_loader = torch.utils.data.DataLoader(
#    dataset=dataset,
#    batch_size=128,
#    shuffle=True,
# )
# test_loader = torch.utils.data.DataLoader(
#    dataset=datasets.CIFAR10(
#        root='data/',
#        train=False,
#        download=True,
#        transform=transform,
#        ),
#    batch_size=128,
#    shuffle=True,
# )

# imgs, labels = train_loader.__iter__().next()
# plot_images(imgs, nrow=16, transform=denorm)
# x = torch.linspace(0, 1, 16)
# x=x.reshape((1,1,1,4,4))
# f = transforms.Normalize(mean=0.5, std=0.5)
# g = normalize(clamp=False)
##g = partial(normalize, clamp=False)
# h = partial(denorm, clamp=False)
#
# @curry
# def mul(x,y):
#    return x*y
# ll = fclayer(10, 1, False, 0.0, nn.Sigmoid())

# gauss = mixedGaussianCircular(rho=0.01, sigma=0.6, k=10, j=0)
# mix = distributions.Categorical(torch.ones(10,))
# comp = distributions.Independent(gauss, 0)
# gmm = distributions.MixtureSameFamily(mix, comp)
# samples = gmm.sample((10000,))
# samples.shape
# samples = samples.cpu().numpy()
# x = samples[:,0]
# y = samples[:,1]
# plt.scatter(x,y)
# plt.legend(['star'])

def randomSubset(
        s : int,
        r: float,
        ):
    """
    returns a numpy boolean 1d array of size size,
    with approximately ratio of r*s True values.
    s must be positive integer
    r must be in the range [0,1]
    """
    x = np.random.rand(s)
    x = x <= r
    return x


class Blobs:
    """
    samples gaussian blobs.
    """
    def __init__(
            self,
            means: Tensor = torch.rand(5,2) * 5e0,
            scales: Tensor = torch.rand(5,2) * 2e-1,
            ) -> None:
        self.means = means
        self.scales = scales
        self.nclasses = means.shape[0]
        self.ndim = means.shape[1]
        self.comp = distributions.Normal(means, scales)
        self.mix = distributions.Categorical(torch.ones((self.nclasses,)))
        return
    def sample(self, batch_size=(100,)):
        l = self.mix.sample(batch_size)
        x = self.comp.sample(batch_size)
        s = torch.vstack(
                [x[i,l[i]] for i in range(len(x))]
                )
        return s, l, x
    def plotSample(self, batch_size=(300,)):
        s, l, x = self.sample(batch_size)
        sns.scatterplot(x=s[:,0], y=s[:,1], hue=l,)
        return


class SynteticSampler:
    """
    An object which samples from a mixture distribution.
    Should contain methods that return batched samples.
    """

    def __init__(
        self,
        means: torch.Tensor = torch.rand(5, 2),
        logscales: torch.Tensor = torch.randn(5) * 5e-1,
        noiseLevel: float = 5e-2,
    ) -> None:
        self.means = means
        self.logscales = logscales
        self.scales = logscales.exp()
        self.n_dim = means.shape[1]
        self.n_classes = means.shape[0]
        #self.m = m = distributions.Normal(
        #    loc=means,
        #    scale=logscales.exp(),
        #)
        self.noiseLevel = noiseLevel
        return

    def sample(self, batch_size=(100,)):
        m = distributions.Categorical(probs=torch.ones(self.n_classes))
        labels = m.sample(batch_size)
        locs = torch.stack(
                [self.means[labels[i]] for i in range(len(labels))], dim=0)
        scales = torch.stack(
                [self.scales[labels[i]] for i in range(len(labels))],
                dim=0).unsqueeze(1)
        noise = torch.randn_like(locs) * self.noiseLevel
        theta = torch.rand_like(labels*1e-1) * pi * 2
        data = torch.zeros_like(locs)
        data[:,0] = theta.cos()
        data[:,1] = theta.sin()
        data = data * scales + locs + noise
        return data, labels, locs, scales

    def plotData(
        self,
    ) -> None:
        # data = self.m.sample((1000,))
        data = torch.rand((300, self.n_classes)) * pi * 2
        fig = plt.figure()
        for idx in range(self.n_classes):
            color = plt.cm.Set1(idx)
            # x = data[:,idx,0]
            # y = data[:,idx,1]
            x = (
                data[:, idx].cos()
                * self.logscales[idx].exp()
                + self.means[idx, 0]
                + torch.randn(300) * self.noiseLevel
            )
            y = (
                data[:, idx].sin()
                * self.logscales[idx].exp()
                + self.means[idx, 1]
                + torch.randn(300) * self.noiseLevel
            )
            plt.scatter(
                x,
                y,
                color=color,
                s=10,
                cmap="viridis",
            )
        plt.legend([str(i) for i in range(self.n_classes)])
        plt.title("data plot")

    def plotSampleData(
        self,
    ) -> None:
        data, labels, _, _  = self.sample((1000,))
        fig = plt.figure()
        for idx in range(self.n_classes):
            color = plt.cm.Set1(idx)
            # x = data[:,idx,0]
            # y = data[:,idx,1]
            x = data[labels == idx][:, 0]
            y = data[labels == idx][:, 1]
            plt.scatter(
                x,
                y,
                color=color,
                s=10,
                cmap="viridis",
            )
        plt.legend([str(i) for i in range(self.n_classes)])
        plt.title("data plot")

class SynteticDataSet(torch.utils.data.Dataset):
    def __init__(self, data : Tensor, labels : Tensor,):
        super().__init__()
        self.data = data
        self.labels = labels
        return
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]
    def __len__(self):
        return len(self.labels)

def diffMatrix(A : np.ndarray, alpha : float = 0.25):
    """
    Returns the diffusion Kernel K for a given adjacency matrix 
    A, and restart pobability alpha.
    K = ??[I - (1 - ??)AD^-1]^-1
    """
    #D = np.diag(A.sum(0))
    T = A / A.sum(0)
    I = np.eye(A.shape[0])
    K = alpha * np.linalg.inv(I - (1 - alpha)*T)
    return K

def diffCluster(
        A : np.ndarray,
        num_clusters : int = 3,
        alpha : float=0.25, ):
    """
    A : adj matrix
    alpha: restart prob
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(A)))
    K = diffMatrix(A, alpha)
    # pageranking (increasing)
    pr = np.argsort(
            K @ np.ones(len(A)))
    n = len(K)
    nodes = list(np.arange(n))
    i = 0
    K = K.T
    #while n > num_clusters:
    while nx.number_connected_components(G) > num_clusters:
        s = pr[i]
        nodes.remove(s)
        t = K[s,nodes].argmax()
        x = nodes[t]
        G.add_edge(s, x)
        i = i+1
    clusters = np.zeros(len(A))
    i = 0
    for c in nx.connected_components(G):
        for j in c:
            clusters[j] = i
        i = i+1
    return G, clusters

def diffCluster2(
        A : np.ndarray,
        num_neighbors: int = 3,
        alpha : float=0.25, ):
    """
    A : adj matrix
    alpha: restart prob
    """
    G = nx.Graph()
    G.add_nodes_from(np.arange(len(A)))
    K = diffMatrix(A, alpha)
    # pageranking (increasing)
    n = len(K)
    nodes = list(np.arange(n))
    K = K.T
    i = 0
    for i in range(n):
       l = np.argpartition(-K[i], num_neighbors)[:num_neighbors]
       for j in l:
           G.add_edge(i,j)
    clusters = np.zeros(len(A))
    i = 0
    for c in nx.connected_components(G):
        for j in c:
            clusters[j] = i
        i = i+1
    return G, clusters
