import gdown
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
import numpy as np
import scanpy as sc
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from anndata.experimental.pytorch import AnnLoader
#https://anndata-tutorials.readthedocs.io/en/latest/annloader.html

from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO
from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
import my_torch_utils as ut

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, out_dim):
        super().__init__()

        modules = []
        for in_size, out_size in zip([input_dim]+hidden_dims, hidden_dims):
            modules.append(nn.Linear(in_size, out_size))
            modules.append(nn.LayerNorm(out_size))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=0.05))
        modules.append(nn.Linear(hidden_dims[-1], out_dim))
        self.fc = nn.Sequential(*modules)

    def forward(self, *inputs):
        input_cat = torch.cat(inputs, dim=-1)
        return self.fc(input_cat)


class CVAE(nn.Module):
    # The code is based on the scarches trVAE model
    # https://github.com/theislab/scarches/blob/v0.3.5/scarches/models/trvae/trvae.py
    # and on the pyro.ai Variational Autoencoders tutorial
    # http://pyro.ai/examples/vae.html
    def __init__(self, input_dim, n_conds, n_classes, hidden_dims, latent_dim):
        super().__init__()

        self.encoder = MLP(input_dim+n_conds, hidden_dims, 2*latent_dim) # output - mean and logvar of z

        self.decoder = MLP(latent_dim+n_conds, hidden_dims[::-1], input_dim)
        self.theta = nn.Linear(n_conds, input_dim, bias=False)

        self.classifier = nn.Linear(latent_dim, n_classes)

        self.latent_dim = latent_dim

    def model(self, x, batches, classes, size_factors):
        pyro.module("cvae", self)

        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            z_loc = x.new_zeros((batch_size, self.latent_dim))
            z_scale = x.new_ones((batch_size, self.latent_dim))
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))

            classes_probs = self.classifier(z).softmax(dim=-1)
            pyro.sample("class", dist.Categorical(probs=classes_probs), obs=classes)

            dec_mu = self.decoder(z, batches).softmax(dim=-1) * size_factors[:, None]
            dec_theta = torch.exp(self.theta(batches))

            logits = (dec_mu + 1e-6).log() - (dec_theta + 1e-6).log()

            pyro.sample("obs", dist.NegativeBinomial(total_count=dec_theta, logits=logits).to_event(1), obs=x.int())

    def guide(self, x, batches, classes, size_factors):
        batch_size = x.shape[0]

        with pyro.plate("data", batch_size):
            z_loc_scale = self.encoder(x, batches)

            z_mu = z_loc_scale[:, :self.latent_dim]
            z_var = torch.sqrt(torch.exp(z_loc_scale[:, self.latent_dim:]) + 1e-4)

            pyro.sample("latent", dist.Normal(z_mu, z_var).to_event(1))

cdata = sc.read("./data/limb_sce_alternative.h5ad",)

n_conds = len(cdata.obs['subtissue'].cat.categories)
n_classes = len(cdata.obs['cell_ontology_class'].cat.categories)

cvae = CVAE(cdata.n_vars, n_conds=n_conds, n_classes=n_classes,
        hidden_dims=[128,128], latent_dim=10)


enc = OneHotEncoder(sparse=False, dtype=np.float32)
enc.fit(cdata.obs['subtissue'].to_numpy()[:,None])

enc_ct = LabelEncoder()
enc_ct.fit(cdata.obs['cell_ontology_class'])

use_cuda = torch.cuda.is_available()


convert = {
        'obs' : {
            'subtissue' : lambda s: enc.transform(s.to_numpy()[:, None]),
            'cell_ontology_class': enc_ct.transform,
            }
        }

cdataloader = AnnLoader(cdata, batch_size=128, shuffle=True, convert=convert,
        use_cuda=use_cuda)

temp = cdataloader.dataset[:10]
temp.obs['subtissue']
temp.obs['cell_ontology_class']

optimizer = pyro.optim.Adam({"lr": 1e-3})
svi = pyro.infer.SVI(cvae.model, cvae.guide, optimizer, loss=pyro.infer.TraceMeanField_ELBO())


def train(svi, train_loader):
    epoch_loss = 0.
    for batch in train_loader:
        epoch_loss += svi.step(batch.X, batch.obs['subtissue'], batch.obs['cell_ontology_class'], batch.obs['sizeFactor'])
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    return total_epoch_loss_train

NUM_EPOCHS = 210

cvae.cuda()
for epoch in range(NUM_EPOCHS):
    total_epoch_loss_train = train(svi, cdataloader)
    if epoch % 40 == 0 or epoch == NUM_EPOCHS-1:
        print("[epoch %03d]  average training loss: %.4f" % (epoch, total_epoch_loss_train))


batch = cdataloader.dataset[:10]

print('X:', batch.X.device, batch.X.dtype)
print('subtissue:', batch.obs['subtissue'].device, batch.obs['subtissue'].dtype)
print('cell_ontology_class:', batch.obs['cell_ontology_class'].device, batch.obs['cell_ontology_class'].dtype)


full_data = cdataloader.dataset[:]
means = cvae.encoder(full_data.X, full_data.obs['subtissue'])[:, :10]

cdata.obsm['X_cvae'] = means.data.cpu().numpy()

sc.pp.neighbors(cdata, use_rep='X_cvae')
sc.tl.umap(cdata)

sc.pl.umap(cdata, color=['subtissue', 'cell_ontology_class'], wspace=0.35)

adata = sc.read("./data/limb_sce_alternative.h5ad",)

adata.raw.X = adata.X
adata.X = adata.layers['logcounts']

sc.pp.neighbors(adata)
sc.tl.umap(adata)

sc.pl.umap(adata, color=['subtissue', 'cell_ontology_class'])




kld = lambda mu, logvar: -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

class SigmaVAE(nn.Module):
    def __init__(
        self,
        nz: int = 20,
        nh: int = 2*1024,
        nx: int = 28 ** 2,
        ny: int = 2,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nin = nx + ny
        nin = nx + ny
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
        )
        self.xmu = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                )
        self.yp = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,ny),
                nn.LogSoftmax(dim=1),
                )
        self.zmu =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        self.zlogvar =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        #self.log_sigma = torch.nn.Parameter(torch.zeros(1)[0], requires_grad=True)
        # per pixel sigma:
        self.log_sigma = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)

    def reparameterize(self, mu, logsig, ):
        eps = torch.randn(mu.shape).to(mu.device)
        sigma = (logsig).exp()
        return mu + sigma * eps

    def encode(self, x,y):
        w = torch.cat((x,y), dim=1)
        h = self.encoder(w)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return mu, logvar

    def decode(self, z):
        h = self.decoder(z)
        mu = self.xmu(h)
        p = self.yp(h)
        return mu, p

    def forward(self, x, y):
        zmu, zlogvar = self.encode(x, y)
        z = self.reparameterize(zmu, 0.5*zlogvar)
        xmu, p = self.decode(z)
        return zmu, zlogvar, xmu, p

    def reconstruction_loss(self, x, xmu, log_sigma):
        # log_sigma is the parameter for 'global' variance on x
        #result = gaussian_nll(xmu, xlogsig, x).sum()
        result = -log_gaussian_prob(x, xmu, log_sigma).sum()
        return result
    
    def loss_function(self, x, xmu, log_sigma, zmu, zlogvar, y, p):
        batch_size = x.size(0)
        rec = self.reconstruction_loss(x, xmu, log_sigma) / batch_size
        kl = kld(zmu, zlogvar) / batch_size
        celoss = nn.NLLLoss(reduction='sum')(p, y) / batch_size
        return rec, kl, celoss

    def init_kmeans(self, nclusters, data):
        """
        initiate the kmeans cluster heads
        """
        self.cpu()
        lattent_data, _ = self.encode(data)
        kmeans = KMeans(nclusters, n_init=20)
        y_pred = kmeans.fit_predict(lattent_data.detach().numpy())
        #self.mu.data.copy_(torch.Tensor(kmeans.cluster_centers_))
        self.y_pred = y_pred
        #self.q = q = self.soft_assign(lattent_data)
        #self.p = p = self.target_distribution(q)
        self.kmeans = kmeans

    def fit(self, train_loader, num_epochs=10, lr=1e-3,
            optimizer = None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        if not optimizer:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            #for idx, (data, labels) in enumerate(train_loader):
            for idx, data in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                log_sigma = ut.softclip(self.log_sigma, -2, 2)
                #self.log_sigma.fill_(log_sigma)
                #x = data.flatten(1).to(device)
                x = data.X.float().to(device)
                y = data.obs['subtissue']
                ytarget = torch.argmax(y, dim=1)
                zmu, zlogvar, xmu, p = self.forward(x,y)
                rec, kl, ce = self.loss_function(x, xmu, log_sigma, zmu, zlogvar, ytarget, p)
                loss = rec + kl + ce
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print("loss = ",
                            loss.item(),
                            kl.item(),
                            rec.item(),
                            ce.item(),
                            )
        self.cpu()
        optimizer = None
        print('done training')
        return None

xdim = batch.X.shape[1]
#ydim = batch.obs['cell_ontology_class'].shape[0]
ydim = batch.obs['subtissue'].shape[1]
xdim
ydim

model = SigmaVAE(nz=20, nh=2048, nx=xdim, ny=ydim)

foo = iter(cdataloader).next()

model.cuda()



xs = foo.X
ys = foo.obs['subtissue']

zmu, zlogvar = model.encode(xs,ys)
zmu, zlogvar, xmu, p = model(xs, ys)

nn.NLLLoss()(p, ys[:,0].long())



adata = sc.read("./data/limb_sce_alternative.h5ad",)
adata.X = adata.layers['logcounts']
dataloader = AnnLoader(adata, batch_size=128, shuffle=True, convert=convert,
        use_cuda=use_cuda)

model.fit(dataloader)

model.cuda()

mu, logvar = model.encode(dataloader.dataset[:].X.float(), dataloader.dataset[:].obs['subtissue'])

mu.shape

adata.obsm['X_cvae'] = mu.data.cpu().numpy()

sc.pp.neighbors(adata, use_rep='X_cvae')
sc.tl.umap(adata)

sc.pl.umap(adata, color=['subtissue', 'cell_ontology_class'], wspace=0.35)

sc.pp.neighbors(adata, use_rep='UMAP')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['subtissue', 'cell_ontology_class'], wspace=0.35)

class SigmaVAE2(nn.Module):
    def __init__(
        self,
        nz: int = 20,
        nh: int = 2*1024,
        nx: int = 28 ** 2,
        ny: int = 2,
    ) -> None:
        super(self.__class__, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.nin = nx + ny
        nin = nx + ny
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nin, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
        )
        self.decoder = nn.Sequential(
            nn.Linear(nz+ny, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
            nn.Linear(nh, nh),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=nh),
        )
        self.xmu = nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nx),
                )
        self.zmu =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        self.zlogvar =  nn.Sequential(
                nn.Linear(nh,nh),
                nn.LeakyReLU(),
                nn.Linear(nh,nz),
                )
        #self.log_sigma = torch.nn.Parameter(torch.zeros(1)[0], requires_grad=True)
        # per pixel sigma:
        self.log_sigma = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)

    def reparameterize(self, mu, logsig, ):
        eps = torch.randn(mu.shape).to(mu.device)
        sigma = (logsig).exp()
        return mu + sigma * eps

    def encode(self, x,y):
        w = torch.cat((x,y), dim=1)
        h = self.encoder(w)
        mu = self.zmu(h)
        logvar = self.zlogvar(h)
        return mu, logvar

    def decode(self, z, y):
        w = torch.cat((z,y), dim=1)
        h = self.decoder(w)
        mu = self.xmu(h)
        return mu

    def forward(self, x, y):
        zmu, zlogvar = self.encode(x, y)
        z = self.reparameterize(zmu, 0.5*zlogvar)
        xmu = self.decode(z, y)
        return zmu, zlogvar, xmu

    def reconstruction_loss(self, x, xmu, log_sigma):
        # log_sigma is the parameter for 'global' variance on x
        #result = gaussian_nll(xmu, xlogsig, x).sum()
        result = -log_gaussian_prob(x, xmu, log_sigma).sum()
        return result
    
    def loss_function(self, x, xmu, log_sigma, zmu, zlogvar):
        batch_size = x.size(0)
        rec = self.reconstruction_loss(x, xmu, log_sigma) / batch_size
        kl = kld(zmu, zlogvar) / batch_size
        return rec, kl

    def fit(self, train_loader, num_epochs=10, lr=1e-3,
            optimizer = None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(device)
        if not optimizer:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(10):
            #for idx, (data, labels) in enumerate(train_loader):
            for idx, data in enumerate(train_loader):
                self.train()
                self.requires_grad_(True)
                optimizer.zero_grad()
                log_sigma = ut.softclip(self.log_sigma, -2, 2)
                #self.log_sigma.fill_(log_sigma)
                #x = data.flatten(1).to(device)
                x = data.X.float().to(device)
                y = data.obs['subtissue']
                ytarget = torch.argmax(y, dim=1)
                zmu, zlogvar, xmu = self.forward(x,y)
                rec, kl = self.loss_function(x, xmu, log_sigma, zmu, zlogvar)
                loss = rec + kl
                loss.backward()
                optimizer.step()
                if idx % 300 == 0:
                    print("loss = ",
                            loss.item(),
                            kl.item(),
                            rec.item(),
                            )
        self.cpu()
        optimizer = None
        print('done training')
        return None

model2 = SigmaVAE2(nz=20, nh=2048, nx=xdim, ny=ydim)
model2.fit(dataloader)

model2.cuda()

mu, logvar = model2.encode(dataloader.dataset[:].X.float(), dataloader.dataset[:].obs['subtissue'])

adata = sc.read("./data/limb_sce_alternative.h5ad",)
adata.X = adata.layers['logcounts']

adata.obsm['X_cvae2'] = mu.data.cpu().numpy()

sc.pp.neighbors(adata, use_rep='X_cvae2')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['subtissue', 'cell_ontology_class'], wspace=0.35)

sc.pp.neighbors(adata, use_rep='UMAP')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['subtissue', 'cell_ontology_class'], wspace=0.35)
