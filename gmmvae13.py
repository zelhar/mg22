# best of models
# sources:
# https://github.com/psanch21/VAE-GMVAE
# https://arxiv.org/abs/1611.02648
# http://ruishu.io/2016/12/25/gmvae/
# https://github.com/RuiShu/vae-clustering
# https://github.com/hbahadirsahin/gmvae
import gdown
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pyro
import pyro.distributions as pyrodist
import scanpy as sc
import seaborn as sns
import time
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision.utils as vutils
import umap
from abc import abstractmethod
from anndata.experimental.pytorch import AnnLoader
from importlib import reload
from math import pi, sin, cos, sqrt, log
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO

# from pyro.optim import Adam
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from toolz import partial, curry
from torch import nn, optim, distributions, Tensor
from torch.nn.functional import one_hot, binary_cross_entropy_with_logits
from torch.nn.functional import cross_entropy
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO

# my own sauce
# from my_torch_utils import denorm, normalize, mixedGaussianCircular
# from my_torch_utils import fclayer, init_weights
# from my_torch_utils import fnorm, replicate, logNorm, log_gaussian_prob
# from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
# from my_torch_utils import scsimDataset
import my_torch_utils as ut

print(torch.cuda.is_available())

def preTrain(
    model,
    train_loader: torch.utils.data.DataLoader,
    #test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of pre train (unsupervised)
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                y=None,
            )
            # output = model.forward(x,y, training=True)
            losses = output["losses"]
            q_y = output["q_y"]
            p_y = torch.ones_like(q_y) / model.nclasses
            loss_q_y = (p_y * q_y.log()).sum(-1).mean() * model.yscale
            loss_q_y2 = (p_y - q_y).abs().sum(-1).mean() * model.yscale * 1e0
            losses["loss_q_y"] = loss_q_y
            losses["loss_q_y2"] = loss_q_y2
            loss = (
                    losses["rec"]
                    #+ losses["loss_z"]
                    + losses["loss_pretrain_z"]
                    + losses["loss_w"]
                    + loss_q_y
                    + loss_q_y2
                    )
            losses["pretrain_loss"] = loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                #model.printDict(output["losses"])
                model.printDict(losses)
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
    model.cpu()
    model.eval()
    optimizer = None
    model.load_state_dict(best_result)
    return

def preTrainLoop(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of basicTrainLoopCond
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        preTrain(
            model,
            train_loader,
            num_epochs,
            lr,
            device,
            wt,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )

    print("done training")
    return

def advancedTrain(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
    advanced_semi : bool = True,
    cc_extra_sclae : float = 1e1,
) -> None:
    """
    non-conditional version of advancedTrain train (unsupervised)
    does supervised training using generated samples ans unsupervised
    training using the data_loader.
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                y=None,
            )
            # output = model.forward(x,y, training=True)
            loss = output["losses"][loss_type]
            if advanced_semi:
                batch_size = x.shape[0]
                #ww = torch.randn(batch_size,model.nw).to(device)
                #w = output["w"].detach()
                mu_w = output["mu_w"].detach().to(device)
                logvar_w = output["logvar_w"].detach().to(device)
                noise = torch.randn_like(mu_w).to(device)
                std_w = (0.5 * logvar_w).exp()
                ww = mu_w + noise * std_w
                zz = model.Pz(ww)[:,:,:model.nz]
                rr = model.Px(zz.reshape(batch_size * model.nclasses, model.nz))
                yy = model.justPredict(rr).to(device)
                cc = torch.eye(model.nclasses, device=device)
                cc = cc.repeat(batch_size,1)
                loss_cc = model.cc_scale * (yy - cc).abs().sum(-1).mean()
                loss_cc = loss_cc - model.cc_scale * (cc * yy.log()).sum(-1).mean()
                loss_cc = loss_cc * cc_extra_sclae
                loss = loss + loss_cc
                output["losses"]["loss_cc"] = loss_cc
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
            #if advanced_semi:
            #    model.train()
            #    model.to(device)
            #    model.requires_grad_(True)
            #    batch_size = x.shape[0]
            #    ww = torch.randn(batch_size,model.nw).to(device)
            #    zz = model.Pz(ww)[:,:,:model.nz]
            #    rr = model.Px(zz.reshape(batch_size * model.nclasses, model.nz))
            #    yy = model.justPredict(rr).to(device)
            #    cc = torch.eye(model.nclasses, device=device)
            #    cc = cc.repeat(batch_size,1)
            #    loss_cc = model.yscale * (yy - cc).abs().sum(-1).mean()
            #    loss_cc = loss_cc - model.yscale * (cc * yy.log()).sum(-1).mean()
            #    #model.eval()
            #    #model.to(device)
            #    #model.requires_grad_(False)
            #    #batch_size = x.shape[0]
            #    #ww = torch.randn(batch_size,model.nw).to(device)
            #    #zz = model.Pz(ww)[:,:,:model.nz]
            #    #rr = model.Px(zz.reshape(batch_size * model.nclasses, model.nz))
            #    #yy = model.justPredict(rr).to(device)
            #    #cc = torch.eye(model.nclasses, device=device)
            #    #cc = cc.repeat(batch_size,1)
            #    #model.train()
            #    #model.requires_grad_(True)
            #    #output = model.forward(rr, y=cc)
            #    #loss = output["losses"]["total_loss"]
            #    optimizer.zero_grad()
            #    loss_cc.backward()
            #    optimizer.step()
    model.cpu()
    model.eval()
    optimizer = None
    model.load_state_dict(best_result)
    return

def advancedTrainLoop(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    advanced_semi : bool = True,
    cc_extra_sclae : float = 1e1,
) -> None:
    """
    non-conditional version of advancedTrainLoop
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        advancedTrain(
            model,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
            advanced_semi = advanced_semi,
            cc_extra_sclae = cc_extra_sclae,
        )

    print("done training")
    return

def basicTrain(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of basic train (unsupervised)
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                y=None,
            )
            # output = model.forward(x,y, training=True)
            loss = output["losses"][loss_type]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
    model.cpu()
    model.eval()
    optimizer = None
    model.load_state_dict(best_result)
    return

def basicTrainCond(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    best_loss: float = 1e6,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    conditional version of basic train (unsupervised)
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        # print("training phase")
        # for idx, (data, labels) in enumerate(train_loader):
        for idx, data in enumerate(train_loader):
            x = data[0].flatten(1).to(device)
            # y = labels.to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            # x = data.to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                cond1=cond1,
                y=None,
            )
            # output = model.forward(x,y, training=True)
            loss = output["losses"][loss_type]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("training phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    model.cpu()
                    model.eval()
                    #w = model.w_prior.sample((16,))
                    #cond = model.y_prior.sample((16,)) * 0
                    w = model.w_prior.sample((16,))
                    w = w.repeat(model.nc1, 1)
                    cond = torch.eye(model.nc1)
                    cond = cond.repeat(16,1)
                    z = model.Pz(torch.cat([w, cond], dim=-1))
                    #mu = z[:, :, : model.nz].reshape(16 * model.nclasses, model.nz)
                    mu = z[:, :, : model.nz].reshape(16 * model.nc1 * model.nclasses, model.nz)
                    cond = cond.repeat(model.nclasses,1)
                    rec = model.Px(torch.cat([mu, cond],dim=-1)).reshape(-1, 1, 28, 28)
                    if model.reclosstype == "Bernoulli":
                        rec = rec.sigmoid()
                    ut.plot_images(rec, model.nclasses * model.nc1)
                    plt.pause(0.05)
                    plt.savefig("tmp.png")
                    model.train()
                    model.to(device)
                if test_accuracy:
                    model.eval()
                    r, p, s = ut.estimateClusterImpurityLoop(
                        model,
                        x,
                        y,
                        device,
                    )
                    print(p, "\n", r.mean(), "\n", r)
                    print(
                        (r * s).sum() / s.sum(),
                        "\n",
                    )
                    model.train()
                    model.to(device)
    model.cpu()
    optimizer = None
    model.load_state_dict(best_result)
    return

def basicTrainLoop(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of basicTrainLoopCond
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        basicTrain(
            model,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )

    print("done training")
    return

def basicTrainLoopCond(
    model,
    train_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    num_epochs: int = 10,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt: float = 1e-4,
    loss_type: str = "total_loss",
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    conditional version of basicTrainLoopCond
    """
    for lr in lrs:
        print(
            "epoch's lr = ",
            lr,
        )
        basicTrainCond(
            model,
            train_loader,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            loss_type,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )

    print("done training")
    return

def trainSemiSuper(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lr=1e-3,
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_eval: bool = True,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    best_loss: float = 1e6,
) -> None:
    """
    non-conditional version of trainSemiSuper
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader_labeled):
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(x, y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not do_unlabeled:
                if loss < best_loss:
                    best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("labeled phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    ut.do_plot_helper(model, device)
                    model.train()
                    model.to(device)
                if test_accuracy:
                    ut.test_accuracy_helper(model, x, y, device)
                model.train()
                model.to(device)
        # for idx, (data, labels) in enumerate(train_loader_unlabeled):
        for idx, data in enumerate(train_loader_unlabeled):
            if do_unlabeled == False:
                break
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                y=None,
            )
            # output = model.forward(x,y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("unlabeled phase")
                model.printDict(output["losses"])
                print()
            # possibly need to reconsider the following:
            if idx >= len(train_loader_labeled):
                break
        for idx, data in enumerate(test_loader):
            if do_eval == False:
                break
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            model.eval()
            output = model.forward(x, y=None, )
            loss = output["losses"]["total_loss"]
            q_y = output["q_y"]
            ce_loss = (y * q_y.log()).sum(-1).mean()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("eval phase")
                model.printDict(output["losses"])
                print("ce loss:", ce_loss.item())
                print()
    model.cpu()
    model.load_state_dict(best_result)
    # optimizer = None
    del optimizer
    print("done training")
    return None

def trainSemiSuperCond(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lr=1e-3,
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_eval: bool = True,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
    best_loss: float = 1e6,
) -> None:
    """
    conditional version of trainSemiSuper
    """
    model.train()
    model.to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wt,
    )
    best_result = model.state_dict()
    for epoch in range(num_epochs):
        for idx, data in enumerate(train_loader_labeled):
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            model.train()
            model.requires_grad_(True)
            output = model.forward(x, cond1=cond1, y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not do_unlabeled:
                if loss < best_loss:
                    best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("epoch " + str(epoch))
                print("labeled phase")
                model.printDict(output["losses"])
                print()
                if do_plot:
                    model.cpu()
                    model.eval()
                    w = model.w_prior.sample((16,))
                    z = model.Pz(w)
                    mu = z[:, :, : model.nz].reshape(16 * model.nclasses, model.nz)
                    rec = model.Px(mu).reshape(-1, 1, 28, 28)
                    if model.reclosstype == "Bernoulli":
                        rec = rec.sigmoid()
                    ut.plot_images(rec, model.nclasses)
                    plt.pause(0.05)
                    plt.savefig("tmp.png")
                    model.train()
                    model.to(device)
                if test_accuracy:
                    model.eval()
                    r, p, s = ut.estimateClusterImpurityLoop(
                        model,
                        x,
                        y,
                        device,
                    )
                    print(p, "\n", r.mean(), "\n", r)
                    print(
                        (r * s).sum() / s.sum(),
                        "\n",
                    )
                    model.train()
                    model.to(device)
        # for idx, (data, labels) in enumerate(train_loader_unlabeled):
        for idx, data in enumerate(train_loader_unlabeled):
            if do_unlabeled == False:
                break
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            model.train()
            model.requires_grad_(True)
            output = model.forward(
                x,
                cond1=cond1,
                y=None,
            )
            # output = model.forward(x,y=y)
            loss = output["losses"]["total_loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if loss < best_loss:
                best_result = model.state_dict()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("unlabeled phase")
                model.printDict(output["losses"])
                print()
            # possibly need to reconsider the following:
            if idx >= len(train_loader_labeled):
                break
        for idx, data in enumerate(test_loader):
            if do_eval == False:
                break
            x = data[0].flatten(1).to(device)
            y = data[1].to(device)
            if len(data) > 2:
                cond1 = data[2].to(device)
            else:
                cond1 = None
            model.eval()
            # output = model.forward(x,)
            # output = model.forward(x,y=y,cond1=cond1)
            output = model.forward(x, y=None, cond1=cond1)
            loss = output["losses"]["total_loss"]
            q_y = output["q_y"]
            ce_loss = (y * q_y.log()).sum(-1).mean()
            if epoch % report_interval == 0 and idx % 1500 == 0:
                print("eval phase")
                model.printDict(output["losses"])
                print("ce loss:", ce_loss.item())
                print()
    model.cpu()
    model.load_state_dict(best_result)
    # optimizer = None
    del optimizer
    print("done training")
    return None

def trainSemiSuperLoop(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_validation: bool = True,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    non-conditional version of trainSemiSuperLoop
    """
    for lr in lrs:
        trainSemiSuper(
            model,
            train_loader_labeled,
            train_loader_unlabeled,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            do_unlabeled,
            do_validation,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )


def trainSemiSuperLoopCond(
    model,
    train_loader_labeled: torch.utils.data.DataLoader,
    train_loader_unlabeled: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    # test_loader: Optional[torch.utils.data.DataLoader],
    num_epochs=15,
    lrs: Iterable[float] = [
        1e-3,
    ],
    device: str = "cuda:0",
    wt=1e-4,
    do_unlabeled: bool = True,
    do_validation: bool = True,
    report_interval: int = 3,
    do_plot: bool = False,
    test_accuracy: bool = False,
) -> None:
    """
    conditional version of trainSemiSuperLoop
    """
    for lr in lrs:
        trainSemiSuperCond(
            model,
            train_loader_labeled,
            train_loader_unlabeled,
            test_loader,
            num_epochs,
            lr,
            device,
            wt,
            do_unlabeled,
            do_validation,
            report_interval,
            do_plot=do_plot,
            test_accuracy=test_accuracy,
        )


class GenericNet(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input):
        raise NotImplementedError()

class GenericClusterAE(GenericNet):
    def __init__(
        self,
    ) -> None:
        """
        Contains all the common features of most the models 
        I am working with:
        w,z,y latent variables.
        """
        super().__init__()
        return

    def forward(self, input):
        raise NotImplementedError()

    def justPredict(self, input):
        raise NotImplementedError()



class VAE_Dirichlet_Type1300(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), Q(d|w,z)
    relax option
    resnet option
    version 5 nw
    MI?
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nw+nz, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    [nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        #q_d = nn.Softmax(dim=-1)(d_logits)
        #q_d = (eps/self.nclasses +  (1 - eps) * q_d)
        #output["q_d"] = q_d
        #loss_mi = -ut.mutualInfo(q_y, q_d) * self.mi_scale
        #losses["loss_mi"] = loss_mi
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z, ], dim=1))
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=p_y,
                    )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                + loss_y_alt
                + loss_y_alt2
                #+ loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_Dirichlet_Type1300D(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), Q(d|w,z)
    relax option
    resnet option
    version 5 nw
    MI?
    deterministic
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nw+nz, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    [nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        #q_d = nn.Softmax(dim=-1)(d_logits)
        #q_d = (eps/self.nclasses +  (1 - eps) * q_d)
        #output["q_d"] = q_d
        #loss_mi = -ut.mutualInfo(q_y, q_d) * self.mi_scale
        #losses["loss_mi"] = loss_mi
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z, ], dim=1))
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=p_y,
                    )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                + loss_y_alt
                + loss_y_alt2
                #+ loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        ww = torch.zeros_like(w)
        zz = self.Pz(ww)[0,:,:self.nz]
        rr = self.Px(zz)
        yy = self.justPredict(rr).to(x.device)
        cc = torch.eye(self.nclasses, device=x.device)
        loss_cc = losses["loss_cc"] = self.cc_scale * (yy - cc).abs().sum()
        total_loss = total_loss + loss_cc
        output["losses"] = losses
        return output

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y


class CVAE_Dirichlet_Type1301(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), Q(d|w,z)
    relax option
    resnet option
    version 5 nw
    MI?
    hyrarchical?
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nc1: int = 10, # level 1 categories
        nc2: int = 10, # level 2 categories
        nclasses : int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.nc1 = nc1
        self.nc2 = nc2
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz + nc1] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw + nc1] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx + nc1, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx + nc1] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw + nc1, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz + nc1] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nw+nz + nc1, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    [nw + nz + nc1] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, cond1=None, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,cond1], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,cond1], dim=1))
        output["d_logits"] = d_logits
        D_y = distributions.Dirichlet(d_logits.exp())
        #q_d = nn.Softmax(dim=-1)(d_logits)
        #q_d = (eps/self.nclasses +  (1 - eps) * q_d)
        #output["q_d"] = q_d
        #loss_mi = -ut.mutualInfo(q_y, q_d) * self.mi_scale
        #losses["loss_mi"] = loss_mi
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z, cond1], dim=1))
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,cond1], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = D_y.rsample()
        p_y = (eps/self.nclasses +  (1 - eps) * p_y)
        Py = distributions.OneHotCategorical(probs=p_y)
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=p_y,
                    )
        else:
            Py = distributions.OneHotCategorical(probs=p_y)
        output["Py"] = Py
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            #loss_z = (p_y*loss_z).sum(-1).mean()
            loss_y_alt = self.yscale * (q_y * (
                    q_y.log() - p_y.log())).sum(-1).mean()
            loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        Pd = distributions.Dirichlet(torch.ones_like(q_y) * self.concentration)
        loss_d = self.dscale * distributions.kl_divergence(D_y, Pd).mean()
        losses["loss_d"] = loss_d = loss_d
        losses["loss_y_alt"] = loss_y_alt
        losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_d
                + loss_y_alt
                + loss_y_alt2
                #+ loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output
    def justPredict(self, input, cond1=None, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,cond1], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_GMM_Type1302(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
    relax option
    resnet option
    version 5 nw
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z,], dim=1))
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_GMM_Type1302D(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
    relax option
    resnet option
    version 5 nw
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z,], dim=1))
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                )
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        ww = torch.randn(10,self.nw).to(x.device)
        zz = self.Pz(ww)[:,:,:self.nz]
        rr = self.Px(zz.reshape(10 * self.nclasses, self.nz))
        yy = self.justPredict(rr).to(x.device)
        cc = torch.eye(self.nclasses, device=x.device)
        cc = cc.repeat(10,1)
        loss_cc = losses["loss_cc"] = (yy - cc).abs().sum() * self.cc_scale
        total_loss = total_loss + loss_cc
        losses["total_loss"] = total_loss
        output["losses"] = losses
        return output

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y


class CVAE_GMM_Type1303(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), 
    relax option
    resnet option
    version 5 nw
    MI?
    hyrarchical?
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nc1: int = 10, # level 1 categories
        nc2: int = 10, # level 2 categories
        nclasses : int = 10,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.nc1 = nc1
        self.nc2 = nc2
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz + nc1] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw + nc1] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx + nc1, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx + nc1] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw + nc1, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz + nc1] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, cond1=None, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,cond1], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z, cond1], dim=1))
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,cond1], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        logvar_z_w = z_w[:,:,self.nz:]
        std_z_w = (0.5*logvar_z_w).exp()
        Pz = distributions.Normal(
                loc=mu_z_w,
                scale=std_z_w,
                )
        output["Pz"] = Pz
        loss_z = self.zscale * ut.kld2normal(
                mu=mu_z.unsqueeze(1),
                logvar=logvar_z.unsqueeze(1),
                mu2=mu_z_w,
                logvar2=logvar_z_w,
                ).sum(-1)
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            loss_y = -Qy.log_prob(y).mean() * self.yscale
        losses["loss_y"] = loss_y
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(self, input, cond1=None, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        if cond1 == None:
            cond1 = torch.zeros((batch_size, self.nc1), device=x.device)
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,cond1], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,cond1], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y

class VAE_MI_Type1304(nn.Module):
    """
    made some changes to the forward functions,
    loss_y_alt, not tanh on logvars
    dscale: scale of loss_d
    concentraion: scale of the symmetric dirichlet prior
    trying to tweek 807 model
    back to batchnorm
    reclosstype: 'gauss' is default. other options: 'Bernoulli', and 'mse'
    Q(y|w,z,x), Q(d|w,z)
    relax option
    resnet option
    version 5 nw
    MI?
    """

    def __init__(
        self,
        nx: int = 28 ** 2,
        nh: int = 1024,
        nhq: int = 1024,
        nhp: int = 1024,
        nz: int = 64,
        nw: int = 32,
        nclasses : int = 10,
        dscale : float = 1e0,
        wscale : float = 1e0,
        yscale : float = 1e0,
        zscale : float = 1e0,
        mi_scale : float = 1e0,
        cc_scale : float = 1e1,
        concentration : float = 5e-1,
        numhidden : int = 2,
        numhiddenq : int = 2,
        numhiddenp : int = 2,
        dropout : float = 0.3,
        bn : bool = True,
        reclosstype : str = "Gauss",
        temperature : float = 0.1,
        relax : bool = False,
        use_resnet : bool = False,
        softargmax: bool = False,
        eps : float = 1e-9,
        restrict_w : bool = False,
    ) -> None:
        super().__init__()
        self.nx = nx
        self.nh = nh
        self.nhq = nhq
        self.nhp = nhp
        self.nz = nz
        self.nw = nw
        self.eps = eps
        self.nclasses = nclasses
        self.numhidden = numhidden
        self.numhiddenq = numhiddenq
        self.numhiddenp = numhiddenp
        # Dirichlet constant prior:
        self.dscale = dscale
        self.wscale = wscale
        self.yscale = yscale
        self.zscale = zscale
        self.cc_scale = cc_scale
        self.mi_scale = mi_scale
        self.concentration = concentration
        self.temperature = torch.tensor([temperature])
        self.relax = relax
        self.restrict_w = restrict_w
        self.softargmax = softargmax
        self.use_resnet = use_resnet
        self.dir_prior = distributions.Dirichlet(dscale*torch.ones(nclasses))
        self.logsigma_x = torch.nn.Parameter(torch.zeros(nx), requires_grad=True)
        self.reclosstype = reclosstype
        self.kld_unreduced = lambda mu, logvar: -0.5 * (
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        #self.y_prior = distributions.OneHotCategorical(probs=torch.ones(nclasses))
        self.y_prior = distributions.RelaxedOneHotCategorical(
                probs=torch.ones(nclasses), temperature=self.temperature,)
        self.w_prior = distributions.Normal(
            loc=torch.zeros(nw),
            scale=torch.ones(nw),
        )
        ## P network
        self.Px = ut.buildNetworkv5(
                [nz] + numhiddenp * [nhp] + [nx],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz = ut.buildNetworkv5(
                [nw] + numhiddenp * [nhp] + [2*nclasses*nz],
                dropout=dropout, 
                activation=nn.LeakyReLU(),
                batchnorm=bn,
                )
        self.Pz.add_module(
                "unflatten", 
                nn.Unflatten(1, (nclasses, 2*nz)))
        ## Q network
        if use_resnet:
            resnet_wz = models.resnet18()
            #resnet_wz = models.resnet34()
            resnet_wz.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qwz = nn.Sequential(
                    nn.Flatten(1),
                    nn.Linear(nx, 64**2),
                    #nn.Unflatten(1, (1,28,28)),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_wz,
                    nn.Linear(1000, 2*nw + 2*nz),
                    )
        else:
            self.Qwz = ut.buildNetworkv5(
                    [nx] + numhiddenq*[nhq] + [2*nw + 2*nz],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        self.Qwz.add_module(
                "unflatten", 
                nn.Unflatten(1, (2, nz + nw)))
        if use_resnet:
            resnet_y = models.resnet18()
            #resnet_y = models.resnet34()
            resnet_y.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qy = nn.Sequential(
                    nn.Linear(nx+nz+nw, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_y,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qy = ut.buildNetworkv5(
                    [nx + nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        #self.Qy.add_module( "softmax", nn.Softmax(dim=-1))
        if use_resnet:
            resnet_d = models.resnet18()
            #resnet_d = models.resnet34()
            resnet_d.conv1 = nn.Conv2d(1, 64, (7,7), (2,2), (3,3), bias=False)
            self.Qd = nn.Sequential(
                    nn.Linear(nw+nz, 64**2),
                    #nn.Flatten(1),
                    nn.Unflatten(1, (1,64,64)),
                    resnet_d,
                    nn.Linear(1000, nclasses),
                    )
        else:
            self.Qd = ut.buildNetworkv5(
                    [nw + nz] + numhiddenq*[nhq] + [nclasses],
                    dropout=dropout, 
                    activation=nn.LeakyReLU(),
                    batchnorm=bn,
                    )
        return

    def printDict(self, d: dict):
        for k, v in d.items():
            print(k + ":", v.item())
        return

    def forward(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        #w = mu_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        #logvar_z = wz[:,1,self.nw:]
        #std_z = (0.5 * logvar_z).exp()
        #noise = torch.randn_like(mu_z).to(x.device)
        #z = mu_z + noise * std_z
        z = mu_z
        #Qz = distributions.Normal(loc=mu_z, scale=std_z)
        #output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        #output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        d_logits = self.Qd(torch.cat([w,z,], dim=1))
        output["d_logits"] = d_logits
        q_d = nn.Softmax(dim=-1)(d_logits)
        q_d = (eps/self.nclasses +  (1 - eps) * q_d)
        output["q_d"] = q_d
        loss_mi = -ut.mutualInfo(q_y, q_d) * self.mi_scale
        losses["loss_mi"] = loss_mi
        if self.relax:
            Qy = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_y,
                    )
        else:
            Qy = distributions.OneHotCategorical(probs=q_y)
        output["q_y"] = q_y
        output["w"]=w
        output["z"]=z
        rec = self.Px(torch.cat([z, ], dim=1))
        if self.reclosstype == "Bernoulli":
            logits = rec
            rec = logits.sigmoid()
            bce = nn.BCEWithLogitsLoss(reduction="none")
            loss_rec = bce(logits, x).sum(-1).mean()
        elif self.reclosstype == "mse":
            mse = nn.MSELoss(reduction="none")
            loss_rec = mse(rec, x).sum(-1).mean()
        else:
            logsigma_x = ut.softclip(self.logsigma_x, -8, 8)
            sigma_x = logsigma_x.exp()
            Qx = distributions.Normal(loc=rec, scale=sigma_x)
            loss_rec = -Qx.log_prob(x).sum(-1).mean()
        output["rec"]= rec
        losses["rec"] = loss_rec
        z_w = self.Pz(torch.cat([w,], dim=-1))
        mu_z_w = z_w[:,:,:self.nz]
        #logvar_z_w = z_w[:,:,self.nz:]
        #std_z_w = (0.5*logvar_z_w).exp()
        #Pz = distributions.Normal(
        #        loc=mu_z_w,
        #        scale=std_z_w,
        #        )
        #output["Pz"] = Pz
        #loss_z = self.zscale * ut.kld2normal(
        #        mu=mu_z.unsqueeze(1),
        #        logvar=logvar_z.unsqueeze(1),
        #        mu2=mu_z_w,
        #        logvar2=logvar_z_w,
        #        ).sum(-1)
        mse = nn.MSELoss(reduction="none")
        loss_z = self.zscale * mse(mu_z.unsqueeze(1), mu_z_w).sum(-1)
        losses["loss_pretrain_z"] = loss_z.mean(-1).mean()
        if self.relax:
            Py = distributions.RelaxedOneHotCategorical(
                    temperature=self.temperature.to(x.device),
                    probs=q_d,
                    )
        else:
            Py = distributions.OneHotCategorical(probs=q_d)
        output["Py"] = Py
        p_y = torch.ones_like(q_y) / self.nclasses
        if y == None:
            loss_z = (q_y*loss_z).sum(-1).mean()
            loss_y = self.yscale * (
                    q_y.mean(0) * (
                        q_y.mean(0).log() - p_y.mean(0).log()
                        )).sum()
            #loss_y_alt = self.yscale * (q_y * (
            #        q_y.log() - q_d.log())).sum(-1).mean()
            #loss_y_alt2 = torch.tensor(0)
        else:
            loss_z = (y*loss_z).sum(-1).mean()
            if self.relax:
                y = (eps/self.nclasses +  (1 - eps) * y)
            #loss_y_alt = self.yscale * -Py.log_prob(y).mean()
            #loss_y_alt2 = self.yscale * -Qy.log_prob(y).mean()
            loss_y = self.yscale * -Qy.log_prob(y).mean()
        losses["loss_z"] = loss_z
        loss_w = self.wscale * self.kld_unreduced(
                mu=mu_w,
                logvar=logvar_w).sum(-1).mean()
        losses["loss_w"]=loss_w
        loss_cluster = -1e0 * q_y.max(-1)[0].mean()
        losses["loss_cluster"] = loss_cluster
        losses["loss_y"] = loss_y
        #losses["loss_y_alt"] = loss_y_alt
        #losses["loss_y_alt2"] = loss_y_alt2
        total_loss = (
                loss_rec
                + loss_z 
                + loss_w
                + loss_y
                #+ loss_y_alt
                #+ loss_y_alt2
                + loss_mi
                )
        losses["total_loss"] = total_loss
        losses["num_clusters"] = torch.sum(torch.threshold(q_y, 0.5, 0).sum(0) > 0)
        output["losses"] = losses
        return output

    def justPredict(self, input, y=None,):
        x = nn.Flatten()(input)
        batch_size = x.shape[0]
        losses = {}
        output = {}
        #eps=1e-6
        eps = self.eps
        wz = self.Qwz(torch.cat([x,], dim=-1))
        mu_w = wz[:,0,:self.nw]
        logvar_w = wz[:,1,:self.nw]
        if self.restrict_w:
            mu_w = mu_w.tanh()
            logvar_w = logvar_w.tanh()
        std_w = (0.5 * logvar_w).exp()
        noise = torch.randn_like(mu_w).to(x.device)
        w = mu_w + noise * std_w
        Qw = distributions.Normal(loc=mu_w, scale=std_w)
        output["Qw"] = Qw
        mu_z = wz[:,0,self.nw:]
        logvar_z = wz[:,1,self.nw:]
        std_z = (0.5 * logvar_z).exp()
        noise = torch.randn_like(mu_z).to(x.device)
        z = mu_z + noise * std_z
        Qz = distributions.Normal(loc=mu_z, scale=std_z)
        output["Qz"] = Qz
        output["wz"] = wz
        output["mu_z"] = mu_z
        output["mu_w"] = mu_w
        output["logvar_z"] = logvar_z
        output["logvar_w"] = logvar_w
        q_y_logits = self.Qy(torch.cat([w,z,x,], dim=1))
        q_y = nn.Softmax(dim=-1)(q_y_logits)
        if self.softargmax:
            q_y = ut.softArgMaxOneHot(q_y,)
        q_y = (eps/self.nclasses +  (1 - eps) * q_y)
        return q_y


