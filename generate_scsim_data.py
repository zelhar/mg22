import argparse
from importlib import reload
import matplotlib.pyplot as plt
import my_torch_utils as ut
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import torch
import torch.utils.data
import torchvision.utils as vutils
import umap
from math import pi, sin, cos, sqrt, log
from toolz import partial, curry
from torch import nn, optim, distributions
from torch.nn.functional import one_hot
from torchvision import datasets, transforms, models
from torchvision.utils import save_image, make_grid
from typing import Callable, Iterator, Union, Optional, TypeVar
from typing import List, Set, Dict, Tuple
from typing import Mapping, MutableMapping, Sequence, Iterable
from typing import Union, Any, cast, IO, TextIO
# my own sauce
from my_torch_utils import denorm, normalize, mixedGaussianCircular
from my_torch_utils import fclayer, init_weights
from my_torch_utils import plot_images, save_reconstructs, save_random_reconstructs
from my_torch_utils import fnorm, replicate, logNorm
from my_torch_utils import scsimDataset
import scsim.scsim as scsim
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceGraph_ELBO
from pyro.optim import Adam
from toolz import take, drop
import opt_einsum
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans


import scanpy as sc
import anndata as ad

import pickle

print(torch.cuda.is_available())

ngenes = 25 * 10**3
ncells = 30 * 10**3
K=10
deprob = .025
progdeloc = deloc = deval = 1
descale = 1.0
progcellfrac = .35
deprob = .025
doubletfrac = .0
ndoublets=int(doubletfrac*ncells)
nproggenes = 400
nproggroups = int(K/3)
proggroups = list(range(1, nproggroups+1))
randseed = 42


simulator = scsim.scsim(
    ngenes=ngenes,
    ncells=ncells,
    ngroups=K,
    libloc=7.64,
    libscale=0.78,
    mean_rate=7.68,
    mean_shape=0.34,
    expoutprob=0.00286,
    expoutloc=6.15,
    expoutscale=0.49,
    diffexpprob=deprob,
    diffexpdownprob=0.0,
    diffexploc=deloc,
    diffexpscale=descale,
    bcv_dispersion=0.448,
    bcv_dof=22.087,
    ndoublets=ndoublets,
    nproggenes=nproggenes,
    progdownprob=0.0,
    progdeloc=progdeloc,
    progdescale=descale,
    progcellfrac=progcellfrac,
    proggoups=proggroups,
    minprogusage=0.1,
    maxprogusage=0.7,
    seed=randseed,
)

simulator.simulate()

path_prefix_name = "data/scrnasim/fooo"

#simulator.cellparams.to_csv(
#        path_prefix_name + "_cell_params.tsv.gz",
#        sep='\t',
#        compression="gzip",
#        )
#simulator.counts.to_csv(
#        path_prefix_name + "_counts.tsv.gz",
#        sep='\t',
#        compression="gzip"
#        )
#simulator.geneparams.to_csv(
#        path_prefix_name + "_gene_params.tsv.gz", sep='\t',
#        compression="gzip"
#        )
#
#df = pd.read_csv(
#        path_prefix_name + "_cell_params.tsv.gz",
#        sep='\t',
#        index_col=0,
#        compression='gzip', 
#        )
#
#with open(path_prefix_name + "pickle", 'wb') as f:
#    pickle.dump(simulator, f)
#
#with open(path_prefix_name + "pickle", 'rb') as f:
#    my_sim = pickle.load(f)

if __name__ == "__main__":
    path_prefix_name = "data/scrnasim/my_scsim_data"

    print("simulation results will be saved in: ",
            path_prefix_name,
            )

    simulator.simulate()
    print("done simulating. saving...")

    simulator.cellparams.to_csv(
            path_prefix_name + "_cell_params.tsv.gz",
            sep='\t',
            compression="gzip",
            )
    simulator.counts.to_csv(
            path_prefix_name + "_counts.tsv.gz",
            sep='\t',
            compression="gzip"
            )
    simulator.geneparams.to_csv(
            path_prefix_name + "_gene_params.tsv.gz", sep='\t',
            compression="gzip"
            )

    with open(path_prefix_name + ".pickle", 'wb') as f:
        pickle.dump(simulator, f)

        print("done.")




