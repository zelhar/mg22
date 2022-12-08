import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import toolz
import anndata
import scanpy as sc
import plotly.express as px
from mpl_toolkits.mplot3d import Axes3D

X = torch.randn(1000,2)
X = torch.concat([X, X+1],)
cond = 1000*["control"] + 1000 * ["treatment"]
label = toolz.concat( 200 * [np.arange(10)])
label = [str(x) for x in label]
df = pd.DataFrame()
df["cond"] = cond
df[["x","y"]] = X.numpy()
df["label"] = label

sns.set_palette(sns.color_palette("pastel"),)
ax=sns.relplot(
        data=df,
        x="x",
        y="y",
        kind="scatter",
        hue="label",
        style="cond",
        legend="brief",
        )
sns.move_legend(ax,loc="upper right")
ax.savefig("foo.png",)


#%paste

def blobs(
        nx : int = 2, # dimensions
        nc : int = 2, # number of conditions
        ny : int = 5, # number of blobs per condition
        ns : int = 500, # number of samples per blop per cond
        effect : float = 15e-1
        ):
    """
    create Gaussian blobs.
    nx : dimensions of the space
    nc: number of conditions
    ny: number of components
    ns: number of samples per blob per condition
    effect: approx shift effect of treatment
    """
    mu1 = torch.rand(ny, nx)*1e1
    std1 = torch.rand(ny,nx)*5e-1
    #shift = 5e0 * torch.rand(ny,nx)
    shift = effect + torch.randn(ny,nx)*5e-1
    mu2 = mu1 + shift
    std2 = std1 + torch.randn_like(std1)*1e-2
    mu = torch.concat(
            [mu1,mu2], dim=0,).unsqueeze(0)
    std = torch.concat(
            [std1,std2], dim=0,).unsqueeze(0)
    X1 = torch.randn(ns, ny, nx)*std1 + mu1
    X2 = torch.randn(ns, ny, nx)*std2 + mu2
    X = torch.concat(
            [X1,X2], dim=0,).reshape(-1,nx).numpy()
    df = pd.DataFrame()
    adata = sc.AnnData(X=X)
    condition = ns * ny * ['ctrl'] + ns*ny*['trtmnt']
    label = [str(x) for x in toolz.concat(
        ns*nc * [np.arange(ny)]) ]
    df["label"] = label
    df["cond"] = condition
    if nx == 2:
        df[["x","y"]] = X
    elif nx == 3:
        df[["x","y", "z"]] = X
    else:
        df[["x","y", "z"]] = X[:,:3]
    adata.obs = df
    return adata


adata = blobs(ns=100, effect=1.5, nx=6)
df=adata.obs

sns.set_style("darkgrid")
sns.set_palette(sns.color_palette(),)

ax=sns.relplot(
        data=df,
        x="x",
        y="y",
        kind="scatter",
        hue="cond",
        style="label",
        #hue="label",
        #style="cond",
        legend="brief",
        #legend="full",
        )
sns.move_legend(ax,loc="upper right")

fig = px.scatter_3d(df,
                    x="x",y="y",z="z",
                    color="label",
                    symbol="cond",
                    )
# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

sns.set_style("darkgrid")
sns.set_palette(sns.color_palette(),)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d',)
ax.scatter(df.obs["x"],df.obs["y"],df.obs["z"],c=list(toolz.concat([np.arange(5)]*200)),)

sc.pl.scatter(df,x="x",y=["y","z"], projection='3d', color="label",)

