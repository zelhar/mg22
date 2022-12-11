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
import plotly.io as pio
from dash import Dash, html, dcc

sns.set_palette(sns.color_palette("pastel"),)
sns.set(rc={"figure.dpi":200, 'savefig.dpi':100})

pio.renderers.default = "png"

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

adata = blobs(ns=350, effect=1.7, nx=16)
df=adata.obs
sns.set_palette(sns.color_palette("pastel"),)
#sns.set_palette(sns.color_palette("viridis"),)
sns.set_style("darkgrid")

fig = px.scatter_3d(df,
                    x="x",y="y",z="z",
                    color="label",
                    symbol="cond",
                    size_max=8,
                    size=np.ones(len(df))*1,
                    opacity=0.4
                    )
# tight layout
fig.update_layout(
    margin=dict(l=0, r=0, b=0, t=0),
    height=900,
    #width=1500,
    #title="\nMy 3d graph\n",
)
fig.update_coloraxes(
        colorscale="viridis",
        )


#app = Dash("foodash")
app = Dash(__name__)

app.layout = html.Div(children = [
    html.H1(
        children="hello world",
        ),
    html.Div(children='''
    this is dash
    '''),
    dcc.Graph(
        id="plotly graph",
        figure=fig
        )
    ])

fig.write_image("fooo.png")


if __name__ == '__main__':
    app.run_server(port=8888,debug=True,use_reloader=True,)
