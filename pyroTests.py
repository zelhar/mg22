import logging
import os
import matplotlib
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import logging
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

import pyro
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

# http://pyro.ai/examples/intro_long.html
# in jupyter:
#%matplotlib inline
#%matplotlib notebook

#from pyro.distributions import Normal


DATA_URL = "https://d2hg8soec8ck9v.cloudfront.net/datasets/rugged_data.csv"
data = pd.read_csv(DATA_URL, encoding="ISO-8859-1")
df = data[["cont_africa", "rugged", "rgdppc_2000"]]

train = torch.tensor(df.values, dtype=torch.float)

# linear regression model:
# mean = a + b_a * is_african + b_r * ruggedness + b_ar * is_african *
# ruggedness


def simple_model(is_african, ruggedness, log_gdp=None):
    a = pyro.param("a", lambda: torch.randn(()))
    b_a = pyro.param("bA", lambda : torch.randn(()))
    b_r = pyro.param("bR", lambda : torch.randn(()))
    b_ar = pyro.param("bAR", lambda : torch.randn(()))
    sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)

    mean = a + b_a * is_african + b_r * ruggedness + b_ar * is_african * ruggedness

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

is_african, ruggedness, log_gdp = train[:,0], train[:,1], train[:,2]

g = pyro.render_model(simple_model, model_args=(is_african, ruggedness, log_gdp))
g.view()

g.save('./a_graph.gv')
g.render(format='png', outfile='./a_graph.png')

simple_model(is_african, ruggedness, )
simple_model(is_african, ruggedness, log_gdp)

def simpler_model(is_african, ruggedness):
    a = pyro.param("a", lambda: torch.randn(()))
    b_a = pyro.param("bA", lambda : torch.randn(()))
    b_r = pyro.param("bR", lambda : torch.randn(()))
    b_ar = pyro.param("bAR", lambda : torch.randn(()))
    sigma = pyro.param("sigma", lambda: torch.ones(()), constraint=constraints.positive)

    mean = a + b_a * is_african + b_r * ruggedness + b_ar * is_african * ruggedness

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma))

conditioned_model = pyro.condition(simpler_model, data={"obs": log_gdp})


def model(is_cont_africa, ruggedness, log_gdp=None):
    a = pyro.sample("a", dist.Normal(0., 10.))
    b_a = pyro.sample("bA", dist.Normal(0., 1.))
    b_r = pyro.sample("bR", dist.Normal(0., 1.))
    b_ar = pyro.sample("bAR", dist.Normal(0., 1.))
    sigma = pyro.sample("sigma", dist.Uniform(0., 10.))

    mean = a + b_a * is_cont_africa + b_r * ruggedness + b_ar * is_cont_africa * ruggedness

    with pyro.plate("data", len(ruggedness)):
        return pyro.sample("obs", dist.Normal(mean, sigma), obs=log_gdp)

pyro.render_model(model, model_args=(is_african, ruggedness, log_gdp))

auto_guide = pyro.infer.autoguide.AutoNormal(model)

adam = pyro.optim.Adam({"lr" : 2e-3})
elbo = pyro.infer.Trace_ELBO()

