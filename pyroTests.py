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
pyro.render_model(simple_model, model_args=(is_african, ruggedness, log_gdp))

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
