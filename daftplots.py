import daft
import pyro
import torch
import torch.nn.functional as F
import pyro.distributions as dist
import pyro.distributions.constraints as constraints

import graphviz

def model(data):
    with pyro.plate("N", len(data)):
        z = pyro.sample("z", dist.Normal(0,1))
        pyro.sample("x", dist.Normal(z,1), obs=data)

data = torch.ones(10)
g = pyro.render_model(model, model_args=(data,), filename="plots/vae_basic_p.pdf")

g.render("./plots/vae_basic_p.pdf")


g = graphviz.Source.from_file("./plots/vaeBasicP.gv")
g.render()
g = graphviz.Source.from_file("./plots/vaeBasicQ.gv")
g.render()

g = graphviz.Source.from_file("./plots/vaegmmP.gv")
g.render()

g = graphviz.Source.from_file("./plots/vaegmmQ.gv")
g.render()
