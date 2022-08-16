import daft
import pyro
import torch
import torch.nn.functional as F
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import matplotlib.pyplot as plt

import graphviz


def model(data):
    with pyro.plate("N", len(data)):
        z = pyro.sample("z", dist.Normal(0, 1))
        pyro.sample("x", dist.Normal(z, 1), obs=data)


data = torch.ones(10)
g = pyro.render_model(model, model_args=(
    data,), filename="plots/vae_basic_p.pdf")

g.render("./plots/vae_basic_p.pdf")


g = graphviz.Source.from_file("./plots/vaeBasicP.gv")
g.render()
g = graphviz.Source.from_file("./plots/vaeBasicQ.gv")
g.render()

g = graphviz.Source.from_file("./plots/vaegmmP.gv")
g.render()

g = graphviz.Source.from_file("./Digraph.gv")

# renders both aaa.gv and aaa.pdf
g.render("aaaa",)

g.render("aaaa", format='png', formatter='cairo', renderer='cairo')

g.render("aaaaa", format='png', )

scale = 1.0
pgm = daft.PGM(shape=(16,10), aspect=1,)
#pgm = daft.PGM(aspect=1,)
#pgm.add_node("z", r"$z$", 0, 1, scale)
pgm.add_plate([0,0,4,6])
pgm.add_node("z", r"$z$", 2, 4, scale=1, aspect=1, )
pgm.add_node("x", "$x$", 2, 1, scale=1, observed=True,aspect=1,)
pgm.add_node("theta", r"$\theta$", -1.5, 1, scale=0.5, alternate=True,aspect=1,shape="rectangle",)
pgm.add_edge("z", "x",)
#pgm.add_plate([1.5,0,1,4])
pgm.render()

#%paste

%*clipboard-unnamedplus*paste

pgm.savefig("abbbb.png",)

g = graphviz.Source.from_file("./plots/gmm.gv")
g.render()

g = graphviz.Source.from_file("./plots/gmm_vanilla.gv")
g.render()


g = graphviz.Source.from_file("./plots/dirichlet_gmm.gv")
g.render()

g = graphviz.Source.from_file("./plots/dirichlet_gmm_p.gv")
g.render()

g = graphviz.Source.from_file("./plots/dirichlet_gmm_q.gv")
g.render()
