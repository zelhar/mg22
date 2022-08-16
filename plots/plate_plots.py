import daft
import pyro
import torch
import torch.nn.functional as F
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import matplotlib.pyplot as plt

import graphviz

g = graphviz.Source.from_file('./vae.gv',)
g.render(format='png', )
g.render(format='pdf', )

g = graphviz.Source.from_file('./vae_p.gv',)
g.render(format='png', )
g.render(format='pdf', )

g = graphviz.Source.from_file('./vae_q.gv',)
g.render(format='png', )
g.render(format='pdf', )

g = graphviz.Source.from_file('./dirichlet_gmm.gv',)
g.render(format='png', )
g.render(format='pdf', )

g = graphviz.Source.from_file('./dirichlet_gmm_p.gv',)
g.render(format='png', )
g.render(format='pdf', )

g = graphviz.Source.from_file('./dirichlet_gmm_q.gv',)
g.render(format='png', )
g.render(format='pdf', )

g = graphviz.Source.from_file('./gmm.gv',)
g.render(format='png', )
g.render(format='pdf', )

g = graphviz.Source.from_file('./gmm_vanilla.gv',)
g.render(format='png', )
g.render(format='pdf', )



