
# %%
import numpy as np
import pandas as pd
from contools import Cascade_Analyzer

# set up toy adjacency matrices

adj = np.array([[0, 50, 100, 0, 0],
                [0, 0, 0, 100, 0],
                [0, 0, 0, 100, 0],
                [0, 0, 0, 0, 100],
                [0, 0, 0, 0, 0]])

adj = pd.DataFrame(adj)

adj_NT = np.array([[0, 100, 100, 0, 0],
                [0, 0, 0, -30, 0],
                [0, 0, 0, 100, 0],
                [0, 0, 0, 0, 100],
                [0, 0, 0, 0, 0]])

adj_NT = pd.DataFrame(adj_NT)

# %%
# standard cascade (excitatory only)
from contools import cascade, TraverseDispatcher, Cascade

p = 0.05
max_hops = 4
n_init = 100
simultaneous = True
source_indices = [0]
stop_indices = [4]

transition_probs = cascade.to_transmission_matrix(adj.values, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs,
    stop_nodes = stop_indices,
    max_hops=max_hops+1, # +1 because max_hops includes hop 0
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

casc = Cascade_Analyzer.run_cascade(start_nodes = source_indices, cdispatch = cdispatch)
print(casc)
# %%
# inhibitory and excitatory cascade, single node activation

transition_probs_NT = cascade.to_transmission_matrix(adj_NT.values, p)

cdispatch = TraverseDispatcher(
    Cascade,
    transition_probs_NT,
    stop_nodes = stop_indices,
    max_hops=max_hops+1, # +1 because max_hops includes hop 0
    allow_loops = False,
    n_init=n_init,
    simultaneous=simultaneous,
)

casc_NT = Cascade_Analyzer.run_cascade(start_nodes = source_indices, cdispatch = cdispatch)
print(casc_NT)
# %%
