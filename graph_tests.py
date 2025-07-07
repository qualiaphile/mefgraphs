################
# graph_tests.py
# C.Hillar, 2025
################

import numpy as np
from math import comb
from basic_graphs import *

graphs = [Graph(V=3), Cycle(V=5, K=3), Clique(V=6, K=3), Chain(V=4, K=3), Random(V=5, p=.8), Random(V=5, p=.3), \
Bipartite(V=6, K=3), Johnson(n=4, K=2), Circulant(V=5, jumps=[2, 2]), Paley(V=4)]

for G in graphs:
    print("---------------------")
    print("Graph class:", G.name)
    print("N:", G.N, "V:", G.V)
    print("Edges:", G.edges)
    print("Random Action:", G.action())
    print("Orbit:", G.orbit())
    print("Orbit with Dictionary:", G.orbit(returnDict=True))
    print("Isomorphism class size:", G.iso_count())