###################
# basic_graphs.py: Basic graph class
# C. Hillar, 2025
###################

import numpy as np
import itertools
from itertools import combinations
from math import comb

from basic_utils import ind, all_k_cliques


def key(bin):
    """ takes numpy binary array bin and makes a string """
    return "".join(str(k) for k in bin.astype(np.int64).ravel())

def rkey(key):
    return np.array([np.int64(k) for k in list(key)])

class Graph(object):
    """ Basic graph class 
        default: empty graph """
    def __init__(self, V=None):
        self.name = "Empty"
        self.V = V
        if self.V is not None:
            self.N = self.V * (self.V - 1) // 2
            self.edges = np.zeros(self.N)
        else:
            self.N = 0
            self.edges = None

    def sample(self, M=None, all=False, returnDict=False):
        """ Gives M samples of the graph under permutation action on vertices
        M: number of samples of graph acted on by permutation on verticews
        all: if True returns all graphs isomorphic to base graph
        returnDict: if True return dictionary of graph->counts vs array graphs (possibly repeated)
        """
        if M is None: M = 1
        if not all:
            samples = [self.action() for i in range(M)]
            if not returnDict:
                return samples
        else:
            all_perms = itertools.permutations(range(self.V), self.V)
            samples = [self.action(perm=p) for p in all_perms]
        d = dict()
        for s in samples:
            if key(s) in d:
                d[key(s)] += 1
            else:
                d[key(s)] = 1
        if returnDict:
            return d
        return [rkey(key) for key in d.keys()]

    def action(self, perm=None):
        """ Returns action of random perm on graph """
        if perm is None: perm = np.random.permutation(self.V)
        edges = self.edges.copy()
        for i in range(self.V):
            for j in range(self.V):
                if i != j:
                    edges[ind(i, j, self.V)] = self.edges[ind(perm[i], perm[j], self.V)]

        return edges

    def orbit(self, returnDict=False):
        """ Returns the orbit under the permutation group acting on the graph """
        self._sample = self.sample(M=None, all=True, returnDict=returnDict)
        return self._sample

    def iso_count(self):
        return len(self.orbit())

class Random(Graph):
    """ Random graph class ER with p = .5 default """
    def __init__(self, V=None, p=.5):
        super().__init__(V=V)
        self.name = "Random"
        self.p = p
        self.edges += (np.random.random(self.N) < p).astype(np.double)

class Clique(Graph):
    """ Cycle graph class """
    def __init__(self, V=None, K=None):
        super().__init__(V=V)
        self.name = "Clique"
        self.K = K
        self.Vs = None
        self.edges = np.zeros(self.N)
        clique_array = range(self.K)  # clique vertices
        for i in clique_array:
            for j in clique_array:
                if i != j:
                    self.edges[ind(i, j, V)] = 1

    def orbit(self, returnDict=False):
        """ Returns the orbit under the permutation group acting on the graph """
        all_graphs = all_k_cliques(self.K, self.V)
        if returnDict:
            d = dict()
            for graph in all_graphs:
                d[key(graph)] = int(np.math.factorial(self.V) / self.iso_count())
            return d
        return np.array([graph for graph in all_graphs])

    def iso_count(self):
        return comb(self.V, self.K)

###################
# TODO: Write overidding orbit and iso_count functions for all below:
###################

class Cycle(Graph):
    """ Cycle graph class """
    def __init__(self, V=None, K=None):
        super().__init__(V=V)
        self.name = "Cycle"
        self.K = K
        self.Vs = None
        self.edges = np.zeros(self.N)
        for i in range(self.K - 1):
            self.edges[ind(i, i + 1, V)] = 1
        self.edges[ind(0, self.K - 1, V)] = 1

class Chain(Graph):
    """ Chain graph class o-o-o-o """
    def __init__(self, V=None, K=None):
        super().__init__(V=V)
        self.name = "Chain"
        self.K = K
        self.Vs = None
        self.edges = np.zeros(self.N)
        for i in range(self.K - 1):
            self.edges[ind(i, i + 1, V)] = 1

class Bipartite(Graph):
    """ Bipartite graph class """
    def __init__(self, V=None, K=None):
        super().__init__(V=V)
        self.name = "Bipartite"
        self.K = K
        left_verts = range(self.K)
        self.adj = np.zeros((V, V))
        self.edges = np.zeros(self.N)
        for i in left_verts:
            for j in range(self.K, self.V):
                self.adj[i, j] = 1; self.adj[j, i] = 1
                self.edges[ind(i, j, V)] = 1

class Johnson(Graph):
    """ Johnson graph class: https://en.wikipedia.org/wiki/Johnson_graph
        Parameters:
        n (int): The size of the set from which subsets are chosen.
        k (int): The size of each subset.
        Note: k = 1 or n-1 are the complete graph
              n = 4, 
    """
    def __init__(self, n=4, K=2):
        super().__init__(V=comb(n, K))
        self.name = "Johnson"
        self.n = n
        self.K = K
        nodes = list(combinations(range(self.n), self.K))
        for i, u in enumerate(nodes):
            for j, v in enumerate(nodes):
                if len(set(u).difference(set(v))) == 1:
                    self.edges[ind(i, j, self.V)] = 1

class Circulant(Graph):
    """ Constructs a Circulant Graph with n nodes and given jump connections.
    https://en.wikipedia.org/wiki/Circulant_graph
    
    Parameters:
    n (int): Number of nodes in the graph.
    jumps (list of int): List of jump distances (mod n). """
    def __init__(self, V=4, jumps=[2, 2]):
        super().__init__(V=V)
        self.name = "Circulant"
        self.V = V
        self.jumps = jumps
        for i in range(V):
            for jump in jumps:
                neighbor = (i + jump) % self.V
                self.edges[ind(i, neighbor, self.V)] = 1

class Paley(Graph):
    """Construct the Paley graph of order q.
       https://en.wikipedia.org/wiki/Paley_graph """
    def __init__(self, V=5):
        super().__init__(V=V)
        self.name = "Paley"
        self.V = V
        residues = self.quadratic_residues(self.V)
        for i in range(self.V):
            for j in range(i + 1, self.V):
                if (j - i) % self.V in residues:
                    self.edges[ind(i, j, self.V)] = 1

    def quadratic_residues(self, V):
        """Compute the set of quadratic residues modulo q."""
        return {pow(x, 2, V) for x in range(1, V)}
