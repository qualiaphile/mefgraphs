###################
# basic_utils.py: 
# C.Hillar, 2025
###################

import numpy as np
import itertools


def ind(i, j, V):
   """ returns index into (1D) internal array edges of V(V-1)/2 binary vector """
   if i < j:
       return (i * V - i*(i+1) // 2 + j - i - 1)
   return (j * V - j * (j + 1) // 2 + i - j - 1)

def all_ksparse_vector_all_ones(sparsity, dim_vector):
   """ iterates over all (sparsity)-sparse (dim_vector)-dimensional binary vectors """
   for k_subset in itertools.combinations(range(dim_vector), sparsity):
       v = np.zeros(dim_vector,dtype='int')
       for k in range(sparsity):
           v[k_subset[k]] = 1
       yield v

def all_k_cliques(K, V):
    """ iterator for all K-cliques in a V node graph
        viewed as binary vectors of length E = V(V-1)/2 
        usage: ac = all_k_cliques(3,7)
               for clique in ac:
                  print clique
    """
    for vec in all_ksparse_vector_all_ones(K, V):
        pattern = np.zeros((V*(V-1)) // 2,dtype='int')
        cnt = 0
        for i in range(V):
            for j in range(1,V-i):
                if vec[i] != 0 and vec[i+j] != 0:
                    pattern[cnt] = 1
                cnt += 1
        yield pattern

def matrix_proj(V, J):
    """ Returns x,y = average of two types of edge pairs """
    N = V*(V-1)/2   # number of edges
    x = []; y = []
    ac1 = itertools.combinations(range(V),2)  # all edges
    for i,j in ac1:
        ac2 = itertools.combinations(range(V),2)  # all edges
        for k,l in ac2:
            set1 = {i,j}
            set2 = {k,l}
            intersect_size = len(set1.intersection(set2))
            if intersect_size == 1:
                x.append(J[ind(i,j,V),ind(k,l,V)])
            if intersect_size == 0:
                y.append(J[ind(i,j,V),ind(k,l,V)])
            if intersect_size == 2:
                pass
    return np.array(x).mean(), np.array(y).mean()

def project_to_3param(V, J, theta, normalize=True):
    """ Returns x,y,z = average of two types of edge pairs and the threshold
        normalize: if True first divides parameters by abs(avg thresholds) [unless avg is zero]
    """
    if normalize:
        theta_abs_mean = np.abs(theta.mean())
        if theta_abs_mean > 0:
            theta_normalized = theta / theta_abs_mean
            J_normalized = J / theta_abs_mean
        return *matrix_proj(V, J_normalized), theta_normalized.mean()
    return *matrix_proj(V, J), theta.mean()
