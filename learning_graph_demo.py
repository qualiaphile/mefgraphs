###################
# learning_graph_demo.py
# Iso(G) learning Experiment (Small V) ###
# C. Hillar, July 2025
#
# To run, call in ipython:
#
# run learning_graph_demo.py
#
# Computes:
#
# Pick a GRAPH type on V vertices, perhaps with K as an extra parameter
# Compute its Full isomorphism class C (its orbit under Permutation group Sv)
# Over T trials do:
#    Choose an ordering of C
#    Set TestSet = last 10% of C
#    For M = 1 to |C|:
#       Set TestSet = first M elements of C
#       Train MEF Hopfield network on M samples
#       Compute AttractorsTrain
#       Compute AttractorsTest
#       Compute Bit/Exact scores for Train and Test sets
#
# --------------------- 
# After compute, Displays some basic figures
######

import os
import sys
import numpy as np
from math import comb

from matplotlib import pyplot as plt

###########################
# Be sure to install this:
""" pip install git+ssh://git@github.com/team-hdnet/hdnet.git
    https://github.com/team-hdnet/hdnet """
from hdnet import HopfieldNetMPF
###########################

from basic_graphs import Cycle, Chain, Clique, Random, Bipartite, Johnson, Circulant, Paley
from basic_utils import project_to_3param

###########################
# Specify Graph Class:
G = Random(V=7, p=.2)
#G = Clique(12, 6)
#G = Cycle(9, 5)
#G = Chain(9, 4)
#G = Bipartite(8,4)
#G = Johnson(4,2)
#G = Circulant(7,[2,2])
#G = Paley(6)

print("Graph type:", G.name, "V=", G.V)
all_graphs = np.array(G.orbit())
tot_num_graphs = G.iso_count()
print("Iso size:", G.iso_count())

T = 2  # number trials
test_size = int(.1 * tot_num_graphs)  # ten percent holdout

Ms = np.arange(1, tot_num_graphs + 1)

proj_3params = np.zeros((T, len(Ms), 3))
train_scores = np.zeros((T, len(Ms), 2))
test_scores = np.zeros((T, len(Ms), 2))

print("Training on %s (%d Tot Graphs in isomoprhism class)" % (G.name, tot_num_graphs))
for t in range(T):
    print("Trial %d (%s):" % (t, G.name))
    idx = np.random.permutation(tot_num_graphs)
    for c, M in enumerate(Ms):
        print("M: %d" % M)
        graph_samples = all_graphs[idx[:M]]
        NET = HopfieldNetMPF(G.N)
        NET.learn_all(graph_samples)  # fit network on M samples
        train_scores[t, c, 0] = (NET(graph_samples) == graph_samples).prod(axis=1).mean()
        test_scores[t, c, 0] = (NET(all_graphs[idx[-test_size:]]) == all_graphs[idx[-test_size:]]).prod(axis=1).mean()
        train_scores[t, c, 1] = (NET(graph_samples) == graph_samples).mean()
        test_scores[t, c, 1] = (NET(all_graphs[idx[-test_size:]]) == all_graphs[idx[-test_size:]]).mean()
        proj_3params[t, c] = project_to_3param(G.V, NET.J, NET.theta)
        print("Test Scores (avg/exact)", test_scores[t, c, 0], test_scores[t, c, 1], "proj:", proj_3params[t, c])

NET = HopfieldNetMPF(G.N)
NET.learn_all(all_graphs)  # fit network on all isomorphic graphs

x, y, z = project_to_3param(G.V, NET.J, NET.theta)

# Normalize weights, thresholds
theta_abs_mean = np.abs(NET.theta.mean())
if theta_abs_mean > 0:
    theta_normalized = NET.theta / theta_abs_mean
    J_normalized = NET.J / theta_abs_mean

######### Basic Plots #########
plt.ion()
plt.figure(0)
plt.clf()
plt.title("Scatter (x, y) %s" % (G.name))
plt.ylabel("Y")
plt.xlabel("X")
plt.scatter(proj_3params[:, :, 0].mean(axis=0), proj_3params[:, :, 1].mean(axis=0))
plt.scatter(x, y, c='red', label='All %d' % tot_num_graphs)
plt.legend(frameon=False, loc='best')

plt.figure(1)
plt.clf()
plt.title("Bit/Exact T=%d trials (%s V%d)" % (T, G.name, G.V))
plt.xlabel("Sample count (%d total graphs)" % tot_num_graphs)
plt.ylabel("Accuracy (exact / avg bits correct)")
plt.errorbar(Ms, train_scores[:,:,0].mean(axis=0), yerr=train_scores[:,:,0].std(axis=0), label='Train (Exact)')
plt.errorbar(Ms, test_scores[:,:,0].mean(axis=0), yerr=test_scores[:,:,0].std(axis=0), label='Test (Exact)')
plt.errorbar(Ms, test_scores[:,:,1].mean(axis=0), yerr=test_scores[:,:,1].std(axis=0), label='Test (Bit)')
plt.xlim([1, max(Ms) + 1])
plt.legend(frameon=False, loc='best')

plt.figure(2)
plt.clf()
plt.title("(x, y, z) Projections (%s)" % (G.name))
plt.ylabel("Values")
plt.xlabel("Sample count")
plt.errorbar(Ms, proj_3params[:, :, 0].mean(axis=0), yerr=proj_3params[:, :, 0].std(axis=0), label='x')
plt.errorbar(Ms, proj_3params[:, :, 1].mean(axis=0), yerr=proj_3params[:, :, 1].std(axis=0), label='y')
plt.errorbar(Ms, proj_3params[:, :, 2].mean(axis=0), yerr=proj_3params[:, :, 2].std(axis=0), label='z')
plt.axhline(y=x, color='b', linestyle='--')
plt.axhline(y=y, color='r', linestyle='--')
plt.legend(frameon=False, loc='best')

plt.matshow(J_normalized)
plt.title("Weights (%s V%d)" % (G.name, G.V))
plt.colorbar()