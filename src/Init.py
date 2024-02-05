

#@title Import
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import seaborn as sns

from scipy.linalg import block_diag
import multiprocessing as mp

from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import homogeneity_score, completeness_score

import time as tm

from sklearn import metrics
import networkx as nx



#@title Init Function
def lowRankMat_init(shape, rank, weight_sparsity = 0):
    assert weight_sparsity <= 1

    base = np.random.randn(shape[0], rank)
    u,s,vt = np.linalg.svd(base, full_matrices = False)
    multiplier = np.random.rand(rank, shape[1])

    choice = np.random.choice(2, p = [weight_sparsity,1 - weight_sparsity], size = (rank, shape[1]), replace = True)
    sparse_multiplier = np.multiply(multiplier, choice)
    lowRankMat = u @ sparse_multiplier

    return lowRankMat
