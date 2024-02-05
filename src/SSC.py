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

"""# SSC"""

#@title SSC
def SSC(Y):
    C = cp.Variable((Y.shape[1], Y.shape[1]))

    constraint = [cp.sum_squares(Y - Y @ C) <=1e-2]
    constraint += [C[i,i] == 0 for i in range(Y.shape[1])]
    objective = cp.Minimize(cp.atoms.norm1(C))

    prob = cp.Problem(objective, constraints= constraint)
    result = prob.solve()

    #print(np.round(C.value,2))
    #print(objective.value)

    return C.value, objective.value
