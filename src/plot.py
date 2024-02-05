
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



"""# Helper Functions"""

#@title Weight Plot Function
def plot_function(trace1, trace2, target):
    for j in range(trace1[0].shape[1]):
        plt.figure(figsize = (10,3))
        plt.subplot(1,2,1)
        trace_gamma = [gamma_i[j] for gamma_i in target]
        for i in range(trace1[0].shape[0]):
            trace_b = [beta_i[i,j] for beta_i in trace1]
            plt.plot(trace_b,'b-',  alpha = 0.2)

        plt.plot(trace_gamma, 'r:')


        plt.subplot(1,2,2)
        for i in range(trace1[0].shape[0]):
            trace_b = [beta_i[i,j] for beta_i in trace2]
            plt.plot(trace_b,'g-',  alpha = 0.2)

        plt.plot(trace_gamma, 'r:')

        plt.show()
