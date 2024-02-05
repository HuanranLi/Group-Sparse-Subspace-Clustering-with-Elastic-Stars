
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

from Init import *
from plot import *
from ES_SSC import *
from SSC import *
from EnSC import *



# Parameter
low_rank_shape = (100, 40)
rank = 5
n_clusters = 3
labels_true = [i for i in range(n_clusters) for _ in range(low_rank_shape[1])]
weight_sparsity = 0

# Initialize arrays for results
analyze_C_results = []
analyze_C_EN_results = []
analyze_B_EN_square_results = []

# Initialize arrays for timing if they are not defined elsewhere
Time_C_accuracy, Time_C_EN_accuracy, Time_B_accuracy = [], [], []

for _ in range(1):
    beginning_time = tm.time()

    # Data Initialization
    Xs = [lowRankMat_init(low_rank_shape, rank=rank, weight_sparsity=weight_sparsity) for _ in range(n_clusters)]
    X = np.hstack(Xs)
    X_input = np.array([np.hstack([X[:, :i], X[:, i+1:]]) for i in range(X.shape[1])])
    Y = X.T

    # Calculation and Analysis
    try:
        C, obj = SSC(X)
        print('SSC Done At:', np.round(tm.time() - beginning_time), 's')
        Time_C_accuracy.append(tm.time() - beginning_time)
        analyze_C_results.append(analyze_graph(C))
    except Exception as e:
        print("SSC failed:", e)
        C = None

    try:
        C_EN, obj_EN = SSC_EN(X, lambd=100)
        print('SSC_EN Done At:', np.round(tm.time() - beginning_time), 's')
        Time_C_EN_accuracy.append(tm.time() - beginning_time - Time_C_accuracy[-1])
        analyze_C_EN_results.append(analyze_graph(C_EN))
    except Exception as e:
        print("SSC_EN failed:", e)
        C_EN = None

    B_EN, beta_record, v_record, gamma_record = find_weight(Y, X_input, regularizer_weight=0.01, max_itr=50)
    print('MDSP-EN Done At:', np.round(tm.time() - beginning_time), 's')
    Time_B_accuracy.append(tm.time() - beginning_time)

    B_EN_square = np.empty([B_EN.shape[0], B_EN.shape[0]])
    for i in range(B_EN.shape[0]):
        if i == 0:
            B_EN_square[i] = np.hstack(([0], B_EN[i].flatten()))
        elif i == B_EN.shape[0] - 1:
            B_EN_square[i] = np.hstack((B_EN[i].flatten(), [0]))
        else:
            B_EN_square[i] = np.hstack((B_EN[i, :i].flatten(), [0], B_EN[i, i+1:].flatten()))

    analyze_B_EN_square_results.append(analyze_graph(B_EN_square))

# Results stored in analyze_C_results, analyze_C_EN_results, and analyze_B_EN_square_results


# Calculate mean and std for each result array
mean_C = np.mean(analyze_C_results)
std_C = np.std(analyze_C_results)
print('Mean of C results:', mean_C, 'Std of C results:', std_C)

mean_C_EN = np.mean(analyze_C_EN_results)
std_C_EN = np.std(analyze_C_EN_results)
print('Mean of C_EN results:', mean_C_EN, 'Std of C_EN results:', std_C_EN)

mean_B_EN_square = np.mean(analyze_B_EN_square_results)
std_B_EN_square = np.std(analyze_B_EN_square_results)
print('Mean of B_EN_square results:', mean_B_EN_square, 'Std of B_EN_square results:', std_B_EN_square)

# Save results and times to a .npz file
np.savez('results_and_times.npz', analyze_C_results=analyze_C_results, analyze_C_EN_results=analyze_C_EN_results,
         analyze_B_EN_square_results=analyze_B_EN_square_results, Time_C_accuracy=Time_C_accuracy,
         Time_C_EN_accuracy=Time_C_EN_accuracy, Time_B_accuracy=Time_B_accuracy)

print("All results and times saved to 'results_and_times.npz'")
