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


def analyze_graph(affinity_matrix, threshold=0):
    """
    Analyze the graph represented by the affinity matrix.

    Args:
    affinity_matrix (np.array): An N x N numpy array representing the affinity matrix of the graph.
    threshold (float): Threshold value to consider an element as 'non-zero'.

    Returns:
    tuple: Number of connected components and the sparsity of the affinity matrix.
    """
    # Create a graph from the affinity matrix
    G = nx.convert_matrix.from_numpy_array(affinity_matrix)

    # Calculate the number of connected components
    num_components = nx.number_connected_components(G)

    # Adjust the matrix based on the threshold
    adjusted_matrix = np.where(affinity_matrix > threshold, 1, 0)

    # Calculate the sparsity of the matrix
    size = adjusted_matrix.size
    num_zeros = size - np.count_nonzero(adjusted_matrix)
    sparsity = num_zeros / size

    return num_components, sparsity
