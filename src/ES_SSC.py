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


"""# MDSP Source Code"""

def find_beta(Y,X,v,cummulative_diff, k, regularizer_weight):

    beta_array = [cp.Variable(X.shape[2]) for i in range(X.shape[0])]

    norm_array = cp.vstack([Y[i] - X[i] @ beta_array[i] for i in range(len(Y))])
    diff_array = cp.hstack([beta_array[i] - v[i] + cummulative_diff[i]/k for i in range(X.shape[0])])

    # + lambd*cp.sum_squares(cp.vstack(beta_array))
    objective = cp.Minimize(0.5*cp.sum_squares(norm_array) + k/2* cp.sum_squares(diff_array)+ regularizer_weight*cp.sum_squares(cp.vstack(beta_array)))
    prob = cp.Problem(objective)
    result = prob.solve()

    value_array = np.array([i.value for i in beta_array])

    return value_array, prob.value


#can be optimized with momentum gradient descent
def find_vij(beta_ij, cummulative_diff_ij, k, lambd, gamma_j):

    #grid search to find the best starting point for gradient descent
    search_range = max(abs(beta_ij + cummulative_diff_ij/k), abs(gamma_j))*2
    #x = np.linspace(-1*search_range, search_range, num = 50)
    x = [beta_ij + cummulative_diff_ij/k, 0, gamma_j]
    objs = [ k/2*(v_ij - beta_ij - cummulative_diff_ij/k)**2  + lambd * min(abs(v_ij), abs(v_ij - gamma_j)) for v_ij in x]
    v_ij = x[np.argmin(objs)]

    #gradient descent parameter setup
    step = 0.001
    gradient = 1
    iter = 0

    #gradient descent
    while(abs(gradient) > 1e-10 and iter < 100):
        gradient = k/2*2*(v_ij - beta_ij - cummulative_diff_ij/k)
        if abs(v_ij) < abs(v_ij - gamma_j):
            gradient += lambd * np.sign(v_ij)
        else:
            gradient += lambd * np.sign(v_ij - gamma_j)

        v_ij -= gradient * step
        iter += 1

    return v_ij


def find_obj(obj_base, beta, v, gamma, cummulative_diff, k, lambd):
    obj = obj_base.copy()
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            obj += k/2*(v[i,j] - beta[i,j] - cummulative_diff[i,j]/k)**2  + lambd * min(abs(v[i,j]), abs(v[i,j] - gamma[j]))

    return obj

def find_gamma_j(v_j):

     gamma_range = np.linspace(min(v_j), max(v_j), num = 100)

     value = [ sum( [min(abs(v_ij), abs(v_ij - x)) for v_ij in v_j ]) for x in gamma_range]

     best_gamma = gamma_range[np.argmin(value)]

     return best_gamma

def find_weight(Y,X, regularizer_weight, max_itr = 30):

    early_start_time = tm.time()

    weight_size = X.shape[2]
    sample_size = X.shape[0]
    #initialize
    v = np.zeros((sample_size, weight_size))
    cummulative_diff = np.zeros(v.shape)
    k = 1
    gamma = np.zeros(weight_size)
    lambd = 1e-5

    old_beta = np.zeros(v.shape)
    old_gamma = np.zeros(gamma.shape)
    itr = 0

    obj_record = []
    beta_record = []
    gamma_record = []
    v_record = []


    start_time = tm.time()
    while(1):
        #print(itr)
        #calculate beta

        beta,obj_b = find_beta(Y,X,v,cummulative_diff,k, regularizer_weight)

        #print('Done Finding Beta:', np.round(tm.time() - start_time, 2))

        if itr == 0:
            #update gamma
            for j in range(beta.shape[1]):
                gamma[j] = find_gamma_j(beta[:,j])
                #gamma[j] = 1

        #update v in parrallel
        with mp.Pool(mp.cpu_count()) as pool:
            v_total =  np.array(pool.starmap(find_vij,
                                            ([ [beta[i,j], cummulative_diff[i,j], k, lambd, gamma[j] ]
                                                for i in range(beta.shape[0])
                                                for j in range(beta.shape[1])
                                                ])))

            v = np.reshape(v_total, beta.shape)
            #print('Parrallel - Done Finding v:', np.round(tm.time() - start_time, 2))


        #update v separably
        #for i in range(beta.shape[0]):
        #    for j in range(beta.shape[1]):
        #        v[i,j] = find_vij(beta[i,j], cummulative_diff[i,j], k, lambd, gamma[j])
        #print('Done Finding v:', np.round(tm.time() - start_time, 2))

        #update gamma in parrallel
        #with mp.Pool() as pool:
            #gamma = np.array( pool.map(find_gamma_j, v) )
            #print('Done Finding gamma in parrallel:', np.round(tm.time() - start_time))

        #update gamma
        for j in range(beta.shape[1]):
            gamma[j] = find_gamma_j(v[:,j])
        #print('Done Finding gamma:', np.round(tm.time() - start_time, 2))

        obj = find_obj(obj_b, beta, v, gamma, cummulative_diff, k, lambd)
        obj_record.append(obj)
        #print('Done Finding Objective:', np.round(tm.time() - start_time, 2) )



        #update cummulative diff
        diff = (beta - v)
        cummulative_diff += k * diff

        if np.linalg.norm(old_beta - beta) + np.linalg.norm(old_gamma - gamma) < 1e-5 and np.linalg.norm(diff) < 1e-5:
            break
        elif itr > max_itr:
            break
        #else:
        #    print(np.linalg.norm(old_beta - beta) + np.linalg.norm(old_gamma - gamma))
        #    print(np.linalg.norm(diff))


        report_every_n_iter = min(100, max_itr)
        if itr % report_every_n_iter == report_every_n_iter-1 and False:
            print('ITER:', itr)
            print("obj:", obj)
            print("gamma:", np.round(gamma, 1))
            #print('beta:', np.round(beta, 1))
            plt.figure(figsize = (10,5))
            plt.plot(obj_record)
            plt.title('Objective Function')
            plt.show()


        #print('Iter', itr, '; Time:', np.round( (tm.time() - start_time), 2), 's; Obj = ', obj )
        #print()
        start_time = tm.time()
        beta_record.append(beta.copy())
        gamma_record.append(gamma.copy())
        v_record.append(v.copy())

        old_beta = beta
        old_gamma = gamma
        itr += 1

    #print('Total Time: ',np.round( (tm.time() - early_start_time)/60, 2), 'min')

    return beta, beta_record, v_record, gamma_record
