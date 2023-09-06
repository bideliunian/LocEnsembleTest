# %%

import numpy as np
import sys
import argparse
import pathlib
import time
import multiprocessing as mp

#sys.path.insert(0, 'd:\\Angle\\Code\\Functions')
sys.path.append('d:\\Angle\\Code\\Functions')
from generate_data import *
from permutation import *
from distance import *
from mp_helper import *
from double import *
# parser = argparse.ArgumentParser()
# parser.add_argument('-mod', '--model', default=1, type=int, help='Model number.')
# parser.add_argument('-meth', '--method', default="angle", type=str, help='Method name.')
# parser.add_argument('-met', '--metric', default="Wasserstein2", type=str, help='Metric name.')
# parser.add_argument('-reps', '--reps', default=100, type=int, help='Number of simulation replications.')
# parser.add_argument('-n','--nsample', default=50, type=int, help='Sample size.')
# parser.add_argument('-del', '--delta', default=0, type=float, help='delta.')
# parser.add_argument('-r', '--r', default=1, type=float, help='r.')
# parser.add_argument('-nperm', '--nperm', default=500, type=int, help='Number of permutation replications.')
# parser.add_argument('-np','--nproc', default=8, type=int, help='Number of processes to run in parallel.')
# args = parser.parse_args()

# print("Args: ", args)

# n = args.nsample
# model = args.model
# method = args.method
# metric = args.metric
# delta = args.delta
# r = args.r
# nperm = args.nperm
# reps = args.reps
# nproc = args.nproc


# %%
n = 10
model = 'model1'
methods = ['angle', 'ball', 'graph', 'energy']
metric = 'wasserstein2'
r = 1
nperm = 99
nreps = 8
nprocs = 8
space = 'distribution'
start_idx = 0
deltas = [0, 0.2, 0.4, 0.6, 0.8, 1]

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-mod', '--model', default='model1', type=str, help='Model number.')
parser.add_argument('-met', '--metric', default="wasserstein2", type=str, help='Metric name.')
parser.add_argument('-space', '--space', default="distribution", type=str, help='Space name.')
parser.add_argument('-nreps', '--nreps', default=5, type=int, help='Number of simulation replications.')
parser.add_argument('-startidx', '--startidx', default=0, type=int, help='Start index of replications')
parser.add_argument('-n','--nsample', default=10, type=int, help='Sample size.')
parser.add_argument('-nperm', '--nperm', default=99, type=int, help='Number of permutation replications.')

args = parser.parse_args([])

print("Args: ", args)

n = args.nsample
model = args.model
metric = args.metric
space = args.space
nperm = args.nperm
nreps = args.nreps
start_idx = args.startidx
methods = ['angle', 'ball', 'graph', 'energy']
deltas = [0.2, 0.4, 0.6, 0.8, 1.0]
# %%
print("----------------------------------Starting-------------------------------------")
print("model:", model, ", metric", metric, ", n=", n, ", nperm", nperm)
path = "Results"
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

n_methods = len(methods)
n_deltas = len(deltas)

p_values = np.full((nreps, n_methods, n_deltas), -1.0)
elapsed  = np.full((nreps, n_methods, n_deltas), -1.0)


print("---------------------------------------------------------------------------")

for rep in range(start_idx, start_idx + nreps):
    
    np.random.seed(rep)
    print(rep)
    
    for delta_idx in range(0, n_deltas):

        start_time = time.time()

        delta = deltas[delta_idx]
        x = generate_data_distribution_1d(n = n, m = 100, model='model1', delta=0.)
        y = generate_data_distribution_1d(n = n, m = 100, model='model1', delta=delta)
        
        pooled = np.vstack((x, y))

        # pooled distance matrix (m+n)*(m+n)
        if space == "distribution":
            dist_pooled = pdist_distribution1d(pooled, metric=metric)

        elif space == "spd":
            dist_pooled = pdist_spd(pooled, metric=metric)

        for method_idx in range(0, n_methods):
            method = methods[method_idx]
            p_value = permutation_test_homogeneity_from_distance_matrix(m=n, dist=dist_pooled, func=method, 
                                                method="approximate", num_rounds=nperm, seed=rep)

            elap_time = time.time() - start_time

            p_values[rep, method_idx, delta_idx] = p_value
            elapsed[rep, method_idx, delta_idx] = elap_time

            print("Method ", method, ", delta", delta, "\\Repetition ", rep, ", p-value ", p_value, ", elapsed time", elap_time)

print(np.sum(p_values < 0.05, axis=0)/nreps)
# np.save(path + "/pvalues.npy", p_values)
# np.save(path + "/elapsed.npy", elapsed)


# %%
