# %%

import numpy as np
import sys
import argparse
import pathlib
import time
import multiprocessing as mp
import os

cur_dir = os.path.dirname(__file__)
func_dir = os.path.abspath(os.path.join(cur_dir,'../Functions'))
sys.path.append(func_dir)

from generate_data import *
from permutation import *
from distance import *
from mp_helper import *

# %%
parser = argparse.ArgumentParser()
parser.add_argument('-mod', '--model', default='model1', type=str, help='Model number.')
parser.add_argument('-met', '--metric', default="frobenius", type=str, help='Metric name.')
parser.add_argument('-space', '--space', default="spd", type=str, help='Space name.')
parser.add_argument('-nreps', '--nreps', default=100, type=int, help='Number of simulation replications.')
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
start_idx = args.startidx * nreps
methods = ['angle', 'ball', 'graph', 'energy']
if model == 'model1':
    deltas = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]
elif model == 'model2':
    deltas = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75]

print("----------------------------------Starting-------------------------------------")
print("model:", model, ", deltas:", deltas, ", metric:", metric, ", n=", n, ", nperm", nperm)

path = "Results"
pathlib.Path(path).mkdir(parents=True, exist_ok=True)

n_methods = len(methods)
ran = [start_idx, start_idx + nreps]
nprocs = len(deltas)

# %%
print("---------------------------------------------------------------------------")

start_time = time.time()
mproc_results = []

if __name__ ==  '__main__': 
    pool = mp.Pool(processes=nprocs)
    results = [pool.apply_async(mp_helper, args = (ran, methods, space, metric, model, delta, n, nperm))
                     for delta in deltas]
    mproc_results = [r.get() for r in results]
    pool.close()
    pool.join()
    
    print(mproc_results)

elapsed_time = time.time() - start_time

print(len(mproc_results))
print(type(mproc_results))
#print(mproc_results[0])
# print(np.sum(mproc_results < 0.05, axis=1) / nreps)

print(elapsed_time)
#np.save(path + "/pvalues"+ str(model) + str(start_idx) + ".npy", mproc_results)

# %%
