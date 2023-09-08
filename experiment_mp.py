import numpy as np
import argparse
import multiprocessing as mp
from training import training


parser = argparse.ArgumentParser()
parser.add_argument('-mod', '--model', type=str, help='Model number.')
parser.add_argument('-met', '--metric', default="wasserstein2", type=str, help='Metric name.')
parser.add_argument('-space', '--space', default="distribution", type=str, help='Space name.')
parser.add_argument('-nreps', '--nreps', default=5, type=int, help='Number of simulation replications.')
parser.add_argument('-startidx', '--startidx', default=0, type=int, help='Start index of replications')
parser.add_argument('-n', '--nsample', default=50, type=int, help='Sample size.')
parser.add_argument('-nperm', '--nperm', default=399, type=int, help='Number of permutation replications.')

args = parser.parse_args()

n = args.nsample
model = args.model
metric = args.metric
space = args.space
nperm = args.nperm
nreps = args.nreps
start_idx = args.startidx * nreps
methods = ['angle', 'ball', 'graph', 'energy']

if model == 'model1':
    deltas = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
elif model == 'model2':
    deltas = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
elif model == 'model3':
    deltas = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
else:
    raise Exception("Input model not defined")

print("----------------------------------Starting-------------------------------------")
print("Training args: ", args)

path = "result"
n_methods = len(methods)
ran = [start_idx, start_idx + nreps]
nprocs = len(deltas)
mproc_results = []

if __name__ == '__main__':
    pool = mp.Pool(processes=nprocs)
    results = [pool.apply_async(training, args=(ran, methods, space, metric, model, delta, n, nperm))
               for delta in deltas]
    mproc_results = [r.get() for r in results]
    pool.close()
    pool.join()

print("---------------------------------------------------------------------------")

np.save(path + "/" + str(model) + "/pvalues_" + str(model) + "_" + str(start_idx) + ".npy", mproc_results)
