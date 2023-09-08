import argparse
from training import *


parser = argparse.ArgumentParser()
parser.add_argument('-mod', '--model', type=str, help='Model number.')
parser.add_argument('-met', '--metric', default="wasserstein2", type=str, help='Metric name.')
parser.add_argument('-space', '--space', default="distribution", type=str, help='Space name.')
parser.add_argument('-nreps', '--nreps', default=5, type=int, help='Number of simulation replications.')
parser.add_argument('-startidx', '--startidx', default=0, type=int, help='Start index of replications')
parser.add_argument('-n','--nsample', default=50, type=int, help='Sample size.')
parser.add_argument('-nperm', '--nperm', default=399, type=int, help='Number of permutation replications.')

args = parser.parse_args()

n = args.nsample
model = args.model
metric = args.metric
space = args.space
nperm = args.nperm
nreps = args.nreps
start_idx = args.startidx
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
n_deltas = len(deltas)
ran = [start_idx, start_idx + nreps]
start_time = time.time()
lower = ran[0]
upper = ran[1]
n_reps = upper - lower

p_values = np.full((n_reps, n_methods), -1.0)
elapsed = np.full((n_reps, n_methods), -1.0)

for rep in range(start_idx, start_idx + nreps):
    
    np.random.seed(rep)
    
    for delta_idx in range(0, n_deltas):

        start_time = time.time()

        delta = deltas[delta_idx]
        x = generate_data_distribution_1d(n=n, m=100, model='model1', delta=0.)
        y = generate_data_distribution_1d(n=n, m=100, model='model1', delta=delta)
        
        pooled = np.vstack((x, y))

        # pooled distance matrix (m+n)*(m+n)
        if space == "distribution":
            dist_pooled = pdist_distribution1d(pooled, metric=metric)

        elif space == "spd":
            dist_pooled = pdist_spd(pooled, metric=metric)

        for j in range(n_methods):
            method = methods[j]
            p_value = permutation_test_homogeneity_from_distance_matrix(m=n, dist=dist_pooled, func=method,
                                                                        method="approximate", num_rounds=nperm,
                                                                        seed=rep)
            elap_time = time.time() - start_time

            p_values[rep - lower, j] = p_value
            elapsed[rep - lower, j] = time.time() - start_time

            print("Delta", delta, ", method ", method, ", repetition ", rep, ", p-value ", p_value, ", elapsed time", elap_time)

# print(np.sum(p_values < 0.05, axis=0) / nreps)
# np.save(path + "/" + str(model) + "/pvalues_" + str(model) + "_" + str(start_idx) + ".npy", p_values)

