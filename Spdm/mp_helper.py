import numpy as np
import sys
import time

#sys.path.insert(0, 'd:\\Angle\\Code\\Functions')
sys.path.append('d:\\Angle\\home\\Code\\Functions')
from generate_data import *
from permutation import *
from distance import *


def mp_helper(ran, methods, space, metric, model, delta, n, nperm):
    """ 
    Running a given range of simulations, for given methods and sample size,
    in parallel. 
    Para:
        ran: array [lower, upper] a chunk of repeatments of experiments
        methods: with default ['angle', 'ball', 'energy', 'graph']
        space: 'distribution' or 'spd'
        metric: space='distribution': 'wasserstein' or 'euclidean'
                space='spd': 'frobenius', 'cholesky', 'affineinv'
        model: 'model1' or 'model2'
        delta: float
        r: float > 0.0
        n: int > 0 sample size
        nperm: int > 0 number of permutations with default 399

    Returns:
        p_values : list or numpy array with shape n_reps * n_methods
        elapsed : list or numpy array with shape n_reps * n_methods
    """
    
    lower = ran[0]
    upper = ran[1]
    n_methods = len(methods)
    n_reps = upper - lower

    p_values = np.full((n_reps, n_methods), -1.0)
    elapsed  = np.full((n_reps, n_methods), -1.0)

    for rep in range(lower, upper):
        np.random.seed(rep)
        print(rep)
        start_time = time.time()

        x = generate_data_spd(n=n, d=10, model=model, delta=1.)
        y = generate_data_spd(n=n, d=10, model=model, delta=delta)
        pooled = np.vstack((x, y))

        # pooled distance matrix (m+n)*(m+n)
        if space == "distribution":
            dist_pooled = pdist_distribution1d(pooled, metric=metric)

        elif space == "spd":
            dist_pooled = pdist_spd(pooled, metric=metric)

        for j in range(n_methods):
            method = methods[j]
            p_value = permutation_test_homogeneity_from_distance_matrix(m=n, dist=dist_pooled, func=method, 
                                            method="approximate", num_rounds=nperm, seed=rep)
            elap_time = time.time() - start_time

            p_values[rep-lower, j] = p_value
            elapsed[rep-lower, j] = time.time() - start_time

            print("Method ", method, "\\Repetition ", rep, ", p-value ", p_value, ", elapsed time", elap_time)

    return p_values



# print("----------------------------------Starting-------------------------------------")
# path = "Results"
# print("model:", model, ", delta:", delta, ", r:", r, ", metric", metric, ", n=", n, ", nperm", nperm)
# pathlib.Path(path).mkdir(parents=True, exist_ok=True)

# n_methods = len(methods)

# p_values_all_methods = np.full((nreps, n_methods), -1.0)
# elapsed_all_methods  = np.full((nreps, n_methods), -1.0)

# proc_chunks = []

# print("---------------------------------------------------------------------------")

# Del = nreps // nprocs
# for j in range(nprocs):
#     if j == nprocs-1:
#         proc_chunks.append(( (nprocs-1) * Del, nreps) )

#     else:
#         proc_chunks.append(( (j*Del, (j+1)*Del ) ))

# print(proc_chunks)


# if __name__ ==  '__main__': 
#     with mp.Pool(processes=nprocs) as pool:
#         proc_results = [pool.apply_async(multiprocess_helper, 
#                         args=(chunk, methods, space, metric, model, delta, r, n, nperm))
#                         for chunk in proc_chunks]
#         result_chunks = [r.get() for r in proc_results]

# print(result_chunks)

# for j in range(nprocs):
#     if j == nprocs-1:
#         p_values_all_methods[((nprocs-1)*Del):nreps,] = result_chunks[j][0]
#         elapsed_all_methods [((nprocs-1)*Del):nreps,] = result_chunks[j][1]

#     else:
#         p_values_all_methods[(j*Del):((j+1)*Del),] = result_chunks[j][0]
#         elapsed_all_methods[(j*Del):((j+1)*Del),] = result_chunks[j][1]

# print(np.sum(p_values_all_methods < 0.05, axis=0)/nreps)

# np.save(path + "/pvalues.npy", p_values)
# np.save(path + "/elapsed.npy", elapsed)

