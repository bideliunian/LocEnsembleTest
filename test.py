
# %%
import multiprocessing as mp
from double import *

mproc_results = []
def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    mproc_results.append(result)

if __name__ ==  '__main__': 
    nprocs = 3
    pool = mp.Pool(processes=nprocs)
    for i in range(0, 3):
        pool.apply_async(double, args = (i,2), callback=log_result)
    pool.close()
    pool.join()
    print(mproc_results)
# %%
