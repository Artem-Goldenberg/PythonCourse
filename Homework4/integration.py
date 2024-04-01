import math
import os
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def sumUp(f, a, b, iters):
    acc = 0
    step = (b - a) / iters
    for i in range(iters):
        acc += f(a + i * step) * step
    return acc

def integrate(f, a, b, *, n_jobs=1, PoolCls, n_iter=10000000):
    iterPerJob = n_iter // n_jobs

    interval = (b - a) / n_jobs
    result = 0
    with PoolCls(max_workers=n_jobs) as pool:
        futures = []
        for i in range(n_jobs):
            a1 = a + i * interval
            b1 = a + (i + 1) * interval
            print(f"Starting job in [{a1}; {b1}]")
            r = pool.submit(sumUp, f, a1, b1, iterPerJob)
            futures.append(r)
        for r in futures:
            result += r.result()
    return result


if __name__ == "__main__":
    ncores = os.cpu_count()

    print("Thread Executor Pool:")
    for njobs in range(1, ncores * 2 + 1):
        start = timer()
        r = integrate(math.cos, 0, math.pi / 2, n_jobs=njobs, PoolCls=ThreadPoolExecutor)
        end = timer()
        print(f"{njobs}: {end - start}s, result = {r}\n")

    print("\n\nProcess Executor Pool:")
    for njobs in range(1, ncores * 2 + 1):
        start = timer()
        r = integrate(math.cos, 0, math.pi / 2, n_jobs=njobs, PoolCls=ProcessPoolExecutor)
        end = timer()
        print(f"njobs: {njobs}, time: {end - start}s, result: {r}\n")