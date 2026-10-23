
import os
import time
import math
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

ARRAY_SIZE = 30_000_000
REPEATS = 3

MAX_CORES = os.cpu_count() or 1
CORE_COUNTS = sorted(set([1, 2, MAX_CORES]))

print("TASK 7: Arithmetic Intensity vs Memory Bottlenecks")
print("Array size:", ARRAY_SIZE)
print("Logical cores:", MAX_CORES)
print("Data type: float64")
print()

rng = np.random.default_rng(230103215)
A = rng.uniform(0.1, 1.0, ARRAY_SIZE).astype(np.float64)
B = np.empty_like(A)


def memory_kernel(start, end):
    B[start:end] = A[start:end]


def compute_kernel(start, end):
    values = A[start:end]
    B[start:end] = (
        np.sin(values) * np.cos(values) + np.tan(values)
    )


def run_kernel(kernel, cores):
    boundaries = np.linspace(0, ARRAY_SIZE, cores + 1, dtype=int)

    start_time = time.perf_counter()

    if cores == 1:
        kernel(0, ARRAY_SIZE)
    else:
        with ThreadPoolExecutor(max_workers=cores) as executor:
            futures = []
            for i in range(cores):
                start = boundaries[i]
                end = boundaries[i + 1]
                futures.append(executor.submit(kernel, start, end))

            for future in futures:
                future.result()

    return time.perf_counter() - start_time


def benchmark(name, kernel):
    results = {}
    run_kernel(kernel, 1)

    print(name)
    print("-" * 65)
    print("{:<10} {:<15} {:<15} {:<15}".format(
        "Cores", "Time (s)", "Speedup", "Efficiency"
    ))

    for cores in CORE_COUNTS:
        times = []

        for _ in range(REPEATS):
            elapsed = run_kernel(kernel, cores)
            times.append(elapsed)

        elapsed = statistics.median(times)
        results[cores] = elapsed

    baseline = results[1]

    for cores in CORE_COUNTS:
        elapsed = results[cores]
        speedup = baseline / elapsed
        efficiency = speedup / cores

        print("{:<10} {:<15.4f} {:<15.2f} {:<15.2%}".format(
            cores, elapsed, speedup, efficiency
        ))

    print()
    return results


memory_results = benchmark("KERNEL A: MEMORY-BOUND (Vector Copy)", memory_kernel)
compute_results = benchmark("KERNEL B: COMPUTE-BOUND (sin*cos + tan)", compute_kernel)

print("SCALING ANALYSIS")
print("=" * 65)

memory_speedup = memory_results[1] / memory_results[MAX_CORES]
compute_speedup = compute_results[1] / compute_results[MAX_CORES]

print("Memory kernel speedup at max cores: {:.2f}x".format(memory_speedup))
print("Compute kernel speedup at max cores: {:.2f}x".format(compute_speedup))

if memory_speedup < compute_speedup:
    print(
        "Conclusion: Kernel A scaling are weaker. This is consistent with possible RAM bandwidth saturation."
    )
else:
    print(
        "Conclusion: Kernel A did not show weaker scaling than Kernel B in this run. RAM bandwidth saturation was not demonstrated clearly."
    )

