
import os
import time
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

SIZE = 1024
REPEATS = 3
MAX_CORES = os.cpu_count() or 1

rng = np.random.default_rng(230103215)
A = rng.random((SIZE, SIZE), dtype=np.float64)
B = rng.random((SIZE, SIZE), dtype=np.float64)

print("TASK 8: Parallel Tile Reduction / Compute Geometry")
print("Matrix size:", SIZE, "x", SIZE)
print("CPU logical cores:", MAX_CORES)
print("Data type: float64")
print()

reference = A @ B


def non_tiled_worker(start, end):
    return start, A[start:end] @ B


def run_non_tiled():
    C = np.empty((SIZE, SIZE), dtype=np.float64)
    row_chunks = np.linspace(
        0, SIZE, min(MAX_CORES, SIZE) + 1, dtype=int
    )

    with ThreadPoolExecutor(max_workers=MAX_CORES) as executor:
        futures = []
        for i in range(len(row_chunks) - 1):
            start = row_chunks[i]
            end = row_chunks[i + 1]
            futures.append(executor.submit(non_tiled_worker, start, end))

        for future in futures:
            start, result = future.result()
            C[start:start + result.shape[0]] = result

    return C


def tiled_worker(row_start, tile_size):
    row_end = min(row_start + tile_size, SIZE)
    rows = row_end - row_start
    result_rows = np.zeros((rows, SIZE), dtype=np.float64)

    for j in range(0, SIZE, tile_size):
        j_end = min(j + tile_size, SIZE)
        output_tile = np.zeros((rows, j_end - j), dtype=np.float64)

        for k in range(0, SIZE, tile_size):
            k_end = min(k + tile_size, SIZE)

            output_tile += (
                A[row_start:row_end, k:k_end]
                @ B[k:k_end, j:j_end]
            )

        result_rows[:, j:j_end] = output_tile

    return row_start, result_rows


def run_tiled(tile_size):
    C = np.empty((SIZE, SIZE), dtype=np.float64)
    row_starts = list(range(0, SIZE, tile_size))

    with ThreadPoolExecutor(max_workers=MAX_CORES) as executor:
        futures = [
            executor.submit(tiled_worker, row_start, tile_size)
            for row_start in row_starts
        ]

        for future in futures:
            start, result = future.result()
            C[start:start + result.shape[0]] = result

    return C


def benchmark(name, function):
    # Warm-up
    result = function()
    if not np.allclose(result, reference, rtol=1e-9, atol=1e-9):
        raise ValueError(name + " produced an incorrect result")

    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median_time = statistics.median(times)

    print("{:<20} {:.4f} seconds".format(name, median_time))
    return median_time


print("Benchmark results")
print("-" * 45)

results = {}
results["Non-tiled"] = benchmark("Non-tiled", run_non_tiled)
results["Tile 16x16"] = benchmark(
    "Tile 16x16", lambda: run_tiled(16)
)
results["Tile 64x64"] = benchmark(
    "Tile 64x64", lambda: run_tiled(64)
)

print()
print("Runtime comparison")
print("-" * 45)

baseline = results["Non-tiled"]

for name, elapsed in results.items():
    speedup = baseline / elapsed
    print(
        "{:<20} Speedup: {:.2f}x".format(name, speedup)
    )

fastest = min(results, key=results.get)

print()
print("Fastest method in this run:", fastest)
print("Cache misses: not directly measured.")
print(
    "Tiling may improve cache locality but actual performance depends on CPU cache, NumPy, and BLAS implementation."
)
