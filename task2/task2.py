
import os
import time
import platform
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103215
N = 20_000_000
P = 0.95


def calculate_sum(chunk):
    return np.sum(chunk, dtype=np.float64)


def main():
    print("Student ID:", STUDENT_ID)
    print("Operating System:", platform.platform())
    print("Logical cores:", os.cpu_count())

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)

    max_cores = os.cpu_count()
    worker_counts = sorted(set([1, 2, 4, max_cores, max_cores * 2]))

    results = []

    for workers in worker_counts:
        chunks = np.array_split(data, workers)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(calculate_sum, chunks))

        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            partial_sums = list(
                executor.map(calculate_sum, chunks)
            )

        total_sum = sum(partial_sums)
        elapsed = time.perf_counter() - start_time

        results.append({
            "workers": workers,
            "time": elapsed,
            "sum": total_sum
        })

    baseline_time = results[0]["time"]

    print("\nArray size:", N)
    print("Data type: float64")
    print("Expected sum:", results[0]["sum"])
    print("\nCores | Wall Time (ms) | Speedup | Amdahl Upper Bound")
    print("-" * 62)

    for result in results:
        workers = result["workers"]
        elapsed = result["time"]
        speedup = baseline_time / elapsed
        amdahl = 1 / ((1 - P) + P / workers)

        print(
            f"{workers:5d} | "
            f"{elapsed * 1000:14.2f} | "
            f"{speedup:7.2f} | "
            f"{amdahl:18.2f}"
        )


if __name__ == "__main__":
    main()