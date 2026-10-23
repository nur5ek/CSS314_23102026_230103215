
import os
import time
import platform
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

STUDENT_ID = 230103215
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

A = 2.5
B = 1.5
REPEATS = 5


def serial_operation(x, y):
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def parallel_chunk(args):
    x_part, y_part = args
    np.multiply(x_part, A, out=y_part)
    np.add(y_part, B, out=y_part)


def measure_serial(x, y):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        serial_operation(x, y)
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def measure_parallel(x, y, executor, workers):
    chunks_x = np.array_split(x, workers)
    chunks_y = np.array_split(y, workers)
    tasks = list(zip(chunks_x, chunks_y))

    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()
        list(executor.map(parallel_chunk, tasks))
        times.append(time.perf_counter() - start)

    return statistics.median(times)


def main():
    workers = os.cpu_count() or 1

    print("Task 5: The Overhead Crossover Point")
    print("Student ID:", STUDENT_ID)
    print("Operating System:", platform.platform())
    print("Logical cores:", workers)
    print("Workers used:", workers)
    print("Repeats per test:", REPEATS)

    results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        print("\nN | Serial (ms) | Parallel (ms) | Speedup")
        print("-" * 55)

        for n in SIZES:
            x = np.arange(n, dtype=np.float64)
            y_serial = np.empty_like(x)
            y_parallel = np.empty_like(x)

            serial_operation(x, y_serial)
            measure_parallel(x, y_parallel, executor, workers)

            serial_time = measure_serial(x, y_serial)
            parallel_time = measure_parallel(
                x, y_parallel, executor, workers
            )

            speedup = serial_time / parallel_time

            if not np.allclose(y_serial, y_parallel):
                raise RuntimeError("Results do not match!")

            results.append({
                "n": n,
                "serial": serial_time,
                "parallel": parallel_time,
                "speedup": speedup
            })

            print(
                f"{n:>10} | "
                f"{serial_time * 1000:>11.3f} | "
                f"{parallel_time * 1000:>13.3f} | "
                f"{speedup:>7.3f}x"
            )

    print("\n========== CROSSOVER ANALYSIS ==========")

    slower_sizes = [
        r["n"] for r in results
        if r["parallel"] >= r["serial"]
    ]
    faster_sizes = [
        r["n"] for r in results
        if r["parallel"] < r["serial"]
    ]

    transitions = []

    for i in range(1, len(results)):
        previous = results[i - 1]
        current = results[i]

        if (
            previous["parallel"] >= previous["serial"]
            and current["parallel"] < current["serial"]
        ):
            transitions.append((previous["n"], current["n"]))

    if transitions:
        for low, high in transitions:
            print(
                f"Crossover detected between N={low:,} "
                f"and N={high:,}"
            )
    elif faster_sizes and slower_sizes:
        print(
            "parallel performance changes across the tested sizes but no adjacent also no slower to faster transition was detected."
        )
        print("Slower sizes:", slower_sizes)
        print("Faster sizes:", faster_sizes)
    elif faster_sizes:
        print(
            "Parallel was faster at every tested size. "
            "Crossover is at smallest tested size or below the size."
        )
    else:
        print(
            "Parallel was not faster at any tested size. "
            "Crossover was not found in the tested range."
        )

    print("\nConclusion:")
    if faster_sizes and slower_sizes:
        print("Slowest-to-fastest transition is hardware-dependent.")
        print("Largest tested size where parallel was slower:",
              max(slower_sizes))
        print("Smallest tested size where parallel was faster:",
              min(faster_sizes))
    elif faster_sizes:
        print("In all tested sizes parallel execution wins.")
    else:
        print("Serial execution wins or ties for all tested sizes.")


if __name__ == "__main__":
    main()
