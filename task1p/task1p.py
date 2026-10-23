
import os
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


SIZE = 8000


def get_physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 1
    except ImportError:
        return os.cpu_count() or 1


def calculate_sum(args):
    shm_name, start, end, mode = args

    shm = shared_memory.SharedMemory(name=shm_name)
    matrix = np.ndarray(
        (SIZE, SIZE),
        dtype=np.float64,
        buffer=shm.buf
    )

    if mode == "row":
        result = np.sum(matrix[start:end, :])

    else:
        result = 0.0
        for col in range(start, end):
            result += np.sum(matrix[:, col])

    shm.close()
    return result


def run_benchmark(pool, shm_name, workers, mode):
    if mode == "row":
        total = SIZE
    else:
        total = SIZE

    chunk_size = (total + workers - 1) // workers

    tasks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        tasks.append((shm_name, start, end, mode))

    start_time = time.perf_counter()

    results = pool.map(calculate_sum, tasks)

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return sum(results), elapsed_ms


def main():
    student_id = "230103215"

    if not student_id.isdigit():
        print("Student ID must contain digits only.")
        return

    seed = int(student_id)
    logical_cores = os.cpu_count() or 1
    physical_cores = get_physical_cores()
    workers = min(physical_cores, SIZE)

    print("Creating 8000 x 8000 matrix...")
    rng = np.random.default_rng(seed)
    matrix = rng.random((SIZE, SIZE), dtype=np.float64)

    shm = shared_memory.SharedMemory(
        create=True,
        size=matrix.nbytes
    )

    shared_matrix = np.ndarray(
        matrix.shape,
        dtype=np.float64,
        buffer=shm.buf
    )
    shared_matrix[:] = matrix
    del shared_matrix
    del matrix

    output = []

    try:
        with mp.Pool(processes=workers) as pool:
            print("Running Mode A: Row-Major...")
            sum_a, time_a = run_benchmark(
                pool, shm.name, workers, "row"
            )

            print("Running Mode B: Column-Major...")
            sum_b, time_b = run_benchmark(
                pool, shm.name, workers, "column"
            )

        penalty = time_b / time_a if time_a > 0 else 0

        output.append("TASK 1: SPATIAL CACHE PROFILING")
        output.append(f"Student ID: {student_id}")
        output.append(f"Matrix size: {SIZE} x {SIZE}")
        output.append(f"Logical cores: {logical_cores}")
        output.append(f"Physical cores: {physical_cores}")
        output.append(f"Workers: {workers}")
        output.append(f"Mode A (Row-Major): {time_a:.2f} ms")
        output.append(f"Mode B (Column-Major): {time_b:.2f} ms")
        output.append(f"Cache penalty factor: {penalty:.2f}x")
        output.append(f"Mode A sum: {sum_a:.6f}")
        output.append(f"Mode B sum: {sum_b:.6f}")

        print("\n" + "\n".join(output))

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
