
import multiprocessing as mp
import os
import platform
import time

ITERATIONS = 50_000_000


def get_physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 1
    except ImportError:
        return max(1, (os.cpu_count() or 1) // 2)


def unpadded_worker(shared_counts, index):
    for i in range(ITERATIONS):
        shared_counts[index] += 1


def local_worker(shared_counts, index):
    count = 0
    for i in range(ITERATIONS):
        count += 1
    
    shared_counts[index] = count

def run_test(worker_function, worker_count):
    shared_counts = mp.RawArray("q", worker_count)

    processes = []
    start_time = time.perf_counter()

    for i in range(worker_count):
        process = mp.Process(
            target=worker_function,
            args=(shared_counts, i)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    elapsed = time.perf_counter() - start_time

    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("A worker process failed.")

    expected = ITERATIONS
    if any(shared_counts[i] != expected for i in range(worker_count)):
        raise RuntimeError("Counter result is incorrect.")

    return elapsed


def main():
    physical_cores = get_physical_cores()

    print("Student ID: 230103215")
    print("Operating System:", platform.platform())
    print("Physical cores:", physical_cores)
    print("Iterations per worker:", ITERATIONS)
    print("Workers:", physical_cores)

    print("\nRunning Unpadded Shared test...")
    unpadded_time = run_test(unpadded_worker, physical_cores)

    print("Running Local Variable test...")
    local_time = run_test(local_worker, physical_cores)

    ratio = unpadded_time / local_time if local_time > 0 else 0

    print("\n========== RESULTS ==========")
    print(f"Unpadded Shared: {unpadded_time:.2f} seconds")
    print(f"Local Variable:  {local_time:.2f} seconds")
    print(f"Execution ratio (Unpadded / Local): {ratio:.2f}x")


if __name__ == "__main__":
    mp.freeze_support()
    main()