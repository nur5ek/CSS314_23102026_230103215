import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor


STUDENT_ID = 230103215
TASK_COUNT = 10_000
HASH_ROUNDS = 500
PAYLOAD_SIZE = 1024


def cpu_task(task_id):
    payload = (
        str(task_id).encode() + b"x" * PAYLOAD_SIZE
    )[:PAYLOAD_SIZE]

    result = payload

    # Hash the input 500 times.
    for i in range(HASH_ROUNDS):
        result = hashlib.sha256(payload + result[:32]).digest()

    return result


def run_benchmark(pool_size):
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        results = list(executor.map(cpu_task, range(TASK_COUNT)))

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Check that the tasks produced results.
    if len(results) != TASK_COUNT:
        raise RuntimeError("Task verification failed")

    return elapsed_ms


def main():
    cores = os.cpu_count() or 1

    pool_sizes = [
        1,
        max(1, cores // 2),
        cores,
        cores * 2,
        cores * 4,
        cores * 16,
        256
    ]

    # Remove duplicates and sort by pool size.
    pool_sizes = sorted(set(pool_sizes))

    print("TASK 5: THREAD OVERSUBSCRIPTION")
    print("Student ID:", STUDENT_ID)
    print("Logical CPU cores:", cores)
    print("CPU-bound tasks:", TASK_COUNT)
    print("Hash rounds per task:", HASH_ROUNDS)
    print()

    results = []

    for pool_size in pool_sizes:
        print(f"Testing pool size {pool_size}...")

        elapsed_ms = run_benchmark(pool_size)

        results.append({
            "pool": pool_size,
            "time": elapsed_ms
        })

    baseline = results[0]["time"]
    best_time = baseline
    best_pool = results[0]["pool"]
    first_worse_pool = None

    print("\n========== RESULTS ==========")
    print(
        f"{'Pool Size':>10} | "
        f"{'Execution Time (ms)':>20} | "
        f"{'Speedup':>10} | Status"
    )
    print("-" * 68)

    for item in results:
        pool = item["pool"]
        elapsed = item["time"]
        speedup = baseline / elapsed if elapsed > 0 else 0

        if pool == 1:
            status = "BASELINE"
        elif elapsed < best_time:
            status = "NEW BEST"
            best_time = elapsed
            best_pool = pool
        else:
            status = "SLOWER"

            if first_worse_pool is None:
                first_worse_pool = pool

        print(
            f"{pool:10} | "
            f"{elapsed:20.2f} | "
            f"{speedup:9.2f}x | {status}"
        )

    print("\nBest pool size:", best_pool)
    print(f"Best execution time: {best_time:.2f} ms")

    if first_worse_pool is not None:
        print("First observed worsening pool size:", first_worse_pool)
        print(
            "Note: timing alone cannot prove that context switching "
            "caused the slowdown."
        )
    else:
        print("No worsening was observed in these measurements.")

    print("Verification: PASS")


if __name__ == "__main__":
    main()
