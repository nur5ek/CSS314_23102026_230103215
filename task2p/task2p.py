import math
import random
import threading
import time


STUDENT_ID = 230103215
TOTAL_POINTS = 40_000_000
THREADS = 4
POINTS_PER_THREAD = TOTAL_POINTS // THREADS

shared_rng = random.Random(STUDENT_ID)
rng_lock = threading.Lock()


def shared_worker(thread_index, results):
    inside = 0

    for i in range(POINTS_PER_THREAD):
        with rng_lock:
            x = shared_rng.random()
            y = shared_rng.random()

        if x * x + y * y <= 1.0:
            inside += 1

    results[thread_index] = inside


def independent_worker(thread_index, results):
    seed = (STUDENT_ID * 10) + thread_index
    rng = random.Random(seed)

    inside = 0

    for i in range(POINTS_PER_THREAD):
        x = rng.random()
        y = rng.random()

        if x * x + y * y <= 1.0:
            inside += 1

    results[thread_index] = inside


def run_shared():
    results = [0] * THREADS
    threads = []

    start = time.perf_counter()

    for i in range(THREADS):
        thread = threading.Thread(
            target=shared_worker,
            args=(i, results)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_value = 4.0 * sum(results) / TOTAL_POINTS

    return pi_value, elapsed_ms


def run_independent():
    results = [0] * THREADS
    threads = []

    start = time.perf_counter()

    for i in range(THREADS):
        thread = threading.Thread(
            target=independent_worker,
            args=(i, results)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed_ms = (time.perf_counter() - start) * 1000
    pi_value = 4.0 * sum(results) / TOTAL_POINTS

    return pi_value, elapsed_ms


def main():
    print("TASK 2: PRNG STATE ISOLATION")
    print("Student ID:", STUDENT_ID)
    print("Sample points:", f"{TOTAL_POINTS:,}")
    print("Threads:", THREADS)

    print("\nRunning Shared PRNG...")
    pi_shared, time_shared = run_shared()
    error_shared = abs(pi_shared - math.pi)

    print("\nRunning Independent PRNG...")
    pi_independent, time_independent = run_independent()
    error_independent = abs(pi_independent - math.pi)

    speedup = (
        time_shared / time_independent
        if time_independent > 0
        else 0
    )

    print("\n========== RESULTS ==========")

    print("\nShared PRNG:")
    print(f"Computed Pi: {pi_shared:.10f}")
    print(f"Absolute Error: {error_shared:.10f}")
    print(f"Wall Time: {time_shared:.2f} ms")

    print("\nIndependent PRNG:")
    print(f"Computed Pi: {pi_independent:.10f}")
    print(f"Absolute Error: {error_independent:.10f}")
    print(f"Wall Time: {time_independent:.2f} ms")

    print(f"\nSpeedup Factor: {speedup:.2f}x")


if __name__ == "__main__":
    main()
