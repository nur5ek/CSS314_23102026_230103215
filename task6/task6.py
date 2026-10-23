
import multiprocessing as mp
import time
import platform

THREADS = 4
INCREMENTS = 2_000_000
EXPECTED = THREADS * INCREMENTS


def increment_without_lock(counter, iterations):
    for _ in range(iterations):
        counter.value += 1


def increment_with_lock(counter, iterations):
    for _ in range(iterations):
        with counter.get_lock():
            counter.value += 1


def run_test(worker_function, use_lock):
    if use_lock:
        counter = mp.Value("q", 0, lock=True)
    else:
        counter = mp.Value("q", 0, lock=False)

    processes = []
    start_time = time.perf_counter()

    for _ in range(THREADS):
        process = mp.Process(
            target=worker_function,
            args=(counter, INCREMENTS)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

    elapsed = time.perf_counter() - start_time

    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError("A worker process failed.")

    return counter.value, elapsed


def main():
    print("Task 6: Synchronization Cost & Data Loss")
    print("Operating System:", platform.platform())
    print("Workers:", THREADS)
    print("Increments per worker:", f"{INCREMENTS:,}")
    print("Expected final value:", f"{EXPECTED:,}")

    print("\nRunning without synchronization...")
    actual_unlocked, time_unlocked = run_test(
        increment_without_lock, False
    )

    lost_updates = EXPECTED - actual_unlocked
    lost_percentage = (lost_updates / EXPECTED) * 100

    print("\n--- Without Lock ---")
    print("Actual final value:", f"{actual_unlocked:,}")
    print("Lost updates:", f"{lost_updates:,}")
    print("Lost updates percentage:", f"{lost_percentage:.4f}%")
    print("Wall-clock time:", f"{time_unlocked:.3f} seconds")

    print("\nRunning with Lock...")
    actual_locked, time_locked = run_test(
        increment_with_lock, True
    )

    print("\n--- With Lock ---")
    print("Actual final value:", f"{actual_locked:,}")
    print("Wall-clock time:", f"{time_locked:.3f} seconds")

    penalty = time_locked - time_unlocked
    penalty_percent = (
        (penalty / time_unlocked) * 100
        if time_unlocked > 0 else 0
    )

    print("\n--- Synchronization Cost ---")
    print("Expected final value:", f"{EXPECTED:,}")
    print("Correct result:", actual_locked == EXPECTED)
    print("Time penalty:", f"{penalty:.3f} seconds")
    print("Time penalty percentage:", f"{penalty_percent:.2f}%")


if __name__ == "__main__":
    mp.freeze_support()
    main()
