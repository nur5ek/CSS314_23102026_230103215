import os
import time
import random
import threading


STUDENT_ID = 230103215
ARRAY_SIZE = 2_000_000
THREADS = 4


def serial_prefix_sum(data):
    output = [0] * len(data)
    output[0] = data[0]

    for i in range(1, len(data)):
        output[i] = output[i - 1] + data[i]

    return output


def calculate_chunk(data, output, start, end, chunk_index, chunk_sums):
    total = 0

    for i in range(start, end):
        total += data[i]
        output[i] = total

    chunk_sums[chunk_index] = total


def apply_offset(output, start, end, offset):
    for i in range(start, end):
        output[i] += offset


def parallel_prefix_sum(data):
    size = len(data)
    output = [0] * size
    chunk_sums = [0] * THREADS
    threads = []
    chunk_size = (size + THREADS - 1) // THREADS
    start_time = time.perf_counter()

    for chunk_index in range(THREADS):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, size)

        if start >= size:
            break

        thread = threading.Thread(
            target=calculate_chunk,
            args=(
                data, output, start, end,
                chunk_index, chunk_sums
            )
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    offsets = [0] * THREADS
    running_total = 0

    for i in range(THREADS):
        offsets[i] = running_total
        running_total += chunk_sums[i]

    threads = []

    for chunk_index in range(THREADS):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, size)

        if start >= size:
            break

        if offsets[chunk_index] == 0:
            continue

        thread = threading.Thread(
            target=apply_offset,
            args=(
                output, start, end,
                offsets[chunk_index]
            )
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    return output, elapsed_ms


def main():
    rng = random.Random(STUDENT_ID)
    data = [rng.randint(1, 10) for _ in range(ARRAY_SIZE)]

    print("TASK 3: PREFIX SCAN")
    print("Student ID:", STUDENT_ID)
    print("Array size:", f"{ARRAY_SIZE:,}")
    print("Threads:", THREADS)

    print("\nRunning serial prefix sum...")
    start = time.perf_counter()
    serial_output = serial_prefix_sum(data)
    serial_time = (time.perf_counter() - start) * 1000

    print("Running parallel prefix scan...")
    parallel_output, parallel_time = parallel_prefix_sum(data)

    try:
        assert parallel_output[-1] == serial_output[-1], (
            "Parallel and serial final sums do not match!"
        )
    except AssertionError as error:
        print("Verification Status: FAIL")
        print(error)
        raise SystemExit(1)

    speedup = serial_time / parallel_time if parallel_time > 0 else 0

    print("\n========== RESULTS ==========")
    print(f"Serial Time: {serial_time:.2f} ms")
    print(f"Parallel Time: {parallel_time:.2f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print("Verification Status: PASS")
    print("Serial final sum:", serial_output[-1])
    print("Parallel final sum:", parallel_output[-1])


if __name__ == "__main__":
    main()
