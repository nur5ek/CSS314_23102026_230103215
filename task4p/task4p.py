
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np


STUDENT_ID = 230103215
ARRAY_SIZES = [100_000, 1_000_000, 10_000_000]


def queue_worker(data_queue, result_pipe):
    data = data_queue.get()
    result_pipe.send(int(np.sum(data)))
    result_pipe.close()


def shared_memory_worker(shm_name, size, result_pipe):
    shm = shared_memory.SharedMemory(name=shm_name)

    try:
        data = np.ndarray(
            (size,),
            dtype=np.int64,
            buffer=shm.buf
        )
        result_pipe.send(int(np.sum(data)))
    finally:
        shm.close()
        result_pipe.close()


def benchmark_queue(data):
    data_queue = mp.Queue()
    parent_pipe, child_pipe = mp.Pipe(duplex=False)

    process = mp.Process(
        target=queue_worker,
        args=(data_queue, child_pipe)
    )

    start = time.perf_counter()

    process.start()
    child_pipe.close()

    data_queue.put(data)
    result = parent_pipe.recv()

    process.join()
    elapsed_ms = (time.perf_counter() - start) * 1000

    data_queue.close()
    data_queue.join_thread()
    parent_pipe.close()

    if process.exitcode != 0:
        raise RuntimeError("Queue worker failed")

    return elapsed_ms, result


def benchmark_shared_memory(data):
    shm = shared_memory.SharedMemory(
        create=True,
        size=data.nbytes
    )

    parent_pipe, child_pipe = mp.Pipe(duplex=False)

    process = mp.Process(
        target=shared_memory_worker,
        args=(shm.name, len(data), child_pipe)
    )

    try:
        shared_array = np.ndarray(
            data.shape,
            dtype=np.int64,
            buffer=shm.buf
        )

        start = time.perf_counter()

        process.start()
        child_pipe.close()

        # Copy data into shared memory.
        shared_array[:] = data

        result = parent_pipe.recv()
        process.join()

        elapsed_ms = (time.perf_counter() - start) * 1000

        if process.exitcode != 0:
            raise RuntimeError("Shared memory worker failed")

        return elapsed_ms, result

    finally:
        if process.is_alive():
            process.terminate()
            process.join()

        parent_pipe.close()
        shm.close()
        shm.unlink()


def main():
    rng = np.random.default_rng(STUDENT_ID)

    print("TASK 4: MEMORY TRANSFER PROFILE")
    print("Student ID:", STUDENT_ID)
    print("Method A: IPC Queue with serialization")
    print("Method B: Shared Memory")
    print()

    print(
        f"{'Array Size':>12} | "
        f"{'Queue (ms)':>12} | "
        f"{'Shared Memory (ms)':>18} | "
        f"{'Queue / Shared':>14}"
    )
    print("-" * 68)

    for size in ARRAY_SIZES:
        data = rng.integers(
            1, 101,
            size=size,
            dtype=np.int64
        )

        queue_time, queue_sum = benchmark_queue(data)
        shared_time, shared_sum = benchmark_shared_memory(data)

        if queue_sum != shared_sum:
            raise AssertionError("Transfer verification failed!")

        ratio = (
            queue_time / shared_time
            if shared_time > 0
            else 0
        )

        print(
            f"{size:12,} | "
            f"{queue_time:12.2f} | "
            f"{shared_time:18.2f} | "
            f"{ratio:13.2f}x"
        )

        del data

    print("\nVerification: PASS")
    print("All transferred array sums match.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
