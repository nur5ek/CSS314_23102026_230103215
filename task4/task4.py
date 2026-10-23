import time
import threading
from queue import Queue

ITEMS = 1000
WORKERS = 4

def do_work(i):
    time.sleep(0.0001 * (i % 20))


def run_static():
    finish_times = [0.0] * WORKERS

    def worker(worker_id, start, end):
        for i in range(start, end):
            do_work(i)
        finish_times[worker_id] = time.perf_counter()

    start_time = time.perf_counter()
    threads = []

    for worker_id in range(WORKERS):
        start = worker_id * (ITEMS // WORKERS)
        end = (worker_id + 1) * (ITEMS // WORKERS)

        thread = threading.Thread(
            target=worker,
            args=(worker_id, start, end)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_time = time.perf_counter() - start_time
    completion_delta = max(finish_times) - min(finish_times)

    return total_time, completion_delta

def run_dynamic(chunk_size):
    tasks = Queue()
    for start in range(0, ITEMS, chunk_size):
        chunk = list(range(start, min(start + chunk_size, ITEMS)))
        tasks.put(chunk)

    finish_times = [0.0] * WORKERS

    def worker(worker_id):
        while True:
            try:
                chunk = tasks.get_nowait()
            except Exception:
                break

            for i in chunk:
                do_work(i)

        finish_times[worker_id] = time.perf_counter()

    start_time = time.perf_counter()
    threads = []

    for worker_id in range(WORKERS):
        thread = threading.Thread(target=worker, args=(worker_id,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    total_time = time.perf_counter() - start_time
    completion_delta = max(finish_times) - min(finish_times)

    return total_time, completion_delta


def print_result(name, total_time, delta):
    print(
        f"{name:<22} | "
        f"{total_time * 1000:10.2f} ms | "
        f"{delta * 1000:10.2f} ms"
    )


def main():
    print("Task 4: Work Stealing & Scheduling Under Skew")
    print(f"Items: {ITEMS} | Workers: {WORKERS}")
    print("\nStrategy              | Total Time | Completion Delta")
    print("-" * 58)

    static_time, static_delta = run_static()
    print_result("Static chunks", static_time, static_delta)

    dynamic_time, dynamic_delta = run_dynamic(1)
    print_result("Dynamic (chunk=1)", dynamic_time, dynamic_delta)

    dynamic5_time, dynamic5_delta = run_dynamic(5)
    print_result("Dynamic (chunk=5)", dynamic5_time, dynamic5_delta)


if __name__ == "__main__":
    main()
