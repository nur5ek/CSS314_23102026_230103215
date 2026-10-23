import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import platform
import time
import numpy as np

STUDENT_ID= "230103215"
N = 20_000_000


def main():
    if not STUDENT_ID.isdigit():
        raise ValueError("Replace YOUR_STUDENT_ID with your numeric Student ID.")

    seed = int(STUDENT_ID)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Operating System: {platform.platform()}")

    cpu_model = platform.processor()
    if not cpu_model:
        cpu_model = platform.uname().processor or "Not available"
    print(f"CPU model / platform details: {cpu_model}")

    logical_cores = os.cpu_count()
    print(f"Logical cores detected: {logical_cores}")

    start_time = time.perf_counter()

    rng = np.random.default_rng(seed)
    data = rng.random(N, dtype=np.float64)

    total_sum = np.sum(data, dtype=np.float64)
    standard_deviation = np.std(data, dtype=np.float64, ddof=0)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    print(f"Array size: {N}")
    print("Data type: float64")
    print(f"Sum: {total_sum:.6f}")
    print(f"Standard deviation: {standard_deviation:.6f}")
    print(f"Wall-clock execution time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
