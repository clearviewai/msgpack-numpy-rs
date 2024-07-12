import numpy as np
import msgpack
import msgpack_numpy as m
import time

m.patch()


def bench_serialize(value, iterations):
    start_time = time.time()
    for _ in range(iterations):
        msgpack.packb(value)
    total_time = time.time() - start_time

    return total_time / iterations * 1000


def bench_deserialize(value, iterations):
    packed = msgpack.packb(value)

    start_time = time.time()
    for _ in range(iterations):
        msgpack.unpackb(packed)
    total_time = time.time() - start_time

    return total_time / iterations * 1000


if __name__ == "__main__":
    iterations = 10  # Adjust this number as needed

    # large arrays
    arr = np.arange(1000, dtype=np.float16)
    arrs = [arr for _ in range(10000)]

    # small but many arrays
    # arr = np.arange(100, dtype=np.float16)
    # arrs = [arr for _ in range(100000)]

    print(f"Running benchmarks with {iterations} iterations each...")

    serialize_avg = bench_serialize(arrs, iterations)
    print(f"Serialize average: {serialize_avg:.3f} ms")

    deserialize_avg = bench_deserialize(arrs, iterations)
    print(f"Deserialize average: {deserialize_avg:.3f} ms")
