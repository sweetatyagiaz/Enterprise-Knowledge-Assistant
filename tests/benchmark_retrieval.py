import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ai_core.vector_store as vector_store_module
from ai_core.vector_store import VectorStore


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store_module.get_settings = lambda: SimpleNamespace(
            sqlite_path=Path(temp_dir) / 'benchmark.sqlite3'
        )
        run_benchmark()


def run_benchmark():
    dimensions = 1536
    vector_count = 1000
    repetitions = 30

    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(vector_count, dimensions)).astype(
        np.float32
    )
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query = vectors[0].tolist()

    store = VectorStore()
    store.add_document(
        "benchmark-synthetic.txt",
        [f"Synthetic benchmark chunk {i}" for i in range(vector_count)],
        vectors.tolist(),
    )

    # Warm-up run, excluded from measurements.
    store.search(query, top_k=5)

    timings_ms = []
    for _ in range(repetitions):
        started = time.perf_counter()
        store.search(query, top_k=5)
        timings_ms.append((time.perf_counter() - started) * 1000)

    timings_ms.sort()
    p95_index = min(
        len(timings_ms) - 1,
        int(np.ceil(0.95 * len(timings_ms))) - 1,
    )

    print(f"Vectors: {vector_count}")
    print(f"Dimensions: {dimensions}")
    print(f"Repetitions: {repetitions}")
    print(f"Median search latency: {statistics.median(timings_ms):.2f} ms")
    print(f"P95 search latency: {timings_ms[p95_index]:.2f} ms")


if __name__ == "__main__":
    main()
