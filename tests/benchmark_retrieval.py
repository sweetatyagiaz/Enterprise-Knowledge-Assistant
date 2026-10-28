import gc
import statistics
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ai_core.vector_store as vector_store_module
from ai_core.vector_store import VectorStore


def summarize(label, timings_ms):
    ordered = sorted(timings_ms)
    p95_index = min(
        len(ordered) - 1,
        int(np.ceil(0.95 * len(ordered))) - 1,
    )

    print(f"{label} median: {statistics.median(ordered):.2f} ms")
    print(f"{label} P95: {ordered[p95_index]:.2f} ms")


def run_case(vector_count, dimensions=1536, repetitions=30):
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store_module.get_settings = lambda: SimpleNamespace(
            sqlite_path=Path(temp_dir) / "benchmark.sqlite3"
        )

        rng = np.random.default_rng(42)
        vectors = rng.normal(
            size=(vector_count, dimensions)
        ).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        store = VectorStore()
        store.add_document(
            "benchmark-synthetic.txt",
            [f"Synthetic benchmark chunk {i}" for i in range(vector_count)],
            vectors.tolist(),
        )

        query = vectors[0].tolist()
        del vectors
        gc.collect()

        started = time.perf_counter()
        store.search(query, top_k=5)
        cold_ms = (time.perf_counter() - started) * 1000

        # Warm up numerical-library execution before collecting timings.
        for _ in range(5):
            store.search(query, top_k=5)

        timings_ms = []
        for _ in range(repetitions):
            started = time.perf_counter()
            store.search(query, top_k=5)
            timings_ms.append((time.perf_counter() - started) * 1000)

        print(f"\nVectors: {vector_count:,} | Dimensions: {dimensions}")
        print(f"Cold-cache first search: {cold_ms:.2f} ms")
        summarize("Warm-cache search", timings_ms)


def main():
    print("Vector search scalability baseline")
    for vector_count in (1000, 5000, 10000):
        run_case(vector_count)


if __name__ == "__main__":
    main()
