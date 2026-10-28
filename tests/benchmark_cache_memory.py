import gc
import tempfile
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ai_core.vector_store as vector_store_module
from ai_core.vector_store import VectorStore


def run_case(vector_count, dimensions=1536):
    with tempfile.TemporaryDirectory() as temp_dir:
        vector_store_module.get_settings = lambda: SimpleNamespace(
            sqlite_path=Path(temp_dir) / "memory_benchmark.sqlite3"
        )

        rng = np.random.default_rng(42)
        vectors = rng.normal(
            size=(vector_count, dimensions)
        ).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

        store = VectorStore()
        store.add_document(
            "memory-benchmark.txt",
            [f"Synthetic chunk {i}" for i in range(vector_count)],
            vectors.tolist(),
        )

        del vectors
        gc.collect()

        tracemalloc.start()
        before, _ = tracemalloc.get_traced_memory()

        store.search([1.0] + [0.0] * (dimensions - 1), top_k=5)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(
            f"{vector_count:>6} vectors | "
            f"current cache allocation: {(current - before) / (1024 ** 2):.2f} MiB | "
            f"peak allocation: {(peak - before) / (1024 ** 2):.2f} MiB"
        )


def main():
    print("Cache memory benchmark (1536 dimensions per vector)")
    for count in (1000, 5000, 10000):
        run_case(count)


if __name__ == "__main__":
    main()
