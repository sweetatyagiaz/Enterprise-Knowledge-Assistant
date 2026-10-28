from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from types import SimpleNamespace

import ai_core.vector_store as vector_store_module
from ai_core.vector_store import VectorStore


def test_search_remains_valid_during_concurrent_inserts(
    tmp_path, monkeypatch
):
    database_path = tmp_path / "concurrent.sqlite3"
    monkeypatch.setattr(
        vector_store_module,
        "get_settings",
        lambda: SimpleNamespace(sqlite_path=database_path),
    )

    reader_store = VectorStore()
    writer_store = VectorStore()

    reader_store.add_document(
        "initial.txt",
        ["Initial chunk"],
        [[1.0, 0.0]],
    )

    start = Barrier(2)

    def search_repeatedly():
        start.wait()
        counts = []
        for _ in range(100):
            results = reader_store.search([1.0, 0.0], top_k=100)
            counts.append(len(results))
            assert all("score" in item for item in results)
        return counts

    def insert_repeatedly():
        start.wait()
        for i in range(20):
            writer_store.add_document(
                f"document_{i}.txt",
                [f"Concurrent chunk {i}"],
                [[1.0, 0.0]],
            )

    with ThreadPoolExecutor(max_workers=2) as executor:
        search_future = executor.submit(search_repeatedly)
        insert_future = executor.submit(insert_repeatedly)

        observed_counts = search_future.result()
        insert_future.result()

    final_results = reader_store.search([1.0, 0.0], top_k=100)

    assert all(count >= 1 for count in observed_counts)
    assert len(final_results) == 21
