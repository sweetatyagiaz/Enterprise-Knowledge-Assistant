import json
import sqlite3
from pathlib import Path

import numpy as np

from ai_core.settings import get_settings


class VectorStore:
    """SQLite-backed storage and cosine-similarity search for embeddings."""

    def __init__(self):
        settings = get_settings()
        self.db_path = Path(settings.sqlite_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(str(self.db_path))
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self):
        with self._connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    chunk_text TEXT NOT NULL,
                    embedding TEXT NOT NULL
                )
            """)

    def add_document(
        self,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding.")

        rows = [
            (filename, chunk, json.dumps(vector))
            for chunk, vector in zip(chunks, embeddings)
        ]

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO document_chunks
                    (filename, chunk_text, embedding)
                VALUES (?, ?, ?)
                """,
                rows,
            )

        return len(rows)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        query = np.asarray(query_vector, dtype=np.float32)
        query_norm = np.linalg.norm(query)

        if query_norm == 0:
            return []

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, filename, chunk_text, embedding "
                "FROM document_chunks"
            ).fetchall()

        results = []

        for row in rows:
            vector = np.asarray(
                json.loads(row["embedding"]),
                dtype=np.float32,
            )

            if vector.shape != query.shape:
                continue

            norm = np.linalg.norm(vector)
            if norm == 0:
                continue

            score = float(np.dot(query, vector) / (query_norm * norm))

            results.append({
                "id": row["id"],
                "filename": row["filename"],
                "text": row["chunk_text"],
                "score": score,
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:max(1, top_k)]

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT filename, COUNT(*) AS chunk_count
                FROM document_chunks
                GROUP BY filename
                ORDER BY filename
            """).fetchall()

        return [dict(row) for row in rows]

    def count_chunks(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS total FROM document_chunks"
            ).fetchone()

        return int(row["total"])