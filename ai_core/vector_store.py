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
        self._embedding_cache = None
        self._cache_revision = None
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

            connection.execute("""
                CREATE TABLE IF NOT EXISTS vector_store_metadata (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    revision INTEGER NOT NULL DEFAULT 0
                )
            """)

            connection.execute("""
                INSERT OR IGNORE INTO vector_store_metadata (id, revision)
                VALUES (1, 0)
            """)

            for operation in ("INSERT", "UPDATE", "DELETE"):
                connection.execute(f"""
                    CREATE TRIGGER IF NOT EXISTS
                    document_chunks_revision_{operation.lower()}
                    AFTER {operation} ON document_chunks
                    BEGIN
                        UPDATE vector_store_metadata
                        SET revision = revision + 1
                        WHERE id = 1;
                    END
                """)

    @staticmethod
    def _validate_vector(vector, label="Embedding"):
        try:
            array = np.asarray(vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must contain valid numeric values.") from exc

        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{label} must be a non-empty one-dimensional vector.")

        if not np.isfinite(array).all():
            raise ValueError(f"{label} must contain only finite numeric values.")

        if np.linalg.norm(array) == 0:
            raise ValueError(f"{label} must not be a zero vector.")

        return array

    def add_document(
        self,
        filename: str,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have one embedding.")

        rows = []
        for chunk, vector in zip(chunks, embeddings):
            validated = self._validate_vector(vector)
            rows.append((filename, chunk, json.dumps(validated.tolist())))

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO document_chunks
                    (filename, chunk_text, embedding)
                VALUES (?, ?, ?)
                """,
                rows,
            )

        self._embedding_cache = None
        self._cache_revision = None
        return len(rows)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        try:
            query_array = np.asarray(query_vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Query vector must contain valid numeric values."
            ) from exc

        if query_array.ndim != 1 or query_array.size == 0:
            raise ValueError(
                "Query vector must be a non-empty one-dimensional vector."
            )

        if not np.isfinite(query_array).all():
            raise ValueError(
                "Query vector must contain only finite numeric values."
            )

        if np.linalg.norm(query_array) == 0:
            return []

        query = query_array
        query_norm = np.linalg.norm(query)

        with self._connect() as connection:
            revision = connection.execute(
                "SELECT revision FROM vector_store_metadata WHERE id = 1"
            ).fetchone()[0]

            if (
                self._embedding_cache is None
                or self._cache_revision != revision
            ):
                rows = connection.execute(
                    "SELECT id, filename, chunk_text, embedding "
                    "FROM document_chunks"
                ).fetchall()

                cached_rows = []
                cached_vectors = []

                for row in rows:
                    vector = np.asarray(
                        json.loads(row["embedding"]),
                        dtype=np.float32,
                    )

                    if vector.ndim != 1 or vector.size == 0:
                        continue

                    if not np.isfinite(vector).all():
                        continue

                    norm = np.linalg.norm(vector)
                    if norm == 0:
                        continue

                    cached_rows.append({
                        "id": row["id"],
                        "filename": row["filename"],
                        "text": row["chunk_text"],
                    })
                    cached_vectors.append((vector, norm))

                self._embedding_cache = (
                    cached_rows,
                    cached_vectors,
                )
                self._cache_revision = revision

        cached_rows, cached_vectors = self._embedding_cache
        results = []

        for row, (vector, norm) in zip(cached_rows, cached_vectors):
            if vector.shape != query.shape:
                continue

            score = float(np.dot(query, vector) / (query_norm * norm))

            results.append({
                **row,
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
