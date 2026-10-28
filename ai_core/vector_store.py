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

            # Add the binary column to existing databases without deleting data.
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(document_chunks)"
                ).fetchall()
            }
            if "embedding_blob" not in columns:
                connection.execute(
                    "ALTER TABLE document_chunks ADD COLUMN embedding_blob BLOB"
                )

            # Legacy SQL updates must invalidate the binary copy for that row.
            connection.execute("""
                CREATE TRIGGER IF NOT EXISTS
                document_chunks_clear_blob_after_embedding_update
                AFTER UPDATE OF embedding ON document_chunks
                WHEN NEW.embedding IS NOT OLD.embedding
                BEGIN
                    UPDATE document_chunks
                    SET embedding_blob = NULL
                    WHERE id = NEW.id;
                END
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
            rows.append((
                filename,
                chunk,
                json.dumps(validated.tolist()),
                sqlite3.Binary(validated.tobytes()),
            ))

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO document_chunks
                    (filename, chunk_text, embedding, embedding_blob)
                VALUES (?, ?, ?, ?)
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
            query = np.asarray(query_vector, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Query vector must contain valid numeric values."
            ) from exc

        if query.ndim != 1 or query.size == 0:
            raise ValueError(
                "Query vector must be a non-empty one-dimensional vector."
            )

        if not np.isfinite(query).all():
            raise ValueError(
                "Query vector must contain only finite numeric values."
            )

        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        with self._connect() as connection:
            revision = connection.execute(
                "SELECT revision FROM vector_store_metadata WHERE id = 1"
            ).fetchone()[0]

            if (
                self._embedding_cache is None
                or self._cache_revision != revision
            ):
                rows = connection.execute(
                    "SELECT id, filename, chunk_text, embedding, embedding_blob "
                    "FROM document_chunks ORDER BY id"
                ).fetchall()

                grouped_rows = {}
                grouped_vectors = {}

                for row in rows:
                    blob = row["embedding_blob"]
                    if blob is not None:
                        if len(blob) % np.dtype(np.float32).itemsize != 0:
                            continue
                        vector = np.frombuffer(blob, dtype=np.float32)
                    else:
                        # Backward compatibility for legacy JSON-only rows.
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

                    dimension = vector.size
                    grouped_rows.setdefault(dimension, []).append({
                        "id": row["id"],
                        "filename": row["filename"],
                        "text": row["chunk_text"],
                    })
                    grouped_vectors.setdefault(dimension, []).append(
                        (vector, norm)
                    )

                cached_groups = {}
                for dimension, vectors in grouped_vectors.items():
                    cached_groups[dimension] = (
                        grouped_rows[dimension],
                        np.ascontiguousarray(
                            np.stack([item[0] for item in vectors]),
                            dtype=np.float32,
                        ),
                        np.asarray(
                            [item[1] for item in vectors],
                            dtype=np.float32,
                        ),
                    )

                self._embedding_cache = cached_groups
                self._cache_revision = revision

        group = self._embedding_cache.get(query.size)
        if group is None:
            return []

        cached_rows, matrix, norms = group

        # One matrix-vector operation replaces per-embedding Python calls.
        scores = (matrix @ query) / (norms * query_norm)

        result_count = min(max(1, top_k), len(scores))
        if result_count < len(scores):
            candidate_indices = np.argpartition(
                scores, len(scores) - result_count
            )[-result_count:]
        else:
            candidate_indices = np.arange(len(scores))

        # Sort only the selected candidates, highest cosine similarity first.
        candidate_indices = candidate_indices[
            np.argsort(-scores[candidate_indices], kind="stable")
        ]

        return [
            {
                **cached_rows[int(index)],
                "score": float(scores[index]),
            }
            for index in candidate_indices
        ]

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
