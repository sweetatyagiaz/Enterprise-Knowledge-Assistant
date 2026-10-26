
from pathlib import Path

from ai_core.chunking import chunk_text
from ai_core.embeddings import EmbeddingService
from ai_core.ingestion.loaders import load_document
from ai_core.vector_store import VectorStore


def main():
    filename = "README.md"
    path = Path(filename)

    if not path.exists():
        raise FileNotFoundError(f"Could not find {path.resolve()}")

    content = path.read_bytes()
    text = load_document(filename, content)

    settings_path = Path("data/vector_store.sqlite3")
    if not settings_path.exists():
        raise FileNotFoundError(
            f"Database not found: {settings_path.resolve()}"
        )

    chunks = chunk_text(
        text,
        chunk_size=900,
        chunk_overlap=150,
    )

    if not chunks:
        raise ValueError("No chunks were generated.")

    print(f"Generated {len(chunks)} new chunks.")

    embeddings = EmbeddingService()
    vectors = embeddings.embed_texts(chunks)

    if len(vectors) != len(chunks):
        raise RuntimeError("Embedding count does not match chunk count.")

    store = VectorStore()

    # Replace only README.md chunks; preserve other documents.
    with store._connect() as connection:
        connection.execute(
            "DELETE FROM document_chunks WHERE filename = ?",
            (filename,),
        )

    store.add_document(
        filename=filename,
        chunks=chunks,
        embeddings=vectors,
    )

    print("Re-indexing completed.")
    print("Documents:", store.list_documents())
    print("Total chunks:", store.count_chunks())


if __name__ == "__main__":
    main()

