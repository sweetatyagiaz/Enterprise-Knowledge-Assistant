from ai_core.embeddings import EmbeddingService
from ai_core.settings import get_settings
from ai_core.vector_store import VectorStore
from ai_core.ingestion.loaders import load_document
from ai_core.chunking import chunk_text
from ai_core.retrieval.ranking import hybrid_rank
import re

class KnowledgeAssistant:
    """Coordinates document ingestion, retrieval, and question answering."""

    def __init__(self):
        self.settings = get_settings()
        self.embeddings = EmbeddingService()
        self.store = VectorStore()

    def ingest(self, filename: str, content: bytes) -> dict:
        """Load a document, split it into chunks, embed, and store it."""
        text = load_document(filename, content)

        if not text or not text.strip():
            raise ValueError("The document contains no readable text.")

        chunks = chunk_text(
            text,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        vectors = self.embeddings.embed_texts(chunks)

        self.store.add_document(
            filename=filename,
            chunks=chunks,
            embeddings=vectors,
        )

        return {
            "filename": filename,
            "chunks_created": len(chunks),
            "status": "indexed",
        }

    def ask(self, question: str, top_k: int | None = None) -> dict:
        """Retrieve relevant chunks and generate a grounded answer."""
        query_vector = self.embeddings.embed_query(question)

        # Retrieve a wider candidate pool, then rerank.
        candidate_k = max(top_k or self.settings.top_k, 10)

        candidates = self.store.search(
            query_vector,
            top_k=candidate_k,
        )

        results = hybrid_rank(
            question,
            candidates,
        )[: top_k or self.settings.top_k]

        if not results:
            return {
                "answer": "I could not find relevant information in the uploaded documents.",
                "sources": [],
            }

        question_lower = question.casefold()
        financial_terms = ("revenue", "profit")
        requested_metrics = [
            term for term in financial_terms if term in question_lower
        ]

        if requested_metrics:
            evidence_sentences = re.split(
                r"(?<=[.!?])\\s+|\\n+",
                " ".join(str(item.get("text", "")) for item in results),
            )
            has_metric_evidence = any(
                any(term in sentence.casefold() for term in requested_metrics)
                and re.search(r"\\d", sentence)
                for sentence in evidence_sentences
            )

            if not has_metric_evidence:
                metric = " and ".join(requested_metrics)
                return {
                    "answer": (
                        "The provided document context does not contain "
                        f"information with a numeric value for the company's "
                        f"annual {metric}."
                    ),
                    "sources": results,
                }

        context = "\n\n".join(
            f"[Source: {item['filename']}]\n{item['text']}"
            for item in results
        )

        from openai import OpenAI

        client = OpenAI(api_key=self.settings.openai_api_key)

        response = client.chat.completions.create(
            model=self.settings.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using the supplied document context. "
                        "If the context does not contain the answer, say so. "
                        "Do not invent facts or citations."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Document context:\n{context}\n\n"
                        f"Question: {question}"
                    ),
                },
            ],
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": results,
        }
