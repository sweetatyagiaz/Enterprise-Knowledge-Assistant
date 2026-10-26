from openai import OpenAI

from ai_core.settings import get_settings


class EmbeddingService:
    """Generate embeddings for document chunks and user queries."""

    def __init__(self):
        self.settings = get_settings()
        self.client = OpenAI(api_key=self.settings.openai_api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        if not texts:
            return []

        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
            dimensions=self.settings.embedding_dimensions,
        )

        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query."""
        response = self.client.embeddings.create(
            model=self.settings.embedding_model,
            input=[text],
            dimensions=self.settings.embedding_dimensions,
        )

        return response.data[0].embedding
