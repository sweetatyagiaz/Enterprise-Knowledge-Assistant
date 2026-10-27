from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from ai_core.embeddings import EmbeddingService


@pytest.fixture
def service():
    with patch("ai_core.embeddings.service.get_settings") as mock_settings:
        mock_settings.return_value = SimpleNamespace(
            openai_api_key="test-key",
            embedding_model="text-embedding-3-small",
            embedding_dimensions=1536,
        )
        instance = EmbeddingService()
        instance.client = MagicMock()
        return instance


def test_embed_texts_returns_vectors(service):
    service.client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[0.1, 0.2]),
            SimpleNamespace(embedding=[0.3, 0.4]),
        ]
    )

    result = service.embed_texts(["first chunk", "second chunk"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    service.client.embeddings.create.assert_called_once()


def test_embed_texts_empty_list_skips_api(service):
    assert service.embed_texts([]) == []
    service.client.embeddings.create.assert_not_called()


def test_embed_query_returns_vector(service):
    service.client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])]
    )

    assert service.embed_query("test query") == [0.1, 0.2]


def test_embed_texts_propagates_api_errors(service):
    service.client.embeddings.create.side_effect = RuntimeError("API unavailable")

    with pytest.raises(RuntimeError, match="API unavailable"):
        service.embed_texts(["test chunk"])


def test_embed_query_propagates_api_errors(service):
    service.client.embeddings.create.side_effect = RuntimeError("API unavailable")

    with pytest.raises(RuntimeError, match="API unavailable"):
        service.embed_query("test query")


def test_embed_query_rejects_blank_input(service):
    with pytest.raises(ValueError, match="must not be empty"):
        service.embed_query("   ")

    service.client.embeddings.create.assert_not_called()


def test_embed_texts_rejects_blank_items(service):
    with pytest.raises(ValueError, match="must not be empty"):
        service.embed_texts(["valid text", "   "])

    service.client.embeddings.create.assert_not_called()


def test_embed_query_rejects_empty_api_response(service):
    service.client.embeddings.create.return_value = SimpleNamespace(data=[])

    with pytest.raises(ValueError, match="no embedding returned"):
        service.embed_query("test query")

def test_embed_texts_preserves_input_order(service):
    service.client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(embedding=[1.0, 0.0]),
            SimpleNamespace(embedding=[0.0, 1.0]),
        ]
    )

    result = service.embed_texts(["first chunk", "second chunk"])

    assert result == [[1.0, 0.0], [0.0, 1.0]]


def test_embed_texts_rejects_empty_api_response(service):
    service.client.embeddings.create.return_value = SimpleNamespace(data=[])

    with pytest.raises(ValueError, match="no embedding returned"):
        service.embed_texts(["test chunk"])


def test_embed_query_passes_input_as_single_item_list(service):
    service.client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])]
    )

    service.embed_query("  test query  ")

    request = service.client.embeddings.create.call_args.kwargs
    assert request["input"] == ["  test query  "]


def test_embed_texts_sends_configured_model(service):
    service.client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=[0.1, 0.2])]
    )

    service.embed_texts(["test chunk"])

    request = service.client.embeddings.create.call_args.kwargs
    assert request["model"] == "text-embedding-3-small"

