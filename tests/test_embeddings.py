from types import SimpleNamespace

from ai_core.embeddings import EmbeddingService


def make_embedding_service(create):
    service = object.__new__(EmbeddingService)
    service.settings = SimpleNamespace(
        embedding_model="test-model",
        embedding_dimensions=3,
    )
    service.client = SimpleNamespace(
        embeddings=SimpleNamespace(create=create)
    )
    return service


def test_embed_texts_returns_empty_list_without_api_call():
    def unexpected_call(**kwargs):
        raise AssertionError("API should not be called for empty input")

    service = make_embedding_service(unexpected_call)

    assert service.embed_texts([]) == []


def test_embed_texts_preserves_response_order():
    expected = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    def fake_create(**kwargs):
        assert kwargs["input"] == ["First document", "Second document"]
        return SimpleNamespace(
            data=[
                SimpleNamespace(embedding=vector)
                for vector in expected
            ]
        )

    service = make_embedding_service(fake_create)

    assert service.embed_texts(
        ["First document", "Second document"]
    ) == expected


def test_embed_texts_passes_model_and_dimensions():
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]
        )

    service = make_embedding_service(fake_create)
    service.embed_texts(["Sample document"])

    assert captured["model"] == "test-model"
    assert captured["dimensions"] == 3
    assert captured["input"] == ["Sample document"]


def test_embed_query_returns_single_embedding():
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.4, 0.5, 0.6])]
        )

    service = make_embedding_service(fake_create)

    result = service.embed_query("Find relevant documents")

    assert result == [0.4, 0.5, 0.6]
    assert captured["input"] == ["Find relevant documents"]
    assert captured["model"] == "test-model"
    assert captured["dimensions"] == 3
