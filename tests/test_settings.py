from pathlib import Path

from ai_core.settings import Settings, get_settings


def test_settings_have_expected_defaults(monkeypatch):
    monkeypatch.setattr(Settings, "model_config", Settings.model_config)

    settings = Settings(_env_file=None)

    assert settings.openai_api_key == ""
    assert settings.chat_model == "gpt-4.1-mini"
    assert settings.embedding_model == "text-embedding-3-small"
    assert settings.embedding_dimensions == 1536
    assert settings.data_dir == Path("data")
    assert settings.sqlite_path == Path("data/vector_store.sqlite3")
    assert settings.chunk_size == 900
    assert settings.chunk_overlap == 150
    assert settings.top_k == 5
    assert settings.max_upload_mb == 15


def test_settings_read_environment_variables(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "test-chat-model")
    monkeypatch.setenv("TOP_K", "8")
    monkeypatch.setenv("MAX_UPLOAD_MB", "25")

    settings = Settings(_env_file=None)

    assert settings.chat_model == "test-chat-model"
    assert settings.top_k == 8
    assert settings.max_upload_mb == 25


def test_settings_convert_path_environment_variables(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "custom_data")
    monkeypatch.setenv("SQLITE_PATH", "custom_data/vectors.sqlite3")

    settings = Settings(_env_file=None)

    assert settings.data_dir == Path("custom_data")
    assert settings.sqlite_path == Path("custom_data/vectors.sqlite3")


def test_settings_ignore_unknown_environment_variables(monkeypatch):
    monkeypatch.setenv("UNRECOGNIZED_SETTING", "ignored")

    settings = Settings(_env_file=None)

    assert settings.chat_model == "gpt-4.1-mini"


def test_get_settings_returns_cached_instance():
    get_settings.cache_clear()
    try:
        first = get_settings()
        second = get_settings()

        assert first is second
    finally:
        get_settings.cache_clear()
