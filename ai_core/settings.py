from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""

    chat_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = Field(default=1536, gt=0)

    data_dir: Path = Path("data")
    sqlite_path: Path = Path("data/vector_store.sqlite3")

    chunk_size: int = Field(default=900, gt=0)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, gt=0)
    max_upload_mb: int = Field(default=15, gt=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def validate_chunk_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

