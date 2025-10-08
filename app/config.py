from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import AnyHttpUrl, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )
    openai_assistant_id: str = Field(alias="OPENAI_ASSISTANT_ID")
    assistant_search_top_k: int = Field(default=3, alias="ASSISTANT_SEARCH_TOP_K")

    google_service_account_file: Optional[Path] = Field(
        default=None,
        alias="GOOGLE_SERVICE_ACCOUNT_FILE",
    )
    google_service_account_info: Optional[SecretStr] = Field(
        default=None,
        alias="GOOGLE_SERVICE_ACCOUNT_INFO",
    )
    google_doc_ids: List[str] = Field(alias="GOOGLE_DOC_IDS")
    google_request_interval_seconds: float = Field(
        default=0.25,
        alias="GOOGLE_REQUEST_INTERVAL_SECONDS",
    )
    google_retry_attempts: int = Field(
        default=5,
        alias="GOOGLE_MAX_RETRIES",
    )
    google_retry_initial_delay: float = Field(
        default=1.0,
        alias="GOOGLE_RETRY_INITIAL_DELAY",
    )

    chroma_host: str = Field(default="localhost", alias="CHROMA_HOST")
    chroma_port: int = Field(default=8000, alias="CHROMA_PORT")
    chroma_collection_name: str = Field(
        default="knowledge",
        alias="CHROMA_COLLECTION_NAME",
    )

    embedding_chunk_size: int = Field(default=800, alias="EMBEDDING_CHUNK_SIZE")
    embedding_chunk_overlap: int = Field(
        default=100,
        alias="EMBEDDING_CHUNK_OVERLAP",
    )

    sync_interval_minutes: int = Field(default=15, alias="SYNC_INTERVAL_MINUTES")

    telegram_bot_token: SecretStr = Field(alias="TELEGRAM_BOT_TOKEN")
    telegram_webhook_url: Optional[AnyHttpUrl] = Field(
        default=None,
        alias="TELEGRAM_WEBHOOK_URL",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @model_validator(mode="before")
    @classmethod
    def _prepare_values(cls, data: dict) -> dict:
        if "GOOGLE_DOC_IDS" in data and isinstance(data["GOOGLE_DOC_IDS"], str):
            ids = [doc_id.strip() for doc_id in data["GOOGLE_DOC_IDS"].split(",") if doc_id.strip()]
            data["GOOGLE_DOC_IDS"] = ids

        for key in ("GOOGLE_SERVICE_ACCOUNT_FILE", "GOOGLE_SERVICE_ACCOUNT_INFO"):
            value = data.get(key)
            if isinstance(value, str) and not value.strip():
                data[key] = None

        webhook = data.get("TELEGRAM_WEBHOOK_URL")
        if isinstance(webhook, str) and not webhook.strip():
            data["TELEGRAM_WEBHOOK_URL"] = None
        return data

    @model_validator(mode="after")
    def _validate_values(self) -> "Settings":
        if (
            self.google_service_account_file is None
            and self.google_service_account_info is None
        ):
            raise ValueError(
                "Необходимо указать GOOGLE_SERVICE_ACCOUNT_FILE или GOOGLE_SERVICE_ACCOUNT_INFO.",
            )

        if self.embedding_chunk_overlap >= self.embedding_chunk_size:
            raise ValueError("EMBEDDING_CHUNK_OVERLAP должен быть меньше EMBEDDING_CHUNK_SIZE.")

        if self.sync_interval_minutes <= 0:
            raise ValueError("SYNC_INTERVAL_MINUTES должен быть положительным.")

        if self.chroma_port <= 0 or self.chroma_port > 65535:
            raise ValueError("CHROMA_PORT должен быть в диапазоне 1-65535.")

        if self.google_request_interval_seconds <= 0:
            raise ValueError("GOOGLE_REQUEST_INTERVAL_SECONDS должен быть положительным.")

        if self.google_retry_attempts <= 0:
            raise ValueError("GOOGLE_MAX_RETRIES должен быть положительным.")

        if self.google_retry_initial_delay <= 0:
            raise ValueError("GOOGLE_RETRY_INITIAL_DELAY должен быть положительным.")

        if self.assistant_search_top_k <= 0:
            raise ValueError("ASSISTANT_SEARCH_TOP_K должен быть положительным.")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> None:
    get_settings.cache_clear()
