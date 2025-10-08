from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Sequence

from openai import OpenAI
from openai.pagination import SyncCursorPage

from app.chroma_manager import VectorStoreGateway
from app.config import Settings, get_settings
from app.embeddings import EmbeddingResult, EmbeddingService

logger = logging.getLogger(__name__)


class ThreadStore(Protocol):
    """Интерфейс хранения thread_id для пользователей."""

    def get(self, key: str) -> Optional[str]:
        ...

    def set(self, key: str, thread_id: str) -> None:
        ...


class InMemoryThreadStore:
    """Простое in-memory хранилище для служебных сценариев."""

    def __init__(self) -> None:
        self._storage: Dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._storage.get(key)

    def set(self, key: str, thread_id: str) -> None:
        self._storage[key] = thread_id


@dataclass
class AssistantResponse:
    """Результат отправки сообщения ассистенту."""

    thread_id: str
    answer: str
    context_chunks: List[str]


class AssistantService:
    """Интеграция с OpenAI Assistants API и векторным поиском."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        vector_store: VectorStoreGateway | None = None,
        embedding_service: EmbeddingService | None = None,
        client: OpenAI | None = None,
        thread_store: ThreadStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.vector_store = vector_store or VectorStoreGateway(settings=self.settings)
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)
        self.client = client or OpenAI(api_key=self.settings.openai_api_key.get_secret_value())
        self.thread_store = thread_store or InMemoryThreadStore()

    def send_message(
        self,
        *,
        user_key: str,
        message: str,
        force_thread_id: str | None = None,
    ) -> AssistantResponse:
        thread_id = force_thread_id or self._get_or_create_thread(user_key)
        context_chunks = self.search_context(message)
        payload = self._build_prompt(message, context_chunks)

        logger.debug("Отправка сообщения в ассистента thread_id=%s", thread_id)
        self.client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=payload,
        )
        self.client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=self.settings.openai_assistant_id,
        )

        answer = self._extract_last_assistant_message(thread_id)
        return AssistantResponse(thread_id=thread_id, answer=answer, context_chunks=context_chunks)

    def search_context(self, query: str) -> List[str]:
        results = self.embedding_service.embed_texts([query])
        if not results:
            return []

        embedding = results[0].embedding
        query_result = self.vector_store.query(
            embedding,
            limit=self.settings.assistant_search_top_k,
        )
        documents = query_result.get("documents") or []

        chunks: List[str] = []
        for group in documents:
            chunks.extend(group)

        return [chunk for chunk in chunks if chunk][: self.settings.assistant_search_top_k]

    def _get_or_create_thread(self, user_key: str) -> str:
        existing = self.thread_store.get(user_key)
        if existing:
            return existing

        thread = self.client.beta.threads.create()
        thread_id = thread.id
        logger.debug("Создан новый thread_id=%s для пользователя %s", thread_id, user_key)
        self.thread_store.set(user_key, thread_id)
        return thread_id

    def _build_prompt(self, user_message: str, context_chunks: Iterable[str]) -> str:
        context_text = "\n\n".join(context_chunks).strip()
        if context_text:
            return f"Контекст:\n{context_text}\n\nВопрос:\n{user_message}"
        return user_message

    def _extract_last_assistant_message(self, thread_id: str) -> str:
        logger.debug("Чтение последних сообщений ассистента thread_id=%s", thread_id)
        messages: SyncCursorPage = self.client.beta.threads.messages.list(thread_id=thread_id)
        for message in messages.data:
            if getattr(message, "role", "") != "assistant":
                continue
            for part in getattr(message, "content", []):
                text = getattr(part, "text", None)
                if text and getattr(text, "value", ""):
                    return text.value
        logger.warning("Ответ ассистента не найден, возвращаем пустую строку.")
        return ""
