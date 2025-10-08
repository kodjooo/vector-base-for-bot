from __future__ import annotations

import logging
from typing import Iterable, List, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorStoreGateway:
    """Инкапсулирует работу с ChromaDB."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: ClientAPI | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or self._build_client()
        self._collection: Collection | None = None

    def replace_document(
        self,
        *,
        doc_id: str,
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Iterable[dict] | None = None,
    ) -> None:
        """Удаляет старые записи документа и добавляет новые чанки."""
        if not texts:
            logger.warning("Получен пустой набор текстов для документа %s, запись пропущена.", doc_id)
            return

        collection = self._get_collection()
        logger.debug("Удаление существующих записей документа %s из коллекции.", doc_id)
        collection.delete(where={"doc_id": doc_id})

        ids = [f"{doc_id}-{index}" for index in range(len(texts))]
        metadata_payload = metadatas or [{"doc_id": doc_id, "chunk": index} for index in range(len(texts))]

        logger.debug("Добавление %s чанков для документа %s.", len(texts), doc_id)
        collection.add(
            ids=ids,
            documents=list(texts),
            embeddings=list(embeddings),
            metadatas=list(metadata_payload),
        )

    def delete_document(self, doc_id: str) -> None:
        logger.debug("Удаление документа %s из коллекции.", doc_id)
        self._get_collection().delete(where={"doc_id": doc_id})

    def query(self, embedding: Sequence[float], *, limit: int = 3) -> dict:
        logger.debug("Поиск релевантных документов (limit=%s).", limit)
        return self._get_collection().query(query_embeddings=[embedding], n_results=limit)

    def _get_collection(self) -> Collection:
        if self._collection is None:
            logger.debug(
                "Получение коллекции %s на %s:%s.",
                self.settings.chroma_collection_name,
                self.settings.chroma_host,
                self.settings.chroma_port,
            )
            self._collection = self._client.get_or_create_collection(self.settings.chroma_collection_name)
        return self._collection

    def _build_client(self) -> ClientAPI:
        logger.debug(
            "Инициализация клиента ChromaDB на %s:%s.",
            self.settings.chroma_host,
            self.settings.chroma_port,
        )
        return chromadb.HttpClient(
            host=self.settings.chroma_host,
            port=self.settings.chroma_port,
        )
