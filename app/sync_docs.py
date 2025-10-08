from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import List

from app.chroma_manager import VectorStoreGateway
from app.config import Settings, get_settings
from app.embeddings import EmbeddingService, chunk_text
from app.google_docs import DocumentSnapshot, GoogleDocsService

logger = logging.getLogger(__name__)


@dataclass
class SyncResult:
    """Результат обработки одного документа."""

    doc_id: str
    status: str
    chunks: int = 0
    error: str | None = None


class SyncOrchestrator:
    """Оркестратор обновления эмбеддингов документов Google Docs."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        docs_service: GoogleDocsService | None = None,
        embedding_service: EmbeddingService | None = None,
        vector_gateway: VectorStoreGateway | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.docs_service = docs_service or GoogleDocsService(settings=self.settings)
        self.embedding_service = embedding_service or EmbeddingService(settings=self.settings)
        self.vector_gateway = vector_gateway or VectorStoreGateway(settings=self.settings)

    def sync_documents(self, *, force: bool = False) -> List[SyncResult]:
        results: List[SyncResult] = []
        for doc_id in self.settings.google_doc_ids:
            try:
                result = self._process_document(doc_id, force=force)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Ошибка обработки документа %s", doc_id)
                results.append(SyncResult(doc_id=doc_id, status="failed", error=str(exc)))
            else:
                results.append(result)
        return results

    def _process_document(self, doc_id: str, *, force: bool) -> SyncResult:
        logger.info("Проверка документа %s", doc_id)

        if not force and not self.docs_service.needs_update(doc_id):
            logger.info("Документ %s без изменений, пропускаем.", doc_id)
            return SyncResult(doc_id=doc_id, status="skipped")

        snapshot = self.docs_service.fetch_document(doc_id)
        logger.debug("Документ %s изменён в %s.", doc_id, snapshot.modified_time)

        chunks = self._chunk_snapshot(snapshot)
        if not chunks:
            logger.warning("Документ %s пустой после разбиения; удаляем записи из коллекции.", doc_id)
            self.vector_gateway.delete_document(doc_id)
            self.docs_service.persist_metadata(doc_id, snapshot.modified_time)
            return SyncResult(doc_id=doc_id, status="deleted", chunks=0)

        embeddings = self.embedding_service.embed_texts(chunks)
        vectors = [item.embedding for item in embeddings]
        texts = [item.text for item in embeddings]
        metadata = [{"doc_id": doc_id, "chunk": index} for index in range(len(texts))]

        self.vector_gateway.replace_document(
            doc_id=doc_id,
            texts=texts,
            embeddings=vectors,
            metadatas=metadata,
        )
        self.docs_service.persist_metadata(doc_id, snapshot.modified_time)

        logger.info("Документ %s обновлён, чанков: %s.", doc_id, len(texts))
        return SyncResult(doc_id=doc_id, status="updated", chunks=len(texts))

    def _chunk_snapshot(self, snapshot: DocumentSnapshot) -> List[str]:
        return chunk_text(
            snapshot.text,
            max_tokens=self.settings.embedding_chunk_size,
            overlap=self.settings.embedding_chunk_overlap,
        )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main() -> None:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Синхронизирует документы Google Docs с векторной базой.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Обновить все документы независимо от метаданных.",
    )
    args = parser.parse_args()

    configure_logging(settings.log_level)
    orchestrator = SyncOrchestrator(settings=settings)
    results = orchestrator.sync_documents(force=args.force)

    for result in results:
        if result.status == "failed":
            logger.error("Документ %s завершился с ошибкой: %s", result.doc_id, result.error)
        else:
            logger.info("Документ %s: %s (чанков: %s)", result.doc_id, result.status, result.chunks)


if __name__ == "__main__":
    main()
