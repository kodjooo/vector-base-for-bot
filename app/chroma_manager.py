from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Структурированный результат поиска по базе знаний."""

    text: str
    metadata: dict
    distance: float | None = None
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    score: float = 0.0
    matched_terms: list[str] = field(default_factory=list)


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

    def search(
        self,
        embedding: Sequence[float],
        *,
        query: str,
        limit: int = 3,
        candidate_limit: int | None = None,
        min_score: float = 0.0,
        keyword_limit: int = 2000,
    ) -> list[SearchResult]:
        """Ищет чанки и переранжирует их с учётом точных совпадений слов."""
        raw_limit = candidate_limit or limit
        query_result = self.query(embedding, limit=raw_limit)
        results = self._merge_results(
            self._flatten_query_result(query_result),
            self._keyword_candidates(keyword_limit=keyword_limit),
        )
        ranked = self._rerank_results(query, results)
        if any(item.keyword_score > 0 for item in ranked):
            ranked = [item for item in ranked if item.keyword_score > 0]
        filtered = [item for item in ranked if item.score >= min_score]
        return filtered[:limit]

    def _flatten_query_result(self, query_result: dict) -> list[SearchResult]:
        documents = query_result.get("documents") or []
        metadatas = query_result.get("metadatas") or []
        distances = query_result.get("distances") or []

        results: list[SearchResult] = []
        for group_index, group in enumerate(documents):
            metadata_group = metadatas[group_index] if group_index < len(metadatas) else []
            distance_group = distances[group_index] if group_index < len(distances) else []
            for item_index, text in enumerate(group):
                if not text:
                    continue
                distance = distance_group[item_index] if item_index < len(distance_group) else None
                metadata = metadata_group[item_index] if item_index < len(metadata_group) else {}
                semantic_score = self._semantic_score(distance)
                results.append(
                    SearchResult(
                        text=text,
                        metadata=metadata or {},
                        distance=distance,
                        semantic_score=semantic_score,
                        score=semantic_score,
                    ),
                )
        return results

    def _rerank_results(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        query_terms = _extract_terms(query)
        for result in results:
            text_terms = set(_extract_terms(result.text))
            metadata_terms = set(_extract_terms(" ".join(str(value) for value in result.metadata.values())))
            matched = sorted({term for term in query_terms if term in text_terms or term in metadata_terms})
            result.matched_terms = matched
            result.keyword_score = min(len(matched) * 0.5, 2.0)
            result.keyword_score += _domain_boost(query_terms, result)
            result.score = result.semantic_score + result.keyword_score
        return sorted(results, key=lambda item: item.score, reverse=True)

    def _keyword_candidates(self, *, keyword_limit: int) -> list[SearchResult]:
        payload = self._get_collection().get(
            include=["documents", "metadatas"],
            limit=keyword_limit,
        )
        documents = payload.get("documents") or []
        metadatas = payload.get("metadatas") or []

        results: list[SearchResult] = []
        for index, text in enumerate(documents):
            if not text:
                continue
            metadata = metadatas[index] if index < len(metadatas) else {}
            results.append(SearchResult(text=text, metadata=metadata or {}))
        return results

    @staticmethod
    def _merge_results(*groups: list[SearchResult]) -> list[SearchResult]:
        merged: dict[str, SearchResult] = {}
        for group in groups:
            for item in group:
                existing = merged.get(item.text)
                if existing is None or item.semantic_score > existing.semantic_score:
                    merged[item.text] = item
        return list(merged.values())

    @staticmethod
    def _semantic_score(distance: float | None) -> float:
        if distance is None:
            return 0.0
        if not math.isfinite(distance):
            return 0.0
        return 1 / (1 + max(distance, 0))

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


def _extract_terms(text: str) -> list[str]:
    terms = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower())
    stop_words = {
        "где",
        "могу",
        "можно",
        "увидеть",
        "посмотреть",
        "покажи",
        "найти",
        "какой",
        "какая",
        "какие",
    }
    return [
        _normalize_term(term)
        for term in terms
        if len(term) >= 4 and term not in stop_words
    ]


def _normalize_term(term: str) -> str:
    endings = (
        "иями",
        "ями",
        "ами",
        "ого",
        "ему",
        "ому",
        "ыми",
        "ими",
        "иях",
        "ах",
        "ях",
        "ую",
        "юю",
        "ая",
        "яя",
        "ое",
        "ее",
        "ые",
        "ие",
        "ый",
        "ий",
        "ой",
        "ам",
        "ям",
        "ом",
        "ем",
        "ов",
        "ев",
        "ей",
        "ий",
        "ия",
        "ие",
        "ии",
        "я",
        "а",
        "у",
        "ю",
        "ы",
        "и",
        "е",
    )
    for ending in endings:
        if len(term) > len(ending) + 4 and term.endswith(ending):
            normalized = term[: -len(ending)]
            return normalized[:-1] if normalized.endswith("и") and len(normalized) > 5 else normalized
    return term


def _domain_boost(query_terms: list[str], result: SearchResult) -> float:
    if "удержан" not in query_terms:
        return 0.0

    text = result.text.lower()
    boost = 0.0
    if "строка «удержания" in text or "строку «удержания" in text:
        boost += 1.2
    if "колонка «инфо" in text or "колонке «инфо" in text:
        boost += 1.0
    if "детализация самовыкупа" in text or "детализации самовыкупа" in text:
        boost += 1.0
    if result.metadata.get("section") == "Диаграмма":
        boost -= 0.5
    return boost
