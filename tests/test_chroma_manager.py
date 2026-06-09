from collections import defaultdict

from app.chroma_manager import VectorStoreGateway


class FakeCollection:
    def __init__(self) -> None:
        self.deleted = []
        self.added = []
        self.queries = []

    def delete(self, where):
        self.deleted.append(where)

    def add(self, ids, documents, embeddings, metadatas):
        self.added.append(
            {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings,
                "metadatas": metadatas,
            },
        )

    def query(self, query_embeddings, n_results):
        self.queries.append((query_embeddings, n_results))
        return {
            "documents": [["общий текст", "расшифровка удержаний в дэшборде"]],
            "metadatas": [[{"title": "Склад"}, {"title": "Удержания"}]],
            "distances": [[0.1, 0.3]],
        }

    def get(self, include, limit):
        return {
            "documents": ["детализация самовыкупа показывает сумму удержаний"],
            "metadatas": [{"title": "Самовыкупы"}],
        }


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection()
        self.calls = defaultdict(int)

    def get_or_create_collection(self, name):
        self.calls[name] += 1
        return self.collection


class StubSettings:
    chroma_host = "chroma"
    chroma_port = 8000
    chroma_collection_name = "knowledge"


def test_replace_document_rewrites_embeddings():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    manager.replace_document(
        doc_id="doc1",
        texts=["a", "b"],
        embeddings=[[0.1], [0.2]],
        metadatas=[{"doc_id": "doc1", "chunk": 0}, {"doc_id": "doc1", "chunk": 1}],
    )

    collection = manager._get_collection()
    assert collection.deleted == [{"doc_id": "doc1"}]
    assert collection.added[0]["ids"] == ["doc1-0", "doc1-1"]


def test_query_returns_payload():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    result = manager.query([0.3, 0.4], limit=2)

    assert "documents" in result
    assert manager._get_collection().queries[0][1] == 2


def test_search_boosts_exact_terms():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    result = manager.search(
        [0.3, 0.4],
        query="где расшифровка удержаний",
        limit=1,
        candidate_limit=2,
    )

    assert result[0].text == "расшифровка удержаний в дэшборде"
    assert result[0].matched_terms == ["расшифровк", "удержан"]


def test_search_uses_keyword_candidates_outside_semantic_results():
    manager = VectorStoreGateway(settings=StubSettings(), client=FakeClient())

    result = manager.search(
        [0.3, 0.4],
        query="детализация самовыкупа удержаний",
        limit=1,
        candidate_limit=1,
    )

    assert result[0].text == "детализация самовыкупа показывает сумму удержаний"
