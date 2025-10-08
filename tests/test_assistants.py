from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

from app.assistants import AssistantResponse, AssistantService, InMemoryThreadStore
from app.embeddings import EmbeddingResult


class _Secret:
    def __init__(self, value: str) -> None:
        self._value = value

    def get_secret_value(self) -> str:
        return self._value


@dataclass
class StubSettings:
    openai_api_key: _Secret = _Secret("key")
    openai_assistant_id: str = "assistant-id"
    assistant_search_top_k: int = 3


class EmbeddingStub:
    def __init__(self) -> None:
        self.calls: List[List[str]] = []
        self.embedding = [0.1, 0.2, 0.3]

    def embed_texts(self, texts: List[str]) -> List[EmbeddingResult]:
        self.calls.append(list(texts))
        if not texts:
            return []
        return [EmbeddingResult(text=texts[0], embedding=self.embedding)]


class VectorStub:
    def __init__(self, documents: List[List[str]]) -> None:
        self.documents = documents
        self.queries: List[Dict] = []

    def query(self, embedding, *, limit: int) -> Dict[str, List[List[str]]]:
        self.queries.append({"embedding": embedding, "limit": limit})
        return {"documents": self.documents}


class FakeMessages:
    def __init__(self, parent: "FakeOpenAI") -> None:
        self.parent = parent

    def create(self, *, thread_id: str, role: str, content: str) -> None:
        self.parent.messages_created.append({"thread_id": thread_id, "role": role, "content": content})

    def list(self, thread_id: str):
        value = self.parent.responses.get(thread_id, self.parent.default_response)
        return SimpleNamespace(data=[FakeMessage("assistant", value)])


class FakeRuns:
    def __init__(self, parent: "FakeOpenAI") -> None:
        self.parent = parent

    def create_and_poll(self, *, thread_id: str, assistant_id: str) -> None:
        self.parent.runs.append({"thread_id": thread_id, "assistant_id": assistant_id})


class FakeMessage:
    def __init__(self, role: str, value: str) -> None:
        self.role = role
        self.content = [SimpleNamespace(text=SimpleNamespace(value=value))]


class FakeThreadsAPI:
    def __init__(self, parent: "FakeOpenAI") -> None:
        self.parent = parent
        self._counter = 0
        self.messages = FakeMessages(parent)
        self.runs = FakeRuns(parent)

    def create(self):
        self._counter += 1
        thread_id = f"thread-{self._counter}"
        self.parent.created_threads.append(thread_id)
        self.parent.responses.setdefault(thread_id, self.parent.default_response)
        return SimpleNamespace(id=thread_id)


class FakeOpenAI:
    def __init__(self, default_response: str = "assistant reply") -> None:
        self.default_response = default_response
        self.created_threads: List[str] = []
        self.messages_created: List[Dict] = []
        self.responses: Dict[str, str] = {}
        self.runs: List[Dict] = []
        self.beta = SimpleNamespace(threads=FakeThreadsAPI(self))


def test_search_context_returns_top_chunks() -> None:
    settings = StubSettings(assistant_search_top_k=2)
    embedding = EmbeddingStub()
    vector = VectorStub(documents=[["chunk1", "chunk2", "chunk3"]])
    service = AssistantService(
        settings=settings,
        embedding_service=embedding,
        vector_store=vector,
        client=FakeOpenAI(),
        thread_store=InMemoryThreadStore(),
    )

    contexts = service.search_context("Запрос")

    assert contexts == ["chunk1", "chunk2"]
    assert vector.queries[0]["limit"] == 2
    assert embedding.calls == [["Запрос"]]


def test_send_message_creates_thread_and_returns_answer() -> None:
    settings = StubSettings()
    embedding = EmbeddingStub()
    vector = VectorStub(documents=[["контекст"]])
    fake_client = FakeOpenAI(default_response="Ответ ассистента")
    store = InMemoryThreadStore()
    service = AssistantService(
        settings=settings,
        embedding_service=embedding,
        vector_store=vector,
        client=fake_client,
        thread_store=store,
    )

    response = service.send_message(user_key="user-1", message="Привет")

    assert isinstance(response, AssistantResponse)
    assert response.answer == "Ответ ассистента"
    assert response.context_chunks == ["контекст"]
    assert store.get("user-1") == response.thread_id
    assert fake_client.created_threads == [response.thread_id]
    assert fake_client.messages_created[-1]["content"].startswith("Контекст:\nконтекст")


def test_send_message_uses_existing_thread() -> None:
    settings = StubSettings()
    embedding = EmbeddingStub()
    vector = VectorStub(documents=[[]])
    fake_client = FakeOpenAI(default_response="Ответ без контекста")
    store = InMemoryThreadStore()
    store.set("user-2", "thread-existing")
    fake_client.responses["thread-existing"] = "Ответ без контекста"

    service = AssistantService(
        settings=settings,
        embedding_service=embedding,
        vector_store=vector,
        client=fake_client,
        thread_store=store,
    )

    response = service.send_message(user_key="user-2", message="Вопрос")

    assert response.thread_id == "thread-existing"
    assert fake_client.created_threads == []
    assert response.answer == "Ответ без контекста"
    assert fake_client.messages_created[-1]["content"] == "Вопрос"
