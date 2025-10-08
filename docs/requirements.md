Документация: GPT-ассистент с внешней самохостинговой векторной базой и Google Docs (Python)

Цель

Создать систему, в которой GPT-ассистент OpenAI работает на основании
внешней самохостинговой векторной базы знаний, формируемой из нескольких
Google Docs. При изменении документов в Google Drive база автоматически
пересобирается только для обновлённых документов.

------------------------------------------------------------------------

Архитектура системы

Основные компоненты

1.  Google Docs API — получение текста документов.
2.  OpenAI Embeddings API — генерация эмбеддингов.
3.  ChromaDB / Qdrant — локальная или самохостинговая векторная база.
4.  OpenAI Assistants API — логика диалога и хранение переписки (через
    thread_id).
5.  Telegram Bot / Web UI — пользовательский интерфейс.
6.  Scheduler (cron / APScheduler) — проверка изменений и пересборка
    базы.

------------------------------------------------------------------------

Поток данных

1.  Система получает список ID Google Docs.
2.  Проверяет дату последнего изменения каждого документа.
3.  Если документ изменился:
    -   Удаляются старые embeddings, связанные с документом.
    -   Документ считывается, разбивается на чанки.
    -   Для каждого чанка создаются embeddings и записываются в
        Chroma/Qdrant.
4.  Пользователь отправляет сообщение.
5.  Перед обращением к GPT выполняется поиск по векторной базе.
6.  Релевантные фрагменты добавляются в контекст перед отправкой в
    ассистент с thread_id.
7.  GPT выдаёт ответ, комбинируя историю переписки и свежие данные.

------------------------------------------------------------------------

Развёртывание самохостинговой векторной базы

Вариант 1. ChromaDB (самый простой)

1.  Установите Docker и Docker Compose.

2.  Создайте файл docker-compose.yml:

        version: '3'
        services:
          chroma:
            image: ghcr.io/chroma-core/chroma:latest
            ports:
              - "8000:8000"
            volumes:
              - ./chroma_data:/chroma/chroma
            restart: always

3.  Запустите командой:

        docker compose up -d

4.  Проверьте, что сервис работает: http://localhost:8000

Данные сохраняются в папке ./chroma_data, она должна быть на постоянном
диске (не tmp).

Вариант 2. Qdrant (альтернатива, надёжнее при росте)

1.  Файл docker-compose.yml:

        version: '3'
        services:
          qdrant:
            image: qdrant/qdrant
            ports:
              - "6333:6333"
            volumes:
              - ./qdrant_data:/qdrant/storage
            restart: always

2.  Запуск:

        docker compose up -d

3.  Проверка: http://localhost:6333/dashboard

------------------------------------------------------------------------

Настройка безопасности

1.  Ограничьте доступ к порту 8000 или 6333 только локально, если бот и
    база на одном сервере.
2.  Если доступ извне — настройте HTTPS (через nginx + Let’s Encrypt).
3.  При необходимости настройте Basic Auth или токен-доступ на уровне
    обратного прокси.

------------------------------------------------------------------------

Структура проекта

    /vectorbot/
    ├── main.py              # взаимодействие с ассистентом
    ├── sync_docs.py         # обновление базы при изменениях
    ├── chroma_manager.py    # работа с Chroma/Qdrant
    ├── google_docs.py       # чтение Google Docs
    ├── config.py            # ключи и настройки
    ├── docker-compose.yml   # контейнер базы
    ├── chroma_data/         # постоянное хранилище базы
    └── requirements.txt

------------------------------------------------------------------------

Настройка окружения

    python3 -m venv venv
    source venv/bin/activate
    pip install openai chromadb google-api-python-client google-auth python-dotenv requests

Файл config.py:

    OPENAI_API_KEY = "sk-..."
    GOOGLE_SERVICE_FILE = "credentials.json"
    DOC_IDS = ["1AbCdEfGh...", "2XyZkLmN..."]
    CHROMA_HOST = "http://localhost:8000"
    CHROMA_PATH = "./chroma_data"

------------------------------------------------------------------------

Модуль: google_docs.py

    from googleapiclient.discovery import build
    from google.oauth2 import service_account

    def get_doc_text(doc_id, creds_path):
        scopes = ["https://www.googleapis.com/auth/documents.readonly"]
        creds = service_account.Credentials.from_service_account_file(creds_path, scopes=scopes)
        service = build("docs", "v1", credentials=creds)
        doc = service.documents().get(documentId=doc_id).execute()
        text = ""
        for el in doc["body"]["content"]:
            if "paragraph" in el:
                for run in el["paragraph"]["elements"]:
                    if "textRun" in run:
                        text += run["textRun"]["content"]
        return text

------------------------------------------------------------------------

Модуль: chroma_manager.py

    import chromadb
    from openai import OpenAI

    client = OpenAI()
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    collection = chroma_client.get_or_create_collection("knowledge")

    def split_text(text, max_len=800):
        words = text.split()
        chunks, current = [], []
        for w in words:
            if sum(len(x) for x in current) + len(w) < max_len:
                current.append(w)
            else:
                chunks.append(" ".join(current))
                current = [w]
        if current:
            chunks.append(" ".join(current))
        return chunks

    def update_doc_embeddings(doc_id, text):
        collection.delete(where={"doc_id": doc_id})
        chunks = split_text(text)
        embeddings = []
        for chunk in chunks:
            emb = client.embeddings.create(model="text-embedding-3-small", input=chunk).data[0].embedding
            embeddings.append(emb)
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"doc_id": doc_id, "chunk": i} for i in range(len(chunks))],
            ids=[f"{doc_id}-{i}" for i in range(len(chunks))]
        )

------------------------------------------------------------------------

Модуль: sync_docs.py

    import os, json
    from datetime import datetime
    from googleapiclient.discovery import build
    from google.oauth2 import service_account
    from google_docs import get_doc_text
    from chroma_manager import update_doc_embeddings
    from config import GOOGLE_SERVICE_FILE, DOC_IDS

    def sync_docs():
        creds = service_account.Credentials.from_service_account_file(GOOGLE_SERVICE_FILE)
        drive = build("drive", "v3", credentials=creds)

        os.makedirs("./meta", exist_ok=True)

        for doc_id in DOC_IDS:
            meta = drive.files().get(fileId=doc_id, fields="modifiedTime").execute()
            mod_time = meta["modifiedTime"]
            record_path = f"./meta/{doc_id}.json"

            if not os.path.exists(record_path) or json.load(open(record_path))["modifiedTime"] != mod_time:
                text = get_doc_text(doc_id, GOOGLE_SERVICE_FILE)
                update_doc_embeddings(doc_id, text)
                with open(record_path, "w") as f:
                    json.dump({"modifiedTime": mod_time, "updated": str(datetime.now())}, f)
                print(f"✅ Обновлён документ {doc_id}")
            else:
                print(f"⏩ Без изменений: {doc_id}")

    if __name__ == "__main__":
        sync_docs()

------------------------------------------------------------------------

main.py (интеграция с GPT)

    from openai import OpenAI
    from chroma_manager import collection

    client = OpenAI()

    def search_context(query, n=3):
        emb = client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
        results = collection.query(query_embeddings=[emb], n_results=n)
        return [r for r in results["documents"][0]]

    def ask_assistant(thread_id, user_message):
        context = "
    ".join(search_context(user_message))
        full_prompt = f"Контекст:
    {context}

    Вопрос:
    {user_message}"
        client.beta.threads.messages.create(thread_id=thread_id, role="user", content=full_prompt)
        run = client.beta.threads.runs.create_and_poll(thread_id=thread_id, assistant_id="asst_xxxxxx")
        messages = client.beta.threads.messages.list(thread_id=thread_id).data
        answer = messages[0].content[0].text.value
        return answer

------------------------------------------------------------------------

Автоматизация обновлений

Добавьте cron-задачу для регулярного обновления базы:

    */15 * * * * /usr/bin/python3 /path/to/vectorbot/sync_docs.py

------------------------------------------------------------------------

Рекомендации по обслуживанию

1.  Делайте резервное копирование chroma_data или qdrant_data.
2.  Следите за дисковым пространством (чтобы база не переполнила том).
3.  Раз в месяц обновляйте контейнер (pull latest image).
4.  Настройте мониторинг логов (docker logs -f chroma).
5.  При росте нагрузки увеличивайте память до 2–4 ГБ.
6.  Если база доступна извне — добавьте HTTPS и Basic Auth.

------------------------------------------------------------------------

Итого

-   Самохостинговая Chroma/Qdrant полностью бесплатна.
-   Система обновляет embeddings только для изменённых документов.
-   GPT-ассистент использует thread_id и работает с внешним контекстом.
-   Все данные и база находятся под вашим контролем.
