# Vector Base for Bot

Сервис синхронизирует Google Docs в ChromaDB и отдаёт релевантные фрагменты через HTTP API для `support-bot`.

## Запуск

```bash
cp .env.example .env
docker compose up -d --build
```

В `.env` нужно указать `OPENAI_API_KEY`, путь к сервисному аккаунту Google в `GOOGLE_SERVICE_ACCOUNT_FILE` и список документов `GOOGLE_DOC_IDS`. Файл сервисного аккаунта должен быть доступен контейнеру по пути, указанному в `docker-compose.yml`.

## Синхронизация

Плановая синхронизация запускается внутри `app` по `SYNC_INTERVAL_MINUTES`. Для ручной полной пересборки индекса:

```bash
docker compose run --rm app python -m app.sync_docs --force
```

Полная пересборка нужна после изменения логики чанкинга, метаданных или модели эмбеддингов.

## Поиск

API доступен на `POST /search`:

```bash
curl -sS -X POST http://localhost:8080/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Где увидеть расшифровку удержаний?","top_k":3}'
```

Ответ содержит `chunks` для обратной совместимости и подробные `results` с `metadata`, `distance`, `score`, `semantic_score`, `keyword_score`, `matched_terms`.

Параметры качества поиска:

- `SEARCH_TOP_K` — сколько фрагментов вернуть.
- `SEARCH_CANDIDATE_MULTIPLIER` — сколько кандидатов взять из ChromaDB перед переранжированием.
- `SEARCH_MIN_SCORE` — минимальный итоговый балл результата.
- `SEARCH_KEYWORD_LIMIT` — сколько чанков просматривать для точного keyword-поиска.

## Проверка

```bash
docker compose run --rm app pytest
docker compose logs -f app
```
