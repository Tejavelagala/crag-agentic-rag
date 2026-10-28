# CRAG: Corrective Retrieval-Augmented Generation

This repository is the production-grade foundation for a stateful corrective
RAG system. It is intentionally small at this stage: the package provides
typed configuration, centralized logging, application exceptions, and a local
health status model. It also defines validated evidence models and provider
contracts without implementing external integrations or workflows.

## Setup

Requires Python 3.11 or newer.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

`.env` is ignored by Git. Keep credentials out of source control.

## Configuration

Settings use the `CRAG_` environment-variable prefix and also load a local
`.env` file when present. Integrations are disabled by default, so package
imports, tests, and basic local operation do not need API keys. If an
integration is enabled, its required setting is validated with an actionable
error:

- `CRAG_ENABLE_OPENAI=true` requires `CRAG_OPENAI_API_KEY`.
- `CRAG_ENABLE_TAVILY=true` requires `CRAG_TAVILY_API_KEY`.
- `CRAG_ENABLE_QDRANT=true` requires `CRAG_QDRANT_URL`.

Use `crag.config.get_settings()` in application code. The returned settings
object is cached; call `clear_settings_cache()` after intentional process-level
reconfiguration, such as in tests.

## Testing and quality

```powershell
python -m pytest
ruff check .
ruff format --check .
```

Unit tests are deterministic and do not use network access or external
credentials. Future integration tests belong in `tests/integration/`, must be
marked `integration`, and require explicit opt-in and configured services.

## Architecture status

### Implemented

- `src/crag/` package with typed settings, logging setup, exceptions, and a
  local-only health status check.
- Validated document, evidence, web-result, and grading models.
- Deterministic character-based parent-child document ingestion with
  configurable overlapping child chunks.
- Async provider contracts for chat, embeddings, vector storage, and web
  search, with explicit timeout parameters.
- Concrete asynchronous OpenAI, Qdrant, and Tavily adapters with deterministic
  provider-error translation and mocked unit tests.
- A typed LangGraph corrective retrieval workflow that grades internal evidence,
  falls back to web search, and generates grounded answers.
- A FastAPI application factory with `/api/v1/ask`, local liveness, and
  dependency-configuration readiness endpoints.
- Deterministic unit tests and an integration-test convention.
- Packaging, Ruff, pytest, environment, and project engineering configuration.

### Planned

- Internal retrieval and Qdrant vector-store integration.

No planned component is represented as implemented until it has real code and
tests.

## Document ingestion

`crag.ingestion.ingest_document` accepts document text and source metadata and
returns ordered `DocumentChunk` records. It creates contiguous parent sections
and smaller overlapping child chunks linked by `parent_id`; IDs are stable for
the same document identity, content, and chunk position. Chunking is
character-based, not token-aware, and does not normalize text or split on
semantic boundaries.
