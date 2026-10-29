# CRAG: Corrective Retrieval-Augmented Generation

This repository is a production-oriented foundation for a stateful corrective
RAG system. It provides typed configuration, provider adapters, deterministic
document ingestion, a LangGraph workflow, and a FastAPI application factory.

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
- `CRAG_PROVIDER_MODE=local` selects Ollama chat and embeddings.
- `CRAG_PROVIDER_MODE=hosted` selects OpenAI chat and embeddings and requires
  `CRAG_OPENAI_API_KEY`.
- `CRAG_OLLAMA_CHAT_MODEL` and `CRAG_OLLAMA_EMBEDDING_MODEL` select local
  models independently.

Use `crag.config.get_settings()` in application code. The returned settings
object is cached; call `clear_settings_cache()` after intentional process-level
reconfiguration, such as in tests.

The explicit composition root is `crag.composition.build_workflow`. It probes
the selected embedding provider once during startup to determine the actual
Qdrant vector dimension, then selects the embedding-specific collection:
`CRAG_OLLAMA_QDRANT_COLLECTION_NAME` for local mode and
`CRAG_QDRANT_COLLECTION_NAME` for hosted mode. The probe is an explicit
startup operation, not an import-time network call.

For a local-only deployment, leave `CRAG_ENABLE_TAVILY=false`. If internal
evidence is insufficient, the workflow reports an explicit provider error and
does not fabricate web results or an answer. Set `CRAG_ENABLE_TAVILY=true` and
provide `CRAG_TAVILY_API_KEY` only when web fallback is desired.

## Testing and quality

```powershell
python -m pytest
ruff check .
ruff format --check .
```

Unit tests are deterministic and do not use network access or external
credentials. Live integration tests are in `tests/integration/`, are marked
`integration`, and require explicit opt-in and configured services.

### Live integration tests

Live tests are skipped unless `CRAG_RUN_INTEGRATION_TESTS=true` is set in the
current process. They also skip individually when required settings are
missing. OpenAI/Tavily/Qdrant tests require `CRAG_OPENAI_API_KEY`,
`CRAG_TAVILY_API_KEY`, or `CRAG_QDRANT_URL` respectively. Qdrant tests require
`CRAG_INTEGRATION_QDRANT_COLLECTION` to name a dedicated collection beginning
with `crag_integration_`; they never use or modify the default `crag_chunks`
collection. Each test uses unique document and point identifiers and removes
only the point it created when cleanup is supported.

From PowerShell, configure the opt-in and dedicated collection without writing
credentials into the repository:

```powershell
$env:CRAG_RUN_INTEGRATION_TESTS = "true"
$env:CRAG_INTEGRATION_QDRANT_COLLECTION = "crag_integration_local"
pytest -m integration -v
```

Ollama defaults to `http://localhost:11434`, with `qwen2.5:3b` for chat and
`nomic-embed-text` for embeddings. Pull the selected models before running
local live tests:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

Ollama embeddings must use a separate Qdrant collection from OpenAI embeddings.
Use `CRAG_INTEGRATION_OLLAMA_QDRANT_COLLECTION` with a dedicated
`crag_integration_` name when constructing a Qdrant provider with
`OllamaEmbeddingProvider`. Vector dimensions are validated and are never
assumed to match OpenAI.

An optional bounded timeout can be set with
`CRAG_INTEGRATION_TIMEOUT_SECONDS` (the default is 20 seconds). The live suite
makes a small number of OpenAI, Qdrant, Tavily, and local Ollama requests.
Hosted provider calls may incur usage charges; local Ollama calls do not
create hosted-provider usage. The workflow/API test may make additional chat,
embedding, and web-search requests. Skipped tests are reported by pytest and
do not contact external services.

Run the ordinary suite separately to verify the no-network default:

```powershell
pytest -q
```

Mocked unit tests and live integration tests are intentionally separate:
passing unit tests proves deterministic local behavior, while passing live
tests proves only the configured services and credentials worked at test time.

## Architecture status

### Implemented

- `src/crag/` package with typed settings, logging setup, exceptions, and a
  local-only health status check.
- Validated document, evidence, web-result, and grading models.
- Deterministic character-based parent-child document ingestion with
  configurable overlapping child chunks.
- Async provider contracts for chat, embeddings, vector storage, and web
  search, with explicit timeout parameters.
- Concrete asynchronous OpenAI, Ollama, Qdrant, and Tavily adapters with
  deterministic provider-error translation and mocked unit tests.
- A typed LangGraph corrective retrieval workflow that grades internal evidence,
  falls back to web search, and generates grounded answers.
- A FastAPI application factory with `/api/v1/ask`, local liveness, and
  dependency-configuration readiness endpoints.
- Deterministic unit tests and an integration-test convention.
- Explicit Ollama chat and embedding model configuration with separate
  Ollama-Qdrant collection configuration.
- Opt-in live smoke tests for OpenAI, Qdrant, Tavily, and the workflow/API path.
- Packaging, Ruff, pytest, environment, and project engineering configuration.

Live smoke tests do not replace production monitoring or a full end-to-end
deployment test. They require a pre-provisioned, dedicated Qdrant collection;
the adapter may initialize that collection but the tests do not delete or
recreate collections.

## Document ingestion

`crag.ingestion.ingest_document` accepts document text and source metadata and
returns ordered `DocumentChunk` records. It creates contiguous parent sections
and smaller overlapping child chunks linked by `parent_id`; IDs are stable for
the same document identity, content, and chunk position. Chunking is
character-based, not token-aware, and does not normalize text or split on
semantic boundaries.
