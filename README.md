# CRAG: Corrective Retrieval-Augmented Generation

This repository is a production-oriented foundation for a stateful corrective
RAG system. It provides typed configuration, provider adapters, deterministic
document ingestion, a LangGraph workflow, and a FastAPI application factory.

## Prerequisites and setup

Requires Python 3.11 or newer.

Local runtime operation requires:

- Ollama running at the configured `CRAG_OLLAMA_BASE_URL` (by default
  `http://localhost:11434`).
- A reachable Qdrant service configured with `CRAG_QDRANT_URL`.
- The configured Ollama chat and embedding models.

The repository does not start Ollama or Qdrant for you. Hosted OpenAI and
Tavily services are optional and are not required for local provider mode.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

`.env` is ignored by Git. Keep credentials out of source control.

## Environment configuration

Copy `.env.example` to `.env` and set only the values needed for the selected
deployment. The example contains placeholders only; do not copy credentials
into the repository. Settings use the `CRAG_` environment-variable prefix,
and process environment variables override values loaded from `.env`.

The important provider settings are:

- `CRAG_PROVIDER_MODE=local` selects Ollama chat and embeddings. Local mode
  does not require OpenAI or Tavily credentials.
- `CRAG_PROVIDER_MODE=hosted` selects OpenAI chat and embeddings and requires
  `CRAG_OPENAI_API_KEY`.
- `CRAG_OLLAMA_BASE_URL`, `CRAG_OLLAMA_CHAT_MODEL`, and
  `CRAG_OLLAMA_EMBEDDING_MODEL` select the local Ollama service and models.
- `CRAG_QDRANT_URL` selects the Qdrant service used by the configured
  workflow. `CRAG_QDRANT_API_KEY` is optional for deployments that require
  Qdrant authentication.
- `CRAG_ENABLE_TAVILY=true` enables hosted web fallback and requires
  `CRAG_TAVILY_API_KEY`. Leave it false for local-only operation.

The complete setting template is [`.env.example`](.env.example). Disabled
integrations do not require their credentials.

## Ollama setup and verification

The default local models are:

- Chat: `qwen2.5:3b`
- Embeddings: `nomic-embed-text`

Check that the Ollama command-line client can reach the local service and list
the models already installed:

```powershell
ollama list
```

The output should contain the configured chat and embedding model names. This
is a read-only check. If either model is missing, install it using your normal
Ollama administration process before starting the application; this guide
does not download models automatically.

The application performs an embedding dimension probe during startup. That
probe contacts Ollama and is not performed when `crag.main:app` is merely
imported. Qdrant connectivity is exercised when the configured collection is
accessed; readiness does not claim that Qdrant or Ollama are healthy.

## Frontend dashboard

The project includes a same-origin, static Bootstrap 5 and vanilla JavaScript
dashboard served at `/frontend/` by the FastAPI application. It uses the live
health endpoints and `POST /api/v1/ask`, displays retrieved evidence, grading
decisions, stage states, source identifiers, and explicit web-fallback status
returned by the API. No frontend build toolchain or API credentials are
required.

Start the API with the local configuration described above, then open
`http://127.0.0.1:8010/frontend/` (or the port selected for Uvicorn). The
dashboard presents only reported stage states and keeps safe-failure responses
free of fabricated answers. Read-only evaluation data is available through
`GET /api/v1/evaluation` and
`GET /api/v1/evaluation/benchmark` when the fixed generated report files are
present. The dashboard identifies report counts and metrics as generated
evaluation data, including expandable benchmark case classifications and
variant metrics; benchmark metrics are not RAGAS measurements. Chat providers
request zero-temperature sampling for grading and answer generation where the
provider supports it, while valid negative grading decisions remain respected.

## Backend architecture

The backend is organized as a modular monolith under `src/crag/`. Shared
configuration, exceptions, logging, and domain models live in `crag.core`;
corrective retrieval lives in `crag.rag`; extraction and chunking live in
`crag.ingestion`; persistence adapters live in `crag.storage`; provider
adapters remain under `crag.providers`; and application orchestration is under
`crag.services`. Evaluation runners, benchmarks, and report publication are
grouped under `crag.evaluation`. The `crag.api` package preserves the existing
FastAPI factory and route contracts, while historical root module imports
remain compatibility shims.

## Qdrant setup and collection isolation

The repository expects an already running, reachable Qdrant service at
`CRAG_QDRANT_URL`. Qdrant startup and persistence are deployment concerns
outside this repository; no Qdrant service process is started by the package.
Before starting the API, verify the service using the connectivity and
administration checks provided by your Qdrant deployment. Do not use a
production collection for a smoke test.

Collection names are selected by provider mode:

- Local Ollama mode uses `CRAG_OLLAMA_QDRANT_COLLECTION_NAME` (default
  `crag_chunks_ollama`).
- Hosted mode uses `CRAG_QDRANT_COLLECTION_NAME` (default `crag_chunks`).
- Evaluation runs require a name beginning with `crag_evaluation_`.
- Integration tests require a name beginning with `crag_integration_`.

Keep production, evaluation, and integration collections separate. The
embedding model used to write vectors must match the model used for queries.
The adapter creates a missing collection with cosine distance and the probed
embedding dimension, and validates the dimension of an existing collection
before upserting or retrieving. A dimension mismatch is a configuration/data
compatibility error; do not fix it by deleting or recreating an existing
collection without an approved migration plan.

`/health/ready` reports application and workflow-configuration readiness only.
It does not claim that Ollama or Qdrant are healthy; use service-specific
connectivity checks and a safe, dedicated integration test for that purpose.

## Document ingestion and indexing

The application supports PDF, DOCX, and UTF-8 TXT uploads through
`POST /api/v1/documents`. Files are stored under the configured
`CRAG_DOCUMENT_STORAGE_DIR`, registered in SQLite at
`CRAG_DOCUMENT_REGISTRY_PATH`, chunked with the existing deterministic
parent/child implementation, and indexed in the dedicated
`CRAG_UPLOADED_COLLECTION_NAME` collection. Content hashes reject duplicate
uploads. Use `GET /api/v1/documents`, `GET /api/v1/documents/search`,
`GET /api/v1/documents/{id}/chunks`, `POST /api/v1/documents/{id}/reindex`,
and `DELETE /api/v1/documents/{id}` for inspection and lifecycle operations.
The `/api/v1/ask` retrieval path queries both the configured demo/production
collection and the uploaded collection; failed, deleting, and deleted records
are not intentionally retrieved.

Uploads are bounded by `CRAG_MAX_UPLOAD_SIZE_BYTES` and use server-generated
storage names. Scanned PDFs without extractable text are rejected because OCR
is not implemented. Authentication and per-owner document namespaces are not
implemented; the registry therefore applies duplicate detection globally.
The frontend provides basic upload, listing, and delete controls, while
operator-facing authorization remains a deployment responsibility.

`crag.ingestion.ingest_document` remains the reusable ingestion primitive. It
accepts document text, a `document_id`, optional JSON metadata, and character
based parent/child sizing options, and returns deterministic `DocumentChunk`
objects.

Chunking is character-based rather than token-aware or semantic. Parents are
non-overlapping sections; children overlap within each parent and retain
parent/document metadata. The current retrieval path does not provide an
operator-configurable score threshold or parent-level deduplication policy.

## Isolated local demo

The fixed, non-sensitive demo corpus is stored in
[`demo/dataset.json`](demo/dataset.json). It is separate from both evaluation
datasets. The demo utility uses only the existing Ollama, Qdrant, chunking,
and CRAG workflow abstractions; it does not change production routing or
grading behavior.

Install the declared project dependencies in the active virtual environment
before running the utility. The demo requires Ollama with `qwen2.5:3b` and
`nomic-embed-text`, a reachable Qdrant service, and local provider mode. Use a
collection beginning with `crag_demo_`; never point the utility at production,
evaluation, or integration collections.

Prepare writes the fixed demo chunks to the explicitly named demo collection.
It refuses an existing collection when ownership, dataset version, distance,
or vector dimensions cannot be verified. Preparation is never performed during
API startup:

```powershell
$env:CRAG_PROVIDER_MODE = "local"
$env:CRAG_QDRANT_URL = "<approved-local-qdrant-url>"
$env:CRAG_OLLAMA_BASE_URL = "http://localhost:11434"
$env:CRAG_OLLAMA_CHAT_MODEL = "qwen2.5:3b"
$env:CRAG_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
$env:CRAG_ENABLE_TAVILY = "false"
python -m crag.demo prepare --collection crag_demo_local
```

Verify collection metadata, ownership, dimensions, and the prepared point
set. The question below is one of the fixed questions in the demo corpus:

```powershell
python -m crag.demo verify `
  --collection crag_demo_local `
  --question "Which chat and embedding models does the local CRAG demo use?"
```

The verification result distinguishes metadata, ownership, dimension, point,
and workflow checks. A grounded workflow result reports an answer and source
IDs. If the grader rejects evidence or web fallback is disabled, verification
reports the actual safe failure diagnostics instead of bypassing the grader or
fabricating an answer.

Start the existing API separately after preparation:

```powershell
uvicorn crag.main:app
```

The API uses the collection selected by
`CRAG_OLLAMA_QDRANT_COLLECTION_NAME`. The demo utility does not configure the
running process automatically, so set that value to the same approved
`crag_demo_` collection before starting the API.

Cleanup is explicit and point-scoped. It requires the run ID printed by
`prepare`; it verifies the recorded manifest and point metadata before
deleting only those points. It never deletes a collection:

```powershell
python -m crag.demo cleanup `
  --collection crag_demo_local `
  --run-id "<run-id-from-prepare>"
```

The commands above are a future live-demo procedure and were not executed as
part of offline validation. Unit tests use fake providers and clients; live
demo execution requires separately approved Ollama and Qdrant services.

## Run the API locally

From PowerShell, activate the environment and install the project:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Select local Ollama mode and start the application with Uvicorn:

```powershell
$env:CRAG_PROVIDER_MODE = "local"
$env:CRAG_OLLAMA_BASE_URL = "http://localhost:11434"
$env:CRAG_OLLAMA_CHAT_MODEL = "qwen2.5:3b"
$env:CRAG_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
uvicorn crag.main:app
```

The application builds its configured providers during startup. Importing
`crag.main:app` does not contact Ollama, Qdrant, OpenAI, or Tavily. The
embedding dimension probe and Qdrant client are initialized only during the
FastAPI lifespan and the owned Qdrant client is closed during shutdown.

In another PowerShell window, check the local health routes:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

Submit a question using the actual request schema:

```powershell
$body = @{ question = "What does this CRAG deployment do?" } | ConvertTo-Json
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/api/v1/ask `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

The request body accepts one required `question` string and rejects blank
questions, overlong questions, and unknown fields. Validation failures return
HTTP 422 with a safe generic detail. If retrieval or grading cannot produce
grounded evidence and web fallback is disabled, the API returns HTTP 503 with
safe diagnostics such as `web_search_failed`; it does not fabricate an answer.

Run the deterministic test suite with:

```powershell
pytest -m "not integration" -q
```

## Run the local evaluation

The fixed, secret-free dataset is stored in
`evaluation/dataset.json`. The runner inserts its documents into a dedicated
evaluation collection, exercises the real `/api/v1/ask` pipeline, records
machine-readable results, and removes only the points created by that run.
Evaluation output is ignored under `evaluation/results/`.

With Ollama and Qdrant running, use PowerShell:

```powershell
$env:CRAG_PROVIDER_MODE = "local"
$env:CRAG_EVALUATION_QDRANT_COLLECTION = "crag_evaluation_local"
python -m crag.evaluation `
  --dataset evaluation/dataset.json `
  --collection crag_evaluation_local `
  --output evaluation/results/latest.json
Get-Content evaluation/results/latest.json
```

The runner records the dataset version, timestamp, provider mode, configured
model names, collection name, per-case HTTP outcomes, fallback behavior, and
mean/median latency. Its context-relevance, faithfulness, and answer-relevance
values are transparent deterministic expected-fact/source heuristics; they are
not RAGAS scores. Context relevance includes completed retrieval attempts,
including fallback cases, while faithfulness and answer relevance are measured
only for answer-bearing responses. Unexpected HTTP errors contribute to the
error rate; an explicitly expected local-only 503 fallback is recorded as the
expected behavior rather than as an unexpected error. No hosted model or API
credential is required for the local evaluation. Unsupported questions are
expected to return the safe local-only web-search error when Tavily is
disabled.

Each evaluation execution uses a unique run identifier in point metadata and
cleans up only points owned by that run and dataset version. Use a dedicated
evaluation collection and do not run concurrent evaluations against the same
collection.

## Benchmark evidence grading

The versioned `evaluation/grader_benchmark.json` contains 32 balanced,
human-labeled question/evidence pairs covering direct facts, paraphrases,
partial evidence, misleading lexical overlap, and irrelevant evidence. Run the
production prompt and a separate experimental prompt variant with:

```powershell
$env:CRAG_PROVIDER_MODE = "local"
python -m crag.grader_benchmark `
  --benchmark evaluation/grader_benchmark.json `
  --output evaluation/results/grader_benchmark.json
Get-Content evaluation/results/grader_benchmark.json
```

Generated reports preserve the previous fixed-path report under
`evaluation/results/history/` before publishing a new latest report. The
benchmark report records its evaluated timestamp and production prompt
version/hash so the dashboard can identify report freshness and grading parity.

The benchmark reports the confusion matrix, precision, recall, F1, false
rejection and false acceptance rates, invalid structured-output count, and
latency distribution. It does not change workflow grading or routing. Results
are experimental and use the locally configured chat model; no alternative
compatible local chat model is downloaded automatically.

The current benchmark demonstrates a known grader limitation: the local
`qwen2.5:3b` grader has produced false rejections of relevant evidence while
maintaining zero false acceptance in the recorded benchmark runs. Do not
interpret the benchmark as justification for keyword overrides or bypassing
structured grading.

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

## Troubleshooting and limitations

### Ollama is unavailable or a model is missing

Run `ollama list` and confirm that Ollama is reachable and that both
`qwen2.5:3b` and `nomic-embed-text` are installed. Check
`CRAG_OLLAMA_BASE_URL` and the two model settings. Startup performs an
embedding probe, so an unavailable Ollama service or missing embedding model
prevents the configured application from becoming ready.

### Qdrant is unavailable

Confirm that the configured Qdrant service is running and that
`CRAG_QDRANT_URL` is reachable from the API process. Check authentication when
`CRAG_QDRANT_API_KEY` is configured. Use a dedicated integration or evaluation
collection for validation; do not delete or recreate a production collection
as a generic troubleshooting step.

### Vector-dimension mismatch

The embedding provider determines the expected vector dimension at startup,
and the Qdrant adapter validates vectors and existing collection dimensions.
Use the same embedding model for indexing and querying and select the correct
provider-specific collection. A mismatch requires an explicit migration or
new isolated collection decision; it is not silently repaired.

### API startup or readiness failure

Inspect the Uvicorn startup log for configuration or Ollama embedding-probe
failures, and inspect request or collection-operation errors for Qdrant
connectivity failures. `/health/live` checks application liveness. `/health/ready`
checks application and workflow configuration readiness only, not external
dependency health. Importing `crag.main:app` alone should not contact external
services.

### Retrieval rejection and disabled web fallback

The workflow retrieves internal evidence and grades it before generating an
answer. If retrieval fails, evidence is rejected, or no evidence is accepted,
the workflow attempts web fallback. With `CRAG_ENABLE_TAVILY=false`, that
fallback is intentionally unavailable and the API returns a safe 503 rather
than inventing evidence or an answer. The response diagnostics distinguish
retrieval, grading, web-search, and answer-generation failures without exposing
provider secrets.

### Known quality limitations

Ingestion is character-based and not token-aware or semantic. Parent and child
chunks can both be retrieved, and the current implementation does not apply a
retrieval-score threshold or parent-level deduplication. The fixed evaluation
metrics are deterministic project heuristics, not official RAGAS metrics.
The local grader is conservative and has demonstrated false rejections of
relevant evidence; the production prompt and routing remain unchanged until a
controlled experiment demonstrates a safer improvement.

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
