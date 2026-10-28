# CRAG engineering rules

- Target Python 3.11+ and use the `src/` layout.
- Use Pydantic v2 and `pydantic-settings` for typed configuration. Never
  require provider credentials for package imports or deterministic unit tests.
- Keep imports lightweight: no network calls or external-service connections at
  import time.
- Configure standard-library logging through one explicit, idempotent entry
  point. Do not configure logging during module imports.
- Use small modules, precise type annotations, clear errors, and a consistent
  application exception hierarchy. Do not swallow unexpected exceptions or
  fabricate successful results.
- Health checks may report only checks that actually ran. Do not claim an
  external service is healthy without making a real check.
- Future LangGraph workflows must use a typed `TypedDict` state.
- Future document-grading nodes must return structured outputs validated by
  Pydantic schemas, never raw text.
- Future external API calls must use explicit timeouts, appropriate exception
  handling, safe fallbacks, and deterministic tests.
- Keep the foundation free of Redis, Celery, Kubernetes, database services, and
  speculative infrastructure.
- Run Ruff and pytest before review; tests must not require network access or
  real credentials.
