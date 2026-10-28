# Integration tests

Integration tests belong in this directory and must be marked with
`@pytest.mark.integration`. They may exercise configured external services only
when credentials and an explicit opt-in are present. Unit tests remain
deterministic and must never require network access or provider credentials.
