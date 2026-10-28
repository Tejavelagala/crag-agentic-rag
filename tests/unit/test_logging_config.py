"""Logging configuration behavior tests."""

import logging

from crag.logging_config import configure_logging


def test_logging_configuration_is_idempotent() -> None:
    logger = logging.getLogger("crag")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    first = configure_logging("INFO")
    second = configure_logging("DEBUG", json_logs=True)

    assert first is second
    assert len(second.handlers) == 1
    assert second.level == logging.DEBUG
    assert second.handlers[0].name == "crag-console"
    assert '"level"' in second.handlers[0].formatter._fmt
