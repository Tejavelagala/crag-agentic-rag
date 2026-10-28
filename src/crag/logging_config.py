"""Centralized, idempotent logging configuration."""

import logging
import logging.config
from typing import Final

_HANDLER_NAME: Final = "crag-console"
_LOGGER_NAME: Final = "crag"


def configure_logging(
    level: str = "INFO",
    *,
    json_logs: bool = False,
) -> logging.Logger:
    """Configure the CRAG logger once and return it.

    The function only owns handlers attached to the ``crag`` logger. Calling it
    repeatedly updates the level without adding duplicate handlers.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(level.upper())
    logger.propagate = False

    handler = next(
        (item for item in logger.handlers if item.name == _HANDLER_NAME),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.name = _HANDLER_NAME
        logger.addHandler(handler)

    formatter = (
        logging.Formatter(
            '{"level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        )
        if json_logs
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.setFormatter(formatter)
    handler.setLevel(level.upper())
    return logger
