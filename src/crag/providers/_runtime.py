"""Shared runtime validation for concrete provider adapters."""

import asyncio
import math
from collections.abc import Awaitable
from typing import TypeVar

from crag.exceptions import ProviderTimeoutError

ResultT = TypeVar("ResultT")


def validate_timeout(timeout: float) -> None:
    """Require a finite, positive timeout in seconds."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite value greater than zero")


async def with_timeout(operation: Awaitable[ResultT], timeout: float) -> ResultT:
    """Bound an async provider operation and translate timeout failures."""
    validate_timeout(timeout)
    try:
        return await asyncio.wait_for(operation, timeout=timeout)
    except TimeoutError as exc:
        raise ProviderTimeoutError(
            "Provider operation exceeded its configured timeout"
        ) from exc
