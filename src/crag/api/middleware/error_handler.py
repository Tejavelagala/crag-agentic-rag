"""Shared request validation error handling for API applications."""

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_error_handler(
    request: object, exc: RequestValidationError
) -> JSONResponse:
    """Return the public validation contract without leaking request details."""
    return JSONResponse(
        status_code=422, content={"detail": "Request validation failed."}
    )
