"""Local application health and status reporting."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class HealthStatus(BaseModel):
    """Status of checks performed locally by the application."""

    model_config = ConfigDict(frozen=True)

    status: Literal["healthy"] = "healthy"
    service: str = "crag"
    checked_at: datetime
    checks: dict[str, bool]


def get_health_status() -> HealthStatus:
    """Report deterministic local process health.

    External services are intentionally not checked here; they must not be
    reported as healthy until a real integration check is implemented.
    """
    return HealthStatus(
        checked_at=datetime.now(UTC),
        checks={"application": True},
    )
