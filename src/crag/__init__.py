"""CRAG application foundation."""

from crag.services.health import HealthStatus, get_health_status

__all__ = ["HealthStatus", "get_health_status"]
