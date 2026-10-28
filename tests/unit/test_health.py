"""Health behavior tests."""

from datetime import UTC

from crag.health import get_health_status


def test_health_status_reports_only_local_application_check() -> None:
    status = get_health_status()

    assert status.status == "healthy"
    assert status.service == "crag"
    assert status.checks == {"application": True}
    assert status.checked_at.tzinfo == UTC
