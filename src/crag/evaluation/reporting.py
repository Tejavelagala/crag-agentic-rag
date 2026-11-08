"""Safe lifecycle helpers for generated JSON reports."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def write_json_report(report: dict[str, Any], output_path: Path) -> None:
    """Archive an existing report before atomically publishing a new report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        evaluated_at = _report_timestamp(report)
        history_dir = output_path.parent / "history"
        history_dir.mkdir(exist_ok=True)
        archive_path = history_dir / f"{output_path.stem}-{evaluated_at}.json"
        suffix = 1
        while archive_path.exists():
            archive_path = (
                history_dir / f"{output_path.stem}-{evaluated_at}-{suffix}.json"
            )
            suffix += 1
        archive_path.write_bytes(output_path.read_bytes())
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def _report_timestamp(report: dict[str, Any]) -> str:
    value = report.get("evaluated_at")
    if isinstance(value, str):
        normalized = re.sub(r"[^0-9A-Za-z-]", "-", value).strip("-")
        if normalized:
            return normalized[:48]
    return datetime.now().strftime("%Y%m%dT%H%M%S")
