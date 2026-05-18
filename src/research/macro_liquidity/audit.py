"""Audit artifacts for macro/liquidity research runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Mapping, Optional
from uuid import uuid4

import pandas as pd

from .policy import MacroLiquidityPolicy


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


class ResearchRunAudit:
    """Write reproducible research-run artifacts under `data/research_runs`."""

    def __init__(
        self,
        policy: Optional[MacroLiquidityPolicy] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.policy = policy or MacroLiquidityPolicy()
        self.run_id = run_id or pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        self.run_dir = Path(self.policy.audit.run_root) / self.run_id

    def prepare(self) -> Path:
        """Create the run directory and return it."""

        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "model_diagnostics").mkdir(exist_ok=True)
        return self.run_dir

    def write_json(self, name: str, payload: Mapping[str, object]) -> Path:
        """Write a JSON artifact if artifact writing is enabled."""

        self.prepare()
        path = self.run_dir / name
        if not self.policy.audit.write_artifacts:
            return path
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=_json_default)
        return path

    def write_frame(self, name: str, frame: pd.DataFrame) -> Path:
        """Write a DataFrame as CSV for broad local compatibility."""

        self.prepare()
        path = self.run_dir / name
        if not self.policy.audit.write_artifacts:
            return path
        frame.to_csv(path, index=False)
        return path

    def write_markdown(self, name: str, text: str) -> Path:
        """Write a Markdown artifact."""

        self.prepare()
        path = self.run_dir / name
        if not self.policy.audit.write_artifacts:
            return path
        path.write_text(text, encoding="utf-8")
        return path

    def write_run_summary(
        self,
        methodology: Mapping[str, object],
        current_label: str,
        expected_label: str,
        errors: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Write a compact run summary for audit review."""

        payload = {
            "run_id": self.run_id,
            "methodology": dict(methodology),
            "current_label": current_label,
            "expected_label": expected_label,
            "has_provider_errors": bool(errors is not None and not errors.empty),
        }
        return self.write_json("run_summary.json", payload)

