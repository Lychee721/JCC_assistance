from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class DemoScenarioService:
    def __init__(self, path: str | Path = "data/demo/demo_scenarios.json") -> None:
        self.path = Path(path)
        self.payload = self._load()

    def _load(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def list_scenarios(self) -> list[dict[str, Any]]:
        return self.payload["scenarios"]

    def get_scenario(self, scenario_id: str) -> dict[str, Any]:
        for scenario in self.payload["scenarios"]:
            if scenario["scenario_id"] == scenario_id:
                return scenario
        raise KeyError(scenario_id)
