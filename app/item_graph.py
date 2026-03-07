from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ItemGraphRepository:
    def __init__(self, runtime_path: str | Path = "data/processed/item_graph.runtime.json") -> None:
        self.runtime_path = Path(runtime_path)
        self.graph = self._load_graph()

    def _load_graph(self) -> dict[str, Any]:
        fallback = Path("data/seed/item_recipes.example.json")
        source_path = self.runtime_path if self.runtime_path.exists() else fallback
        with source_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @property
    def items(self) -> list[dict[str, Any]]:
        return self.graph["items"]

    @property
    def components(self) -> list[dict[str, Any]]:
        return self.graph["components"]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "version": self.graph.get("version"),
            "source": self.graph.get("source"),
            "set_number": self.graph.get("set_number"),
            "set_name": self.graph.get("set_name"),
        }
