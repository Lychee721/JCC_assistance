from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppPaths:
    item_graph: Path
    component_catalog: Path
    system_prompt: Path
    llm_response_schema: Path


@dataclass
class AppConfig:
    name: str
    host: str
    port: int
    locale: str
    enable_llm_preview: bool
    enable_cnn_input: bool
    paths: AppPaths


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_app_config(config_path: str | None = None) -> AppConfig:
    resolved = Path(config_path or os.getenv("APP_CONFIG", "configs/app.example.yaml"))
    raw = load_yaml(resolved)
    app = raw["app"]
    paths = raw["paths"]
    features = raw["features"]
    return AppConfig(
        name=app["name"],
        host=app["host"],
        port=app["port"],
        locale=app["locale"],
        enable_llm_preview=features["enable_llm_preview"],
        enable_cnn_input=features["enable_cnn_input"],
        paths=AppPaths(
            item_graph=Path(paths["item_graph"]),
            component_catalog=Path(paths["component_catalog"]),
            system_prompt=Path(paths["system_prompt"]),
            llm_response_schema=Path(paths["llm_response_schema"]),
        ),
    )
