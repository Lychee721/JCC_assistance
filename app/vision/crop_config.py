from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SlotBox:
    slot_id: str
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass
class ReplaySlotLayout:
    layout_name: str
    width: int
    height: int
    slots: list[SlotBox]


def load_slot_layout(path: str | Path = "configs/replay_slot_layout.json") -> ReplaySlotLayout:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return ReplaySlotLayout(
        layout_name=payload["layout_name"],
        width=payload["reference_resolution"]["width"],
        height=payload["reference_resolution"]["height"],
        slots=[SlotBox(**slot) for slot in payload["slots"]],
    )
