from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class LLMOrchestrator:
    def __init__(
        self,
        system_prompt_path: str | Path = "prompts/item_recommendation_system.txt",
        user_template_path: str | Path = "prompts/item_recommendation_user_template.txt",
    ) -> None:
        self.system_prompt_path = Path(system_prompt_path)
        self.user_template_path = Path(user_template_path)

    def _read(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    def build_messages(self, request: dict[str, Any], recommendation_payload: dict[str, Any]) -> list[dict[str, str]]:
        system_prompt = self._read(self.system_prompt_path)
        template = self._read(self.user_template_path)
        user_prompt = (
            template.replace("{{ user_question }}", request.get("user_question") or "")
            .replace("{{ intent }}", request.get("intent", "balanced"))
            .replace("{{ target_champion }}", request.get("target_champion") or "")
            .replace("{{ stage }}", request.get("stage") or "")
            .replace("{{ components_json }}", json.dumps(request.get("components", {}), ensure_ascii=False))
            .replace(
                "{{ recommendation_payload_json }}",
                json.dumps(recommendation_payload, ensure_ascii=False, indent=2),
            )
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
