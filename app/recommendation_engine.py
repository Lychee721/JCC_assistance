from __future__ import annotations

from collections import Counter
from typing import Any

from app.item_graph import ItemGraphRepository
from app.scoring_model import LinearPreferenceScorer, ModelContext


INTENT_CN = {
    "carry_ad": "\u7269\u7406\u4e3bC",
    "carry_ap": "\u6cd5\u7cfb\u4e3bC",
    "frontline": "\u524d\u6392",
    "balanced": "\u901a\u7528",
}


class ItemRecommendationEngine:
    def __init__(self, catalog_path: str = "data/processed/item_graph.runtime.json") -> None:
        self.repository = ItemGraphRepository(catalog_path)
        self.scorer = LinearPreferenceScorer()

    def get_craftable_items(self, components: dict[str, int]) -> list[dict[str, Any]]:
        inventory = Counter({key: value for key, value in components.items() if value > 0})
        craftable: list[dict[str, Any]] = []
        for item in self.repository.items:
            required = Counter(item["components"])
            if all(inventory[name] >= count for name, count in required.items()):
                craftable.append(item)
        return craftable

    def build_payload(self, request: dict[str, Any]) -> dict[str, Any]:
        craftable = self.get_craftable_items(request["components"])
        context = ModelContext(
            intent=request.get("intent", "balanced"),
            stage=request.get("stage"),
            target_champion=request.get("target_champion"),
        )

        ranked = []
        for item in craftable:
            score, feature_vector, contributions = self.scorer.score(item, context)
            explanation_text = self._build_item_explanation(item["name"], contributions)
            ranked.append(
                {
                    "item_id": item["item_id"],
                    "stable_id": item.get("stable_id"),
                    "name": item["name"],
                    "components": item["components"],
                    "score": score,
                    "reason_tags": item["tags"],
                    "feature_vector": feature_vector,
                    "score_breakdown": [
                        {
                            "feature": contribution.feature,
                            "value": contribution.value,
                            "weight": contribution.weight,
                            "contribution": contribution.contribution,
                        }
                        for contribution in contributions[:6]
                    ],
                    "explanation_text": explanation_text,
                }
            )
        ranked.sort(key=lambda entry: (-entry["score"], entry["name"]))

        missing_context = []
        if not request.get("target_champion"):
            missing_context.append("\u4f60\u8fd8\u6ca1\u544a\u8bc9\u6211\u8fd9\u5957\u88c5\u5907\u51c6\u5907\u7ed9\u8c01\u3002")
        if not request.get("stage"):
            missing_context.append("\u4f60\u8fd8\u6ca1\u544a\u8bc9\u6211\u5f53\u524d\u9636\u6bb5\uff0c\u6bd4\u5982 3-2 \u6216 4-1\u3002")

        return {
            "craftable_items": ranked,
            "top_recommendations": [entry["item_id"] for entry in ranked[:3]],
            "missing_context": missing_context,
            "answer_text": self._build_preview_answer(ranked, request),
            "model_name": self.scorer.model_name,
            "model_version": self.scorer.model_version,
            "input_summary": self._build_input_summary(request),
            "presentation_notes": self._build_presentation_notes(ranked, request),
            "scenario_id": request.get("scenario_id"),
            "scenario_title": request.get("scenario_title"),
        }

    def _build_preview_answer(self, ranked: list[dict[str, Any]], request: dict[str, Any]) -> str:
        if not ranked:
            return (
                "\u6309\u4f60\u73b0\u5728\u7684\u6563\u4ef6\uff0c\u5f53\u524d\u8fd8\u4e0d\u80fd\u5408\u6210"
                "\u56fe\u8c31\u91cc\u7684\u6709\u6548\u6210\u88c5\u3002\u4f60\u53ef\u4ee5\u8865\u5145\u4e3bC"
                "\u3001\u9636\u6bb5\u6216\u8005\u622a\u56fe\uff0c\u6211\u518d\u7ee7\u7eed\u7ec6\u6392\u3002"
            )

        top = ranked[:3]
        names = "\u3001".join(item["name"] for item in top)
        intent = request.get("intent", "balanced")
        intent_cn = INTENT_CN.get(intent, INTENT_CN["balanced"])
        return (
            f"\u6309\u4f60\u5f53\u524d\u6563\u4ef6\uff0c\u5982\u679c\u8d70{intent_cn}\u601d\u8def\uff0c"
            f"\u4f18\u5148\u770b {names}\u3002\u6211\u5df2\u7ecf\u628a\u6bcf\u4ef6\u88c5\u5907\u7684\u7279\u5f81"
            "\u548c\u5206\u6570\u8d21\u732e\u62c6\u5f00\uff0c\u9002\u5408\u76f4\u63a5\u8bb2\u6a21\u578b\u903b\u8f91\u3002"
        )

    def _build_input_summary(self, request: dict[str, Any]) -> str:
        components = request.get("components", {})
        component_labels = {component["component_id"]: component["name"] for component in self.repository.components}
        component_text = "\u3001".join(
            f"{component_labels.get(name, name)}x{count}"
            for name, count in sorted(components.items())
            if count > 0
        )
        target = request.get("target_champion") or "\u672a\u6307\u5b9a\u4e3bC"
        stage = request.get("stage") or "\u672a\u6307\u5b9a\u9636\u6bb5"
        intent = INTENT_CN.get(request.get("intent", "balanced"), "\u901a\u7528")
        return (
            f"\u8f93\u5165\u6563\u4ef6: {component_text} | \u76ee\u6807: {target} | "
            f"\u610f\u56fe: {intent} | \u9636\u6bb5: {stage}"
        )

    def _build_presentation_notes(self, ranked: list[dict[str, Any]], request: dict[str, Any]) -> list[str]:
        notes = [
            "\u5148\u7528\u5f53\u524d patch item graph \u679a\u4e3e\u6240\u6709\u53ef\u5408\u6210\u88c5\u5907\u3002",
            (
                "\u518d\u628a\u6bcf\u4ef6\u88c5\u5907\u8f6c\u6210\u7a00\u758f\u7279\u5f81\u5411\u91cf\uff0c\u5982 "
                "ad\u3001mana\u3001frontline\u3001utility\u3002"
            ),
            "\u6700\u540e\u7528\u53ef\u89e3\u91ca\u7ebf\u6027\u6a21\u578b\u6309\u610f\u56fe\u548c\u9636\u6bb5\u505a\u6253\u5206\u6392\u5e8f\u3002",
        ]
        if ranked:
            top = ranked[0]
            top_features = [entry["feature"] for entry in top["score_breakdown"][:3]]
            notes.append(
                f"\u5f53\u524d\u7b2c\u4e00\u540d\u662f {top['name']}\uff0c\u4e3b\u8981\u56e0\u4e3a\u7279\u5f81 "
                f"{', '.join(top_features)} \u8d21\u732e\u6700\u9ad8\u3002"
            )
        if not request.get("target_champion"):
            notes.append(
                "\u5982\u679c\u8865\u5145\u4e3bC\u4fe1\u606f\uff0c\u6a21\u578b\u8fd8\u80fd\u52a0\u5165 target match bonus\u3002"
            )
        return notes

    def _build_item_explanation(self, item_name: str, contributions: list[Any]) -> str:
        if not contributions:
            return (
                f"{item_name} \u8fdb\u5165\u5019\u9009\uff0c\u4f46\u5f53\u524d\u6ca1\u6709\u660e\u663e"
                "\u7279\u5f81\u4f18\u52bf\u3002"
            )
        top = contributions[:3]
        reason = "\u3001".join(f"{entry.feature}({entry.contribution})" for entry in top)
        return f"{item_name} \u5f97\u5206\u9ad8\uff0c\u4e3b\u8981\u6765\u81ea {reason}\u3002"
