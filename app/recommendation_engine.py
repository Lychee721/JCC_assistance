from __future__ import annotations

from collections import Counter
from typing import Any

from app.item_graph import ItemGraphRepository
from app.scoring_model import LinearPreferenceScorer, ModelContext


INTENT_CN = {
    "carry_ad": "物理主C",
    "carry_ap": "法系主C",
    "frontline": "前排",
    "balanced": "通用",
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
            missing_context.append("你还没告诉我这套装备准备给谁。")
        if not request.get("stage"):
            missing_context.append("你还没告诉我当前阶段，比如 3-2 或 4-1。")

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
            return "按你现在的散件，当前还不能合成图谱里的有效成装。你可以补充主C、阶段或者截图，我再继续细排。"

        top = ranked[:3]
        names = "、".join(item["name"] for item in top)
        intent = request.get("intent", "balanced")
        intent_cn = INTENT_CN.get(intent, INTENT_CN["balanced"])
        return f"按你当前散件，如果走{intent_cn}思路，优先看 {names}。我已经把每件装备的特征和分数贡献拆开，适合直接讲模型逻辑。"

    def _build_input_summary(self, request: dict[str, Any]) -> str:
        components = request.get("components", {})
        component_labels = {component["component_id"]: component["name"] for component in self.repository.components}
        component_text = "、".join(
            f"{component_labels.get(name, name)}x{count}"
            for name, count in sorted(components.items())
            if count > 0
        )
        target = request.get("target_champion") or "未指定主C"
        stage = request.get("stage") or "未指定阶段"
        intent = INTENT_CN.get(request.get("intent", "balanced"), "通用")
        return f"输入散件: {component_text} | 目标: {target} | 意图: {intent} | 阶段: {stage}"

    def _build_presentation_notes(self, ranked: list[dict[str, Any]], request: dict[str, Any]) -> list[str]:
        notes = [
            "先用当前 patch item graph 枚举所有可合成装备。",
            "再把每件装备转成稀疏特征向量，如 ad、mana、frontline、utility。",
            "最后用可解释线性模型按意图和阶段做打分排序。",
        ]
        if ranked:
            top = ranked[0]
            top_features = [entry["feature"] for entry in top["score_breakdown"][:3]]
            notes.append(f"当前第一名是 {top['name']}，主要因为特征 {', '.join(top_features)} 贡献最高。")
        if not request.get("target_champion"):
            notes.append("如果补充主C信息，模型还能加入 target match bonus。")
        return notes

    def _build_item_explanation(self, item_name: str, contributions: list[Any]) -> str:
        if not contributions:
            return f"{item_name} 进入候选，但当前没有明显特征优势。"
        top = contributions[:3]
        reason = "、".join(f"{entry.feature}({entry.contribution})" for entry in top)
        return f"{item_name} 得分高，主要来自 {reason}。"
