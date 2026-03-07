from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FEATURE_KEYS = [
    "ad",
    "ap",
    "attack_speed",
    "mana",
    "crit",
    "carry",
    "frontline",
    "survivability",
    "sustain",
    "utility",
    "anti_tank",
    "scaling",
    "tempo",
    "emblem",
    "flex",
]

INTENT_WEIGHT_MATRIX: dict[str, dict[str, float]] = {
    "carry_ad": {
        "ad": 2.6,
        "attack_speed": 1.2,
        "crit": 1.8,
        "carry": 1.4,
        "anti_tank": 0.9,
        "tempo": 0.5,
    },
    "carry_ap": {
        "ap": 2.7,
        "mana": 1.8,
        "carry": 1.1,
        "utility": 0.7,
        "scaling": 0.8,
        "tempo": 0.4,
    },
    "frontline": {
        "frontline": 2.8,
        "survivability": 1.9,
        "sustain": 1.0,
        "utility": 0.5,
        "mana": 0.3,
    },
    "balanced": {
        "ad": 0.8,
        "ap": 0.8,
        "attack_speed": 0.6,
        "mana": 0.6,
        "crit": 0.6,
        "carry": 0.8,
        "frontline": 0.8,
        "survivability": 0.8,
        "utility": 0.8,
        "flex": 0.7,
    },
}


@dataclass
class ScoreContribution:
    feature: str
    value: float
    weight: float
    contribution: float


@dataclass
class ModelContext:
    intent: str
    stage: str | None = None
    target_champion: str | None = None


class ItemFeatureExtractor:
    def extract(self, item: dict[str, Any]) -> dict[str, float]:
        tag_set = set(item.get("tags", []))
        features = {key: 0.0 for key in FEATURE_KEYS}
        for key in FEATURE_KEYS:
            if key in tag_set:
                features[key] = 1.0

        effects = item.get("effects", {})
        if effects.get("AD") or effects.get("BonusDamage"):
            features["ad"] = max(features["ad"], 1.0)
        if effects.get("AP") or effects.get("AbilityPower"):
            features["ap"] = max(features["ap"], 1.0)
        if effects.get("AS") or effects.get("AttackSpeed"):
            features["attack_speed"] = max(features["attack_speed"], 1.0)
        if effects.get("Mana") or effects.get("ManaRegen") or effects.get("FlatManaRestore"):
            features["mana"] = max(features["mana"], 1.0)
        if effects.get("CritChance") or effects.get("CritDamageToGive"):
            features["crit"] = max(features["crit"], 1.0)
        if effects.get("Armor") or effects.get("MR") or effects.get("HP"):
            features["frontline"] = max(features["frontline"], 1.0)
        if effects.get("ShieldDuration") or effects.get("ShieldSize") or effects.get("DamageReduction"):
            features["survivability"] = max(features["survivability"], 1.0)
        if effects.get("LifeSteal") or effects.get("Omnivamp"):
            features["sustain"] = max(features["sustain"], 1.0)

        return features


class LinearPreferenceScorer:
    model_name = "interpretable_linear_ranker"
    model_version = "v2"

    def __init__(self) -> None:
        self.extractor = ItemFeatureExtractor()

    def score(self, item: dict[str, Any], context: ModelContext) -> tuple[float, dict[str, float], list[ScoreContribution]]:
        features = self.extractor.extract(item)
        weights = dict(INTENT_WEIGHT_MATRIX.get(context.intent, INTENT_WEIGHT_MATRIX["balanced"]))
        self._apply_stage_adjustments(weights, context.stage)

        contributions: list[ScoreContribution] = []
        score = 0.0
        for feature, value in features.items():
            weight = weights.get(feature, 0.0)
            contribution = round(value * weight, 3)
            if contribution:
                contributions.append(
                    ScoreContribution(
                        feature=feature,
                        value=value,
                        weight=weight,
                        contribution=contribution,
                    )
                )
                score += contribution

        priority_boost = float(item.get("priority_hints", {}).get(context.intent, 0.0)) * 3.5
        if priority_boost:
            contributions.append(
                ScoreContribution(
                    feature="priority_hint",
                    value=1.0,
                    weight=round(priority_boost, 3),
                    contribution=round(priority_boost, 3),
                )
            )
            score += priority_boost

        champion_boost = self._target_match_bonus(item, context.target_champion)
        if champion_boost:
            contributions.append(
                ScoreContribution(
                    feature="target_match",
                    value=1.0,
                    weight=champion_boost,
                    contribution=champion_boost,
                )
            )
            score += champion_boost

        contributions.sort(key=lambda entry: (-entry.contribution, entry.feature))
        return round(score, 2), features, contributions

    def _apply_stage_adjustments(self, weights: dict[str, float], stage: str | None) -> None:
        if not stage:
            return
        if stage.startswith(("2-", "3-")):
            weights["tempo"] = weights.get("tempo", 0.0) + 0.6
            weights["utility"] = weights.get("utility", 0.0) + 0.4
        if stage.startswith(("4-", "5-")):
            weights["scaling"] = weights.get("scaling", 0.0) + 0.5
            weights["carry"] = weights.get("carry", 0.0) + 0.3

    def _target_match_bonus(self, item: dict[str, Any], target_champion: str | None) -> float:
        if not target_champion:
            return 0.0
        target = target_champion.lower()
        for hint in item.get("recommended_for", []):
            if target in hint.lower():
                return 0.6
        return 0.0
