from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class VisionDetection(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)


class RecommendationRequest(BaseModel):
    components: dict[str, int]
    target_champion: str | None = None
    intent: Literal["carry_ad", "carry_ap", "frontline", "balanced"] = "balanced"
    stage: str | None = None
    user_question: str | None = None
    detected_slots: list[VisionDetection] = Field(default_factory=list)


class CraftableItem(BaseModel):
    item_id: str
    stable_id: str | None = None
    name: str
    components: list[str]
    score: float
    reason_tags: list[str]
    feature_vector: dict[str, float] = Field(default_factory=dict)
    score_breakdown: list[dict[str, float | str]] = Field(default_factory=list)
    explanation_text: str = ""


class RecommendationResponse(BaseModel):
    craftable_items: list[CraftableItem]
    top_recommendations: list[str]
    missing_context: list[str]
    answer_text: str
    model_name: str
    model_version: str
    input_summary: str
    presentation_notes: list[str] = Field(default_factory=list)
    scenario_id: str | None = None
    scenario_title: str | None = None


class ReplayInferenceRequest(BaseModel):
    screenshot_path: str
    weights_path: str = "data/vision/artifacts/classic_cnn/best.pt"
    classes_path: str = "data/vision/artifacts/classic_cnn/classes.json"
    target_champion: str | None = None
    intent: Literal["carry_ad", "carry_ap", "frontline", "balanced"] = "balanced"
    stage: str | None = None


class ReplayInferenceResponse(BaseModel):
    screenshot_path: str
    predicted_components: dict[str, int]
    special_items: dict[str, int] = Field(default_factory=dict)
    slot_predictions: list[dict[str, str | float | list[int]]]
    recommendation: RecommendationResponse
