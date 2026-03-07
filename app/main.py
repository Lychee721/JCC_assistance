from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import load_app_config
from app.demo_service import DemoScenarioService
from app.llm_orchestrator import LLMOrchestrator
from app.models import ReplayInferenceRequest, ReplayInferenceResponse, RecommendationRequest, RecommendationResponse
from app.recommendation_engine import ItemRecommendationEngine

config = load_app_config()
engine = ItemRecommendationEngine(config.paths.item_graph)
llm = LLMOrchestrator(config.paths.system_prompt)
demo_service = DemoScenarioService()

app = FastAPI(title=config.name)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/recommend/items", response_model=RecommendationResponse)
def recommend_items(request: RecommendationRequest) -> RecommendationResponse:
    payload = engine.build_payload(request.model_dump())
    _messages = llm.build_messages(request.model_dump(), payload)
    return RecommendationResponse(**payload)


@app.get("/v1/demo/scenarios")
def list_demo_scenarios() -> dict[str, list[dict[str, object]]]:
    scenarios = []
    for scenario in demo_service.list_scenarios():
        scenarios.append(
            {
                "scenario_id": scenario["scenario_id"],
                "title": scenario["title"],
                "goal": scenario["goal"],
                "presentation_focus": scenario["presentation_focus"],
            }
        )
    return {"scenarios": scenarios}


@app.get("/v1/demo/run/{scenario_id}", response_model=RecommendationResponse)
def run_demo_scenario(scenario_id: str) -> RecommendationResponse:
    try:
        scenario = demo_service.get_scenario(scenario_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"unknown scenario: {scenario_id}") from exc

    request_payload = dict(scenario["request"])
    request_payload["scenario_id"] = scenario["scenario_id"]
    request_payload["scenario_title"] = scenario["title"]
    payload = engine.build_payload(request_payload)
    payload["presentation_notes"] = scenario["presentation_focus"] + payload["presentation_notes"]
    return RecommendationResponse(**payload)


@app.post("/v1/recommend/replay-local", response_model=ReplayInferenceResponse)
def recommend_from_replay(request: ReplayInferenceRequest) -> ReplayInferenceResponse:
    from app.vision.inference import ReplayVisionPredictor

    predictor = ReplayVisionPredictor(request.weights_path, request.classes_path)
    vision_payload = predictor.predict_screenshot(request.screenshot_path)
    recommendation_payload = engine.build_payload(
        {
            "components": vision_payload["components"],
            "target_champion": request.target_champion,
            "intent": request.intent,
            "stage": request.stage,
            "user_question": "根据游戏回放截图推荐装备",
        }
    )
    recommendation = RecommendationResponse(**recommendation_payload)
    slot_predictions = []
    for detection in vision_payload["detections"]:
        slot_predictions.append(
            {
                "slot_id": detection["slot_id"],
                "label": detection["label"],
                "confidence": detection["confidence"],
                "box": list(detection["box"]),
            }
        )
    return ReplayInferenceResponse(
        screenshot_path=request.screenshot_path,
        predicted_components=vision_payload["components"],
        special_items=vision_payload.get("special_items", {}),
        slot_predictions=slot_predictions,
        recommendation=recommendation,
    )
