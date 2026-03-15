from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm_orchestrator import LLMOrchestrator
from app.recommendation_engine import ItemRecommendationEngine
from app.vision.cropper import ReplayEquipmentCropper
from app.vision.inference import ReplayVisionPredictor
from app.vision.screen_capture import capture_screen_to_file, parse_bbox


def _local_english_fallback(recommendation_payload: dict[str, Any]) -> str:
    top3 = recommendation_payload.get("craftable_items", [])[:3]
    if not top3:
        return "No craftable item is available from the currently detected components."
    names = ", ".join(item.get("name", "unknown_item") for item in top3)
    return f"Based on current components, prioritize: {names}."


def _resolve_screenshot_path(args: argparse.Namespace) -> Path:
    if args.capture_screen:
        bbox = parse_bbox(args.screen_bbox)
        return capture_screen_to_file(args.capture_output, bbox=bbox)
    if not args.screenshot:
        raise ValueError("Either --screenshot or --capture-screen must be provided.")
    return Path(args.screenshot)


def _run_once(args: argparse.Namespace) -> dict[str, Any]:
    screenshot_path = _resolve_screenshot_path(args)
    if not screenshot_path.exists():
        raise FileNotFoundError(f"screenshot not found: {screenshot_path}")
    if not Path(args.weights).exists():
        raise FileNotFoundError("Missing trained weights. Run scripts/train_cnn.py first.")

    cropper = ReplayEquipmentCropper()
    predictor = ReplayVisionPredictor(args.weights, args.classes)
    engine = ItemRecommendationEngine()
    llm = LLMOrchestrator()

    debug_dir = Path(args.debug_dir)
    debug_image = cropper.draw_debug_layout(screenshot_path, debug_dir / "layout_debug.png")
    cropper.save_crops(screenshot_path, debug_dir / "slots")

    vision_payload = predictor.predict_screenshot(screenshot_path, confidence_threshold=args.confidence_threshold)
    request_payload = {
        "components": vision_payload["components"],
        "target_champion": args.target_champion,
        "intent": args.intent,
        "stage": args.stage,
        "user_question": args.question or "Recommend the top craftable items based on current components.",
    }
    recommendation_payload = engine.build_payload(request_payload)

    if args.use_llm:
        llm_result = llm.generate_response(
            request_payload,
            recommendation_payload,
            user_question=args.question,
        )
        if llm_result["text"]:
            recommendation_payload["answer_text"] = llm_result["text"]
        if llm_result["source"] == "fallback":
            recommendation_payload["answer_text"] = (
                _local_english_fallback(recommendation_payload)
                + f"\n\n[LLM fallback: {llm_result.get('error', 'unknown')}]"
            )

    print("=" * 72)
    print("Replay Demo")
    print("=" * 72)
    print(f"screenshot: {screenshot_path}")
    print(f"debug layout: {debug_image}")
    print("predicted components:")
    print(json.dumps(vision_payload["components"], ensure_ascii=False, indent=2))
    print("special items:")
    print(json.dumps(vision_payload.get("special_items", {}), ensure_ascii=False, indent=2))
    print("-" * 72)
    print("top recommendations:")
    for item in recommendation_payload["craftable_items"][:3]:
        print(f"- {item['name']} | score={item['score']} | {item['explanation_text']}")
    print("-" * 72)
    print(recommendation_payload["answer_text"])

    return {
        "request_payload": request_payload,
        "recommendation_payload": recommendation_payload,
        "capture_path": screenshot_path,
    }


def _interactive_chat(args: argparse.Namespace, context: dict[str, Any]) -> None:
    if not args.interactive:
        return

    llm = LLMOrchestrator()
    if not args.use_llm:
        print("Interactive mode requires --use-llm. Exiting interactive loop.")
        return

    history: list[dict[str, str]] = []
    print("\nInteractive LLM chat started. Commands: 'refresh' recapture/re-run, 'exit' quit.")
    while True:
        question = input("you> ").strip()
        if not question:
            continue
        lowered = question.lower()
        if lowered in {"exit", "quit", "q"}:
            break
        if lowered in {"refresh", "r"}:
            context = _run_once(args)
            history.clear()
            continue
        llm_result = llm.generate_response(
            context["request_payload"],
            context["recommendation_payload"],
            user_question=question,
            chat_history=history,
        )
        reply = llm_result["text"] or context["recommendation_payload"].get("answer_text", "No response.")
        if llm_result["source"] == "fallback":
            reply = _local_english_fallback(context["recommendation_payload"])
            reply += f"\n\n[LLM fallback: {llm_result.get('error', 'unknown')}]"
        print(f"assistant> {reply}")
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": reply})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end replay recognition and optional LLM recommendation.")
    parser.add_argument("--screenshot", default=None)
    parser.add_argument("--capture-screen", action="store_true", help="Capture current desktop screen and run inference.")
    parser.add_argument("--screen-bbox", default=None, help="Optional bbox x1,y1,x2,y2 for game window area.")
    parser.add_argument("--capture-output", default="data/vision/live_capture/latest.png")
    parser.add_argument("--weights", default="data/vision/artifacts/classic_cnn/best.pt")
    parser.add_argument("--classes", default="data/vision/artifacts/classic_cnn/classes.json")
    parser.add_argument("--target-champion", default="main_carry")
    parser.add_argument("--intent", default="carry_ad")
    parser.add_argument("--stage", default="4-1")
    parser.add_argument("--debug-dir", default="data/vision/debug_crops")
    parser.add_argument("--question", default=None, help="Optional user question for first LLM response.")
    parser.add_argument("--use-llm", action="store_true", help="Use Gemini LLM if GEMINI_API_KEY is set.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive follow-up chat loop.")
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    args = parser.parse_args()

    context = _run_once(args)
    _interactive_chat(args, context)


if __name__ == "__main__":
    main()
