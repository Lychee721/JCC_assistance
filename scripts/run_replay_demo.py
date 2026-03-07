from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.recommendation_engine import ItemRecommendationEngine
from app.vision.cropper import ReplayEquipmentCropper
from app.vision.inference import ReplayVisionPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run end-to-end replay screenshot recognition and item recommendation.")
    parser.add_argument("--screenshot", required=True)
    parser.add_argument("--weights", default="data/vision/artifacts/classic_cnn/best.pt")
    parser.add_argument("--classes", default="data/vision/artifacts/classic_cnn/classes.json")
    parser.add_argument("--target-champion", default="物理主C")
    parser.add_argument("--intent", default="carry_ad")
    parser.add_argument("--stage", default="4-1")
    parser.add_argument("--debug-dir", default="data/vision/debug_crops")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError("Missing trained weights. Run scripts/train_cnn.py first.")

    cropper = ReplayEquipmentCropper()
    debug_dir = Path(args.debug_dir)
    debug_image = cropper.draw_debug_layout(args.screenshot, debug_dir / "layout_debug.png")
    cropper.save_crops(args.screenshot, debug_dir / "slots")

    predictor = ReplayVisionPredictor(args.weights, args.classes)
    vision_payload = predictor.predict_screenshot(args.screenshot)

    engine = ItemRecommendationEngine()
    recommendation_payload = engine.build_payload(
        {
            "components": vision_payload["components"],
            "target_champion": args.target_champion,
            "intent": args.intent,
            "stage": args.stage,
            "user_question": "根据截图推荐当前可做的成装",
        }
    )

    print("=" * 72)
    print("Replay Demo")
    print("=" * 72)
    print(f"screenshot: {args.screenshot}")
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


if __name__ == "__main__":
    main()
