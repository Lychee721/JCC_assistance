from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.demo_service import DemoScenarioService
from app.recommendation_engine import ItemRecommendationEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a prepared demo scenario for the item assistant.")
    parser.add_argument("--scenario", default="ad_carry_stage_3", help="Scenario id from data/demo/demo_scenarios.json")
    args = parser.parse_args()

    demo_service = DemoScenarioService()
    engine = ItemRecommendationEngine()
    scenario = demo_service.get_scenario(args.scenario)

    request_payload = dict(scenario["request"])
    request_payload["scenario_id"] = scenario["scenario_id"]
    request_payload["scenario_title"] = scenario["title"]
    payload = engine.build_payload(request_payload)

    print("=" * 72)
    print("金铲铲出装助手 Demo")
    print("=" * 72)
    print(f"场景: {scenario['title']} ({scenario['scenario_id']})")
    print(f"目标: {scenario['goal']}")
    print(payload["input_summary"])
    print(f"模型: {payload['model_name']} {payload['model_version']}")
    print("-" * 72)
    print("Top 3 推荐:")
    for index, item in enumerate(payload["craftable_items"][:3], start=1):
        print(f"{index}. {item['name']} | score={item['score']}")
        print(f"   tags: {', '.join(item['reason_tags'][:6])}")
        print(f"   explain: {item['explanation_text']}")
    print("-" * 72)
    print("展示讲点:")
    for note in scenario["presentation_focus"]:
        print(f"- {note}")
    print("-" * 72)
    print("模型说明:")
    for note in payload["presentation_notes"]:
        print(f"- {note}")
    print("-" * 72)
    print("结论:")
    print(payload["answer_text"])


if __name__ == "__main__":
    main()
