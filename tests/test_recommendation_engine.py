import json
import unittest
from pathlib import Path

from app.demo_service import DemoScenarioService
from app.recommendation_engine import ItemRecommendationEngine


RUNTIME_GRAPH = Path("data/processed/item_graph.runtime.json")


class RecommendationEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ItemRecommendationEngine(RUNTIME_GRAPH)
        self.demo_service = DemoScenarioService()

    def test_runtime_graph_is_current_patch_data(self) -> None:
        payload = json.loads(RUNTIME_GRAPH.read_text(encoding="utf-8"))
        self.assertEqual(payload["source"], "communitydragon_current_set")
        self.assertFalse(payload["is_demo_seed"])
        self.assertGreaterEqual(payload["set_number"], 1)
        self.assertGreaterEqual(len(payload["components"]), 10)
        self.assertGreaterEqual(len(payload["items"]), 40)

    def test_two_swords_can_craft_deathblade(self) -> None:
        payload = self.engine.build_payload(
            {
                "components": {"bf_sword": 2},
                "intent": "carry_ad",
                "target_champion": "物理主C",
                "stage": "3-2",
            }
        )
        ids = [item["item_id"] for item in payload["craftable_items"]]
        self.assertIn("deathblade", ids)
        self.assertEqual(payload["model_name"], "interpretable_linear_ranker")
        self.assertEqual(payload["model_version"], "v2")

    def test_bow_and_tear_can_craft_current_patch_item(self) -> None:
        payload = self.engine.build_payload(
            {
                "components": {"recurve_bow": 1, "tear_of_the_goddess": 1},
                "intent": "balanced",
            }
        )
        ids = [item["item_id"] for item in payload["craftable_items"]]
        self.assertIn("void_staff", ids)

    def test_frying_pan_and_chain_vest_can_craft_emblem(self) -> None:
        payload = self.engine.build_payload(
            {
                "components": {"frying_pan": 1, "chain_vest": 1},
                "intent": "balanced",
            }
        )
        ids = [item["item_id"] for item in payload["craftable_items"]]
        self.assertIn("bastion_emblem", ids)

    def test_payload_contains_feature_vector_and_breakdown(self) -> None:
        payload = self.engine.build_payload(
            {
                "components": {"bf_sword": 2},
                "intent": "carry_ad",
                "stage": "3-2",
            }
        )
        first = payload["craftable_items"][0]
        self.assertIn("feature_vector", first)
        self.assertIn("score_breakdown", first)
        self.assertTrue(first["score_breakdown"])

    def test_demo_scenario_can_run(self) -> None:
        scenario = self.demo_service.get_scenario("ad_carry_stage_3")
        request_payload = dict(scenario["request"])
        request_payload["scenario_id"] = scenario["scenario_id"]
        request_payload["scenario_title"] = scenario["title"]
        payload = self.engine.build_payload(request_payload)
        self.assertEqual(payload["scenario_id"], "ad_carry_stage_3")
        self.assertEqual(payload["scenario_title"], scenario["title"])
        self.assertTrue(payload["presentation_notes"])


if __name__ == "__main__":
    unittest.main()
