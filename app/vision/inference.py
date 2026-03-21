from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from app.vision.classic_cnn import ClassicItemCNN
from app.vision.cropper import ReplayEquipmentCropper
from app.vision.dataset import build_transforms
from app.item_graph import ItemGraphRepository


SPECIAL_NON_COMPONENT_LABELS = {"empty_slot", "consumable_tool", "completed_item", "other_unknown"}


class ReplayVisionPredictor:
    def __init__(self, weights_path: str | Path, classes_path: str | Path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = json.loads(Path(classes_path).read_text(encoding="utf-8"))
        self.model = ClassicItemCNN(num_classes=len(self.class_names))
        self.component_ids = {component["component_id"] for component in ItemGraphRepository().components}
        self.non_component_labels = SPECIAL_NON_COMPONENT_LABELS
        try:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.transform = build_transforms(train=False)
        self.cropper = ReplayEquipmentCropper()

    @torch.no_grad()
    def predict_slot(self, image: Image.Image) -> dict[str, Any]:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)
        top_probs, top_indices = torch.topk(probs, k=min(2, probs.shape[0]))
        confidence = float(top_probs[0].item())
        index = int(top_indices[0].item())
        second_best = float(top_probs[1].item()) if top_probs.shape[0] > 1 else 0.0
        margin = confidence - second_best
        label = self.class_names[index]

        if confidence < 0.45 or margin < 0.08:
            label = "other_unknown"
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "margin": round(margin, 4),
        }

    def _slot_index(self, slot_id: str) -> int:
        return int(slot_id.split("_")[-1])

    def predict_screenshot(
        self,
        screenshot_path: str | Path,
        confidence_threshold: float = 0.55,
        min_texture_std: float = 10.0,
    ) -> dict[str, Any]:
        screenshot = Image.open(screenshot_path).convert("RGB")
        results = []
        component_counter: Counter[str] = Counter()
        special_counter: Counter[str] = Counter()
        second_column_active = self.cropper.is_second_column_active(screenshot)
        for cropped_slot in self.cropper.crop_slots(screenshot):
            slot_id = cropped_slot.slot_id
            slot_index = self._slot_index(slot_id)

            if (slot_index % 2 == 1) and (not second_column_active):
                prediction = {"label": "empty_slot", "confidence": 1.0, "margin": 1.0}
            else:
                grayscale_std = float(np.asarray(cropped_slot.image.convert("L"), dtype=np.float32).std())
                if grayscale_std < min_texture_std:
                    prediction = {"label": "empty_slot", "confidence": 1.0, "margin": 1.0}
                else:
                    prediction = self.predict_slot(cropped_slot.image)

            label = prediction["label"]
            if label in self.component_ids and prediction["confidence"] >= confidence_threshold:
                component_counter[label] += 1
            elif label in self.non_component_labels:
                special_counter[label] += 1
            else:
                special_counter["other_unknown"] += 1
            results.append(
                {
                    "slot_id": slot_id,
                    "box": cropped_slot.pixel_box,
                    **prediction,
                }
            )
        return {
            "detections": results,
            "components": dict(component_counter),
            "special_items": dict(special_counter),
        }
