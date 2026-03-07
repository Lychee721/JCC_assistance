from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from app.vision.cropper import ReplayEquipmentCropper


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract labeled slot crops for CNN training.")
    parser.add_argument("--annotation", default="data/vision/annotations/slot_labels.csv")
    parser.add_argument("--output-root", default="data/vision/datasets/classifier_slots")
    args = parser.parse_args()

    annotation_path = Path(args.annotation)
    output_root = Path(args.output_root)
    cropper = ReplayEquipmentCropper()

    if not annotation_path.exists():
        raise FileNotFoundError(f"Missing annotation file: {annotation_path}")

    split_whitelist = {"train", "val", "test"}
    crop_cache: dict[Path, dict[str, object]] = {}
    index_counters: defaultdict[tuple[str, str], int] = defaultdict(int)
    saved_count = 0
    skipped_empty_label = 0
    skipped_missing_file = 0

    with annotation_path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            screenshot_path = Path(row["screenshot_path"])
            slot_id = row["slot_id"]
            label = row.get("label", "").strip()
            split = row.get("split", "train").strip().lower()
            if split not in split_whitelist:
                split = "train"

            if not label:
                skipped_empty_label += 1
                continue

            if screenshot_path not in crop_cache:
                if not screenshot_path.exists():
                    skipped_missing_file += 1
                    continue
                screenshot = Image.open(screenshot_path).convert("RGB")
                crop_cache[screenshot_path] = {crop.slot_id: crop for crop in cropper.crop_slots(screenshot)}
            crops = crop_cache[screenshot_path]
            if slot_id not in crops:
                raise KeyError(f"Unknown slot_id {slot_id}")

            target_dir = output_root / split / label
            target_dir.mkdir(parents=True, exist_ok=True)
            key = (split, label)
            index = index_counters[key]
            index_counters[key] += 1
            target_path = target_dir / f"{screenshot_path.stem}_{slot_id}_{index:04d}.png"
            crops[slot_id].image.save(target_path)
            saved_count += 1

    print(f"saved labeled crops into: {output_root}")
    print(f"saved samples: {saved_count}")
    print(f"skipped empty label rows: {skipped_empty_label}")
    print(f"skipped missing screenshot rows: {skipped_missing_file}")


if __name__ == "__main__":
    main()
