from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.crop_config import load_slot_layout
from app.vision.inference import ReplayVisionPredictor


def assign_split_by_ratio(
    screenshots: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[Path, str]:
    if not screenshots:
        return {}
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1):
        raise ValueError("train/val ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1 (remaining is test_ratio)")

    rng = random.Random(seed)
    shuffled = screenshots[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    split_map: dict[Path, str] = {}
    for index, screenshot_path in enumerate(shuffled):
        if index < train_count:
            split_map[screenshot_path] = "train"
        elif index < train_count + val_count:
            split_map[screenshot_path] = "val"
        else:
            split_map[screenshot_path] = "test"

    if test_count == 0 and total >= 3:
        last_train = next((p for p in reversed(shuffled) if split_map[p] == "train"), None)
        if last_train:
            split_map[last_train] = "test"
    return split_map


def slot_id_to_index(slot_id: str) -> int:
    match = re.search(r"(\d+)$", slot_id)
    if not match:
        raise ValueError(f"Unsupported slot_id format: {slot_id}")
    return int(match.group(1))


def apply_second_column_rule(layout_slots: list[object], labels_by_slot: dict[str, str]) -> None:
    sorted_slots = sorted(layout_slots, key=lambda slot: slot_id_to_index(slot.slot_id))
    left_slots = [slot.slot_id for slot in sorted_slots[::2]]
    right_slots = [slot.slot_id for slot in sorted_slots[1::2]]
    if not left_slots or not right_slots or len(left_slots) != len(right_slots):
        return

    left_filled = sum(
        1
        for slot_id in left_slots
        if labels_by_slot.get(slot_id, "empty_slot") not in {"empty_slot", "other_unknown"}
    )
    if left_filled < len(left_slots):
        for slot_id in right_slots:
            labels_by_slot[slot_id] = "empty_slot"


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an annotation CSV scaffold for replay screenshots.")
    parser.add_argument("--screenshot-dir", default="data/vision/raw/screenshots")
    parser.add_argument("--output", default="data/vision/annotations/slot_labels.csv")
    parser.add_argument("--split-mode", default="ratio", choices=["ratio", "fixed"])
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], help="Used only when split-mode=fixed")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autolabel", action="store_true", help="Pre-fill label using current CNN weights")
    parser.add_argument("--weights", default="data/vision/artifacts/classic_cnn/best.pt")
    parser.add_argument("--classes", default="data/vision/artifacts/classic_cnn/classes.json")
    parser.add_argument("--disable-second-column-rule", action="store_true")
    args = parser.parse_args()

    screenshot_dir = Path(args.screenshot_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    layout = load_slot_layout()
    screenshots = sorted(
        [path for path in screenshot_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )
    if not screenshots:
        raise FileNotFoundError(f"No screenshots found under: {screenshot_dir}")

    if args.split_mode == "fixed":
        split_map = {path: args.split for path in screenshots}
    else:
        split_map = assign_split_by_ratio(
            screenshots,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    predictor: ReplayVisionPredictor | None = None
    if args.autolabel:
        weights_path = Path(args.weights)
        classes_path = Path(args.classes)
        if not weights_path.exists() or not classes_path.exists():
            raise FileNotFoundError("Autolabel requires valid --weights and --classes files.")
        predictor = ReplayVisionPredictor(weights_path=weights_path, classes_path=classes_path)

    try:
        with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["screenshot_path", "slot_id", "label", "split"])
            total_rows = 0
            for screenshot_path in screenshots:
                split = split_map[screenshot_path]
                labels_for_screenshot: dict[str, str] = {}
                if predictor:
                    detections = predictor.predict_screenshot(screenshot_path)["detections"]
                    labels_for_screenshot = {det["slot_id"]: str(det["label"]) for det in detections}
                    if not args.disable_second_column_rule:
                        apply_second_column_rule(layout.slots, labels_for_screenshot)
                for slot in layout.slots:
                    label = ""
                    if predictor and slot.slot_id in labels_for_screenshot:
                        label = labels_for_screenshot[slot.slot_id]
                    writer.writerow([str(screenshot_path).replace("\\", "/"), slot.slot_id, label, split])
                    total_rows += 1
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write {output_path}. Close the file in Excel/WPS/IDE and retry, "
            "or pass a new path via --output."
        ) from exc

    print(f"annotation scaffold: {output_path}")
    print(f"screenshots: {len(screenshots)}")
    print(f"slots per screenshot: {len(layout.slots)}")
    print(f"total rows: {total_rows}")
    print(
        "split summary: "
        f"train={sum(1 for p in screenshots if split_map[p] == 'train')} "
        f"val={sum(1 for p in screenshots if split_map[p] == 'val')} "
        f"test={sum(1 for p in screenshots if split_map[p] == 'test')}"
    )
    if predictor:
        print("autolabel: enabled")
        if not args.disable_second_column_rule:
            print("rule: second column is forced to empty_slot when first column is not full")
    else:
        print("autolabel: disabled (labels left blank)")


if __name__ == "__main__":
    main()
