from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.vision.cropper import ReplayEquipmentCropper


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw replay equipment crop boxes and export slot crops.")
    parser.add_argument("--screenshot", required=True, help="Path to a replay screenshot")
    parser.add_argument("--output-dir", default="data/vision/debug_crops", help="Where to save debug outputs")
    parser.add_argument(
        "--show-inactive-second-column",
        action="store_true",
        help="Draw second-column boxes even when the first column is not full.",
    )
    args = parser.parse_args()

    cropper = ReplayEquipmentCropper()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    debug_path = cropper.draw_debug_layout(
        args.screenshot,
        output_dir / "layout_debug.png",
        hide_inactive_second_column=not args.show_inactive_second_column,
    )
    crop_paths = cropper.save_crops(args.screenshot, output_dir / "slots")

    print(f"debug layout: {debug_path}")
    print("slot crops:")
    for path in crop_paths:
        print(path)


if __name__ == "__main__":
    main()
