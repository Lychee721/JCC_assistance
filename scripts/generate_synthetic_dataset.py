from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    runtime_catalog = Path("data/processed/component_catalog.runtime.json")
    seed_catalog = Path("data/seed/component_catalog.example.json")
    source_path = runtime_catalog if runtime_catalog.exists() else seed_catalog

    output_path = Path("data/processed/cnn_dataset_manifest.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = []
    for component in payload["components"]:
        rows.append(
            {
                "image_id": f"{component['component_id']}_canonical",
                "label": component["component_id"],
                "split": "train",
                "source": "communitydragon_icon",
                "asset_url": component["icon_url"],
                "augmentations": [
                    "scale_0.85_1.15",
                    "jpeg_compression",
                    "motion_blur_light",
                    "gaussian_noise_light",
                    "random_occlusion_small",
                    "bg_overlay_shop_or_bench",
                ],
            }
        )

    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"source catalog: {source_path}")
    print(f"saved dataset manifest: {output_path}")


if __name__ == "__main__":
    main()
