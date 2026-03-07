from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen


def download_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "jcc-item-assistant/0.1"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a CommunityDragon TFT snapshot.")
    parser.add_argument("--locale", default="zh_cn")
    parser.add_argument("--patch", default="latest")
    parser.add_argument("--output-dir", default="data/raw/cdragon")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    url = f"https://raw.communitydragon.org/{args.patch}/cdragon/tft/{args.locale}.json"
    payload = download_json(url)

    output_path = output_dir / f"tft_{args.patch}_{args.locale}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "url": url,
        "locale": args.locale,
        "patch": args.patch,
        "output_path": str(output_path).replace("\\", "/"),
        "top_level_keys": sorted(payload.keys()),
    }
    manifest_path = output_dir / f"manifest_{args.patch}_{args.locale}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved snapshot: {output_path}")
    print(f"saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
