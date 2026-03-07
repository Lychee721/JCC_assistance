from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_ZH_PATH = Path("data/raw/cdragon/tft_latest_zh_cn.json")
DEFAULT_EN_PATH = Path("data/raw/cdragon/tft_latest_en_us.json")

COMPONENT_ID_ALIASES = {
    "TFT_Item_BFSword": "bf_sword",
    "TFT_Item_ChainVest": "chain_vest",
    "TFT_Item_FryingPan": "frying_pan",
    "TFT_Item_GiantsBelt": "giants_belt",
    "TFT_Item_NeedlesslyLargeRod": "needlessly_large_rod",
    "TFT_Item_NegatronCloak": "negatron_cloak",
    "TFT_Item_RecurveBow": "recurve_bow",
    "TFT_Item_SparringGloves": "sparring_gloves",
    "TFT_Item_Spatula": "spatula",
    "TFT_Item_TearOfTheGoddess": "tear_of_the_goddess",
}

COMPONENT_TAGS = {
    "bf_sword": ["ad", "carry"],
    "chain_vest": ["frontline", "armor"],
    "frying_pan": ["emblem", "flex"],
    "giants_belt": ["frontline", "hp"],
    "needlessly_large_rod": ["ap", "carry"],
    "negatron_cloak": ["frontline", "mr"],
    "recurve_bow": ["attack_speed", "carry"],
    "sparring_gloves": ["crit", "flex"],
    "spatula": ["emblem", "flex"],
    "tear_of_the_goddess": ["mana", "caster"],
}


def download_json(url: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": "jcc-item-assistant/0.2"})
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def ensure_snapshot(path: Path, locale: str) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.communitydragon.org/latest/cdragon/tft/{locale}.json"
    payload = download_json(url)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def normalize_asset_url(raw_path: str, patch: str = "latest") -> str | None:
    if not raw_path:
        return None
    normalized = raw_path
    if normalized.startswith("ASSETS/"):
        normalized = normalized[len("ASSETS/") :]
    normalized = normalized.lower().replace(".tex", ".png")
    return f"https://raw.communitydragon.org/{patch}/game/assets/{normalized}"


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "unknown_item"


def api_name_to_snake(api_name: str) -> str:
    core = api_name
    if core.startswith("TFT_Item_"):
        core = core[len("TFT_Item_") :]
    elif core.startswith("TFT"):
        core = core.replace("TFT", "", 1).lstrip("_")
    core = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", core)
    core = re.sub(r"[^a-zA-Z0-9]+", "_", core)
    return core.lower().strip("_")


def current_set_entry(snapshot: dict[str, Any], set_number: int | None = None) -> dict[str, Any]:
    entries = snapshot.get("setData", [])
    if not entries:
        raise ValueError("snapshot missing setData")
    if set_number is None:
        return entries[0]
    for entry in entries:
        if entry.get("number") == set_number:
            return entry
    raise ValueError(f"set number {set_number} not found in snapshot")


def build_item_index(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["apiName"]: item
        for item in snapshot.get("items", [])
        if isinstance(item, dict) and item.get("apiName")
    }


def infer_item_type(item: dict[str, Any], component_ids: list[str]) -> str:
    api_name = item.get("apiName", "")
    name = item.get("name", "")
    if api_name.endswith("EmblemItem") or "纹章" in name or "Emblem" in api_name:
        return "emblem"
    if api_name in {"TFT_Item_TacticiansScepter", "TFT_Item_ForceOfNature"} or "冠冕" in name:
        return "crown"
    if "frying_pan" in component_ids or "spatula" in component_ids:
        return "special"
    return "completed"


def infer_tags(item: dict[str, Any], component_ids: list[str], item_type: str) -> list[str]:
    effects = item.get("effects") or {}
    desc = (item.get("desc") or "").lower()
    name = (item.get("name") or "").lower()
    tags = set()

    for component_id in component_ids:
        tags.update(COMPONENT_TAGS.get(component_id, []))

    if any(key in effects for key in ("AD", "BonusDamage")):
        tags.update({"ad", "carry"})
    if any(key in effects for key in ("AP", "AbilityPower")):
        tags.update({"ap"})
    if any(key in effects for key in ("AS", "AttackSpeed")):
        tags.update({"attack_speed"})
    if any(key in effects for key in ("Mana", "ManaRegen", "FlatManaRestore")):
        tags.update({"mana", "caster"})
    if any(key in effects for key in ("Armor", "MR", "HP", "Health")):
        tags.update({"frontline"})
    if any(key in effects for key in ("CritChance", "CritDamageToGive")) or "暴击" in desc or "crit" in desc:
        tags.update({"crit"})
    if any(key in effects for key in ("ShieldDuration", "ShieldSize", "ShieldHPPercent", "DamageReduction")) or "护盾" in desc:
        tags.update({"survivability"})
    if any(key in effects for key in ("LifeSteal", "Omnivamp")) or "治疗" in desc or "heal" in desc:
        tags.update({"sustain"})
    if any(key in effects for key in ("MRShred", "ArmorShred")) or "shred" in desc or "破甲" in desc or "破魔抗" in desc:
        tags.update({"shred", "utility"})
    if "每次攻击" in desc or "stack" in desc or "叠加" in desc:
        tags.update({"scaling"})
    if "战斗开始" in desc or "施放技能后" in desc or "击杀" in desc:
        tags.update({"tempo"})
    if "巨人" in name or "高生命值" in desc or "max hp" in desc:
        tags.update({"anti_tank"})

    if item_type == "emblem":
        tags.update({"emblem", "flex"})
    elif item_type == "crown":
        tags.update({"crown", "flex"})
    elif item_type == "special":
        tags.update({"special", "flex"})

    return sorted(tags)


def infer_priority_hints(tags: list[str], item_type: str) -> dict[str, float]:
    hints = {
        "carry_ad": 0.25,
        "carry_ap": 0.25,
        "frontline": 0.25,
        "balanced": 0.25,
    }
    tag_set = set(tags)

    if {"ad", "carry"} & tag_set:
        hints["carry_ad"] += 0.45
    if {"crit", "attack_speed"} & tag_set:
        hints["carry_ad"] += 0.15
    if {"ap", "caster"} & tag_set:
        hints["carry_ap"] += 0.45
    if {"mana", "scaling"} & tag_set:
        hints["carry_ap"] += 0.15
    if {"frontline", "survivability"} & tag_set:
        hints["frontline"] += 0.45
    if {"utility", "flex"} & tag_set:
        hints["balanced"] += 0.35

    if item_type == "emblem":
        hints["balanced"] += 0.2
    if item_type == "crown":
        hints["balanced"] += 0.15

    return {key: round(min(value, 0.98), 2) for key, value in hints.items()}


def build_runtime_graph(
    zh_snapshot: dict[str, Any],
    en_snapshot: dict[str, Any] | None = None,
    set_number: int | None = None,
) -> dict[str, Any]:
    zh_set = current_set_entry(zh_snapshot, set_number)
    en_set = current_set_entry(en_snapshot, zh_set["number"]) if en_snapshot else None

    zh_index = build_item_index(zh_snapshot)
    en_index = build_item_index(en_snapshot) if en_snapshot else {}

    current_api_names = set(zh_set.get("items", []))
    current_items = [zh_index[api_name] for api_name in current_api_names if api_name in zh_index]
    craftable_items = [
        item for item in current_items
        if isinstance(item.get("composition"), list) and len(item["composition"]) == 2
    ]

    component_api_names = sorted({component for item in craftable_items for component in item["composition"]})
    normalized_component_ids = {api_name: COMPONENT_ID_ALIASES.get(api_name, api_name_to_snake(api_name)) for api_name in component_api_names}

    components = []
    for api_name in component_api_names:
        zh_item = zh_index[api_name]
        en_item = en_index.get(api_name, {})
        component_id = normalized_component_ids[api_name]
        components.append(
            {
                "component_id": component_id,
                "name": zh_item.get("name") or component_id,
                "name_en": en_item.get("name"),
                "api_name": api_name,
                "icon_raw_path": zh_item.get("icon"),
                "icon_url": normalize_asset_url(zh_item.get("icon")),
                "tags": COMPONENT_TAGS.get(component_id, []),
            }
        )

    normalized_items = []
    for zh_item in sorted(craftable_items, key=lambda item: item.get("apiName", "")):
        api_name = zh_item["apiName"]
        en_item = en_index.get(api_name, {})
        component_ids = [normalized_component_ids[name] for name in zh_item["composition"]]
        item_type = infer_item_type(zh_item, component_ids)
        english_name = en_item.get("name")
        item_id = slugify(english_name) if english_name else api_name_to_snake(api_name)
        tags = infer_tags(zh_item, component_ids, item_type)
        normalized_items.append(
            {
                "item_id": item_id,
                "stable_id": api_name_to_snake(api_name),
                "name": zh_item.get("name") or item_id,
                "name_en": english_name,
                "api_name": api_name,
                "components": component_ids,
                "component_api_names": zh_item["composition"],
                "item_type": item_type,
                "tags": tags,
                "priority_hints": infer_priority_hints(tags, item_type),
                "recommended_for": [],
                "effects": zh_item.get("effects", {}),
                "desc": zh_item.get("desc"),
                "desc_en": en_item.get("desc"),
                "icon_raw_path": zh_item.get("icon"),
                "icon_url": normalize_asset_url(zh_item.get("icon")),
                "unique": bool(zh_item.get("unique")),
            }
        )

    reverse_index: dict[str, list[str]] = defaultdict(list)
    for item in normalized_items:
        for component_id in item["components"]:
            reverse_index[component_id].append(item["item_id"])

    return {
        "version": f"cdragon-latest-set-{zh_set['number']}",
        "source": "communitydragon_current_set",
        "is_demo_seed": False,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "patch": "latest",
        "set_number": zh_set["number"],
        "set_name": zh_set.get("name"),
        "mutator": zh_set.get("mutator"),
        "components": components,
        "items": normalized_items,
        "reverse_index": dict(sorted(reverse_index.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime item graph from current CommunityDragon TFT snapshots.")
    parser.add_argument("--zh-path", default=str(DEFAULT_ZH_PATH))
    parser.add_argument("--en-path", default=str(DEFAULT_EN_PATH))
    parser.add_argument("--set-number", type=int, default=None)
    parser.add_argument("--output-path", default="data/processed/item_graph.runtime.json")
    parser.add_argument("--component-output-path", default="data/processed/component_catalog.runtime.json")
    args = parser.parse_args()

    zh_snapshot = ensure_snapshot(Path(args.zh_path), "zh_cn")
    en_snapshot = ensure_snapshot(Path(args.en_path), "en_us")

    runtime = build_runtime_graph(zh_snapshot, en_snapshot, args.set_number)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    component_output = Path(args.component_output_path)
    component_output.parent.mkdir(parents=True, exist_ok=True)
    component_output.write_text(
        json.dumps(
            {
                "version": runtime["version"],
                "source": runtime["source"],
                "set_number": runtime["set_number"],
                "components": runtime["components"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"saved runtime graph: {output_path}")
    print(f"saved component catalog: {component_output}")
    print(f"current set: {runtime['set_number']} {runtime['set_name']}")
    print(f"components: {len(runtime['components'])}")
    print(f"craftable items: {len(runtime['items'])}")


if __name__ == "__main__":
    main()
