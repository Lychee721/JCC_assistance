from __future__ import annotations

import json
import random
import sys
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.item_graph import ItemGraphRepository


CLASSIFIER_ROOT = Path("data/vision/datasets/classifier_slots")
IMAGE_SIZE = 64
SPLITS = {"train": 240, "val": 60, "test": 60}
SPECIAL_CLASSES = ["empty_slot", "consumable_tool", "completed_item", "other_unknown"]
RAW_ITEMS_EN_US = Path("data/raw/cdragon/tft_latest_en_us.json")
CONSUMABLE_KEYWORDS = (
    "consumable",
    "reforger",
    "remover",
    "duplicator",
    "dissolver",
    "recombobulator",
)
CONSUMABLE_FALLBACK_RAW_PATHS = [
    "ASSETS/Maps/TFT/Icons/Items/Hexcore/TFT_Consumable_ItemReroller.TFT_Set13.tex",
    "ASSETS/Maps/TFT/Icons/Items/Hexcore/TFT_Consumable_ItemRemover.TFT_Set13.tex",
    "ASSETS/Maps/TFT/Icons/Items/Hexcore/TFT_Consumable_NeekosHelp.TFT_Set13.tex",
    "ASSETS/Maps/TFT/Icons/Items/Hexcore/TFT_Consumable_ChampionDuplicator_III.TFT_Set13.tex",
    "ASSETS/Maps/Particles/TFT/TFT_Item_Consumable_Dissolver.tex",
]


def download_icon(url: str) -> Image.Image:
    request = Request(url, headers={"User-Agent": "jcc-item-assistant/vision-dataset"})
    with urlopen(request, timeout=8) as response:
        return Image.open(BytesIO(response.read())).convert("RGBA")


def raw_icon_to_url(raw_path: str) -> str:
    normalized = raw_path.replace("\\", "/").strip().lstrip("/")
    if normalized.lower().endswith(".tex"):
        normalized = normalized[:-4] + ".png"
    elif not normalized.lower().endswith(".png"):
        normalized = normalized + ".png"
    return f"https://raw.communitydragon.org/latest/game/{normalized.lower()}"


def collect_consumable_icon_urls() -> list[str]:
    urls: list[str] = []
    if RAW_ITEMS_EN_US.exists():
        payload = json.loads(RAW_ITEMS_EN_US.read_text(encoding="utf-8"))
        for item in payload.get("items", []):
            name = str(item.get("name", "")).lower()
            api_name = str(item.get("apiName", "")).lower()
            if not any(keyword in name or keyword in api_name for keyword in CONSUMABLE_KEYWORDS):
                continue
            icon = item.get("icon")
            if isinstance(icon, str) and icon:
                icon_lower = icon.lower()
                if "item_icons" not in icon_lower and "/icons/items/" not in icon_lower:
                    continue
                urls.append(raw_icon_to_url(icon))
    urls.extend(raw_icon_to_url(raw_path) for raw_path in CONSUMABLE_FALLBACK_RAW_PATHS)
    return sorted(set(urls))[:24]


def collect_completed_icon_urls(repository: ItemGraphRepository) -> list[str]:
    completed_types = {"completed", "emblem", "crown"}
    urls = [item["icon_url"] for item in repository.items if item.get("item_type") in completed_types and item.get("icon_url")]
    if not urls:
        urls = [item["icon_url"] for item in repository.items if item.get("icon_url")]
    return sorted(set(urls))


def build_slot_background() -> Image.Image:
    image = Image.new("RGBA", (IMAGE_SIZE, IMAGE_SIZE), (24, 24, 24, 255))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((2, 2, IMAGE_SIZE - 3, IMAGE_SIZE - 3), radius=8, fill=(45, 45, 45, 255), outline=(135, 115, 80, 255), width=3)
    draw.rounded_rectangle((8, 8, IMAGE_SIZE - 9, IMAGE_SIZE - 9), radius=6, fill=(70, 70, 70, 220))
    return image


def augment(image: Image.Image) -> Image.Image:
    canvas = image.copy()
    if random.random() < 0.7:
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.0, 1.2)))
    if random.random() < 0.8:
        canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.85, 1.15))
    if random.random() < 0.8:
        canvas = ImageEnhance.Contrast(canvas).enhance(random.uniform(0.85, 1.2))
    if random.random() < 0.5:
        canvas = canvas.rotate(random.uniform(-8, 8), resample=Image.Resampling.BICUBIC, fillcolor=(20, 20, 20, 255))
    return canvas


def paste_icon(background: Image.Image, icon: Image.Image) -> Image.Image:
    canvas = background.copy()
    target_size = random.randint(40, 52)
    icon = icon.resize((target_size, target_size), Image.Resampling.LANCZOS)
    offset_x = (IMAGE_SIZE - target_size) // 2 + random.randint(-3, 3)
    offset_y = (IMAGE_SIZE - target_size) // 2 + random.randint(-3, 3)
    canvas.alpha_composite(icon, (offset_x, offset_y))
    return augment(canvas).convert("RGB")


def draw_random_unknown(background: Image.Image) -> Image.Image:
    canvas = background.copy()
    draw = ImageDraw.Draw(canvas)
    for _ in range(random.randint(3, 8)):
        color = (
            random.randint(70, 255),
            random.randint(70, 255),
            random.randint(70, 255),
            random.randint(120, 220),
        )
        x1 = random.randint(4, IMAGE_SIZE - 20)
        y1 = random.randint(4, IMAGE_SIZE - 20)
        x2 = random.randint(x1 + 6, min(IMAGE_SIZE - 4, x1 + 24))
        y2 = random.randint(y1 + 6, min(IMAGE_SIZE - 4, y1 + 24))
        if random.random() < 0.5:
            draw.rectangle((x1, y1, x2, y2), fill=color)
        else:
            draw.ellipse((x1, y1, x2, y2), fill=color)
    draw.text((IMAGE_SIZE // 2 - 6, IMAGE_SIZE // 2 - 8), "?", fill=(255, 255, 255, 220))
    return augment(canvas).convert("RGB")


def load_icon_pool(urls: list[str]) -> list[Image.Image]:
    pool: list[Image.Image] = []
    for url in urls:
        try:
            pool.append(download_icon(url))
        except Exception:
            continue
    return pool


def save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")


def main() -> None:
    random.seed(42)
    repository = ItemGraphRepository()
    components = repository.components

    if CLASSIFIER_ROOT.exists():
        for child in CLASSIFIER_ROOT.iterdir():
            if child.is_dir():
                for file_path in child.rglob("*.png"):
                    file_path.unlink()

    class_names = [component["component_id"] for component in components] + SPECIAL_CLASSES
    background = build_slot_background()
    icon_cache: dict[str, Image.Image] = {}
    completed_icon_pool = load_icon_pool(collect_completed_icon_urls(repository))
    consumable_icon_pool = load_icon_pool(collect_consumable_icon_urls())

    for component in components:
        icon_cache[component["component_id"]] = download_icon(component["icon_url"])

    for split, count in SPLITS.items():
        for class_name in class_names:
            for index in range(count):
                if class_name == "empty_slot":
                    image = augment(background).convert("RGB")
                elif class_name == "consumable_tool":
                    if consumable_icon_pool:
                        image = paste_icon(background, random.choice(consumable_icon_pool))
                    else:
                        image = draw_random_unknown(background)
                elif class_name == "completed_item":
                    if completed_icon_pool:
                        image = paste_icon(background, random.choice(completed_icon_pool))
                    else:
                        image = draw_random_unknown(background)
                elif class_name == "other_unknown":
                    image = draw_random_unknown(background)
                else:
                    image = paste_icon(background, icon_cache[class_name])
                save_image(image, CLASSIFIER_ROOT / split / class_name / f"{class_name}_{index:04d}.png")

    classes_path = Path("data/vision/artifacts/classic_cnn/classes.json")
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    classes_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"saved synthetic dataset: {CLASSIFIER_ROOT}")
    print(f"saved classes: {classes_path}")


if __name__ == "__main__":
    main()
