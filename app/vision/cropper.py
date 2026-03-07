from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.vision.crop_config import ReplaySlotLayout, load_slot_layout


@dataclass
class CroppedSlot:
    slot_id: str
    image: Image.Image
    pixel_box: tuple[int, int, int, int]


class ReplayEquipmentCropper:
    def __init__(self, layout: ReplaySlotLayout | None = None) -> None:
        self.layout = layout or load_slot_layout()

    def crop_slots(self, screenshot: Image.Image) -> list[CroppedSlot]:
        width, height = screenshot.size
        cropped_slots: list[CroppedSlot] = []
        for slot in self.layout.slots:
            pixel_box = (
                int(slot.x1 * width),
                int(slot.y1 * height),
                int(slot.x2 * width),
                int(slot.y2 * height),
            )
            cropped_slots.append(
                CroppedSlot(
                    slot_id=slot.slot_id,
                    image=screenshot.crop(pixel_box),
                    pixel_box=pixel_box,
                )
            )
        return cropped_slots

    def _slot_index(self, slot_id: str) -> int:
        return int(slot_id.split("_")[-1])

    def is_second_column_active(self, screenshot: Image.Image, filled_std_threshold: float = 24.0) -> bool:
        left_slots = [slot for slot in self.crop_slots(screenshot) if self._slot_index(slot.slot_id) % 2 == 0]
        if not left_slots:
            return True
        filled_count = 0
        for slot in left_slots:
            grayscale = np.asarray(slot.image.convert("L"), dtype=np.float32)
            if float(grayscale.std()) >= filled_std_threshold:
                filled_count += 1
        return filled_count >= len(left_slots)

    def save_crops(self, screenshot_path: str | Path, output_dir: str | Path) -> list[Path]:
        screenshot = Image.open(screenshot_path).convert("RGB")
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for cropped_slot in self.crop_slots(screenshot):
            slot_path = output_path / f"{cropped_slot.slot_id}.png"
            cropped_slot.image.save(slot_path)
            saved.append(slot_path)
        return saved

    def draw_debug_layout(
        self,
        screenshot_path: str | Path,
        output_path: str | Path,
        hide_inactive_second_column: bool = True,
    ) -> Path:
        screenshot = Image.open(screenshot_path).convert("RGB")
        draw = ImageDraw.Draw(screenshot)
        second_column_active = True
        if hide_inactive_second_column:
            second_column_active = self.is_second_column_active(screenshot)
        for cropped_slot in self.crop_slots(screenshot):
            if hide_inactive_second_column and not second_column_active:
                if self._slot_index(cropped_slot.slot_id) % 2 == 1:
                    continue
            draw.rectangle(cropped_slot.pixel_box, outline=(255, 0, 0), width=3)
            draw.text((cropped_slot.pixel_box[0] + 2, cropped_slot.pixel_box[1] + 2), cropped_slot.slot_id, fill=(255, 255, 0))
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        screenshot.save(output)
        return output
