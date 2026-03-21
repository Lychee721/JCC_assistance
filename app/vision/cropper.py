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


@dataclass
class ViewportBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)


class ReplayEquipmentCropper:
    def __init__(self, layout: ReplaySlotLayout | None = None) -> None:
        self.layout = layout or load_slot_layout()
        self._cached_alignment_key: tuple[int, int, int, int, int, int] | None = None
        self._cached_alignment_shift: tuple[float, float] = (0.0, 0.0)

    def _find_active_start(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        max_scan: int,
        *,
        active_mean_threshold: float,
        active_std_threshold: float,
        stable_lines: int = 5,
    ) -> int:
        consecutive = 0
        last_dark_index = -1
        limit = min(max_scan, len(means))
        for index in range(limit):
            is_active = (float(means[index]) >= active_mean_threshold) or (float(stds[index]) >= active_std_threshold)
            if is_active:
                consecutive += 1
                if consecutive >= stable_lines:
                    return max(0, index - stable_lines + 1)
            else:
                consecutive = 0
                last_dark_index = index
        return max(0, last_dark_index + 1)

    def _find_active_end(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        max_scan: int,
        *,
        active_mean_threshold: float,
        active_std_threshold: float,
        stable_lines: int = 5,
    ) -> int:
        consecutive = 0
        last_dark_offset = -1
        limit = min(max_scan, len(means))
        for offset in range(limit):
            index = len(means) - 1 - offset
            is_active = (float(means[index]) >= active_mean_threshold) or (float(stds[index]) >= active_std_threshold)
            if is_active:
                consecutive += 1
                if consecutive >= stable_lines:
                    return min(len(means), index + stable_lines)
            else:
                consecutive = 0
                last_dark_offset = offset
        return len(means) - max(0, last_dark_offset + 1)

    def detect_active_viewport(self, screenshot: Image.Image) -> ViewportBox:
        width, height = screenshot.size
        grayscale = np.asarray(screenshot.convert("L"), dtype=np.float32)
        row_means = grayscale.mean(axis=1)
        row_stds = grayscale.std(axis=1)
        col_means = grayscale.mean(axis=0)
        col_stds = grayscale.std(axis=0)

        vertical_scan = max(16, int(height * 0.18))
        horizontal_scan = max(16, int(width * 0.18))

        top = self._find_active_start(
            row_means,
            row_stds,
            vertical_scan,
            active_mean_threshold=18.0,
            active_std_threshold=7.0,
        )
        bottom = self._find_active_end(
            row_means,
            row_stds,
            vertical_scan,
            active_mean_threshold=18.0,
            active_std_threshold=7.0,
        )
        left = self._find_active_start(
            col_means,
            col_stds,
            horizontal_scan,
            active_mean_threshold=18.0,
            active_std_threshold=7.0,
        )
        right = self._find_active_end(
            col_means,
            col_stds,
            horizontal_scan,
            active_mean_threshold=18.0,
            active_std_threshold=7.0,
        )

        if right - left < width * 0.7:
            left, right = 0, width
        if bottom - top < height * 0.7:
            top, bottom = 0, height

        return ViewportBox(
            x1=max(0, left),
            y1=max(0, top),
            x2=min(width, right),
            y2=min(height, bottom),
        )

    def _slot_pixel_box(
        self,
        slot,
        viewport: ViewportBox,
        *,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
    ) -> tuple[int, int, int, int]:
        return (
            int(viewport.x1 + (slot.x1 + shift_x) * viewport.width),
            int(viewport.y1 + (slot.y1 + shift_y) * viewport.height),
            int(viewport.x1 + (slot.x2 + shift_x) * viewport.width),
            int(viewport.y1 + (slot.y2 + shift_y) * viewport.height),
        )

    def _score_alignment_shift(
        self,
        grayscale: np.ndarray,
        viewport: ViewportBox,
        *,
        shift_x: float,
        shift_y: float,
    ) -> float:
        left_column_slots = [slot for slot in self.layout.slots if self._slot_index(slot.slot_id) % 2 == 0]
        std_values: list[float] = []
        for slot in left_column_slots:
            x1, y1, x2, y2 = self._slot_pixel_box(slot, viewport, shift_x=shift_x, shift_y=shift_y)
            if x1 < 0 or y1 < 0 or x2 > grayscale.shape[1] or y2 > grayscale.shape[0]:
                continue
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            patch = grayscale[y1:y2, x1:x2]
            std_values.append(float(patch.std()))

        if not std_values:
            return -1e9
        top_k = max(4, min(7, len(std_values)))
        best_values = sorted(std_values, reverse=True)[:top_k]
        # Keep the correction small while still allowing enough movement.
        penalty = (abs(shift_x) * 18.0) + (abs(shift_y) * 10.0)
        return float(np.mean(best_values)) - penalty

    def _first_run_start(self, values: np.ndarray, threshold: float, min_len: int) -> int | None:
        run_start: int | None = None
        run_len = 0
        for index, value in enumerate(values):
            if float(value) >= threshold:
                if run_start is None:
                    run_start = index
                run_len += 1
                if run_len >= min_len:
                    return run_start
            else:
                run_start = None
                run_len = 0
        return None

    def _detect_first_item_anchor(self, screenshot: Image.Image, viewport: ViewportBox) -> tuple[int, int] | None:
        rgb = np.asarray(screenshot.convert("RGB"), dtype=np.uint8)
        y_from = viewport.y1 + int(0.10 * viewport.height)
        y_to = viewport.y1 + int(0.55 * viewport.height)
        x_from = viewport.x1 + int(0.002 * viewport.width)
        x_to = viewport.x1 + int(0.10 * viewport.width)
        if y_to <= y_from or x_to <= x_from:
            return None

        patch = rgb[y_from:y_to, x_from:x_to]
        max_c = patch.max(axis=2)
        min_c = patch.min(axis=2)
        sat = max_c - min_c
        mask = (sat >= 40) & (max_c >= 65)

        row_hits = mask.sum(axis=1)
        col_hits = mask.sum(axis=0)
        row_threshold = max(12, int(mask.shape[1] * 0.12))
        col_threshold = max(12, int(mask.shape[0] * 0.05))

        y_local = self._first_run_start(row_hits, threshold=row_threshold, min_len=8)
        x_local = self._first_run_start(col_hits, threshold=col_threshold, min_len=6)
        if y_local is None or x_local is None:
            return None
        return x_from + int(x_local), y_from + int(y_local)

    def estimate_alignment_shift(self, screenshot: Image.Image, viewport: ViewportBox) -> tuple[float, float]:
        cache_key = (
            screenshot.size[0],
            screenshot.size[1],
            viewport.x1,
            viewport.y1,
            viewport.x2,
            viewport.y2,
        )
        if self._cached_alignment_key == cache_key:
            return self._cached_alignment_shift

        grayscale = np.asarray(screenshot.convert("L"), dtype=np.float32)

        anchor = self._detect_first_item_anchor(screenshot, viewport)
        if anchor is not None:
            slot0 = self.layout.slots[0]
            expected_x = viewport.x1 + int(slot0.x1 * viewport.width)
            expected_y = viewport.y1 + int(slot0.y1 * viewport.height)
            anchor_shift_x = (anchor[0] - expected_x) / float(viewport.width)
            anchor_shift_y = (anchor[1] - expected_y) / float(viewport.height)
            anchor_shift_x = float(np.clip(anchor_shift_x, -0.03, 0.03))
            anchor_shift_y = float(np.clip(anchor_shift_y, -0.10, 0.10))
            coarse_x = np.arange(anchor_shift_x - 0.008, anchor_shift_x + 0.0081, 0.002)
            coarse_y = np.arange(anchor_shift_y - 0.012, anchor_shift_y + 0.0121, 0.002)
        else:
            coarse_x = np.arange(-0.025, 0.026, 0.0025)
            coarse_y = np.arange(-0.09, 0.091, 0.003)

        best_score = -1e9
        best_shift = (0.0, 0.0)
        for shift_x in coarse_x:
            for shift_y in coarse_y:
                score = self._score_alignment_shift(grayscale, viewport, shift_x=float(shift_x), shift_y=float(shift_y))
                if score > best_score:
                    best_score = score
                    best_shift = (float(shift_x), float(shift_y))

        fine_x = np.arange(best_shift[0] - 0.003, best_shift[0] + 0.0031, 0.001)
        fine_y = np.arange(best_shift[1] - 0.004, best_shift[1] + 0.0041, 0.001)
        for shift_x in fine_x:
            for shift_y in fine_y:
                score = self._score_alignment_shift(grayscale, viewport, shift_x=float(shift_x), shift_y=float(shift_y))
                if score > best_score:
                    best_score = score
                    best_shift = (float(shift_x), float(shift_y))

        self._cached_alignment_key = cache_key
        self._cached_alignment_shift = best_shift
        return best_shift

    def crop_slots(self, screenshot: Image.Image) -> list[CroppedSlot]:
        viewport = self.detect_active_viewport(screenshot)
        shift_x, shift_y = self.estimate_alignment_shift(screenshot, viewport)
        cropped_slots: list[CroppedSlot] = []
        for slot in self.layout.slots:
            pixel_box = self._slot_pixel_box(slot, viewport, shift_x=shift_x, shift_y=shift_y)
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
        viewport = self.detect_active_viewport(screenshot)
        shift_x, shift_y = self.estimate_alignment_shift(screenshot, viewport)
        second_column_active = True
        if hide_inactive_second_column:
            second_column_active = self.is_second_column_active(screenshot)
        draw.rectangle((viewport.x1, viewport.y1, viewport.x2, viewport.y2), outline=(0, 255, 255), width=3)
        draw.text(
            (viewport.x1 + 4, viewport.y1 + 4),
            f"active_viewport shift=({shift_x:+.3f},{shift_y:+.3f})",
            fill=(0, 255, 255),
        )
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
