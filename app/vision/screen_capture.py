from __future__ import annotations

import ctypes
from ctypes import wintypes
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageGrab, ImageStat


def parse_bbox(raw: str | None) -> tuple[int, int, int, int] | None:
    if not raw:
        return None
    parts = [segment.strip() for segment in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("screen_bbox must be 'x1,y1,x2,y2'")
    x1, y1, x2, y2 = [int(value) for value in parts]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("screen_bbox must satisfy x2>x1 and y2>y1")
    return x1, y1, x2, y2


def _normalize_keywords(raw_keywords: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for keyword in raw_keywords:
        value = keyword.strip().lower()
        if value:
            normalized.append(value)
    return normalized


def _get_window_title(hwnd: int) -> str:
    user32 = ctypes.windll.user32
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value.strip()


def _get_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    rect = wintypes.RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return None
    x1, y1, x2, y2 = int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _get_client_rect_screen(hwnd: int) -> tuple[int, int, int, int] | None:
    user32 = ctypes.windll.user32
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    top_left = wintypes.POINT(rect.left, rect.top)
    bottom_right = wintypes.POINT(rect.right, rect.bottom)
    if not user32.ClientToScreen(hwnd, ctypes.byref(top_left)):
        return None
    if not user32.ClientToScreen(hwnd, ctypes.byref(bottom_right)):
        return None
    x1, y1, x2, y2 = int(top_left.x), int(top_left.y), int(bottom_right.x), int(bottom_right.y)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def find_window_by_title_keywords(
    raw_keywords: Iterable[str],
) -> tuple[int, str, tuple[int, int, int, int]] | None:
    if not hasattr(ctypes, "WINFUNCTYPE"):
        return None
    keywords = _normalize_keywords(raw_keywords)
    if not keywords:
        return None

    user32 = ctypes.windll.user32
    if not hasattr(user32, "EnumWindows"):
        return None

    found: list[tuple[int, str, tuple[int, int, int, int]]] = []
    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.IsIconic(hwnd):
            return True

        title = _get_window_title(hwnd).lower()
        if not title:
            return True
        if not any(keyword in title for keyword in keywords):
            return True

        rect = _get_window_rect(hwnd)
        if rect is None:
            return True
        found.append((hwnd, title, rect))
        return False

    user32.EnumWindows(EnumWindowsProc(callback), 0)
    return found[0] if found else None


def _crop_to_client_area(window_image: Image.Image, hwnd: int) -> Image.Image:
    window_rect = _get_window_rect(hwnd)
    client_rect = _get_client_rect_screen(hwnd)
    if window_rect is None or client_rect is None:
        return window_image

    win_w = window_rect[2] - window_rect[0]
    win_h = window_rect[3] - window_rect[1]
    img_w, img_h = window_image.size
    if abs(img_w - win_w) > 6 or abs(img_h - win_h) > 6:
        return window_image

    left = max(0, client_rect[0] - window_rect[0])
    top = max(0, client_rect[1] - window_rect[1])
    right = min(img_w, client_rect[2] - window_rect[0])
    bottom = min(img_h, client_rect[3] - window_rect[1])
    if right <= left or bottom <= top:
        return window_image
    return window_image.crop((left, top, right, bottom))


def _looks_blank_capture(image: Image.Image) -> bool:
    grayscale = image.convert("L")
    stat = ImageStat.Stat(grayscale)
    mean_value = float(stat.mean[0]) if stat.mean else 0.0
    std_value = float(stat.stddev[0]) if stat.stddev else 0.0
    darkest, brightest = grayscale.getextrema()
    return mean_value <= 4.0 and std_value <= 3.0 and brightest <= 16 and darkest <= 8


def capture_screen_to_file(output_path: str | Path, bbox: tuple[int, int, int, int] | None = None) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = ImageGrab.grab(bbox=bbox, all_screens=True)
    except Exception as exc:
        raise RuntimeError("Screen capture failed. Check game window visibility and screenshot permissions.") from exc
    image.save(output)
    return output


def capture_window_to_file(output_path: str | Path, hwnd: int) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = ImageGrab.grab(window=hwnd, all_screens=True)
    except Exception as exc:
        raise RuntimeError("Window capture failed. Check game window visibility and permissions.") from exc
    image = _crop_to_client_area(image.convert("RGB"), hwnd)
    if _looks_blank_capture(image):
        raise RuntimeError("Window capture returned a blank frame.")
    image.save(output)
    return output


def capture_window_by_keywords_to_file(
    output_path: str | Path,
    title_keywords: Iterable[str],
) -> tuple[Path, tuple[int, int, int, int] | None, str | None, str]:
    match = find_window_by_title_keywords(title_keywords)
    if match is None:
        path = capture_screen_to_file(output_path, bbox=None)
        return path, None, None, "full_screen_fallback"
    hwnd, title, rect = match
    client_rect = _get_client_rect_screen(hwnd) or rect
    try:
        path = capture_window_to_file(output_path, hwnd)
        return path, client_rect, title, "window"
    except Exception:
        path = capture_screen_to_file(output_path, bbox=client_rect)
        return path, client_rect, title, "client_bbox"
