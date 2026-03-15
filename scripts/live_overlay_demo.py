from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import tkinter as tk
from tkinter import ttk
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm_orchestrator import LLMOrchestrator
from app.recommendation_engine import ItemRecommendationEngine
from app.vision.inference import ReplayVisionPredictor
from app.vision.screen_capture import (
    capture_screen_to_file,
    capture_window_by_keywords_to_file,
    parse_bbox,
)


def _enable_click_through(hwnd: int, enabled: bool) -> None:
    if not sys.platform.startswith("win"):
        return
    import ctypes

    GWL_EXSTYLE = -20
    WS_EX_LAYERED = 0x80000
    WS_EX_TRANSPARENT = 0x20
    current_style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    new_style = (current_style | WS_EX_LAYERED | WS_EX_TRANSPARENT) if enabled else (current_style & ~WS_EX_TRANSPARENT)
    ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)


@dataclass
class RuntimeSettings:
    use_llm: bool
    interval: float
    confidence_threshold: float
    min_texture_std: float


class LiveOverlayApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.predictor = ReplayVisionPredictor(args.weights, args.classes)
        self.engine = ItemRecommendationEngine()
        self.llm = LLMOrchestrator()
        self.capture_bbox = parse_bbox(args.screen_bbox)
        self.capture_mode = args.capture_mode
        self.window_title_keywords = [part.strip() for part in args.window_title_keywords.split(",") if part.strip()]
        self.last_capture_bbox: tuple[int, int, int, int] | None = self.capture_bbox
        self.last_capture_note = "startup"
        self.last_frame_hash = ""
        self.stale_frame_count = 0

        self.stop_event = threading.Event()
        self.refresh_event = threading.Event()
        self.result_queue: queue.Queue[dict[str, str]] = queue.Queue(maxsize=8)
        self.running = True

        self.settings_lock = threading.Lock()
        self.chat_lock = threading.Lock()
        self.latest_context_lock = threading.Lock()
        self.chat_history: list[dict[str, str]] = []
        self.latest_context: dict[str, dict] | None = None

        self.last_llm_text = ""
        self.last_llm_ts = 0.0
        self.last_llm_source = "none"
        self.last_llm_error = ""

        self.settings = RuntimeSettings(
            use_llm=args.use_llm,
            interval=max(0.8, args.interval),
            confidence_threshold=args.confidence_threshold,
            min_texture_std=args.min_texture_std,
        )

        self.base_intent = args.intent
        self.base_stage = args.stage
        self.base_target = args.target_champion
        self.base_question = args.question or "Recommend the top 3 craftable items for the current board."

        self.root = tk.Tk()
        self.root.title("JCC Overlay Control Panel")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.panel_width = 500
        self.panel_height = 430

        screen_w = self.root.winfo_screenwidth()
        margin = 24
        overlay_x = args.overlay_x if args.overlay_x is not None else max(margin, screen_w - args.overlay_width - margin)
        overlay_y = args.overlay_y
        panel_x = args.panel_x if args.panel_x is not None else max(margin, overlay_x - self.panel_width - 12)
        panel_y = args.panel_y
        self.root.geometry(f"{self.panel_width}x{self.panel_height}+{panel_x}+{panel_y}")

        self.overlay = tk.Toplevel(self.root)
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", args.overlay_alpha)
        self.overlay.geometry(f"{args.overlay_width}x{args.overlay_height}+{overlay_x}+{overlay_y}")
        self.overlay.configure(bg="#101014")

        self.overlay_text = tk.Label(
            self.overlay,
            text="Starting first inference...",
            justify="left",
            anchor="nw",
            bg="#101014",
            fg="#F6F8FA",
            font=("Microsoft YaHei UI", 12),
            wraplength=args.overlay_width - 24,
            padx=12,
            pady=10,
        )
        self.overlay_text.pack(fill="both", expand=True)

        self._build_control_panel()
        self._set_click_through(args.overlay_click_through)

    def _build_control_panel(self) -> None:
        self.var_use_llm = tk.BooleanVar(value=self.settings.use_llm)
        self.var_interval = tk.StringVar(value=f"{self.settings.interval:.1f}")
        self.var_conf = tk.StringVar(value=f"{self.settings.confidence_threshold:.2f}")
        self.var_texture_std = tk.StringVar(value=f"{self.settings.min_texture_std:.1f}")
        self.var_click_through = tk.BooleanVar(value=self.args.overlay_click_through)
        self.var_chat_input = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="status: initializing...")

        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill="both", expand=True)

        row0 = ttk.Frame(frame)
        row0.pack(fill="x", pady=2)
        ttk.Label(row0, text="interval").pack(side="left")
        ttk.Entry(row0, textvariable=self.var_interval, width=7).pack(side="left", padx=4)
        ttk.Label(row0, text="conf").pack(side="left")
        ttk.Entry(row0, textvariable=self.var_conf, width=7).pack(side="left", padx=4)
        ttk.Label(row0, text="texture_std").pack(side="left")
        ttk.Entry(row0, textvariable=self.var_texture_std, width=7).pack(side="left", padx=4)
        ttk.Checkbutton(row0, text="Use LLM", variable=self.var_use_llm).pack(side="left", padx=4)
        ttk.Checkbutton(row0, text="Click-through", variable=self.var_click_through, command=self._on_toggle_click_through).pack(side="left", padx=4)

        row1 = ttk.Frame(frame)
        row1.pack(fill="x", pady=(6, 8))
        self.toggle_btn = ttk.Button(row1, text="Pause", command=self._toggle_running)
        self.toggle_btn.pack(side="left", padx=4)
        ttk.Button(row1, text="Refresh", command=self._refresh_now).pack(side="left", padx=4)
        ttk.Button(row1, text="Apply", command=self._apply_settings).pack(side="left", padx=4)
        ttk.Button(row1, text="Save Debug", command=self._save_debug_now).pack(side="left", padx=4)
        ttk.Button(row1, text="Hide Overlay", command=self._toggle_overlay).pack(side="left", padx=4)
        ttk.Button(row1, text="Quit", command=self._on_close).pack(side="left", padx=4)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=6)

        ttk.Label(frame, text="Interactive chat").pack(anchor="w")
        chat_wrap = ttk.Frame(frame)
        chat_wrap.pack(fill="both", expand=True, pady=4)

        self.chat_box = tk.Text(chat_wrap, height=9, wrap="word")
        self.chat_box.pack(side="left", fill="both", expand=True)
        chat_scroll = ttk.Scrollbar(chat_wrap, orient="vertical", command=self.chat_box.yview)
        chat_scroll.pack(side="right", fill="y")
        self.chat_box.configure(yscrollcommand=chat_scroll.set, state="disabled")

        chat_input_row = ttk.Frame(frame)
        chat_input_row.pack(fill="x", pady=4)
        ttk.Entry(chat_input_row, textvariable=self.var_chat_input, width=48).pack(side="left", padx=4)
        ttk.Button(chat_input_row, text="Send", command=self._send_chat).pack(side="left", padx=4)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", pady=4)
        ttk.Label(frame, textvariable=self.status_var, wraplength=self.panel_width - 24).pack(fill="x")

    def _append_chat(self, role: str, content: str) -> None:
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"{role}: {content}\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _enqueue_message(self, message: dict[str, str]) -> None:
        # Multiple producer threads write to this queue.
        # Drop oldest on overflow so the inference thread never crashes on queue.Full.
        while True:
            try:
                self.result_queue.put_nowait(message)
                return
            except queue.Full:
                try:
                    _ = self.result_queue.get_nowait()
                except queue.Empty:
                    return

    def _set_click_through(self, enabled: bool) -> None:
        try:
            self.overlay.update_idletasks()
            _enable_click_through(self.overlay.winfo_id(), enabled)
        except Exception as exc:
            self.status_var.set(f"status: click-through unavailable ({exc})")

    def _on_toggle_click_through(self) -> None:
        self._set_click_through(self.var_click_through.get())

    def _toggle_running(self) -> None:
        self.running = not self.running
        self.toggle_btn.config(text="Pause" if self.running else "Resume")
        if self.running:
            self.refresh_event.set()

    def _refresh_now(self) -> None:
        self.refresh_event.set()

    def _toggle_overlay(self) -> None:
        if self.overlay.state() == "withdrawn":
            self.overlay.deiconify()
        else:
            self.overlay.withdraw()

    def _apply_settings(self) -> None:
        try:
            interval = max(0.8, float(self.var_interval.get().strip()))
            conf = min(0.99, max(0.30, float(self.var_conf.get().strip())))
            texture_std = min(60.0, max(2.0, float(self.var_texture_std.get().strip())))
        except ValueError:
            self.status_var.set("status: invalid numeric setting")
            return

        with self.settings_lock:
            self.settings.use_llm = bool(self.var_use_llm.get())
            self.settings.interval = interval
            self.settings.confidence_threshold = conf
            self.settings.min_texture_std = texture_std
        self.status_var.set("status: settings applied")
        self.refresh_event.set()

    def _settings_snapshot(self) -> RuntimeSettings:
        with self.settings_lock:
            return RuntimeSettings(
                use_llm=self.settings.use_llm,
                interval=self.settings.interval,
                confidence_threshold=self.settings.confidence_threshold,
                min_texture_std=self.settings.min_texture_std,
            )

    def _infer_once(self, settings: RuntimeSettings) -> dict[str, str]:
        screenshot_path, capture_note = self._capture_frame()
        frame_hash = hashlib.sha1(screenshot_path.read_bytes()).hexdigest()
        if frame_hash == self.last_frame_hash:
            self.stale_frame_count += 1
        else:
            self.stale_frame_count = 0
            self.last_frame_hash = frame_hash

        vision_payload = self.predictor.predict_screenshot(
            screenshot_path,
            confidence_threshold=settings.confidence_threshold,
            min_texture_std=settings.min_texture_std,
        )
        request_payload = {
            "components": vision_payload["components"],
            "target_champion": self.base_target,
            "intent": self.base_intent,
            "stage": self.base_stage,
            "user_question": self.base_question,
        }
        recommendation_payload = self.engine.build_payload(request_payload)

        top3 = recommendation_payload.get("craftable_items", [])[:3]
        top_text = "\n".join(f"{idx + 1}. {item['name']} (score={item['score']})" for idx, item in enumerate(top3)) if top3 else "No craftable items."
        answer_text = self._build_local_english_fallback(recommendation_payload)

        with self.latest_context_lock:
            self.latest_context = {"request_payload": request_payload, "recommendation_payload": recommendation_payload}

        comp_text = json.dumps(vision_payload.get("components", {}), ensure_ascii=False)
        sp_text = json.dumps(vision_payload.get("special_items", {}), ensure_ascii=False)
        ts = time.strftime("%H:%M:%S")

        overlay_text = (
            f"JCC Live Assistant | {ts}\n"
            f"capture: {capture_note}\n"
            f"components: {comp_text}\n"
            f"special: {sp_text}\n"
            f"top3:\n{top_text}\n\n"
            f"recommendation:\n{answer_text}"
        )
        status = (
            f"status: ok @ {ts} | conf>={settings.confidence_threshold:.2f} "
            f"std>={settings.min_texture_std:.1f} | llm={'enabled(chat_only)' if settings.use_llm else 'disabled'} "
            f"| stale_frames={self.stale_frame_count}"
        )
        if self.stale_frame_count >= 3:
            status += " | warning: captured frame unchanged (check capture mode / fullscreen / window title)"
        return {"type": "inference", "overlay": overlay_text, "status": status}

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            if not self.running:
                if not self.refresh_event.wait(timeout=0.2):
                    continue
                self.refresh_event.clear()

            settings = self._settings_snapshot()
            start_time = time.time()
            try:
                payload = self._infer_once(settings)
            except Exception as exc:
                ts = time.strftime("%H:%M:%S")
                payload = {"type": "inference", "overlay": f"JCC Live Assistant error @ {ts}\n{exc}", "status": f"status: error @ {ts} | {exc}"}
                print(f"[live-overlay-error] {exc}", flush=True)

            self._enqueue_message(payload)

            elapsed = time.time() - start_time
            wait_seconds = max(0.1, settings.interval - elapsed)
            if self.refresh_event.wait(timeout=wait_seconds):
                self.refresh_event.clear()

    def _send_chat(self) -> None:
        question = self.var_chat_input.get().strip()
        if not question:
            return
        self.var_chat_input.set("")
        self._append_chat("you", question)
        threading.Thread(target=self._chat_worker, args=(question,), daemon=True).start()

    def _chat_worker(self, question: str) -> None:
        with self.latest_context_lock:
            context = dict(self.latest_context) if self.latest_context else None
        if not context:
            self._enqueue_message({"type": "chat", "role": "assistant", "text": "No inference context yet. Click Refresh first."})
            return

        request_payload = dict(context["request_payload"])
        recommendation_payload = dict(context["recommendation_payload"])
        with self.chat_lock:
            history_snapshot = list(self.chat_history)

        llm_result = self.llm.generate_response(
            request_payload,
            recommendation_payload,
            user_question=question,
            chat_history=history_snapshot,
        )
        reply = llm_result["text"] or recommendation_payload.get("answer_text", "No response.")
        if llm_result["source"] == "fallback":
            fallback_reason = llm_result.get("error", "unknown")
            local_text = self._build_local_english_fallback(recommendation_payload)
            reply = f"{local_text}\n\n[LLM fallback: {fallback_reason}]"

        with self.chat_lock:
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": reply})
        self._enqueue_message({"type": "chat", "role": "assistant", "text": reply})

    def _save_debug_now(self) -> None:
        settings = self._settings_snapshot()
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            debug_root = Path("data/vision/debug_crops/live_overlay")
            debug_root.mkdir(parents=True, exist_ok=True)
            shot_path, capture_note = self._capture_frame(output_path=debug_root / f"capture_{ts}.png")
            layout_path = self.predictor.cropper.draw_debug_layout(shot_path, debug_root / f"layout_{ts}.png")
            slots_dir = debug_root / f"slots_{ts}"
            self.predictor.cropper.save_crops(shot_path, slots_dir)
            self.status_var.set(
                f"status: debug saved -> {layout_path} | {capture_note} | conf={settings.confidence_threshold:.2f} std={settings.min_texture_std:.1f}"
            )
        except Exception as exc:
            self.status_var.set(f"status: debug save failed ({exc})")

    def _build_local_english_fallback(self, recommendation_payload: dict) -> str:
        top3 = recommendation_payload.get("craftable_items", [])[:3]
        if not top3:
            return "No craftable item is available from the currently detected components."
        names = ", ".join(item.get("name", "unknown_item") for item in top3)
        return f"Based on current components, prioritize: {names}."

    def _ui_pump(self) -> None:
        latest_overlay: dict[str, str] | None = None
        try:
            while True:
                message = self.result_queue.get_nowait()
                if message.get("type") == "chat":
                    self._append_chat(message.get("role", "assistant"), message.get("text", ""))
                else:
                    latest_overlay = message
        except queue.Empty:
            pass

        if latest_overlay:
            self.overlay_text.config(text=latest_overlay["overlay"])
            self.status_var.set(latest_overlay["status"])

        if not self.stop_event.is_set():
            self.root.after(120, self._ui_pump)

    def _on_close(self) -> None:
        self.stop_event.set()
        try:
            self.overlay.destroy()
        except Exception:
            pass

    def _capture_frame(self, output_path: str | Path | None = None) -> tuple[Path, str]:
        def _attach_size_note(path: Path, note: str) -> str:
            try:
                width, height = Image.open(path).size
                return f"{note} size={width}x{height}"
            except Exception:
                return note

        out = output_path or self.args.capture_output
        mode = self.capture_mode
        if mode == "auto":
            if self.capture_bbox:
                path = capture_screen_to_file(out, bbox=self.capture_bbox)
                self.last_capture_bbox = self.capture_bbox
                self.last_capture_note = _attach_size_note(path, f"bbox={self.capture_bbox}")
                return path, self.last_capture_note
            if self.window_title_keywords:
                path, bbox, title, source = capture_window_by_keywords_to_file(out, self.window_title_keywords)
                if bbox:
                    self.last_capture_bbox = bbox
                    title_note = f" title={title}" if title else ""
                    self.last_capture_note = _attach_size_note(path, f"{source} bbox={bbox}{title_note}")
                    return path, self.last_capture_note
            path = capture_screen_to_file(out, bbox=None)
            self.last_capture_bbox = None
            self.last_capture_note = _attach_size_note(path, "full_screen(fallback)")
            return path, self.last_capture_note

        if mode == "bbox":
            path = capture_screen_to_file(out, bbox=self.capture_bbox)
            self.last_capture_bbox = self.capture_bbox
            self.last_capture_note = _attach_size_note(path, f"bbox={self.capture_bbox}")
            return path, self.last_capture_note

        if mode == "window":
            path, bbox, title, source = capture_window_by_keywords_to_file(out, self.window_title_keywords)
            if bbox:
                self.last_capture_bbox = bbox
                title_note = f" title={title}" if title else ""
                self.last_capture_note = _attach_size_note(path, f"{source} bbox={bbox}{title_note}")
                return path, self.last_capture_note
            path = capture_screen_to_file(out, bbox=None)
            self.last_capture_bbox = None
            self.last_capture_note = _attach_size_note(path, "window_not_found->full_screen")
            return path, self.last_capture_note

        path = capture_screen_to_file(out, bbox=None)
        self.last_capture_bbox = None
        self.last_capture_note = _attach_size_note(path, "full_screen")
        return path, self.last_capture_note
        try:
            self.root.destroy()
        except Exception:
            pass

    def run(self) -> None:
        self.refresh_event.set()
        worker = threading.Thread(target=self._worker_loop, daemon=True)
        worker.start()
        self._ui_pump()
        if sys.platform.startswith("win"):
            print("Tip: Use Borderless Windowed mode; exclusive fullscreen may hide overlay.")
        self.root.mainloop()
        self.stop_event.set()
        worker.join(timeout=1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live overlay with control panel and chat.")
    parser.add_argument("--weights", default="data/vision/artifacts/classic_cnn/best.pt")
    parser.add_argument("--classes", default="data/vision/artifacts/classic_cnn/classes.json")
    parser.add_argument("--capture-output", default="data/vision/live_capture/latest.png")
    parser.add_argument("--screen-bbox", default=None, help="x1,y1,x2,y2. Omit for full screen.")
    parser.add_argument("--capture-mode", choices=["auto", "full", "bbox", "window"], default="full")
    parser.add_argument(
        "--window-title-keywords",
        default="金铲铲,云顶之弈,TFT,League of Legends,MuMu,腾讯手游助手,WeGame",
        help="Comma-separated window title keywords used when capture-mode is window/auto.",
    )

    # Kept as CLI-only context (removed from panel UI)
    parser.add_argument("--target-champion", default="main_carry")
    parser.add_argument("--intent", default="carry_ad")
    parser.add_argument("--stage", default="4-1")
    parser.add_argument("--question", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-min-interval", type=float, default=8.0)

    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--min-texture-std", type=float, default=10.0)

    parser.add_argument("--overlay-width", type=int, default=620)
    parser.add_argument("--overlay-height", type=int, default=360)
    parser.add_argument("--overlay-x", type=int, default=None)
    parser.add_argument("--overlay-y", type=int, default=28)
    parser.add_argument("--overlay-alpha", type=float, default=0.90)
    parser.add_argument("--overlay-click-through", action="store_true")

    parser.add_argument("--panel-x", type=int, default=None)
    parser.add_argument("--panel-y", type=int, default=28)

    args = parser.parse_args()
    LiveOverlayApp(args).run()


if __name__ == "__main__":
    main()
