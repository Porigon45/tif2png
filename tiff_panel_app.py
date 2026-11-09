"""Local browser tool for preparing DIC / Red / Green / Merge figure rows."""

from __future__ import annotations

import argparse
import io
import json
import re
import webbrowser
import zipfile
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from merge_tiff_picture import load_panel_channel, suggest_green_shift

CHANNELS = ("dic", "red", "green")
PANELS = (*CHANNELS, "merge")
DEFAULT_TONE = [0, 100]


def discover(root: Path) -> tuple[dict, dict]:
    groups = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".tif", ".tiff") or not path.is_file():
            continue
        prefix, _, channel = path.stem.rpartition("-")
        if channel.lower() in CHANNELS and prefix:
            key = (path.parent / prefix).relative_to(root).as_posix()
            groups.setdefault(key, {}).setdefault(channel.lower(), []).append(path)
    complete, skipped = {}, {}
    for key, files in groups.items():
        issues = [name for name in CHANNELS if len(files.get(name, [])) != 1]
        if issues:
            skipped[key] = issues
        else:
            complete[key] = {name: files[name][0] for name in CHANNELS}
    return complete, skipped


@lru_cache(maxsize=9)
def read_channel(path: Path, channel: str, modified: int, size: int) -> np.ndarray:
    return load_panel_channel(path, channel)


def load_channels(paths: dict) -> dict:
    channels = {}
    for name, path in paths.items():
        stat = path.stat()
        channels[name] = read_channel(path, name, stat.st_mtime_ns, stat.st_size)
    return channels


def integer_list(values, count: int) -> list[int]:
    if not isinstance(values, list) or len(values) != count or any(type(v) is not int for v in values):
        raise ValueError("整数の設定値が必要です。")
    return values


def display_pixels(arr: np.ndarray, levels: list[int]) -> np.ndarray:
    low, high = integer_list(levels, 2)
    if not 0 <= low < high <= 1000:
        raise ValueError("表示範囲は 0 ≤ Black < White ≤ 1000 です。")
    return np.rint(np.clip((arr * 1000 - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def adjust_tone(pixels: np.ndarray, tone: list[int]) -> np.ndarray:
    brightness, contrast = integer_list(tone, 2)
    if not -100 <= brightness <= 100 or not 0 <= contrast <= 200:
        raise ValueError("明るさは -100〜100、コントラストは 0〜200% です。")
    # Contrast is centered at 127.5. Match the integer rounding in the editor.
    values = np.arange(256, dtype=np.int32)
    lut = ((2 * values - 255) * contrast + 25500 + 200 * brightness + 100) // 200
    return np.clip(lut, 0, 255).astype(np.uint8)[pixels]


def render_panel(channels: dict, recipe: dict) -> Image.Image:
    x, y, width, height = integer_list(recipe["roi"], 4)
    if width <= 0 or height <= 0:
        raise ValueError("切り抜き枠の幅と高さは正の整数にしてください。")
    crops = {}
    for name in CHANNELS:
        dx, dy = (0, 0) if name == "dic" else integer_list(recipe["offsets"][name], 2)
        sx, sy = x - dx, y - dy
        arr = channels[name]
        h, w = arr.shape
        if sx < 0 or sy < 0 or sx + width > w or sy + height > h:
            raise ValueError(f"切り抜き枠が {name.upper()} の画像範囲外です。")
        crops[name] = display_pixels(arr[sy : sy + height, sx : sx + width], recipe["levels"][name])
    tones = recipe.get("tones", {})
    dic, red, green = (adjust_tone(crops[name], tones.get(name, DEFAULT_TONE)) for name in CHANNELS)
    merge_tone = tones.get("merge", DEFAULT_TONE)
    zero = np.zeros_like(dic)
    tiles = [
        np.stack([dic, dic, dic], axis=-1),
        np.stack([red, zero, zero], axis=-1),
        np.stack([zero, green, zero], axis=-1),
        np.stack([adjust_tone(red, merge_tone), adjust_tone(green, merge_tone), zero], axis=-1),
    ]
    return Image.fromarray(np.concatenate(tiles, axis=1))


def export_zip(panel: Image.Image, recipe: dict) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for extension, fmt in (("png", "PNG"), ("tif", "TIFF")):
            image = io.BytesIO()
            panel.save(image, format=fmt)
            archive.writestr(f"panel.{extension}", image.getvalue())
        archive.writestr("recipe.json", json.dumps(recipe, ensure_ascii=False, indent=2))
    return output.getvalue()


def handler_for(root: Path):
    groups, skipped = discover(root)
    html = Path(__file__).with_suffix(".html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, payload: bytes, content_type: str, status: int = 200, filename: str = ""):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(payload)

        def json(self, value, status: int = 200):
            self.reply(json.dumps(value, ensure_ascii=False).encode(), "application/json", status)

        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    self.reply(html, "text/html; charset=utf-8")
                elif url.path == "/groups":
                    self.json({"groups": list(groups), "skipped": skipped})
                elif url.path == "/image":
                    group, name = query["group"][0], query["channel"][0]
                    path = groups[group][name]
                    stat = path.stat()
                    arr = read_channel(path, name, stat.st_mtime_ns, stat.st_size)
                    pixels = display_pixels(arr, [int(query["low"][0]), int(query["high"][0])])
                    output = io.BytesIO()
                    Image.fromarray(pixels).save(output, format="PNG")
                    self.reply(output.getvalue(), "image/png")
                elif url.path == "/suggest":
                    data = load_channels(groups[query["group"][0]])
                    dx, dy, score = suggest_green_shift(data["red"], data["green"])
                    self.json({"shift": [dx, dy], "score": score})
                else:
                    self.json({"error": "ページが見つかりません。"}, 404)
            except (KeyError, ValueError, OSError) as exc:
                self.json({"error": str(exc)}, 400)

        def do_POST(self):
            if self.path != "/export":
                self.json({"error": "ページが見つかりません。"}, 404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 32768:
                    raise ValueError("設定データのサイズが不正です。")
                request = json.loads(self.rfile.read(length))
                group = request["group"]
                paths = groups[group]
                tones = request.get("tones", {})
                if not isinstance(tones, dict):
                    raise ValueError("明るさ・コントラストの設定が不正です。")
                recipe = {
                    "version": 2,
                    "group": group,
                    "sources": {name: path.relative_to(root).as_posix() for name, path in paths.items()},
                    "roi": request["roi"],
                    "offsets": request["offsets"],
                    "levels": request["levels"],
                    "tones": {name: tones.get(name, DEFAULT_TONE) for name in PANELS},
                }
                panel = render_panel(load_channels(paths), recipe)
                filename = re.sub(r"[^a-zA-Z0-9._-]", "_", group.replace("/", "_"))
                self.reply(export_zip(panel, recipe), "application/zip", filename=f"{filename}-panel.zip")
            except (KeyError, ValueError, TypeError, OSError) as exc:
                self.json({"error": str(exc)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="TIFFフォルダ。サブフォルダも検索します。")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"フォルダが見つかりません: {root}")
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(root)) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"{url} をブラウザで開いてください。終了: Ctrl+C", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
