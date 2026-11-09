"""X-red.tif と X-green.tif を位置合わせして重ね、
合成 TIFF (X-merge.tif) および合成 PNG (png/X-merge.png) を作成。

使い方:
    python merge_tiff_picture.py "C:\\path\\to\\images"
    python merge_tiff_picture.py "C:\\path\\to\\images" --dx 2 --dy -1
    python merge_tiff_picture.py  # カレントディレクトリを処理
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def load_image_gray(path: Path) -> np.ndarray:
    """Load a TIFF/PNG/etc. and return float32 grayscale image in [0, 1]."""
    img = Image.open(path)
    # If multi-page TIFF, take the first frame.
    try:
        img.seek(0)
    except Exception:
        pass

    # Convert alpha-containing images safely to RGB first.
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
        bg.alpha_composite(img.convert("RGBA"))
        img = bg.convert("RGB")
    elif img.mode not in ("L", "I;16", "I", "F", "RGB"):
        img = img.convert("RGB")

    arr = np.asarray(img)

    if arr.ndim == 3:
        # RGB -> grayscale (luminance)
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    arr = arr.astype(np.float32)

    # Normalize raw bit depth roughly into [0, 1]
    maxv = np.nanmax(arr) if arr.size else 1.0
    if maxv > 0:
        arr = arr / maxv
    return arr


def load_panel_channel(path: Path, channel: str) -> np.ndarray:
    """Read the first TIFF frame on a fixed 0–1 intensity scale."""
    with Image.open(path) as img:
        if img.mode in ("L", "I", "F") or img.mode.startswith("I;16"):
            arr = np.asarray(img)
        elif channel == "dic":
            arr = np.asarray(img.convert("L"))
        else:
            rgb = np.asarray(img.convert("RGB"))
            arr = rgb[..., {"red": 0, "green": 1}[channel]]
    if np.issubdtype(arr.dtype, np.integer):
        white = np.iinfo(arr.dtype).max
    else:
        white = 1.0  # Floating-point TIFF input is expected to use [0, 1].
    return np.clip(np.nan_to_num(arr.astype(np.float32) / white), 0, 1)


def percentile_normalize(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    lo = np.percentile(img, p_low)
    hi = np.percentile(img, p_high)
    if hi <= lo:
        hi = lo + 1e-6
    out = (img - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def crop_to_same_shape(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


def estimate_translation_phase_corr(ref: np.ndarray, mov: np.ndarray, max_shift: int = 30) -> Tuple[int, int, float]:
    """
    Estimate translation (dx, dy) to move 'mov' onto 'ref' using phase correlation.
    Returned shift is integer pixels.
    """
    ref, mov = crop_to_same_shape(ref, mov)

    # Remove global background offsets.
    ref0 = ref - np.mean(ref)
    mov0 = mov - np.mean(mov)

    # Windowing reduces edge artifacts.
    h, w = ref0.shape
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    win = wy * wx
    ref0 = ref0 * win
    mov0 = mov0 * win

    F = np.fft.fft2(ref0)
    G = np.fft.fft2(mov0)
    R = F * np.conj(G)
    denom = np.abs(R)
    denom[denom == 0] = 1e-8
    R /= denom
    corr = np.fft.ifft2(R)
    corr = np.abs(corr)
    corr = np.fft.fftshift(corr)

    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    y0 = max(0, cy - max_shift)
    y1 = min(corr.shape[0], cy + max_shift + 1)
    x0 = max(0, cx - max_shift)
    x1 = min(corr.shape[1], cx + max_shift + 1)
    local = corr[y0:y1, x0:x1]
    peak_idx = np.unravel_index(np.argmax(local), local.shape)
    py = y0 + peak_idx[0]
    px = x0 + peak_idx[1]
    if corr[py, px] == 0:
        return 0, 0, 0.0

    dy = py - cy
    dx = px - cx
    score = corr[py, px] / (corr.mean() + 1e-8)
    # F * conj(G) already peaks at the shift to apply to mov.
    return int(dx), int(dy), float(score)


def suggest_green_shift(red: np.ndarray, green: np.ndarray, max_shift: int = 30) -> Tuple[int, int, float]:
    """Estimate a candidate using interior pixels, excluding edge annotations."""
    red, green = crop_to_same_shape(red, green)
    h, w = red.shape
    interior = np.s_[h // 10 : max(1, 4 * h // 5), w // 10 : max(1, 9 * w // 10)]
    # Limit the influence of the brightest cells on the estimate.
    return estimate_translation_phase_corr(
        percentile_normalize(red[interior]),
        percentile_normalize(green[interior]),
        max_shift=max_shift,
    )


def shift_image(img: np.ndarray, dx: int = 0, dy: int = 0, fill: float = 0.0) -> np.ndarray:
    """Shift image without wraparound. +dx -> right, +dy -> down."""
    h, w = img.shape[:2]
    out = np.full_like(img, fill)

    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx) if dx >= 0 else w
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx) if dx <= 0 else w

    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy) if dy >= 0 else h
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy) if dy <= 0 else h

    if src_x1 > src_x0 and src_y1 > src_y0 and dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
    return out


def build_rgb(red: np.ndarray, green: np.ndarray) -> np.ndarray:
    red, green = crop_to_same_shape(red, green)
    r = (percentile_normalize(red) * 255).astype(np.uint8)
    g = (percentile_normalize(green) * 255).astype(np.uint8)
    b = np.zeros_like(r)
    return np.dstack([r, g, b])


def save_grayscale_preview(img: np.ndarray, path: Path) -> None:
    Image.fromarray((percentile_normalize(img) * 255).astype(np.uint8), mode="L").save(path)


def find_green_path(red_path: Path) -> Optional[Path]:
    folder = red_path.parent
    stem = red_path.stem
    if not stem.lower().endswith("-red"):
        return None
    prefix = stem[:-4]

    # 同一拡張子・tif/tiff の候補を順に確認
    candidates = [
        folder / f"{prefix}-green{red_path.suffix}",
        folder / f"{prefix}-green.tif",
        folder / f"{prefix}-green.tiff",
        folder / f"{prefix}-GREEN{red_path.suffix}",
        folder / f"{prefix}-GREEN.tif",
        folder / f"{prefix}-GREEN.tiff",
    ]
    for c in candidates:
        if c.is_file():
            return c

    # 大文字小文字の違いにも柔軟に対応
    target_stem_lower = f"{prefix.lower()}-green"
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}:
            if p.stem.lower() == target_stem_lower:
                return p
    return None


def merge_pair(
    red_path: Path,
    green_path: Path,
    dx: int = 0,
    dy: int = 0,
    no_auto_align: bool = False,
    max_shift: int = 30,
    save_aligned_grayscale: bool = False,
) -> bool:
    try:
        red = load_panel_channel(red_path, "red")
        green = load_panel_channel(green_path, "green")

        if no_auto_align:
            dx_g_auto = dy_g_auto = 0
            score_g = float("nan")
        else:
            dx_g_auto, dy_g_auto, score_g = suggest_green_shift(
                red, green, max_shift=max_shift
            )

        dx_g = dx_g_auto + dx
        dy_g = dy_g_auto + dy
        green_aligned = shift_image(green, dx=dx_g, dy=dy_g)

        rgb = build_rgb(red, green_aligned)
        folder = red_path.parent
        prefix = red_path.stem[:-4]

        # DPI 情報を引き継ぐ（存在する場合）
        dpi = None
        try:
            with Image.open(red_path) as img:
                dpi = img.info.get("dpi")
        except Exception:
            pass

        img_out = Image.fromarray(rgb, mode="RGB")

        # TIFF 保存 (<指定ディレクトリ>/X-merge.tif)
        output_tif = folder / f"{prefix}-merge.tif"
        tif_options = {}
        if dpi is not None:
            tif_options["dpi"] = dpi
        img_out.save(output_tif, **tif_options)

        # PNG 保存 (<指定ディレクトリ>/png/X-merge.png)
        png_dir = folder / "png"
        png_dir.mkdir(parents=True, exist_ok=True)
        output_png = png_dir / f"{prefix}-merge.png"
        png_options = {"format": "PNG", "optimize": True, "compress_level": 9}
        if dpi is not None:
            png_options["dpi"] = dpi
        img_out.save(output_png, **png_options)

        print(f"[OK] {red_path.name} + {green_path.name}")
        print(f"     → TIFF: {output_tif.name}")
        print(f"     → PNG : {output_png.parent.name}/{output_png.name}")
        if not no_auto_align:
            print(f"     green auto shift = (dx={dx_g_auto}, dy={dy_g_auto}), score={score_g:.3f}")
        print(f"     green final shift = (dx={dx_g}, dy={dy_g})")

        if save_aligned_grayscale:
            save_grayscale_preview(red, png_dir / f"{prefix}_red_aligned.png")
            save_grayscale_preview(green_aligned, png_dir / f"{prefix}_green_aligned.png")

        return True
    except Exception as exc:
        print(f"[ERROR] {red_path.name}: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="X-red.tif と X-green.tif を位置合わせして重ね、RGB合成 TIFF (X-merge.tif) および PNG (png/X-merge.png) を作成します。"
    )
    parser.add_argument(
        "target",
        type=Path,
        nargs="?",
        default=Path("."),
        help="対象フォルダ、または特定の *-red.tif ファイル（デフォルト: カレントディレクトリ）",
    )
    parser.add_argument("--dx", type=int, default=0, help="green画像に追加でかけるX方向シフト（右が+）")
    parser.add_argument("--dy", type=int, default=0, help="green画像に追加でかけるY方向シフト（下が+）")
    parser.add_argument("--no-auto-align", action="store_true", help="自動位置合わせをしない")
    parser.add_argument("--max-shift", type=int, default=30, help="自動位置合わせで許容する最大シフト(pixel)")
    parser.add_argument(
        "--save-aligned-grayscale",
        action="store_true",
        help="位置合わせ後の各チャンネルをグレースケールPNGでも保存する",
    )
    args = parser.parse_args()

    target = args.target.expanduser()
    if not target.exists():
        parser.error(f"指定されたパスが見つかりません: {target}")

    if target.is_file():
        if not target.stem.lower().endswith("-red"):
            parser.error(f"指定されたファイル名が *-red.tif 形式ではありません: {target.name}")
        red_files = [target]
    else:
        red_files = sorted(
            p for p in target.iterdir()
            if p.is_file() and p.suffix.lower() in {".tif", ".tiff"} and p.stem.lower().endswith("-red")
        )

    if not red_files:
        print(f"*-red.tif ファイルが見つかりませんでした: {target}")
        return

    merged_count = 0
    skipped_count = 0

    for red_path in red_files:
        green_path = find_green_path(red_path)
        prefix = red_path.stem[:-4]
        if green_path is None:
            print(f"[SKIP] 対応する緑画像 ({prefix}-green.tif) が見つかりません: {red_path.name}")
            skipped_count += 1
            continue

        success = merge_pair(
            red_path=red_path,
            green_path=green_path,
            dx=args.dx,
            dy=args.dy,
            no_auto_align=args.no_auto_align,
            max_shift=args.max_shift,
            save_aligned_grayscale=args.save_aligned_grayscale,
        )
        if success:
            merged_count += 1
        else:
            skipped_count += 1

    print(f"\n完了: 合成 {merged_count}件 / スキップ・失敗 {skipped_count}件")


if __name__ == "__main__":
    main()
