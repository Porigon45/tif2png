"""フォルダ直下の TIFF を PNG に一括変換（元の TIFF は保持）。
変換された PNG は <指定ディレクトリ>/png/ に保存されます。

使い方:
    python tif2png.py "C:\\path\\to\\images"
    python tif2png.py "C:\\path\\to\\images" --max-size 2048
    python tif2png.py "C:\\path\\to\\images" --max-size 0  # リサイズなし
"""

from pathlib import Path
import argparse

from PIL import Image, UnidentifiedImageError


def convert_one(tif_path: Path, output_dir: Path, max_size: int) -> tuple[int, int, int]:
    """成功画像数、スキップ数、エラー数を返す。"""
    converted = skipped = errors = 0

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(tif_path) as tif:
            n_frames = getattr(tif, "n_frames", 1)
            dpi = tif.info.get("dpi")

            for index in range(n_frames):
                # 複数ページTIFFは _p001.png, _p002.png のように別々に保存
                suffix = f"_p{index + 1:03d}" if n_frames > 1 else ""
                output = output_dir / f"{tif_path.stem}{suffix}.png"

                if output.exists():
                    print(f"[SKIP] 既存ファイル: {output.parent.name}/{output.name}")
                    skipped += 1
                    continue

                tif.seek(index)
                image = tif.copy()
                original_size = image.size

                # 不要な透明度チャンネルを外す（RGBA → RGB）
                if image.mode == "RGBA" and image.getchannel("A").getextrema() == (255, 255):
                    image = image.convert("RGB")

                # CMYK は PNG 非対応のため RGB に変換
                if image.mode == "CMYK":
                    image = image.convert("RGB")

                # 16bitグレースケールは維持。浮動小数点画像は誤変換防止のためスキップ
                if image.mode == "F":
                    raise ValueError("浮動小数点TIFFは画素値を維持したPNGへ直接変換できません")

                if max_size > 0 and max(image.size) > max_size:
                    # 16bit画像は画素値の補間による変化を避けるため最近傍法を使う
                    resampling = (Image.Resampling.NEAREST if image.mode.startswith("I;16")
                                  else Image.Resampling.LANCZOS)
                    image.thumbnail((max_size, max_size), resample=resampling)

                options = {"format": "PNG", "optimize": True, "compress_level": 9}
                if dpi is not None:
                    options["dpi"] = dpi
                image.save(output, **options)

                before = tif_path.stat().st_size
                after = output.stat().st_size
                # 複数ページTIFFのサイズ比較は全ページ合計に対する参考値
                reduction = 100 * (1 - after / before)
                print(f"[OK] {tif_path.name} → {output.parent.name}/{output.name} "
                      f"({original_size[0]}×{original_size[1]} → "
                      f"{image.width}×{image.height}, "
                      f"{after / 1024**2:.2f} MB, 元TIFF比 {reduction:+.1f}%削減)")
                converted += 1

    except (OSError, ValueError, UnidentifiedImageError) as exc:
        print(f"[ERROR] {tif_path.name}: {exc}")
        errors += 1

    return converted, skipped, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="TIFF画像をまとめてPNGへ変換します")
    parser.add_argument("folder", type=Path, help="TIFFが入ったフォルダ")
    parser.add_argument("--max-size", type=int, default=1600,
                        help="画像の長辺の上限（デフォルト1600、0で縮小なし）")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="PNGの保存先フォルダ（デフォルト: <指定ディレクトリ>/png）")
    args = parser.parse_args()

    folder = args.folder.expanduser()
    if not folder.is_dir():
        parser.error(f"フォルダが見つかりません: {folder}")
    if args.max_size < 0:
        parser.error("--max-size は0以上を指定してください")

    files = sorted(p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() in {".tif", ".tiff"})
    if not files:
        print("TIFFファイルが見つかりませんでした。")
        return

    output_dir = (args.output_dir.expanduser() if args.output_dir
                  else folder / "png")
    output_dir.mkdir(parents=True, exist_ok=True)

    converted = skipped = errors = 0
    for tif_path in files:
        ok, skip, err = convert_one(tif_path, output_dir, args.max_size)
        converted += ok
        skipped += skip
        errors += err

    print(f"\n完了: 変換 {converted}枚 / スキップ {skipped}枚 / エラー {errors}件")
    print(f"保存先: {output_dir}")


if __name__ == "__main__":
    main()
