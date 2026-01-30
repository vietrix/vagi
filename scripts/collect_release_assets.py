"""Collect release assets into a single folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect release assets.")
    parser.add_argument("--out-dir", type=str, default="release_assets")
    parser.add_argument("--dist-dir", type=str, default="dist")
    parser.add_argument("--exports-dir", type=str, default="exports")
    return parser.parse_args()


def _copy_if_exists(paths: Iterable[Path], out_dir: Path) -> int:
    count = 0
    for path in paths:
        if path.exists():
            target = out_dir / path.name
            shutil.copy2(path, target)
            count += 1
    return count


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_dir = Path(args.dist_dir)
    exports_dir = Path(args.exports_dir)

    _copy_if_exists(dist_dir.glob("*"), out_dir)

    export_globs = [
        "manifest.json",
        "*.safetensors",
        "*.onnx",
        "*.trt",
        "*.meta.json",
    ]
    for pattern in export_globs:
        _copy_if_exists(exports_dir.glob(pattern), out_dir)

    print(f"Collected assets in {out_dir}")


if __name__ == "__main__":
    main()
