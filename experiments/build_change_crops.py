"""Change crop generation placeholder.

The final implementation will create paired crops from pre-change and
post-change images using one of four modes: none, random, auto_diff, oracle_mask.
This placeholder records the intended CLI without generating synthetic results.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired change crops.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("none", "random", "auto_diff", "oracle_mask"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    missing = [path for path in (args.image_root, args.metadata) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input path(s): {joined}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Crop generation must be implemented after dataset image naming and mask "
        "schemas are verified."
    )


if __name__ == "__main__":
    main()
