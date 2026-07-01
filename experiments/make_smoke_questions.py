"""Create a small CDChat question file for smoke-test inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small question JSON.")
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.questions.exists():
        raise FileNotFoundError(args.questions)

    with args.questions.open("r", encoding="utf-8") as handle:
        data: Any = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("CDChat question files are expected to be JSON lists.")

    subset = data[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(subset, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Wrote {len(subset)} question(s) to {args.out}")


if __name__ == "__main__":
    main()
