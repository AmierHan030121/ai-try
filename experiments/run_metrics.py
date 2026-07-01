"""Metric runner placeholder for CDChat reproduction outputs.

This file intentionally does not fabricate metrics. It defines the command-line
interface and validates input paths so the next implementation step can add
BLEU, METEOR, ROUGE-L, CIDEr, and count metrics against real outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CDChat JSONL outputs.")
    parser.add_argument("--pred", required=True, type=Path, help="Prediction JSONL file.")
    parser.add_argument("--references", required=True, type=Path, help="Reference caption JSON.")
    parser.add_argument("--out", required=True, type=Path, help="Output CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    missing = [path for path in (args.pred, args.references) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {joined}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(
        "Metric computation must be implemented after real prediction and reference "
        "schemas are inspected."
    )


if __name__ == "__main__":
    main()
