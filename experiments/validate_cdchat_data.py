"""Validate CDChat image-folder structure for a question file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CDChat A/B/label files.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--out-missing", type=Path)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def normalize_image_id(value: Any) -> str:
    return Path(str(value)).name


def load_question_ids(path: Path, limit: int | None) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Question file must be a JSON list.")

    image_ids: list[str] = []
    seen: set[str] = set()
    for record in data:
        if not isinstance(record, dict):
            continue
        image_id = record.get("img_id") or record.get("image_id") or record.get("file_name")
        if image_id is None:
            continue
        key = normalize_image_id(image_id)
        if key not in seen:
            image_ids.append(key)
            seen.add(key)
        if limit is not None and len(image_ids) >= limit:
            break
    if not image_ids:
        raise ValueError(f"No image IDs found in {path}")
    return image_ids


def expected_paths(image_root: Path, image_id: str) -> dict[str, Path]:
    return {
        "A": image_root / "A" / image_id,
        "B": image_root / "B" / image_id,
        "label": image_root / "label" / image_id,
    }


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1")
    if not args.image_root.exists():
        raise FileNotFoundError(args.image_root)
    if not args.questions.exists():
        raise FileNotFoundError(args.questions)

    image_ids = load_question_ids(args.questions, args.limit)
    missing: list[dict[str, str]] = []
    for image_id in image_ids:
        for role, path in expected_paths(args.image_root, image_id).items():
            if not path.exists():
                missing.append({"image_id": image_id, "role": role, "path": str(path)})

    if args.out_missing:
        args.out_missing.parent.mkdir(parents=True, exist_ok=True)
        with args.out_missing.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "role", "path"])
            writer.writeheader()
            writer.writerows(missing)

    checked_files = len(image_ids) * 3
    present_files = checked_files - len(missing)
    print(f"Checked {len(image_ids)} image id(s), {present_files}/{checked_files} files present.")
    if missing:
        preview = ", ".join(item["path"] for item in missing[:8])
        raise SystemExit(f"Missing {len(missing)} file(s): {preview}")


if __name__ == "__main__":
    main()
