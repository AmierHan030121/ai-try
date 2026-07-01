"""Build paired change crops for CDChat-style dual-temporal datasets.

Expected image root layout:

    image_root/
      A/<img_id>.png
      B/<img_id>.png
      label/<img_id>.png

The output is a JSONL file plus optional cropped image pairs. The JSONL is the
stable interface consumed by later prompt/inference wrappers.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps


@dataclass(frozen=True)
class CropRecord:
    image_id: str
    mode: str
    crop_index: int
    box_xyxy: tuple[int, int, int, int]
    score: float
    crop_a: str
    crop_b: str
    crop_label: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paired change crops.")
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument(
        "--metadata",
        required=True,
        type=Path,
        help="Question JSON/list or caption JSON with images[].",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("none", "random", "auto_diff", "oracle_mask"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--metadata-out",
        type=Path,
        help="Defaults to out-dir/crops_<mode>.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit for smoke tests.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_image_id(value: Any) -> str:
    return Path(str(value)).name


def load_image_ids(path: Path) -> list[str]:
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        records = data["images"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Metadata must be a list or a dict with an 'images' list.")

    image_ids: list[str] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        image_id = (
            record.get("img_id")
            or record.get("file_name")
            or record.get("image_id")
            or record.get("filename")
        )
        if image_id is None:
            continue
        key = normalize_image_id(image_id)
        if key not in seen:
            image_ids.append(key)
            seen.add(key)
    if not image_ids:
        raise ValueError(f"No image IDs found in metadata file: {path}")
    return image_ids


def path_for(image_root: Path, folder: str, image_id: str) -> Path:
    return image_root / folder / image_id


def validate_dataset_paths(image_root: Path, image_ids: list[str], require_label: bool) -> None:
    missing: list[Path] = []
    for image_id in image_ids:
        for folder in ("A", "B"):
            path = path_for(image_root, folder, image_id)
            if not path.exists():
                missing.append(path)
        label_path = path_for(image_root, "label", image_id)
        if require_label and not label_path.exists():
            missing.append(label_path)
    if missing:
        preview = ", ".join(str(path) for path in missing[:12])
        extra = "" if len(missing) <= 12 else f" ... and {len(missing) - 12} more"
        raise FileNotFoundError(f"Missing dataset file(s): {preview}{extra}")


def clamp_box(
    center_x: int,
    center_y: int,
    crop_size: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    half = crop_size // 2
    left = max(0, min(center_x - half, width - crop_size))
    top = max(0, min(center_y - half, height - crop_size))
    right = min(width, left + crop_size)
    bottom = min(height, top + crop_size)
    if right - left < crop_size:
        left = max(0, right - crop_size)
    if bottom - top < crop_size:
        top = max(0, bottom - crop_size)
    return (left, top, right, bottom)


def sliding_boxes(width: int, height: int, crop_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    crop_size = min(crop_size, width, height)
    xs = list(range(0, max(width - crop_size, 0) + 1, stride))
    ys = list(range(0, max(height - crop_size, 0) + 1, stride))
    if not xs or xs[-1] != width - crop_size:
        xs.append(max(width - crop_size, 0))
    if not ys or ys[-1] != height - crop_size:
        ys.append(max(height - crop_size, 0))
    return [(x, y, x + crop_size, y + crop_size) for y in ys for x in xs]


def score_boxes(
    heatmap: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[tuple[int, int, int, int], float]]:
    scored = []
    for box in boxes:
        left, top, right, bottom = box
        patch = heatmap[top:bottom, left:right]
        scored.append((box, float(patch.mean()) if patch.size else 0.0))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored


def iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    inter = max(ix2 - ix1, 0) * max(iy2 - iy1, 0)
    if inter == 0:
        return 0.0
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    return inter / (left_area + right_area - inter)


def top_non_overlapping(
    scored: list[tuple[tuple[int, int, int, int], float]],
    top_k: int,
    max_iou: float = 0.5,
) -> list[tuple[tuple[int, int, int, int], float]]:
    selected: list[tuple[tuple[int, int, int, int], float]] = []
    for box, score in scored:
        if all(iou(box, prev_box) <= max_iou for prev_box, _ in selected):
            selected.append((box, score))
        if len(selected) == top_k:
            break
    return selected


def diff_heatmap(image_a: Image.Image, image_b: Image.Image) -> np.ndarray:
    gray_a = ImageOps.grayscale(image_a)
    gray_b = ImageOps.grayscale(image_b)
    diff = ImageChops.difference(gray_a, gray_b).filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.asarray(diff, dtype=np.float32)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def mask_heatmap(label: Image.Image) -> np.ndarray:
    arr = np.asarray(ImageOps.grayscale(label), dtype=np.float32)
    if arr.max() > 0:
        arr /= arr.max()
    return arr


def random_boxes(
    width: int,
    height: int,
    crop_size: int,
    top_k: int,
    rng: random.Random,
) -> list[tuple[tuple[int, int, int, int], float]]:
    crop_size = min(crop_size, width, height)
    boxes: list[tuple[tuple[int, int, int, int], float]] = []
    max_x = max(width - crop_size, 0)
    max_y = max(height - crop_size, 0)
    attempts = 0
    while len(boxes) < top_k and attempts < top_k * 20:
        attempts += 1
        x = rng.randint(0, max_x) if max_x else 0
        y = rng.randint(0, max_y) if max_y else 0
        box = (x, y, x + crop_size, y + crop_size)
        if all(iou(box, prev_box) <= 0.5 for prev_box, _ in boxes):
            boxes.append((box, 0.0))
    while len(boxes) < top_k:
        boxes.append(((0, 0, crop_size, crop_size), 0.0))
    return boxes


def select_boxes(
    mode: str,
    image_a: Image.Image,
    image_b: Image.Image,
    label: Image.Image | None,
    top_k: int,
    crop_size: int,
    stride: int,
    rng: random.Random,
) -> list[tuple[tuple[int, int, int, int], float]]:
    width, height = image_a.size
    if mode == "none":
        return []
    if mode == "random":
        return random_boxes(width, height, crop_size, top_k, rng)
    boxes = sliding_boxes(width, height, crop_size, stride)
    if mode == "auto_diff":
        heatmap = diff_heatmap(image_a, image_b)
    elif mode == "oracle_mask":
        if label is None:
            raise ValueError("oracle_mask mode requires label images.")
        heatmap = mask_heatmap(label)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return top_non_overlapping(score_boxes(heatmap, boxes), top_k=top_k)


def relpath(path: Path, base: Path) -> str:
    return str(path.resolve().relative_to(base.resolve()))


def save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.crop(box).save(out_path)


def process_image(
    image_root: Path,
    out_dir: Path,
    mode: str,
    image_id: str,
    top_k: int,
    crop_size: int,
    stride: int,
    rng: random.Random,
) -> list[CropRecord]:
    image_a = Image.open(path_for(image_root, "A", image_id)).convert("RGB")
    image_b = Image.open(path_for(image_root, "B", image_id)).convert("RGB")
    label_path = path_for(image_root, "label", image_id)
    label = Image.open(label_path) if label_path.exists() else None

    selected = select_boxes(mode, image_a, image_b, label, top_k, crop_size, stride, rng)
    records: list[CropRecord] = []
    for idx, (box, score) in enumerate(selected):
        stem = Path(image_id).stem
        suffix = Path(image_id).suffix or ".png"
        crop_a = out_dir / mode / "A" / f"{stem}_crop{idx:02d}{suffix}"
        crop_b = out_dir / mode / "B" / f"{stem}_crop{idx:02d}{suffix}"
        crop_label = None
        save_crop(image_a, box, crop_a)
        save_crop(image_b, box, crop_b)
        if label is not None:
            label_out = out_dir / mode / "label" / f"{stem}_crop{idx:02d}{suffix}"
            save_crop(label.convert("L"), box, label_out)
            crop_label = relpath(label_out, out_dir)
        records.append(
            CropRecord(
                image_id=image_id,
                mode=mode,
                crop_index=idx,
                box_xyxy=box,
                score=score,
                crop_a=relpath(crop_a, out_dir),
                crop_b=relpath(crop_b, out_dir),
                crop_label=crop_label,
            )
        )
    return records


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.crop_size < 1:
        raise ValueError("--crop-size must be at least 1")
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    missing = [path for path in (args.image_root, args.metadata) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input path(s): {joined}")

    image_ids = load_image_ids(args.metadata)
    if args.limit is not None:
        image_ids = image_ids[: args.limit]
    validate_dataset_paths(
        args.image_root,
        image_ids,
        require_label=args.mode == "oracle_mask",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_out = args.metadata_out or args.out_dir / f"crops_{args.mode}.jsonl"
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with metadata_out.open("w", encoding="utf-8") as handle:
        for image_id in image_ids:
            for record in process_image(
                args.image_root,
                args.out_dir,
                args.mode,
                image_id,
                args.top_k,
                args.crop_size,
                args.stride,
                rng,
            ):
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
