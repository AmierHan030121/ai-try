"""Evaluate CDChat-style JSONL predictions against caption references.

The script accepts the prediction JSONL produced by CDChat's batch evaluator
and a COCO-like reference JSON with an ``images`` list. It prefers the official
``pycocoevalcap`` scorers when available and can fall back to internal lexical
metrics for smoke tests.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[a-z0-9]+")
NUMBER_RE = re.compile(r"\b\d+\b")
NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}
NO_CHANGE_MARKERS = (
    "no change",
    "no changes",
    "no difference",
    "without change",
    "unchanged",
    "remain unchanged",
    "remains unchanged",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CDChat JSONL outputs.")
    parser.add_argument("--pred", required=True, type=Path, help="Prediction JSONL file.")
    parser.add_argument(
        "--references",
        required=True,
        type=Path,
        help="Reference caption JSON, such as CDChat data_files/levir_captions.json.",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output summary CSV path.")
    parser.add_argument(
        "--per-sample-out",
        type=Path,
        help="Optional per-sample CSV with matched prediction/reference/count fields.",
    )
    parser.add_argument(
        "--metric-backend",
        choices=("auto", "pycoco", "internal"),
        default="auto",
        help="Use pycocoevalcap, internal smoke-test metrics, or auto fallback.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any prediction image_id has no reference caption.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_image_id(value: Any) -> str:
    if value is None:
        raise ValueError("Missing image identifier")
    return Path(str(value)).name


def as_caption_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        captions: list[str] = []
        for item in value:
            if isinstance(item, str):
                captions.append(item)
            elif isinstance(item, dict) and isinstance(item.get("caption"), str):
                captions.append(item["caption"])
        return captions
    return []


def load_references(path: Path) -> tuple[dict[str, list[str]], dict[str, int]]:
    data = read_json(path)
    if isinstance(data, dict) and isinstance(data.get("images"), list):
        records = data["images"]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError(
            "Reference JSON must be a list or a dict with an 'images' list."
        )

    references: dict[str, list[str]] = {}
    region_counts: dict[str, int] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        image_id = (
            record.get("file_name")
            or record.get("img_id")
            or record.get("image_id")
            or record.get("filename")
        )
        if image_id is None:
            continue
        key = normalize_image_id(image_id)
        captions = as_caption_list(record.get("captions") or record.get("caption"))
        if captions:
            references.setdefault(key, []).extend(captions)
        attribute = record.get("attribute")
        if isinstance(attribute, dict) and "num_regions" in attribute:
            try:
                region_counts[key] = int(attribute["num_regions"])
            except (TypeError, ValueError):
                pass
    if not references:
        raise ValueError(f"No reference captions found in {path}")
    return references, region_counts


def load_predictions(path: Path) -> list[dict[str, str]]:
    predictions: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            image_id = (
                item.get("image_id")
                or item.get("img_id")
                or item.get("file_name")
                or item.get("filename")
            )
            answer = item.get("answer") or item.get("prediction") or item.get("text")
            if image_id is None or answer is None:
                raise ValueError(
                    f"Prediction line {line_no} must contain image_id and answer-like text."
                )
            predictions.append(
                {
                    "image_id": normalize_image_id(image_id),
                    "answer": str(answer),
                    "question_id": str(item.get("question_id", "")),
                }
            )
    if not predictions:
        raise ValueError(f"No predictions found in {path}")
    return predictions


def align_samples(
    predictions: list[dict[str, str]],
    references: dict[str, list[str]],
    region_counts: dict[str, int],
    strict: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    samples: list[dict[str, Any]] = []
    unmatched: list[str] = []
    for idx, pred in enumerate(predictions):
        image_id = pred["image_id"]
        refs = references.get(image_id)
        if refs is None:
            unmatched.append(image_id)
            continue
        samples.append(
            {
                "sample_id": f"{image_id}#{idx}",
                "image_id": image_id,
                "answer": pred["answer"],
                "references": refs,
                "question_id": pred["question_id"],
                "target_count": region_counts.get(image_id),
            }
        )
    if unmatched and strict:
        preview = ", ".join(unmatched[:10])
        raise ValueError(f"{len(unmatched)} prediction(s) lack references: {preview}")
    if not samples:
        raise ValueError("No predictions could be matched to references.")
    return samples, unmatched


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def best_ref_length(pred_len: int, ref_tokens: list[list[str]]) -> int:
    return min((len(ref) for ref in ref_tokens), key=lambda ref_len: abs(ref_len - pred_len))


def internal_bleu(samples: list[dict[str, Any]], max_n: int = 4) -> dict[str, float]:
    clipped = [0 for _ in range(max_n)]
    totals = [0 for _ in range(max_n)]
    pred_len_total = 0
    ref_len_total = 0
    for sample in samples:
        pred_tokens = tokenize(sample["answer"])
        ref_tokens = [tokenize(ref) for ref in sample["references"]]
        pred_len_total += len(pred_tokens)
        ref_len_total += best_ref_length(len(pred_tokens), ref_tokens)
        for n in range(1, max_n + 1):
            pred_ngrams = ngrams(pred_tokens, n)
            totals[n - 1] += sum(pred_ngrams.values())
            max_ref_counts: Counter[tuple[str, ...]] = Counter()
            for ref in ref_tokens:
                ref_ngrams = ngrams(ref, n)
                for gram, count in ref_ngrams.items():
                    max_ref_counts[gram] = max(max_ref_counts[gram], count)
            clipped[n - 1] += sum(
                min(count, max_ref_counts[gram]) for gram, count in pred_ngrams.items()
            )

    if pred_len_total == 0:
        return {f"BLEU-{n}": 0.0 for n in range(1, max_n + 1)}
    brevity = 1.0 if pred_len_total > ref_len_total else math.exp(1 - ref_len_total / pred_len_total)
    scores: dict[str, float] = {}
    log_precision_sum = 0.0
    for n in range(1, max_n + 1):
        precision = (clipped[n - 1] + 1.0) / (totals[n - 1] + 1.0)
        log_precision_sum += math.log(precision)
        scores[f"BLEU-{n}"] = brevity * math.exp(log_precision_sum / n)
    return scores


def lcs_length(left: list[str], right: list[str]) -> int:
    prev = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for idx, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(prev[idx - 1] + 1)
            else:
                current.append(max(prev[idx], current[-1]))
        prev = current
    return prev[-1]


def internal_rouge_l(samples: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for sample in samples:
        pred = tokenize(sample["answer"])
        if not pred:
            scores.append(0.0)
            continue
        best = 0.0
        for ref_text in sample["references"]:
            ref = tokenize(ref_text)
            if not ref:
                continue
            lcs = lcs_length(pred, ref)
            precision = lcs / len(pred)
            recall = lcs / len(ref)
            if precision + recall:
                best = max(best, 2 * precision * recall / (precision + recall))
        scores.append(best)
    return sum(scores) / len(scores)


def internal_meteor_exact(samples: list[dict[str, Any]]) -> float:
    scores: list[float] = []
    for sample in samples:
        pred = tokenize(sample["answer"])
        if not pred:
            scores.append(0.0)
            continue
        pred_counts = Counter(pred)
        best = 0.0
        for ref_text in sample["references"]:
            ref = tokenize(ref_text)
            if not ref:
                continue
            matches = sum((pred_counts & Counter(ref)).values())
            precision = matches / len(pred)
            recall = matches / len(ref)
            if precision + recall:
                best = max(best, 10 * precision * recall / (recall + 9 * precision))
        scores.append(best)
    return sum(scores) / len(scores)


def compute_internal_metrics(samples: list[dict[str, Any]]) -> dict[str, tuple[float, str]]:
    values = internal_bleu(samples)
    metrics = {name: (value, "internal_exact") for name, value in values.items()}
    metrics["ROUGE-L"] = (internal_rouge_l(samples), "internal_exact")
    metrics["METEOR"] = (internal_meteor_exact(samples), "internal_exact")
    metrics["CIDEr"] = (float("nan"), "not_available_internal")
    return metrics


def compute_pycoco_metrics(samples: list[dict[str, Any]]) -> dict[str, tuple[float, str]]:
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as exc:
        raise RuntimeError("pycocoevalcap is not installed") from exc

    gts = {sample["sample_id"]: sample["references"] for sample in samples}
    res = {sample["sample_id"]: [sample["answer"]] for sample in samples}
    metrics: dict[str, tuple[float, str]] = {}
    scorers = [
        (Bleu(4), ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]),
        (Meteor(), ["METEOR"]),
        (Rouge(), ["ROUGE-L"]),
        (Cider(), ["CIDEr"]),
    ]
    for scorer, names in scorers:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                score, _ = scorer.compute_score(gts, res)
            if isinstance(score, list):
                for name, value in zip(names, score):
                    metrics[name] = (float(value), "pycocoevalcap")
            else:
                metrics[names[0]] = (float(score), "pycocoevalcap")
        finally:
            meteor_proc = getattr(scorer, "meteor_p", None)
            if meteor_proc is not None:
                meteor_proc.kill()
    return metrics


def pred_count_from_text(text: str) -> int | None:
    lower = text.lower()
    if any(marker in lower for marker in NO_CHANGE_MARKERS):
        return 0
    digit_match = NUMBER_RE.search(lower)
    if digit_match:
        return int(digit_match.group(0))
    tokens = tokenize(lower)
    for token in tokens:
        if token in NUMBER_WORDS:
            return NUMBER_WORDS[token]
    return None


def compute_count_metrics(samples: list[dict[str, Any]]) -> dict[str, tuple[float, str]]:
    covered = []
    for sample in samples:
        target = sample.get("target_count")
        pred_count = pred_count_from_text(sample["answer"])
        sample["pred_count"] = pred_count
        if target is None or pred_count is None:
            sample["count_abs_error"] = None
            continue
        abs_error = abs(pred_count - int(target))
        sample["count_abs_error"] = abs_error
        covered.append(abs_error)
    if not covered:
        return {
            "count_coverage": (0.0, "attribute_num_regions"),
            "count_accuracy": (float("nan"), "attribute_num_regions"),
            "count_mae": (float("nan"), "attribute_num_regions"),
        }
    return {
        "count_coverage": (len(covered) / len(samples), "attribute_num_regions"),
        "count_accuracy": (
            sum(1 for error in covered if error == 0) / len(covered),
            "attribute_num_regions",
        ),
        "count_mae": (sum(covered) / len(covered), "attribute_num_regions"),
    }


def write_summary_csv(
    path: Path,
    metrics: dict[str, tuple[float, str]],
    counts: dict[str, tuple[float, str]],
    n_predictions: int,
    n_matched: int,
    n_unmatched: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, float | int, str]] = [
        ("num_predictions", n_predictions, "count"),
        ("num_matched", n_matched, "count"),
        ("num_unmatched", n_unmatched, "count"),
    ]
    rows.extend((name, value, source) for name, (value, source) in metrics.items())
    rows.extend((name, value, source) for name, (value, source) in counts.items())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "source"])
        for metric, value, source in rows:
            writer.writerow([metric, value, source])


def write_per_sample_csv(path: Path, samples: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "image_id",
                "question_id",
                "answer",
                "references",
                "pred_count",
                "target_count",
                "count_abs_error",
            ],
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "image_id": sample["image_id"],
                    "question_id": sample["question_id"],
                    "answer": sample["answer"],
                    "references": " ||| ".join(sample["references"]),
                    "pred_count": sample.get("pred_count"),
                    "target_count": sample.get("target_count"),
                    "count_abs_error": sample.get("count_abs_error"),
                }
            )


def main() -> None:
    args = parse_args()
    missing = [path for path in (args.pred, args.references) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing input file(s): {joined}")

    references, region_counts = load_references(args.references)
    predictions = load_predictions(args.pred)
    samples, unmatched = align_samples(
        predictions, references, region_counts, strict=args.strict
    )

    if args.metric_backend == "internal":
        metrics = compute_internal_metrics(samples)
    else:
        try:
            metrics = compute_pycoco_metrics(samples)
        except Exception:
            if args.metric_backend == "pycoco":
                raise
            metrics = compute_internal_metrics(samples)
    count_metrics = compute_count_metrics(samples)
    write_summary_csv(
        args.out,
        metrics,
        count_metrics,
        n_predictions=len(predictions),
        n_matched=len(samples),
        n_unmatched=len(unmatched),
    )
    if args.per_sample_out:
        write_per_sample_csv(args.per_sample_out, samples)


if __name__ == "__main__":
    main()
