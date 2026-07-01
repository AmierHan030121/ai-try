from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]


class ExperimentToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image_root = self.root / "images"
        for folder in ("A", "B", "label"):
            (self.image_root / folder).mkdir(parents=True)
        self._write_image_pair("sample_001.png", x_offset=0)
        self._write_image_pair("sample_002.png", x_offset=24)
        questions = [
            {"img_id": "sample_001.png", "question": "Describe changes."},
            {"img_id": "sample_002.png", "question": "Describe changes."},
        ]
        self.questions = self.root / "questions.json"
        self.questions.write_text(json.dumps(questions), encoding="utf-8")
        refs = {
            "images": [
                {
                    "file_name": "sample_001.png",
                    "captions": ["one building appeared ."],
                    "attribute": {"num_regions": 1},
                },
                {
                    "file_name": "sample_002.png",
                    "captions": ["one bright object appeared ."],
                    "attribute": {"num_regions": 1},
                },
            ]
        }
        self.references = self.root / "refs.json"
        self.references.write_text(json.dumps(refs), encoding="utf-8")
        self.predictions = self.root / "pred.jsonl"
        self.predictions.write_text(
            '{"image_id":"sample_001.png","answer":"one building appeared ."}\n'
            '{"image_id":"sample_002.png","answer":"one object appeared ."}\n',
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_image_pair(self, image_id: str, x_offset: int) -> None:
        before = Image.new("RGB", (256, 256), (20, 20, 20))
        after = Image.new("RGB", (256, 256), (20, 20, 20))
        ImageDraw.Draw(after).rectangle(
            (80 + x_offset, 80, 140 + x_offset, 140),
            fill=(240, 240, 240),
        )
        label = Image.new("L", (256, 256), 0)
        ImageDraw.Draw(label).rectangle(
            (80 + x_offset, 80, 140 + x_offset, 140),
            fill=255,
        )
        before.save(self.image_root / "A" / image_id)
        after.save(self.image_root / "B" / image_id)
        label.save(self.image_root / "label" / image_id)

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, *args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )

    def test_make_smoke_questions_and_validate_data(self) -> None:
        smoke = self.root / "smoke.json"
        self.run_script(
            "experiments/make_smoke_questions.py",
            "--questions",
            str(self.questions),
            "--out",
            str(smoke),
            "--limit",
            "1",
        )
        self.assertEqual(len(json.loads(smoke.read_text(encoding="utf-8"))), 1)

        result = self.run_script(
            "experiments/validate_cdchat_data.py",
            "--image-root",
            str(self.image_root),
            "--questions",
            str(self.questions),
        )
        self.assertIn("6/6 files present", result.stdout)

    def test_run_metrics_internal_backend(self) -> None:
        out = self.root / "metrics.csv"
        per_sample = self.root / "per_sample.csv"
        self.run_script(
            "experiments/run_metrics.py",
            "--pred",
            str(self.predictions),
            "--references",
            str(self.references),
            "--out",
            str(out),
            "--per-sample-out",
            str(per_sample),
            "--metric-backend",
            "internal",
        )
        with out.open(newline="", encoding="utf-8") as handle:
            rows = {row["metric"]: row for row in csv.DictReader(handle)}
        self.assertEqual(rows["num_matched"]["value"], "2")
        self.assertEqual(rows["count_accuracy"]["value"], "1.0")

    def test_build_change_crops_auto_diff(self) -> None:
        out_dir = self.root / "crops"
        self.run_script(
            "experiments/build_change_crops.py",
            "--image-root",
            str(self.image_root),
            "--metadata",
            str(self.questions),
            "--out-dir",
            str(out_dir),
            "--mode",
            "auto_diff",
            "--top-k",
            "2",
            "--limit",
            "1",
        )
        lines = (out_dir / "crops_auto_diff.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["image_id"], "sample_001.png")
        self.assertTrue((out_dir / first["crop_a"]).exists())
        self.assertTrue((out_dir / first["crop_b"]).exists())


if __name__ == "__main__":
    unittest.main()
