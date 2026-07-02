#!/usr/bin/env python3
"""Verify local CDChat checkpoint files against observed Hugging Face metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "checkpoints/cdchat/pretrain_mm_projector/mm_projector.bin": {
        "size": 50_349_693,
        "sha256": "649112857bc8683b508749e53bf09843afc2725487dd397b13799e260822b85e",
    },
    "checkpoints/cdchat/model_weights_cdchat/pytorch_model-00001-of-00002.bin": {
        "size": 9_976_634_558,
        "sha256": "6f6e941c126d913a889a7a6ed255ed130b27444c2268e35fb907a5a1a67e882d",
    },
    "checkpoints/cdchat/model_weights_cdchat/pytorch_model-00002-of-00002.bin": {
        "size": 4_158_750_994,
        "sha256": "42c4bf4528254a0544894673eecd2fbed6a26d6705740a8483c2aabaca1e3e18",
    },
    "checkpoints/cdchat/model_weights_cdchat/tokenizer.model": {
        "size": 499_723,
        "sha256": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    all_ok = True
    for rel_path, expected in EXPECTED.items():
        path = ROOT / rel_path
        if not path.exists():
            print(f"MISSING {rel_path}")
            all_ok = False
            continue

        size = path.stat().st_size
        size_ok = size == expected["size"]
        print(f"{'OK' if size_ok else 'BAD'} size {rel_path}: {size}/{expected['size']}")
        if not size_ok:
            all_ok = False
            continue

        actual_hash = sha256(path)
        hash_ok = actual_hash == expected["sha256"]
        print(f"{'OK' if hash_ok else 'BAD'} sha256 {rel_path}: {actual_hash}")
        all_ok = all_ok and hash_ok

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
