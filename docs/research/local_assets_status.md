# Local Asset Status

Last updated: 2026-07-02 UTC.

## CDChat Repository

- Local path: `third_party/cdchat`
- Source: https://github.com/techmn/cdchat
- Status: cloned locally and ignored by git.

## LEVIR Data

Usable local source data was found from an existing workspace:

- A/B images: `/sunjie/AmierHan/Delta-VLM/DataSet/LevirCC/images/test`
- A/B/label images: `/sunjie/AmierHan/Delta-VLM/DataSet/LEVIR-MCI/images/test`

CDChat's official LEVIR question file contains 4064 questions over 2032 unique images, but `levir_captions.json` contains references for 1827 test images. For reproducible evaluation, use the matched subset:

- Matched question file: `results/raw/eval_questions_levir_test_matched.json`
- Smoke question file: `results/raw/smoke_levir_matched_10q.json`
- CDChat image root: `data/cdchat/levir_matched`
- Symlink count: 5481 files, equal to 1827 images x `A/B/label`
- Validation status: `experiments/validate_cdchat_data.py` reports `5481/5481 files present`.

The full CDChat LEVIR question file should not be used for metric reporting unless the missing 205 references are resolved.

## SYSU Data

- Official source repository: `third_party/SYSU-CD`
- Official download options: BaiduYun and OneDrive links in `third_party/SYSU-CD/README.md`
- Local image data status: not found locally.

CDChat's SYSU question file contains 8000 questions over 4000 unique images, while `sysu_test_captions.json` contains references for 3774 images. Use the matched subset after the images are downloaded:

- Matched question file: `results/raw/eval_questions_sysu_test_matched.json`
- Expected image root after organization: `data/cdchat/sysu_matched`

## CDChat Weights

Model repository: https://huggingface.co/mubashir04/cdchat
Pinned revision: `bf08270f943114eee92c5fcd93daf5009d460af4`

Current local files:

- Complete: `checkpoints/cdchat/model_weights_cdchat/tokenizer.model` (499,723 bytes)
- Missing final file: `checkpoints/cdchat/pretrain_mm_projector/mm_projector.bin` (expected 50,349,693 bytes)
- Resumable partial: `checkpoints/cdchat/pretrain_mm_projector/mm_projector.bin.part` (0 bytes after the failed proxy preflight)
- Missing/incomplete: `checkpoints/cdchat/model_weights_cdchat/pytorch_model-00001-of-00002.bin` (expected 9,976,634,558 bytes)
- Missing: `checkpoints/cdchat/model_weights_cdchat/pytorch_model-00002-of-00002.bin`

Expected SHA-256 hashes are encoded in `scripts/verify_cdchat_weights.py`. Size-only checks are not sufficient. During testing, an interrupted `aria2c` transfer produced a 50,349,693-byte `mm_projector.bin` with the wrong SHA-256, so that file was quarantined as `checkpoints/cdchat/pretrain_mm_projector/mm_projector.bin.bad_20260702_0409` and must not be used for inference.

The download script now writes checkpoint payloads to `*.part` files and only promotes them to the final checkpoint path after size and SHA-256 checks pass. This prevents interrupted transfers from being mistaken for usable model weights.

Current proxy diagnosis on 2026-07-02 UTC: `127.0.0.1:7890` is not listening in this shell, and Hugging Face preflight fails with `curl: (7) Failed to connect to 127.0.0.1 port 7890: Connection refused`. Large-file download remains blocked until the proxy process is started or a working proxy is exported.

Download and verify:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

scripts/download_cdchat_weights.sh
scripts/download_cdchat_weights.sh --verify-only
```

To resume one file at a time:

```bash
scripts/download_cdchat_weights.sh pretrain_mm_projector/mm_projector.bin
scripts/download_cdchat_weights.sh model_weights_cdchat/pytorch_model-00001-of-00002.bin
scripts/download_cdchat_weights.sh model_weights_cdchat/pytorch_model-00002-of-00002.bin
```

The script performs a small Hugging Face preflight before downloading large files. Do not set `CDCHAT_SKIP_PREFLIGHT=1` unless you are deliberately debugging network behavior.

Do not run CDChat inference until `scripts/download_cdchat_weights.sh --verify-only` reports `OK` for both size and SHA-256 on all four required files.

## Prepared Smoke Artifacts

- `results/raw/smoke_levir_10q.json`: generated from the original official question file.
- `results/raw/smoke_levir_matched_10q.json`: generated from the matched LEVIR subset.
- `results/crops/levir_matched_smoke_auto_diff/crops_auto_diff.jsonl`: generated successfully.
- `results/crops/levir_matched_smoke_oracle/crops_oracle_mask.jsonl`: generated successfully.

These are ignored by git and should not be reported as experimental results.
