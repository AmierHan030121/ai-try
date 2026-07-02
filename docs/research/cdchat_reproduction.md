# CDChat Reproduction Notes

## Verified External Resources

- Paper: https://arxiv.org/abs/2409.16261
- Official repository: https://github.com/techmn/cdchat
- Model repository: https://huggingface.co/mubashir04/cdchat
- HF model revision observed: `bf08270f943114eee92c5fcd93daf5009d460af4`
- HF license tag: Apache-2.0
- HF storage observed through API: about 14.19 GB
- Local environment note: `huggingface-cli` is deprecated here; use `hf download`.

## Official Setup

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

git clone https://github.com/techmn/cdchat.git third_party/cdchat
cd third_party/cdchat
conda create -n cdchat python=3.10 -y
conda activate cdchat
pip install --upgrade pip
pip install -e .
```

Training-only dependencies from the official README:

```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

For the first inference-only reproduction, do not install `flash-attn` until the basic environment runs.

## Checkpoint Download Gate

Download the pinned CDChat weights through the local proxy:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890

scripts/download_cdchat_weights.sh
```

The script pins Hugging Face revision `bf08270f943114eee92c5fcd93daf5009d460af4`, performs a small Hugging Face proxy preflight, downloads with resumable `wget -c` into `*.part` files, and promotes files to their final checkpoint paths only after size and SHA-256 checks pass. It then runs:

```bash
scripts/download_cdchat_weights.sh --verify-only
```

Inference is blocked until the verifier reports both correct size and correct SHA-256 for:

- `checkpoints/cdchat/pretrain_mm_projector/mm_projector.bin`
- `checkpoints/cdchat/model_weights_cdchat/pytorch_model-00001-of-00002.bin`
- `checkpoints/cdchat/model_weights_cdchat/pytorch_model-00002-of-00002.bin`
- `checkpoints/cdchat/model_weights_cdchat/tokenizer.model`

## Data Layout Required by CDChat Evaluation

The official evaluator loads three files for every `img_id`:

```text
/path/to/image-folder/
  A/<img_id>.png
  B/<img_id>.png
  label/<img_id>.png
```

The `label` file is read by the evaluator/model input path, so missing masks cause inference failures. It is not only an evaluation-side artifact.

Official question files:

- `data_files/eval_questions_levir_test.json`
- `data_files/eval_questions_sysu_test.json`

Official caption references:

- `data_files/levir_captions.json`
- `data_files/sysu_test_captions.json`

Local matched files prepared for this workspace:

- `results/raw/eval_questions_levir_test_matched.json`
- `results/raw/smoke_levir_matched_10q.json`
- `results/raw/eval_questions_sysu_test_matched.json`

Use the matched files for metric reporting. The original LEVIR question file contains 2032 unique images, but the official LEVIR reference file contains references for 1827 test images. The original SYSU question file contains 4000 unique images, but the official SYSU reference file contains references for 3774 images.

## Smoke Test

Create a 10-question subset:

```bash
python experiments/make_smoke_questions.py \
  --questions results/raw/eval_questions_levir_test_matched.json \
  --out results/raw/smoke_levir_matched_10q.json \
  --limit 10
```

Validate that all image files exist:

```bash
python experiments/validate_cdchat_data.py \
  --image-root data/cdchat/levir_matched \
  --questions results/raw/smoke_levir_matched_10q.json \
  --out-missing results/logs/smoke_levir_matched_missing.csv
```

Run inference with a wrapper that passes `model_base=None` for the HF merged weights:

```bash
cd third_party/cdchat
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from argparse import Namespace
from cdchat.eval.batch_cdchat_vqa import eval_model

args = Namespace(
    model_path="/path/to/model_weights_cdchat",
    model_base=None,
    mm_projector_path=None,
    image_folder="../../data/cdchat/levir_matched",
    question_file="../../results/raw/smoke_levir_matched_10q.json",
    answers_file="../../results/raw/smoke_levir_matched_10q_pred.jsonl",
    conv_mode="llava_v1",
    num_chunks=1,
    chunk_idx=0,
    temperature=0.2,
    top_p=None,
    num_beams=1,
    batch_size=1,
)
eval_model(args)
PY
```

Notes:

- The official README shows `--answer-file`, but the code path uses `--answers-file`.
- The HF `model_weights_cdchat` directory appears to contain merged/full weights. Running with the evaluator default `model_base` can send loading down the wrong branch.
- Use `batch_size=1` for the first test on one NVIDIA L20. After a successful smoke test, split full inference across two GPUs with `num_chunks=2` and `chunk_idx=0/1`.

## Metric Computation

```bash
python experiments/run_metrics.py \
  --pred results/raw/smoke_levir_matched_10q_pred.jsonl \
  --references third_party/cdchat/data_files/levir_captions.json \
  --out results/tables/smoke_levir_matched_10q_metrics.csv \
  --per-sample-out results/tables/smoke_levir_matched_10q_per_sample.csv
```

The script uses `pycocoevalcap` for BLEU, METEOR, ROUGE-L, and CIDEr when available. Count metrics use `attribute.num_regions` from the reference file and a conservative number extractor from the generated answer.

## Crop Metadata

```bash
python experiments/build_change_crops.py \
  --image-root data/cdchat/levir_matched \
  --metadata results/raw/smoke_levir_matched_10q.json \
  --out-dir results/crops/levir_matched_smoke_auto_diff \
  --mode auto_diff \
  --top-k 3 \
  --crop-size 128
```

Modes:

- `random`: control condition for extra visual evidence.
- `auto_diff`: deployable proposed condition using image differencing.
- `oracle_mask`: upper bound using `label/<img_id>.png`; not valid as a deployable method.
- `none`: no crop records, used to keep metadata interfaces consistent.

## Current Local Status

- `third_party/cdchat` has been cloned locally at commit `8b59976`.
- `data/cdchat/levir_matched` contains symlinks for 1827 LEVIR images with `A/B/label`.
- `results/raw/smoke_levir_matched_10q.json` has been generated and validated.
- CDChat weights are not yet verified. A size-matching but SHA-mismatched `mm_projector.bin` was quarantined; see `docs/research/local_assets_status.md` before running inference.

## Immediate Risks

- CDChat checkpoint download is the immediate blocker for smoke inference under the current proxy.
- On 2026-07-02 UTC, `127.0.0.1:7890` was not listening and Hugging Face preflight failed with connection refused. Start the proxy before retrying checkpoint download.
- SYSU image download is still required before SYSU reproduction; the CDChat repository contains JSON files but not raw SYSU images.
- Official dependency pins are older; keep inference and training environments separate.
- Full training is not recommended under the current project constraints. If needed later, use LoRA/QLoRA with reduced batch size and a hard 24-hour cap.
