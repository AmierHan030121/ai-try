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

## Smoke Test

Create a 10-question subset:

```bash
python experiments/make_smoke_questions.py \
  --questions third_party/cdchat/data_files/eval_questions_levir_test.json \
  --out results/raw/smoke_levir_10q.json \
  --limit 10
```

Validate that all image files exist:

```bash
python experiments/validate_cdchat_data.py \
  --image-root /path/to/LEVIR_or_SYSU_root \
  --questions results/raw/smoke_levir_10q.json \
  --out-missing results/logs/smoke_levir_missing.csv
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
    image_folder="/path/to/LEVIR_or_SYSU_root",
    question_file="../../results/raw/smoke_levir_10q.json",
    answers_file="../../results/raw/smoke_levir_10q_pred.jsonl",
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
  --pred results/raw/smoke_levir_10q_pred.jsonl \
  --references third_party/cdchat/data_files/levir_captions.json \
  --out results/tables/smoke_levir_10q_metrics.csv \
  --per-sample-out results/tables/smoke_levir_10q_per_sample.csv
```

The script uses `pycocoevalcap` for BLEU, METEOR, ROUGE-L, and CIDEr when available. Count metrics use `attribute.num_regions` from the reference file and a conservative number extractor from the generated answer.

## Crop Metadata

```bash
python experiments/build_change_crops.py \
  --image-root /path/to/LEVIR_or_SYSU_root \
  --metadata results/raw/smoke_levir_10q.json \
  --out-dir results/crops/levir_smoke \
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
- `results/raw/smoke_levir_10q.json` has been generated from the official LEVIR question file.
- `data/cdchat/levir` currently lacks LEVIR image files, so validation correctly reports missing `A/B/label` files.
- A first `hf download mubashir04/cdchat --local-dir checkpoints/cdchat` attempt retrieved small config files but did not finish the large model shards before it was interrupted. Re-run the command when a stable download window is available.

## Immediate Risks

- Dataset download is the likely blocker, because the CDChat repository contains JSON files but not raw LEVIR/SYSU images.
- Official dependency pins are older; keep inference and training environments separate.
- Full training is not recommended under the current project constraints. If needed later, use LoRA/QLoRA with reduced batch size and a hard 24-hour cap.
