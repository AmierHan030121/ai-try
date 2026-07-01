# Experiments

This directory will contain executable scripts for reproducing CDChat and evaluating the proposed change-guided prompting method.

## External Setup

Use the proxy before accessing GitHub or Hugging Face:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
```

## Planned Workflow

1. Clone CDChat into `third_party/cdchat`.
2. Install CDChat environment.
3. Download model weights and dataset images.
4. Run official inference on LEVIR-CD and SYSU-CD test files.
5. Compute metrics from saved JSONL outputs.
6. Generate change-aware crops.
7. Re-run inference under crop conditions.
8. Export tables to `results/`.

## Commands

The commands below are templates. Replace paths after data and model files are available.

Create a small question subset:

```bash
python experiments/make_smoke_questions.py \
  --questions third_party/cdchat/data_files/eval_questions_levir_test.json \
  --out results/raw/smoke_levir_10q.json \
  --limit 10
```

Validate the image folder before inference:

```bash
python experiments/validate_cdchat_data.py \
  --image-root /path/to/levir_or_sysu_root \
  --questions results/raw/smoke_levir_10q.json \
  --out-missing results/logs/smoke_levir_missing.csv
```

Run official CDChat inference. The official README shows `--answer-file`, but the
current evaluator uses `--answers-file`.

```bash
python third_party/cdchat/cdchat/eval/batch_cdchat_vqa.py \
  --model-path /path/to/cdchat/model \
  --question-file third_party/cdchat/data_files/eval_questions_levir_test.json \
  --answers-file results/raw/cdchat_levir_none.jsonl \
  --image-folder /path/to/levir/images
```

For the HF `model_weights_cdchat` directory, prefer the wrapper in
`docs/research/cdchat_reproduction.md` because the merged weights should be
loaded with `model_base=None`.

```bash
python experiments/run_metrics.py \
  --pred results/raw/cdchat_levir_none.jsonl \
  --references third_party/cdchat/data_files/levir_captions.json \
  --out results/tables/cdchat_levir_none_metrics.csv \
  --per-sample-out results/tables/cdchat_levir_none_per_sample.csv
```

Generate change-guided crop metadata:

```bash
python experiments/build_change_crops.py \
  --image-root /path/to/levir_or_sysu_root \
  --metadata third_party/cdchat/data_files/eval_questions_levir_test.json \
  --out-dir results/crops/levir_auto_diff \
  --mode auto_diff \
  --top-k 3 \
  --crop-size 128
```

Do not report metrics from smoke-test outputs in the paper. Smoke tests only
verify environment, data paths, and schema compatibility.
