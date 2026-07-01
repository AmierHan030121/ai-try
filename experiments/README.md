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

```bash
python third_party/cdchat/cdchat/eval/batch_cdchat_vqa.py \
  --model-path /path/to/cdchat/model \
  --question-file third_party/cdchat/data_files/eval_questions_levir_test.json \
  --answer-file results/raw/cdchat_levir_none.jsonl \
  --image-folder /path/to/levir/images
```

```bash
python experiments/run_metrics.py \
  --pred results/raw/cdchat_levir_none.jsonl \
  --references third_party/cdchat/data_files/levir_captions.json \
  --out results/tables/cdchat_levir_none_metrics.csv
```
