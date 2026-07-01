# CDChat Reproduction and Crop Prompting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce CDChat inference on public change-description benchmarks, then evaluate lightweight change-guided crop prompting without long training runs.

**Architecture:** Keep official CDChat code under ignored `third_party/cdchat`, store large weights/data under ignored `checkpoints/` and `data/`, and keep this repository focused on reproducibility wrappers, metric scripts, crop metadata, result tables, figures, and manuscript files.

**Tech Stack:** Python 3.10, PyTorch/CDChat official environment, Hugging Face Hub, pycocoevalcap, Pillow, NumPy, matplotlib for later figures.

---

### Task 1: Acquire CDChat Resources

**Files:**
- Read: `docs/research/cdchat_reproduction.md`
- Create locally, ignored by git: `third_party/cdchat/`, `checkpoints/cdchat/`, `data/cdchat/`

- [ ] **Step 1: Configure proxy**

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
```

- [ ] **Step 2: Clone or refresh CDChat**

```bash
mkdir -p third_party
if [ -d third_party/cdchat/.git ]; then
  git -C third_party/cdchat pull --ff-only
else
  git clone https://github.com/techmn/cdchat.git third_party/cdchat
fi
```

- [ ] **Step 3: Download CDChat weights**

```bash
mkdir -p checkpoints/cdchat
hf download mubashir04/cdchat \
  --local-dir checkpoints/cdchat
```

- [ ] **Step 4: Prepare image data layout**

Place LEVIR/SYSU images under one root per dataset:

```text
data/cdchat/levir/
  A/<img_id>.png
  B/<img_id>.png
  label/<img_id>.png

data/cdchat/sysu/
  A/<img_id>.png
  B/<img_id>.png
  label/<img_id>.png
```

### Task 2: Run Reproduction Smoke Test

**Files:**
- Use: `experiments/make_smoke_questions.py`
- Use: `experiments/validate_cdchat_data.py`
- Write ignored outputs: `results/raw/`, `results/logs/`

- [ ] **Step 1: Create 10-question subset**

```bash
python experiments/make_smoke_questions.py \
  --questions third_party/cdchat/data_files/eval_questions_levir_test.json \
  --out results/raw/smoke_levir_10q.json \
  --limit 10
```

- [ ] **Step 2: Validate image/mask files**

```bash
python experiments/validate_cdchat_data.py \
  --image-root data/cdchat/levir \
  --questions results/raw/smoke_levir_10q.json \
  --out-missing results/logs/smoke_levir_missing.csv
```

- [ ] **Step 3: Run CDChat smoke inference**

```bash
cd third_party/cdchat
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from argparse import Namespace
from cdchat.eval.batch_cdchat_vqa import eval_model

args = Namespace(
    model_path="../../checkpoints/cdchat/model_weights_cdchat",
    model_base=None,
    mm_projector_path=None,
    image_folder="../../data/cdchat/levir",
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
cd ../..
```

### Task 3: Validate Metrics and Crop Metadata

**Files:**
- Use: `experiments/run_metrics.py`
- Use: `experiments/build_change_crops.py`
- Write ignored outputs: `results/tables/`, `results/crops/`

- [ ] **Step 1: Compute smoke metrics for schema validation**

```bash
python experiments/run_metrics.py \
  --pred results/raw/smoke_levir_10q_pred.jsonl \
  --references third_party/cdchat/data_files/levir_captions.json \
  --out results/tables/smoke_levir_10q_metrics.csv \
  --per-sample-out results/tables/smoke_levir_10q_per_sample.csv
```

- [ ] **Step 2: Generate crop metadata for smoke images**

```bash
python experiments/build_change_crops.py \
  --image-root data/cdchat/levir \
  --metadata results/raw/smoke_levir_10q.json \
  --out-dir results/crops/levir_smoke_auto_diff \
  --mode auto_diff \
  --top-k 3 \
  --crop-size 128 \
  --limit 5
```

- [ ] **Step 3: Generate oracle-mask upper-bound crops**

```bash
python experiments/build_change_crops.py \
  --image-root data/cdchat/levir \
  --metadata results/raw/smoke_levir_10q.json \
  --out-dir results/crops/levir_smoke_oracle \
  --mode oracle_mask \
  --top-k 3 \
  --crop-size 128 \
  --limit 5
```

### Task 4: Run Full Inference and Ablations

**Files:**
- Write ignored outputs: `results/raw/*.jsonl`, `results/tables/*.csv`, `results/crops/*`
- Update after real results exist: `paper/manuscript.md`

- [ ] **Step 1: Split official baseline inference across two GPUs**

```bash
cd third_party/cdchat
CUDA_VISIBLE_DEVICES=0 python - <<'PY' &
from argparse import Namespace
from cdchat.eval.batch_cdchat_vqa import eval_model

eval_model(Namespace(
    model_path="../../checkpoints/cdchat/model_weights_cdchat",
    model_base=None,
    mm_projector_path=None,
    image_folder="../../data/cdchat/levir",
    question_file="data_files/eval_questions_levir_test.json",
    answers_file="../../results/raw/cdchat_levir_none_chunk0.jsonl",
    conv_mode="llava_v1",
    num_chunks=2,
    chunk_idx=0,
    temperature=0.2,
    top_p=None,
    num_beams=1,
    batch_size=1,
))
PY

CUDA_VISIBLE_DEVICES=1 python - <<'PY' &
from argparse import Namespace
from cdchat.eval.batch_cdchat_vqa import eval_model

eval_model(Namespace(
    model_path="../../checkpoints/cdchat/model_weights_cdchat",
    model_base=None,
    mm_projector_path=None,
    image_folder="../../data/cdchat/levir",
    question_file="data_files/eval_questions_levir_test.json",
    answers_file="../../results/raw/cdchat_levir_none_chunk1.jsonl",
    conv_mode="llava_v1",
    num_chunks=2,
    chunk_idx=1,
    temperature=0.2,
    top_p=None,
    num_beams=1,
    batch_size=1,
))
PY
wait
cd ../..
```

- [ ] **Step 2: Merge chunked JSONL predictions**

```bash
cat results/raw/cdchat_levir_none_chunk0.jsonl \
    results/raw/cdchat_levir_none_chunk1.jsonl \
  > results/raw/cdchat_levir_none.jsonl
```

- [ ] **Step 3: Compute full baseline metrics**

```bash
python experiments/run_metrics.py \
  --pred results/raw/cdchat_levir_none.jsonl \
  --references third_party/cdchat/data_files/levir_captions.json \
  --out results/tables/cdchat_levir_none_metrics.csv \
  --per-sample-out results/tables/cdchat_levir_none_per_sample.csv
```

- [ ] **Step 4: Run crop conditions**

Generate `random`, `auto_diff`, and `oracle_mask` metadata with `build_change_crops.py`. Use `oracle_mask` only as an upper bound and do not describe it as deployable.

### Task 5: Manuscript, Figures, and Export

**Files:**
- Update: `paper/manuscript.md`
- Update: `paper/references.bib`
- Create after real CSVs exist: `results/figures/*.svg`, `results/figures/*.pdf`, `paper/*.docx`, `paper/*.pdf`

- [ ] **Step 1: Generate figures from real CSVs only**

Use Python/matplotlib. Figure 1 can be a method schematic; Figures 2-4 must use real `results/tables/*.csv` and qualitative outputs from `results/raw/*.jsonl`.

- [ ] **Step 2: Fill Results section**

Report only values backed by CSV files in `results/tables/`. Include baseline, random-crop control, auto-diff crop method, and oracle-mask upper bound.

- [ ] **Step 3: Export manuscript**

```bash
pandoc paper/manuscript.md \
  --bibliography paper/references.bib \
  -o paper/change_guided_cdchat.docx

pandoc paper/manuscript.md \
  --bibliography paper/references.bib \
  -o paper/change_guided_cdchat.pdf
```

- [ ] **Step 4: Final review**

Run citation, data availability, figure, and manuscript-polishing checks. Remove any claim that is not supported by a result file or verified source.
