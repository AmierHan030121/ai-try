# AI Try: Remote Sensing Change Description with VLMs

This repository hosts the first-version research package for a lightweight, reproducible remote sensing + vision-language model study.

## Topic

**Change-guided multimodal prompting for remote sensing image change description**

The planned study reproduces CDChat, then tests whether change-aware crop selection can improve dual-temporal remote sensing change descriptions without long training runs.

## Research Constraints

- Use papers with public code and public data/model paths.
- Reproduce an existing method before proposing modifications.
- Avoid long single-run training jobs. Target: under 10 hours where possible; hard limit: 24 hours.
- Use two available GPUs efficiently, but do not require large-scale pretraining.
- Do not report generated results until scripts have produced them from local runs.

## Primary Baseline

- CDChat: <https://github.com/techmn/cdchat>
- Paper: <https://arxiv.org/abs/2409.16261>
- Model weights: linked from the CDChat repository

## Planned Outputs

- Reproducible experiment scripts under `experiments/`
- Metric tables under `results/`
- Paper manuscript under `paper/`
- Research notes and literature matrix under `docs/research/`

## Proxy

When accessing GitHub, Hugging Face, or other external resources from this environment, use:

```bash
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
```
