# Research Design

## Working Title

Change-Guided Visual Prompting for Remote Sensing Image Change Description

## Motivation

Remote sensing change description requires a model to compare two images and produce a concise description of semantic changes. Large multimodal models such as CDChat adapt LLaVA-style architectures to this task, but high-resolution dual-temporal images contain many unchanged regions. A lightweight mechanism that directs the model toward candidate change regions may improve generation quality without requiring large-scale retraining.

## Research Question

Can change-aware crop selection improve CDChat-style remote sensing change descriptions under a low-compute reproduction setting?

## Main Hypothesis

Compared with full-image prompting alone, adding automatically selected change crops to the prompt context improves change description metrics and count consistency because the VLM spends more visual context on changed regions.

## Baseline and Supporting Methods

| Role | Work | Public Link | Why It Is Used |
| --- | --- | --- | --- |
| Main baseline | CDChat | https://github.com/techmn/cdchat | LLaVA/GeoChat-style remote sensing change description with public code, data files, and model link |
| Architecture precedent | GeoChat | https://github.com/mbzuai-oryx/GeoChat | Grounded remote sensing VLM used as CDChat's codebase inspiration |
| Traditional RSICC baseline | PromptCC | https://github.com/Chen-Yang-Liu/PromptCC | Strong TGRS change captioning method with training and evaluation code |
| Related temporal VLM | TEOChat | https://github.com/ermongroup/TEOChat | Recent temporal earth observation VLM for related work |
| Related RS instruction model | RS-LLaVA | https://github.com/BigData-KSU/RS-LLaVA | Small remote sensing instruction-tuning baseline for related work |
| Related RS vision-language foundation model | RemoteCLIP | https://github.com/ChenDelong1999/RemoteCLIP | Retrieval/classification foundation model context |

## Proposed Method

The proposed method is **Change-Guided Multi-Crop Prompting**:

1. Given a pre-change image `A` and post-change image `B`, generate a difference map.
2. Select top-k candidate change boxes from the difference map.
3. Crop the paired regions from `A` and `B`.
4. Prompt the VLM with the original image pair plus textual references to the selected change regions.
5. Compare output quality against full-image-only prompting and control crops.

## Experimental Conditions

| Condition | Description | Purpose |
| --- | --- | --- |
| `none` | Original CDChat prompt and image pair | Main reproduction baseline |
| `random` | Same number of random paired crops | Tests whether gains come from extra crops alone |
| `auto_diff` | Crops from automatic image differencing | Main proposed method |
| `oracle_mask` | Crops from ground-truth change mask when available | Upper bound only, not the main method |

## Metrics

- BLEU-4
- METEOR
- ROUGE-L
- CIDEr
- Change count accuracy
- Change count MAE
- Inference latency per sample
- Peak GPU memory, if available

## Compute Plan

1. Run inference-only reproduction first.
2. Implement crop-based prompting without training.
3. Fine-tune only if inference-only results justify it.
4. If fine-tuning is used, apply LoRA or QLoRA only, with rank 8 and 16 as the first ablation.

## Known Risks

- CDChat data download paths may require manual dataset acquisition.
- Using ground-truth masks for prompt construction can leak label information; report this only as an upper bound.
- Caption metrics may not fully capture factual change correctness, so include count-based checks.
- Official scores may not match exactly if tokenization, dataset versions, or decoding settings differ.
