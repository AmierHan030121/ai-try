# Change-Guided Visual Prompting for Remote Sensing Image Change Description

## Abstract

Remote sensing image change description requires a model to compare dual-temporal imagery and generate a faithful natural-language summary of semantic changes. Recent multimodal large language models, including CDChat, adapt vision-language assistants to this task, but full-image prompting can allocate substantial visual context to unchanged regions. We propose to evaluate a lightweight change-guided prompting strategy that adds candidate change crops to the original image-pair prompt. The study first reproduces CDChat on public LEVIR-CD and SYSU-CD evaluation files, then compares full-image prompting with random crops, automatically selected change crops, and oracle mask crops used only as an upper bound. This first manuscript draft defines the experimental protocol and reporting structure; quantitative results will be inserted only after local reproduction runs are completed.

## Keywords

remote sensing; change captioning; vision-language model; multimodal large language model; prompt engineering; reproducibility

## 1. Introduction

Remote sensing change description converts paired pre-event and post-event images into natural-language descriptions of what has changed. The task is useful for urban monitoring, disaster assessment, and geographic information updating because it can express semantic changes more directly than pixel-level change maps.

Large multimodal models provide a flexible interface for remote sensing interpretation, but high-resolution dual-temporal scenes contain many unchanged regions. When a model processes the whole image pair, visual context may be spent on regions that are irrelevant to the target change. This study tests whether a lightweight change-guided visual prompting strategy can improve change description without long training runs.

## 2. Related Work

CDChat is the primary baseline because it provides a public implementation for remote sensing change description. GeoChat provides the remote sensing LLaVA-style precedent that CDChat builds upon. PromptCC is included as a non-conversational change captioning reference method. TEOChat, RS-LLaVA, and RemoteCLIP provide broader context for temporal earth observation VLMs, remote sensing instruction tuning, and remote sensing vision-language foundation models.

## 3. Method

The proposed method augments CDChat-style prompting with selected paired crops. Given a pre-change image and a post-change image, a crop selector identifies candidate changed regions. The model receives the original image pair and a prompt that focuses the answer on these candidate regions. Four conditions are planned: no crops, random crops, automatic difference crops, and oracle mask crops.

Oracle crops are not treated as deployable inputs because they can use ground-truth change masks. They are reported only to estimate the upper bound of region-focused prompting.

## 4. Experiments

### 4.1 Datasets

The planned evaluation uses the CDChat release files for LEVIR-CD and SYSU-CD. Dataset access, file versions, and preprocessing details will be reported after local setup is complete.

### 4.2 Baselines

The main baseline is CDChat under its official evaluation setting. PromptCC is used as a change-captioning reference baseline when the required model and data files are available.

### 4.3 Metrics

The planned metrics are BLEU-4, METEOR, ROUGE-L, CIDEr, count accuracy, and count MAE. Inference time and peak memory will be recorded where available.

### 4.4 Implementation Details

The first reproduction will use inference only. Fine-tuning will be attempted only after inference-only experiments establish a valid baseline. If fine-tuning is used, it will use LoRA or QLoRA with rank 8 and 16 under a single-run limit of 24 hours.

## 5. Results

Results are intentionally omitted from this first draft. Tables and figures will be added after local experiments produce reproducible outputs.

## 6. Discussion

The central expected analysis is whether improvements, if observed, come from change localization rather than from simply adding more visual crops. The random crop and oracle crop controls are therefore essential.

## 7. Limitations

This study does not attempt large-scale pretraining. Results may depend on dataset-specific image alignment quality, mask availability, and caption metric sensitivity. Human evaluation may be needed if automatic metrics disagree with factual correctness.

## Data Availability

This study uses public datasets and files linked by the CDChat and PromptCC repositories. No new dataset is released in this first version. Exact download paths and checksums will be added after local data acquisition.

## Code Availability

The first-version code and manuscript files are hosted in this repository. External baseline code remains under the licenses of the original repositories.

## Ethics Declaration

The planned experiments use public remote sensing benchmarks. No human-subject data are collected in this study.

## Conflict of Interest

The author declares no conflict of interest.

## Funding

No funding information has been provided for this first version.

## Author Contributions

Conceptualization, methodology, software, validation, writing, and visualization assignments will be updated before submission.

## AI Disclosure

AI assistance was used to plan the study structure, organize references, and draft non-final manuscript text. All citations, experiments, and quantitative claims must be verified by the author before submission.
