# Figure Prompts

## Architecture Figure Candidate Prompt

Generate a clean scientific architecture diagram for a remote sensing vision-language model paper. The diagram should show a dual-temporal image pair, image A before change and image B after change, entering a change crop selector. The selector produces top-k paired change crops. The original image pair and selected crops are passed into a CDChat-style multimodal large language model with a vision encoder, projector, and language model. The output is a natural-language change description and optional count estimate. Use a white background, thin black lines, muted blue and green accents, clear panel labels, no decorative gradients, and Nature/Remote Sensing journal style.

## Planned Result Figures

- Figure 1: Method overview.
- Figure 2: Quantitative comparison across crop modes.
- Figure 3: Ablation for top-k crops and LoRA rank.
- Figure 4: Qualitative examples with input image pair, selected crops, generated description, and reference caption.
