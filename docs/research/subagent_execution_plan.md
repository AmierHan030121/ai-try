# Subagent Execution Plan

This project should be executed with independent agents and review gates.

## Agent Roles

| Agent | Responsibility | Outputs |
| --- | --- | --- |
| Literature agent | Verify references, DOI, venue, code links, dataset links | Updated `docs/research/literature_matrix.md`, `paper/references.bib` |
| Reproduction agent | Set up CDChat, download weights/data, run official inference | `results/raw/*.jsonl`, setup notes |
| Metrics agent | Implement metric computation after schema inspection | `experiments/run_metrics.py`, `results/tables/*.csv` |
| Method agent | Implement crop generation and crop-aware prompting | `experiments/build_change_crops.py`, crop metadata |
| Figure agent | Generate Python publication figures from real CSV outputs | `results/figures/*`, figure captions |
| Paper agent | Fill manuscript with verified methods and results | `paper/manuscript.md`, exported DOCX/PDF |
| Review agent | Check reproducibility, citations, claims, and manuscript quality | Review report and fix list |

## Review Gates

1. Reproduction gate: official CDChat inference must run on at least 10 samples before full evaluation.
2. Metrics gate: metric script must pass a tiny hand-checkable fixture before evaluating real outputs.
3. Method gate: crop metadata must be visually spot-checked before model inference.
4. Paper gate: no quantitative claim may enter the manuscript unless backed by `results/tables/*.csv`.
5. Citation gate: every bibliography entry must be traceable to DOI, arXiv, venue page, or official repository.

## Execution Order

1. Literature and reproduction agents start first.
2. Metrics agent starts after the first JSONL output schema is available.
3. Method agent starts after dataset image naming is confirmed.
4. Figure and paper agents start after the first complete metric table exists.
5. Review agent runs after each major stage and before PDF/DOCX export.
