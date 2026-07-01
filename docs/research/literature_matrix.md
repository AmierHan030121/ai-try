# Literature Matrix

| Citation Key | Work | Venue/Year | Code | Data/Model Availability | Relevance |
| --- | --- | --- | --- | --- | --- |
| `noman2025cdchat` | CDChat: A Large Multimodal Model for Remote Sensing Change Description | IGARSS 2025 | https://github.com/techmn/cdchat | Model link and data files in repository | Main baseline |
| `deng2024changechat` | ChangeChat: An Interactive Model for Remote Sensing Change Analysis via Multimodal Instruction Tuning | arXiv 2024 / ICASSP 2025 candidate | https://github.com/hanlinwu/ChangeChat | ChangeChat-87k, code and pretrained weights linked by paper/repo | Strong interactive VLM follow-up candidate |
| `liu2024changeagent` | Change-Agent: Towards Interactive Comprehensive Remote Sensing Change Interpretation and Analysis | IEEE TGRS 2024 | https://github.com/Chen-Yang-Liu/Change-Agent | LEVIR-MCI code/data release path | Multi-task change interpretation baseline |
| `kuckreja2024geochat` | GeoChat: Grounded Large Vision-Language Model for Remote Sensing | CVPR 2024 | https://github.com/mbzuai-oryx/GeoChat | Model, dataset, evaluation scripts | Architecture and training precedent |
| `liu2023promptcc` | A Decoupling Paradigm With Prompt Learning for Remote Sensing Image Change Captioning | IEEE TGRS 2023 | https://github.com/Chen-Yang-Liu/PromptCC | LEVIR-CC preparation and HF checkpoint links | Traditional RSICC baseline |
| `karaca2025secondcc` | Robust Change Captioning in Remote Sensing: SECOND-CC Dataset and MModalCC Framework | IEEE JSTARS 2025 | https://github.com/ChangeCapsInRS/SecondCC | SECOND-CC dataset and MModalCC code path | Robustness dataset and non-LLM baseline |
| `irvin2025teochat` | TEOChat: A Large Vision-Language Assistant for Temporal Earth Observation Data | ICLR 2025 | https://github.com/ermongroup/TEOChat | HF model and TEOChatlas dataset | Related temporal VLM |
| `bazi2024rsllava` | RS-LLaVA: Large Vision Language Model for Joint Captioning and Question Answering in Remote Sensing Imagery | Remote Sensing 2024 | https://github.com/BigData-KSU/RS-LLaVA | HF model and RS-instructions dataset | Related remote sensing instruction tuning |
| `liu2024remoteclip` | RemoteCLIP: A Vision Language Foundation Model for Remote Sensing | IEEE TGRS 2024 | https://github.com/ChenDelong1999/RemoteCLIP | HF checkpoints and training dataset | Related vision-language foundation model |
| `yang2025kcfi` | Enhancing Perception of Key Changes in Remote Sensing Image Change Captioning | IEEE TIP 2025 | https://github.com/yangcong356/KCFI | LEVIR-CC-oriented code repository | Candidate modern change-captioning baseline |
| `rscc2025` | RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events | NeurIPS Datasets and Benchmarks 2025 | https://huggingface.co/datasets/BiliSakura/RSCC | RSCC and RSCCM dataset/model pages | New disaster-event data source to inspect after CDChat |

## Citation Integrity Notes

- All listed references have public project pages or repositories.
- DOI and venue metadata must be verified again before final submission.
- No result should be cited as reproduced until generated in this repository.
- Main reproduction remains CDChat because it exactly targets remote sensing change description and provides evaluation question/caption files.
- ChangeChat and Change-Agent are prioritized as next-stage comparison candidates after CDChat inference is working.
