# D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition

**A training-free framework that adapts image-pretrained VLMs to video understanding — achieving SOTA on 7 benchmarks through dynamic compression and question decomposition, with no fine-tuning required.**

[![EMNLP 2025](https://img.shields.io/badge/EMNLP-2025-blue.svg)](https://aclanthology.org/2025.emnlp-main.597/)
[![arXiv](https://img.shields.io/badge/arXiv-2510.08818-b31b1b.svg)](https://arxiv.org/abs/2510.08818)
[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](https://arxiv.org/pdf/2510.08818)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://hukcc.github.io/D-CoDe/)
[![Code](https://img.shields.io/badge/GitHub-Code-black.svg?logo=github)](https://github.com/hukcc/D-CoDe)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

<p align="center">
    <img src="pipeline.png" width="90%">
</p>

## Key Results

D-CoDe achieves state-of-the-art performance across **7 video understanding benchmarks** — all **without any training**.

<table>
<tr>
<td>

**Multiple-Choice VideoQA** (↑ higher is better)

| Method | LLM | NExT-QA | EgoSchema | IntentQA |
|:---|:---:|:---:|:---:|:---:|
| SF-LLaVA | 7B | 64.2 | 47.2 | 60.1 |
| TS-LLaVA | 7B | 66.5 | 50.2 | 61.7 |
| **D-CoDe** | **7B** | **68.3** | **58.0** | **64.2** |

</td>
<td>

**Open-Ended VideoQA — Accuracy** (↑ higher is better)

| Method | MSVD | MSRVTT | TGIF | ANet |
|:---|:---:|:---:|:---:|:---:|
| SF-LLaVA | 79.1 | **65.8** | 78.7 | 55.5 |
| TS-LLaVA | 79.0 | 65.1 | 77.7 | 56.7 |
| **D-CoDe** | **80.0** | 64.2 | **79.1** | 56.4 |

</td>
</tr>
</table>

> **Highlight:** On the challenging long-video benchmark **EgoSchema** (5-min videos), D-CoDe achieves **58.0%** accuracy — a **+7.8%** improvement over the previous best training-free method (TS-LLaVA 50.2%).

## Quick Start

```python
from Dcode import generate_subquestions, supp_frame_selection, token_select_and_merge, load_clip_model

# 1. Question Decomposition (requires OPENAI_API_KEY environment variable)
subquestions = generate_subquestions(
    question="What did the person do after picking up the cup?",
    prompt_variant="original"
)

# 2. Frame Selection (based on semantic diversity)
clip_processor, clip_model = load_clip_model()
selected_frames, frame_idxs = supp_frame_selection(
    video_frames,           # List of PIL Images
    N=15,                   # Number of frames to select
    uniform_ratio=0.85,     # Ratio for uniform sampling
    clip_model=clip_model,
    clip_processor=clip_processor
)

# 3. Token Selection and Merge
merged_features = token_select_and_merge(
    image_features,                  # Tensor (T, N, D)
    top_k=288,                       # Tokens to keep per frame
    merge_strategy="mean",           # Options: "mean", "max", "weighted_mean"
    similarity_threshold=0.8         # Similarity threshold for merging
)
```

### Run Full Evaluation

```bash
# Open-ended VideoQA (e.g., EgoSchema)
bash scripts/run_eval_egoschema.sh

# Other benchmarks
bash scripts/run_eval_nextqa.sh
bash scripts/run_eval_intentqa.sh
bash scripts/run_eval_msvd.sh
bash scripts/run_eval_msrvtt.sh
bash scripts/run_eval_tgif.sh
bash scripts/run_eval_activitynet.sh
```

## Installation

```bash
conda create -n d_code python=3.10.12
conda activate d_code
bash setup_env.sh
```

Set up your OpenAI API key for question decomposition:

```bash
export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
```

Download pre-trained LLaVA-NeXT weights:

```bash
git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.6-vicuna-7b
```

## Data Preparation

<details>
<summary><b>Click to expand full data setup instructions</b></summary>

### Ground-Truth QA Files

GT question and answer CSV files are already included in [playground/gt_qa_files](playground/gt_qa_files):
MSVD-QA, MSRVTT-QA, TGIF-QA, ActivityNet-QA, NExT-QA, EgoSchema, IntentQA.

### Download Raw Videos

- **Open-ended VideoQA**
    - [Recommended] Follow the instruction in [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) to download raw videos.
    - Or download directly: [MSVD-QA](https://github.com/xudejing/video-question-answering?tab=readme-ov-file) · [MSRVTT-QA](https://github.com/xudejing/video-question-answering?tab=readme-ov-file) · [TGIF-QA](https://github.com/YunseokJANG/tgif-qa?tab=readme-ov-file) · [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa)

- **Multiple-Choice VideoQA**
    - [NExT-QA](https://github.com/doc-doc/NExT-QA) · [EgoSchema](https://egoschema.github.io) · [IntentQA](https://github.com/JoseponLee/IntentQA)

### Expected Directory Structure

```
playground/data/
├── video_qa/
│   ├── MSVD_Zero_Shot_QA/videos/
│   ├── MSRVTT_Zero_Shot_QA/videos/all/
│   ├── TGIF_Zero_Shot_QA/mp4/
│   └── Activitynet_Zero_Shot_QA/all_test/
└── multiple_choice_qa/
    ├── NExTQA/video/
    ├── EgoSchema/video/
    └── IntentQA/video/
```

</details>

## Detailed Results

<details>
<summary><b>Module Ablation (EgoSchema)</b></summary>

| Module | Acc. (%) |
|:---|:---:|
| Baseline | 44.8 |
| + Dynamic Spatial Token Compression | 50.6 |
| + Dynamic Temporal Frame Selection | 51.8 |
| + Question Decomposition | **58.0** |

</details>

<details>
<summary><b>Full Module Ablation (All Benchmarks)</b></summary>

| Module | NExT-QA | IntentQA | MSVD | MSRVTT | TGIF | ANet |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Baseline | 65.4 | 61.3 | 77.8/4.0 | 62.8/3.5 | 76.9/4.0 | 54.2/3.3 |
| + Spatial Compression | 66.7 | 62.2 | 79.4/4.0 | 63.6/3.5 | 78.9/4.1 | 55.4/3.3 |
| + Temporal Selection | 67.0 | 62.9 | **80.0**/4.1 | **64.2**/3.5 | **79.1**/4.1 | **56.4**/3.4 |
| + Question Decomposition | **68.3** | **64.2** | 72.4/3.8 | 62.2/3.5 | 75.7/4.0 | 53.8/3.3 |

</details>

<details>
<summary><b>Efficiency Analysis (EgoSchema)</b></summary>

| Module | Acc. (%) | s/sample |
|:---|:---:|:---:|
| Baseline | 44.8 | 3.927 |
| + Dynamic Compression | 51.8 | 6.115 |
| + Question Decomposition | 58.0 | 37.395 |

</details>

## Core Components

The core implementation is in `Dcode.py`:

| Function | Description | Paper Method |
|:---|:---|:---|
| `generate_subquestions()` | Decompose questions into sub-questions using GPT-3.5 | Question Decomposition |
| `supp_frame_selection()` | Select frames based on CLIP semantic similarity | Dynamic Compression (Frame) |
| `token_select_and_merge()` | Select and merge visual tokens to reduce redundancy | Dynamic Compression (Token) |

## Acknowledgement

We extend our gratitude to the following projects: [LLaVA](https://github.com/haotian-liu/LLaVA), [IG-VLM](https://github.com/imagegridworth/IG-VLM), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [SF-LLaVA](https://github.com/apple/ml-slowfast-llava) and [TS-LLaVA](https://github.com/tingyu215/TS-LLaVA).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{huang-etal-2025-code,
    title = "{D}-{C}o{D}e: Scaling Image-Pretrained {VLM}s to Video via Dynamic Compression and Question Decomposition",
    author = "Huang, Yiyang  and
      Wang, Yizhou  and
      Fu, Yun",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    year = "2025",
    pages = "11798--11811",
}
```

arXiv version:

```bibtex
@article{huang2025d,
    title={D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition},
    author={Huang, Yiyang and Wang, Yizhou and Fu, Yun},
    journal={arXiv preprint arXiv:2510.08818},
    year={2025}
}
```

## License

This project is released under the [Apache 2.0 License](LICENSE).
