# D-CoDe
[EMNLP 2025🔥] D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition

This is the official implementation for [D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition](https://aclanthology.org/2025.emnlp-main.597/)

by [Yiyang Huang](https://hukcc.github.io/), [Yizhou Wang](https://wyzjack.github.io/), [Yun Fu](https://www1.ece.neu.edu/~yunfu/).


<p align="center">
    <img src="pipeline.png">
</p>

D-CoDe is a training-free framework for adapting image-pretrained vision-language models (VLMs) to video understanding. It achieves strong performance across multiple benchmarks, especially on long-video tasks, demonstrating its potential for complex video-language understanding.

## Table of contents
- [Core Components](#core-components)
    - [Quick Start](#quick-start)
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
- [Inference and Evaluation](#inference-and-evaluation)
    - [Output Structures](#output-structures)
- [Acknowledgement](#Acknowledgement)
- [Citations](#citations)

## Core Components

The core implementation is in `Dcode.py`, which provides three main functions:

| Function | Description | Paper Method |
|----------|-------------|--------------|
| `generate_subquestions()` | Decompose questions into sub-questions using GPT-3.5 | Question Decomposition |
| `supp_frame_selection()` | Select frames based on CLIP semantic similarity | Dynamic Compression (Frame) |
| `token_select_and_merge()` | Select and merge visual tokens to reduce redundancy | Dynamic Compression (Token) |

### Quick Start

```python
from Dcode import generate_subquestions, supp_frame_selection, token_select_and_merge, load_clip_model

# 1. Question Decomposition (requires OPENAI_API_KEY environment variable)
subquestions = generate_subquestions(
    question="What did the person do after picking up the cup?",
    prompt_variant="original"  # Options: "original", "no_background", "no_temporal_focus", "re"
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


## Getting Started

### Installation

- The code is developed with CUDA 11.7, ***Python >= 3.10.12***, ***PyTorch >= 2.1.0***

    0. [Optional but recommended] Create a new conda environment.
        ```
        conda create -n d_code python=3.10.12
        ```
        And activate the environment.
        ```
        conda activate d_code
        ```

    1. Install the requirements.
        ```
        bash setup_env.sh
        ```

    2. Add OpenAI key and organization to the system environment to use GPT-3.5-turbo for model evaluation.
        ```
        export OPENAI_API_KEY=$YOUR_OPENAI_API_KEY
        export OPENAI_ORG=$YOUR_OPENAI_ORG  # optional
        ```

    3. Download pre-trained LLaVA-NeXT weights from [`HuggingFace`](https://huggingface.co/collections/liuhaotian/llava-16-65b9e40155f60fd046a5ccf2), and put them under the [`Dcode`](./) folder.
        ```
        git lfs clone https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b liuhaotian/llava-v1.6-vicuna-7b
        ```

### Data Preparation

1. **Ground-truth QA Files**: The ground-truth question and answer CSV files are already included in [playground/gt_qa_files](playground/gt_qa_files). These files are prepared based on [`IG-VLM`](https://github.com/imagegridworth/IG-VLM/tree/main).

    Available datasets:
    - MSVD-QA (`MSVDQA.csv`)
    - MSRVTT-QA (`MSRVTTQA.csv`)
    - TGIF-QA (`TGIFFrameQA.csv`)
    - ActivityNet-QA (`ActivityNetQA.csv`)
    - NExT-QA (`Next_QA.csv`)
    - EgoSchema (`EgoSchema.csv`)
    - IntentQA (`IntentQA.csv`)

2. **Download Raw Videos**: Download the raw videos from the official websites.

    - Open-end VideoQA

        - [Recomanded] Option 1: Follow the instruction in [`Video-LLaVA`](https://github.com/PKU-YuanGroup/Video-LLaVA/blob/main/TRAIN_AND_VALIDATE.md) to download raw videos.
        - Option 2: Download videos from the data owners.
            - [`MSVD-QA`](https://github.com/xudejing/video-question-answering?tab=readme-ov-file)
            - [`MSRVTT-QA`](https://github.com/xudejing/video-question-answering?tab=readme-ov-file)
            - [`TGIF-QA`](https://github.com/YunseokJANG/tgif-qa?tab=readme-ov-file)
            - [`ActivityNet-QA`](https://github.com/MILVLG/activitynet-qa)

    - Multiple Choice VideoQA

        - Download datasets from the data owners.
            - [`NExT-QA`](https://github.com/doc-doc/NExT-QA)
            - [`EgoSchema`](https://egoschema.github.io)
            - [`IntentQA`](https://github.com/JoseponLee/IntentQA)


3. **Organize Videos**: Organize the raw videos under [playground/data](playground/data).

    - To directly use our data loaders without changing paths, please organize your datasets as follows

        ```
        $ Dcode/playground/data
            ├── video_qa
                ├── MSVD_Zero_Shot_QA
                    ├── videos
                        ├── ...
                ├── MSRVTT_Zero_Shot_QA
                    ├── videos
                        ├── all
                            ├── ...
                ├── TGIF_Zero_Shot_QA
                   ├── mp4
                       ├── ...
                ├── Activitynet_Zero_Shot_QA
                   ├── all_test
                       ├── ...
            ├── multiple_choice_qa
                ├── NExTQA
                    ├── video
                       ├── ...
                ├── EgoSchema
                    ├── video
                       ├── ...
                ├── IntentQA
                    ├── video
                       ├── ...
        ```

## Inference and Evaluation

D-CoDe is a training-free method, so we can directly do the inference and evaluation without model training.

By default, we use 4 GPUs for the model inference. We can modify the `CUDA_VISIBLE_DEVICES` in the config file to accommodate your own settings.

```
cd Dcode
python run_inference.py --exp_config $PATH_TO_CONFIG_FILE
```

- This is optional, but use `export PYTHONWARNINGS="ignore"` if you want to suppress the warnings.

### Output Structures

- The inference outputs will be stored under [`outputs/artifacts`](outputs/artifacts). <br>
- The intermediate outputs of GPT-3.5-turbo will be stored under [`outputs/eval_save_dir`](outputs/eval_save_dir). <br>
- The evaluation results will be stored under [`outputs/logs`](outputs/logs). <br>
- All of these can be changed in the config file.

## Acknowledgement
We extend our gratitude to the following awesome projects: [LLaVA](https://github.com/haotian-liu/LLaVA), [IG-VLM](https://github.com/imagegridworth/IG-VLM), [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA), [SF-LLaVA](https://github.com/apple/ml-slowfast-llava) and [TS-LLaVA](https://github.com/tingyu215/TS-LLaVA).



## Citations

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
