# D-CoDe
[EMNLP 2025🔥] D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition


**Code will be released soon.**

This is the official implementation for [D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition](https://arxiv.org/abs/2510.08818)

by [Yiyang Huang](https://hukcc.github.io/), [Yizhou Wang](https://wyzjack.github.io/), [Yun Fu](https://www1.ece.neu.edu/~yunfu/).


<p align="center">
    <img src="pipeline.png">
</p>

D-CoDe is a training-free framework for adapting image-pretrained vision-language models (VLMs) to video understanding. It achieves strong performance across multiple benchmarks, especially on long-video tasks, demonstrating its potential for complex video-language understanding.

## Table of contents
- [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
- [Configuration](#configuration)
- [Inference and Evaluation](#inference-and-evaluation)
    - [Output Structures](#output-structures)
- [Acknowledgement](#Acknowledgement)
- [Citations](#citations)


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

1. We prepare the ground-truth question and answer files based on [`IG-VLM`](https://github.com/imagegridworth/IG-VLM/tree/main), and put them under [playground/gt_qa_files](playground/gt_qa_files).

    - MSVD-QA
        - Download the `MSVD_QA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/MSVD_QA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_msvd_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - MSRVTT-QA
        - Download the `MSRVTT_QA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/MSRVTT_QA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_msrvtt_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - TGIF-QA
        - Download the `TGIF_FrameQA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/TGIF_FrameQA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_tgif_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - Activitynet-QA
        - Download the `Activitynet_QA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/open_ended_qa/ActivityNet_QA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_activitynet_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - NExT-QA
        - Download the `NExT_QA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/multiple_choice_qa/NExT_QA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_nextqa_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - EgoSchema
        - Download the `EgoSchema.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/multiple_choice_qa/EgoSchema.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_egoschema_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```
    - IntentQA
        - Download the `IntentQA.csv` from the [`here`](https://github.com/imagegridworth/IG-VLM/blob/main/data/multiple_choice_qa/IntentQA.csv)
        - Reformat the files by running
            ```
            python scripts/data/prepare_intentqa_qa_file.py --qa_file $PATH_TO_CSV_FILE
            ```

    

2. Download the raw videos from the official websites.

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


3. Organize the raw videos under [playground/data](playground/data).

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

    @article{huang2025dcode,
        title={D-CoDe: Scaling Image-Pretrained VLMs to Video via Dynamic Compression and Question Decomposition}, 
        author={Yiyang Huang and Yizhou Wang and Yun Fu},
        year={2025},
        journal={arXiv preprint},
    }
