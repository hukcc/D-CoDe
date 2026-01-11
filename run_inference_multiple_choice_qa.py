#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import argparse
import os
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, os.path.join(Path(__file__).parent.as_posix(), "slowfast_llava"))

import json
import warnings
warnings.filterwarnings('ignore')

import torch
from tqdm import tqdm

from slowfast_llava.llava.constants import IMAGE_TOKEN_INDEX
from slowfast_llava.llava.model.builder import load_pretrained_model
from slowfast_llava.llava.utils import disable_torch_init
from slowfast_llava.llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from dataset import load_video
from prompt import get_multiple_choice_prompt
from utils import get_chunk
import Dcode

def llava_inference(
    video_frames,
    question,
    candidates,
    conv_mode,
    model,
    tokenizer,
    image_processor,
    image_sizes,
    temperature,
    top_p,
    num_beams,
    max_new_tokens=64,
    cot_caption=None,
    caption=None,
):
    # Get multiple choice prompt
    prompt = get_multiple_choice_prompt(model, conv_mode, question, candidates, cot_caption, caption)

    # Get text inputs
    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    ).unsqueeze(0).cuda()

    image_tensor = process_images(video_frames, image_processor, model.config)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens, 
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return outputs


def run_inference(args):
    """
    Run inference on Video QA Dataset.

    Args:
        args: Command-line arguments.
    """

    disable_torch_init()

    # Load tokenizer, model and image processor
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        device=torch.cuda.current_device(),
        device_map="cuda",
        rope_scaling_factor=args.rope_scaling_factor,
    )

    # Override image aspect ratio if needed
    if args.image_aspect_ratio:
        model.config.image_aspect_ratio = args.image_aspect_ratio

    # Set D-CoDe token selection parameters
    model.config.dcode_top_k = args.dcode_top_k
    model.config.dcode_merge_strategy = args.dcode_merge_strategy
    model.config.dcode_similarity_threshold = args.dcode_similarity_threshold

    # Load questions and answers
    gt_qa_pairs = json.load(open(args.gt_file, "r"))
    gt_qa_pairs = get_chunk(gt_qa_pairs, args.num_chunks, args.chunk_idx)

    os.makedirs(args.output_dir, exist_ok=True)
    ans_file = open(
        os.path.join(args.output_dir, f"{args.output_name}.json"), "w")
    
    # Pre-load CLIP model for frame selection (avoid reloading per sample)
    clip_processor, clip_model = Dcode.load_clip_model()

    # Iterate over each sample in the ground truth file
    for index, sample in enumerate(tqdm(gt_qa_pairs)):

        task_name = sample["task_name"]
        video_name = sample["video_name"]
        question_id = sample["question_id"]
        question = sample["question"]
        answer_number = sample["answer_number"]
        candidates = sample["candidates"]
        answer = sample["answer"]

        
        sample_set = {
            "task_name": task_name,
            "question": question,
            "id": question_id,
            "answer_number": answer_number,
            "candidates": candidates,
            "answer": answer,
        }

        # Load video
        video_path = os.path.join(args.video_dir, video_name)
                
        if os.path.exists(video_path): 
            # 20250305: NextQA=36, IntentQA=50, EgoSchema=40?
            video_frames, sizes = load_video(video_path, num_frms=args.num_frames)
                
            # Supp Frame selection based on semantic similarity (using CLIP visual encoder)
            video_frames, frame_idxs = Dcode.supp_frame_selection(
                video_frames, 
                N=args.select_frames,
                uniform_ratio=args.uniform_ratio,
                clip_model=clip_model,
                clip_processor=clip_processor
            )

            # CoT (Chain-of-Thought) question decomposition
            CoT_captions = []
            if args.enable_cot:
                CoT_questions = Dcode.extract_list_from_text(Dcode.generate_subquestions(question=question, api_key=args.openai_api_key))
                
                for i in range(len(CoT_questions)):
                    curr_question = CoT_questions[i]

                    # Run inference on the video (CoT caption generation)
                    output = llava_inference(
                        video_frames,
                        curr_question,
                        candidates,
                        args.conv_mode,
                        model,
                        tokenizer,
                        image_processor,
                        sizes,
                        args.temperature,
                        args.top_p,
                        args.num_beams,
                        args.cot_max_new_tokens,
                        cot_caption=True
                    )
                    CoT_captions.append(output.replace("image", "video").replace("Image", "Video"))

            # Run inference on the video (final answer generation)
            output = llava_inference(
                video_frames,
                question,
                candidates,
                args.conv_mode,
                model,
                tokenizer,
                image_processor,
                sizes,
                args.temperature,
                args.top_p,
                args.num_beams,
                args.answer_max_new_tokens,
                caption=" ".join(CoT_captions) if CoT_captions else None
            )
            output = output.replace("In the image", "In the video")
            # print(output)
            sample_set["pred"] = output
            
            ans_file.write(json.dumps(sample_set) + "\n")

    ans_file.close()



def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", help="Directory containing video files.", default='playground/data/multiple_choice_qa/IntentQA/video')
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question and answer.", default='playground/gt_qa_files/IntentQA/val_qa.json')
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", default='outputs/artifacts/IntentQA/slowfast_llava_7b-resize-slow_10frms_spatial_1d_max_pool_fast_4x4-50_frms')
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", default='test')
    parser.add_argument("--model_path", type=str, default='liuhaotian/llava-v1.6-vicuna-7b/')
    parser.add_argument("--model_base", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default="multiple_choice_allvideo_v4") 
    parser.add_argument("--num_chunks", type=int, default=1)
    parser.add_argument("--chunk_idx", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--input_structure", type=str, default="image_seq")
    parser.add_argument("--image_aspect_ratio", type=str, default='resize')
    parser.add_argument("--rope_scaling_factor", type=int, default=2)
    parser.add_argument("--enable_cot", action='store_true', help="Enable CoT question decomposition (requires OpenAI API key)")
    parser.add_argument("--openai_api_key", type=str, default=os.environ.get("OPENAI_API_KEY", None), help="OpenAI API key for CoT")
    parser.add_argument("--cot_max_new_tokens", type=int, default=64, help="Max new tokens for CoT caption generation")
    parser.add_argument("--answer_max_new_tokens", type=int, default=16, help="Max new tokens for final answer generation")
    # D-CoDe frame selection parameters
    parser.add_argument("--select_frames", type=int, default=15, help="Number of frames to select (N)")
    parser.add_argument("--uniform_ratio", type=float, default=0.85, help="Ratio for uniform sampling (e.g., 0.85)")
    # D-CoDe token selection parameters
    parser.add_argument("--dcode_top_k", type=int, default=288, help="Top-k tokens to select (288 for 7B)")
    parser.add_argument("--dcode_merge_strategy", type=str, default="mean", choices=["mean", "max", "weighted_mean"], help="Token merge strategy")
    parser.add_argument("--dcode_similarity_threshold", type=float, default=0.8, help="Similarity threshold for token merging")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
