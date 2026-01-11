"""
D-CoDe: Decomposed Chain-of-Thought for Video Understanding
Core utility functions for frame selection and question decomposition.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, Path(__file__).parent.as_posix())
sys.path.insert(0, os.path.join(Path(__file__).parent.as_posix(), "slowfast_llava"))

import re
import ast
import torch
import numpy as np
import openai
from transformers import CLIPProcessor, CLIPModel

# OpenAI API Key (read from environment variable)
API_KEY = os.environ.get("OPENAI_API_KEY", None)


def extract_list_from_text(text):
    """
    Extract a Python list from text containing list format.
    
    Args:
        text: String containing a Python list
    
    Returns:
        Extracted list or empty list if not found
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)  
    if match:
        return ast.literal_eval(match.group(0)) 
    return []


def load_clip_model(model_name="openai/clip-vit-large-patch14-336"):
    """
    Load CLIP model and processor for frame selection.
    
    Args:
        model_name: HuggingFace model name for CLIP
    
    Returns:
        clip_processor: CLIP processor
        clip_model: CLIP model on CUDA
    """
    clip_processor = CLIPProcessor.from_pretrained(model_name)
    clip_model = CLIPModel.from_pretrained(model_name).cuda()
    return clip_processor, clip_model






def generate_subquestions(question, prompt_variant="original", api_key=API_KEY):
    assert api_key is not None, "You must provide a valid OpenAI API key."

    prompts = {
        "original": f"""
            I am working on a video understanding task. Your job is to break down the given question into a series of subquestions that guide the model toward solving the problem. The subquestions should focus on temporal and dynamic aspects of the video, rather than just static information that could be answered from a single frame. I will provide a question, and you should output the corresponding subquestions in English.

            Question: "{question}"

            Output the subquestions as a Python list of strings.
            """,

        "no_background": f"""
            Your job is to break down the given question into a series of subquestions that guide the model toward solving the problem. The subquestions should focus on temporal and dynamic aspects of the video, rather than just static information.

            Question: "{question}"

            Output the subquestions as a Python list of strings.
            """,

        "no_temporal_focus": f"""
            I am working on a video understanding task. Your job is to break down the given question into subquestions that guide the model toward solving the problem.

            Question: "{question}"

            Output the subquestions as a Python list of strings.
            """,

        "re": f"""
            You're assisting with a task that involves understanding events in videos over time. When given a question, your role is to transform it into several focused subquestions that explore how things change, move, or unfold throughout the video. Avoid questions that could be answered by looking at a single frame—prioritize those that require analyzing sequences, transitions, actions, or time-based dependencies.

            I'll provide a main question about the video. Your output should be a list of specific, time-aware subquestions in English, formatted as a Python list of strings.

            Main Question: “{question}”

            Your Output (Python list of strings):

            """
    }

    if prompt_variant not in prompts:
        raise ValueError(f"Invalid prompt_variant: {prompt_variant}")
    
    prompt = prompts[prompt_variant]

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=[
            {"role": "system", "content": "You are an expert in video understanding."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    subquestions = response.choices[0].message.content
    return subquestions.replace('%', '')

def supp_frame_selection(video_frames, N=15, uniform_ratio=0.85, clip_model=None, clip_processor=None):
    """
    Supplementary frame selection based on semantic similarity.
    
    Args:
        video_frames: List of video frames
        N: Number of frames to select
        uniform_ratio: Ratio for uniform sampling (e.g., NextQA=16/0.85, IntentQA=20/0.75)
        clip_model: Pre-loaded CLIP model (optional, will auto-load if None)
        clip_processor: Pre-loaded CLIP processor (optional, will auto-load if None)
    
    Returns:
        selected_frames: List of selected frames
        frame_idxs: Indices of selected frames
    """
    # Load CLIP model if not provided
    if clip_processor is None:
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    if clip_model is None:
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").cuda()
    
    # Extract frame features
    with torch.no_grad():
        CLS_list = []
        for frame in video_frames:
            clip_inputs = clip_processor(images=frame, return_tensors="pt").to('cuda')
            outputs = clip_model.get_image_features(**clip_inputs)
            CLS_list.append(outputs)
    
    # Select representative frames
    frame_idxs = _select_representative_frames(CLS_list, N=N, uniform_ratio=uniform_ratio)
    selected_frames = [video_frames[i] for i in frame_idxs]
    
    return selected_frames, frame_idxs


def _select_representative_frames(CLS_list, N, uniform_ratio=0.5):
    """
    Internal function to select representative frames based on similarity.
    
    Args:
        CLS_list: List of CLIP features
        N: Number of frames to select
        uniform_ratio: Ratio for uniform sampling
    
    Returns:
        selected_indices: List of selected frame indices
    """
    CLS_matrix = torch.cat(CLS_list, dim=0).cpu().numpy()  # (num_frames, feature_dim)
    
    # Normalize CLS vectors (prevent division by zero)
    eps = 1e-8
    norms = np.linalg.norm(CLS_matrix, axis=1, keepdims=True)
    CLS_matrix = CLS_matrix / (norms + eps)

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(CLS_matrix, CLS_matrix.T)  # (num_frames, num_frames)

    num_frames = len(CLS_list)
    
    # Step 1: Uniformly select a portion of frames
    num_uniform = int(N * uniform_ratio)
    uniform_indices = np.linspace(0, num_frames - 1, num=num_uniform, dtype=int).tolist()
    
    selected_indices = list(uniform_indices)
    remaining_N = N - num_uniform

    if remaining_N > 0:
        # Step 2: Select remaining frames based on similarity
        for _ in range(remaining_N):
            min_sim_to_selected = np.full(num_frames, 1.0)
            for idx in range(num_frames):
                if idx in selected_indices:
                    continue
                # Compute mean similarity to already selected frames
                min_sim_to_selected[idx] = np.mean(similarity_matrix[idx, selected_indices])

            # Select frame with minimum similarity to ensure diversity
            next_index = np.argmin(min_sim_to_selected)
            selected_indices.append(next_index)

    selected_indices = sorted(selected_indices)
    return selected_indices


# =============================================================================
# Token Selection and Merge Functions (for Vision Features)
# =============================================================================

def token_select_and_merge(image_features, top_k=288, merge_strategy="mean", similarity_threshold=0.8):
    """
    Select and merge tokens from image features to reduce redundancy.
    
    This function performs two main operations:
    1. Select top-k tokens based on L2 activation values
    2. Merge similar tokens to remove redundancy
    
    Args:
        image_features: Tensor of shape (T, N, D) where T=frames, N=tokens, D=dim
        top_k: Maximum number of tokens to select per frame (default: 288 for 7B, 320 for 34B)
        merge_strategy: Strategy for merging similar tokens ("mean", "max", "weighted_mean")
        similarity_threshold: Cosine similarity threshold for considering tokens as duplicates
    
    Returns:
        merged_features: Tensor of shape (1, total_tokens, D) with selected and merged tokens
    """
    T, N, D = image_features.shape
    top_k = min(top_k, N)
    
    selected_features_list = []
    
    for frame_features in image_features:
        # Compute L2 activation values
        activations = torch.norm(frame_features, dim=-1)
        
        # Select top-k tokens
        _, topk_indices = torch.topk(activations, k=top_k, dim=-1)
        selected_tokens = frame_features[topk_indices, :]  # [top_k, D]
        
        # Remove redundant tokens
        selected_tokens, keep_indices = _merge_similar_tokens(
            selected_tokens, 
            activations[topk_indices],
            top_k,
            merge_strategy, 
            similarity_threshold
        )
        
        selected_features_list.append(selected_tokens)
    
    # Concatenate all selected tokens
    merged_features = torch.cat(selected_features_list, dim=0).unsqueeze(0)  # [1, total_tokens, D]
    
    return merged_features


def _merge_similar_tokens(selected_tokens, activations, top_k, merge_strategy="mean", threshold=0.8):
    """
    Internal function to merge similar tokens based on cosine similarity.
    
    Args:
        selected_tokens: Tensor of shape (top_k, D)
        activations: Tensor of activation values for weighting
        top_k: Number of tokens
        merge_strategy: "mean", "max", or "weighted_mean"
        threshold: Cosine similarity threshold
    
    Returns:
        merged_tokens: Tensor with merged tokens
        keep_indices: Boolean tensor indicating which tokens to keep
    """
    # Normalize tokens for cosine similarity
    normalized_tokens = torch.nn.functional.normalize(selected_tokens, p=2, dim=-1)
    cosine_sim_matrix = torch.mm(normalized_tokens, normalized_tokens.T)  # [top_k, top_k]
    
    # Find similar tokens
    mask = cosine_sim_matrix >= threshold  # [top_k, top_k]
    
    keep_indices = torch.ones(top_k, dtype=torch.bool, device=selected_tokens.device)
    
    for i in range(top_k):
        if keep_indices[i]:
            similar_indices = mask[i].nonzero(as_tuple=True)[0]
            
            if len(similar_indices) > 1:
                # Merge similar tokens based on strategy
                if merge_strategy == "mean":
                    selected_tokens[i] = selected_tokens[similar_indices].mean(dim=0)
                elif merge_strategy == "max":
                    selected_tokens[i] = selected_tokens[similar_indices].max(dim=0).values
                elif merge_strategy == "weighted_mean":
                    weights = activations[similar_indices] / activations[similar_indices].sum()
                    selected_tokens[i] = torch.sum(weights[:, None] * selected_tokens[similar_indices], dim=0)
                
                # Discard duplicate tokens (keep only the first one)
                keep_indices[similar_indices[1:]] = False
    
    merged_tokens = selected_tokens[keep_indices]
    
    return merged_tokens, keep_indices
