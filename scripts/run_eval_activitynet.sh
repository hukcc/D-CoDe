#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
ROOT_DIR=${ROOT_DIR:-"$(dirname "$(dirname "$(readlink -f "$0")")")"}
CONFIG_NAME=${CONFIG_NAME:-"D-CoDe_llava_7b"}
DATA_DIR=${DATA_DIR:-"${ROOT_DIR}/playground/data/video_qa"}
GT_QA_DIR=${GT_QA_DIR:-"${ROOT_DIR}/playground/gt_qa_files"}
MODEL_PATH=${MODEL_PATH:-"${ROOT_DIR}/liuhaotian/llava-v1.6-vicuna-7b/"}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT_DIR}/outputs/artifacts"}
TEMP_DIR=${TEMP_DIR:-"${ROOT_DIR}/outputs/eval_save_dir"}
CONV_MODE=${CONV_MODE:-"image_seq_v3"}
INPUT_STRUCTURE=${INPUT_STRUCTURE:-"image_seq"}
IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-"resize"}
ROPE_SCALING_FACTOR=${ROPE_SCALING_FACTOR:-"2"}
# D-CoDe frame selection parameters
NUM_FRAMES=${NUM_FRAMES:-"50"}
SELECT_FRAMES=${SELECT_FRAMES:-"25"}
UNIFORM_RATIO=${UNIFORM_RATIO:-"0.8"}
# D-CoDe token selection parameters
DCODE_TOP_K=${DCODE_TOP_K:-"288"}
DCODE_MERGE_STRATEGY=${DCODE_MERGE_STRATEGY:-"mean"}
DCODE_SIMILARITY_THRESHOLD=${DCODE_SIMILARITY_THRESHOLD:-"0.8"}
# Generation parameters
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-"64"}
ANSWER_MAX_NEW_TOKENS=${ANSWER_MAX_NEW_TOKENS:-"128"}
# CoT parameters (requires OpenAI API key)
ENABLE_COT=${ENABLE_COT:-"false"}
OPENAI_API_KEY=${OPENAI_API_KEY:-""}

################################# Run ##################################

gpu_list="${CUDA_VISIBLE_DEVICES:-0,0,1,1,2,2,3,3}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Build CoT arguments
COT_ARGS=""
if [ "${ENABLE_COT}" = "true" ]; then
    COT_ARGS="--enable_cot --openai_api_key ${OPENAI_API_KEY}"
fi

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python3 ${ROOT_DIR}/run_inference_video_qa.py \
      --video_dir ${DATA_DIR}/Activitynet_Zero_Shot_QA/all_test \
      --gt_file_question ${GT_QA_DIR}/Activitynet_Zero_Shot_QA/test_q.json \
      --gt_file_answers ${GT_QA_DIR}/Activitynet_Zero_Shot_QA/test_a.json \
      --output_dir ${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/${CONFIG_NAME} \
      --output_name ${CHUNKS}_${IDX} \
      --model_path ${MODEL_PATH} \
      --conv_mode ${CONV_MODE} \
      --num_chunks ${CHUNKS} \
      --chunk_idx ${IDX} \
      --num_frames ${NUM_FRAMES} \
      --temperature 0 \
      --input_structure ${INPUT_STRUCTURE} \
      --image_aspect_ratio ${IMAGE_ASPECT_RATIO} \
      --rope_scaling_factor ${ROPE_SCALING_FACTOR} \
      --select_frames ${SELECT_FRAMES} \
      --uniform_ratio ${UNIFORM_RATIO} \
      --dcode_top_k ${DCODE_TOP_K} \
      --dcode_merge_strategy ${DCODE_MERGE_STRATEGY} \
      --dcode_similarity_threshold ${DCODE_SIMILARITY_THRESHOLD} \
      --cot_max_new_tokens ${COT_MAX_NEW_TOKENS} \
      --answer_max_new_tokens ${ANSWER_MAX_NEW_TOKENS} \
      ${COT_ARGS} &
done

wait

output_dir=${OUTPUT_DIR}/Activitynet_Zero_Shot_QA/${CONFIG_NAME}
output_file=${output_dir}/merge.jsonl
temp_dir=${TEMP_DIR}/Activitynet_Zero_Shot_QA/${CONFIG_NAME}

# Clear out the output file if it exists.
> "${output_file}"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${output_dir}/${CHUNKS}_${IDX}.json >> "${output_file}"
done

################################# Eval ##################################

gpt_version="gpt-3.5-turbo-0125"
num_tasks=25

python3 ${ROOT_DIR}/eval/eval_video_qa.py \
    --pred_path ${output_file} \
    --output_dir ${temp_dir}/${gpt_version} \
    --output_json ${output_dir}/results.json \
    --gpt_version ${gpt_version} \
    --num_tasks ${num_tasks}
