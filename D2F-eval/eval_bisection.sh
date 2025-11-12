#!/bin/bash

# Bisection Sampling Evaluation Script
# Simplified from the original - removes progressive decoding parameters

tasks="gsm8k mbpp minerva_math"
nshots="4 3 0"
max_lengths="1024 1024 10224"  # Total sequence length including prompt
temperatures="0 0 0"
limits="10000 10000 10000"
block_sizes="32 32 32"  # Block size used during training
top_ps="0.95 0.95 0.95"
dtypes="bfloat16 bfloat16 bfloat16"

# HumanEval specific settings
humaneval_nshots="0"
humaneval_max_lengths="2048"
humaneval_temperatures="0"
humaneval_limits="10000"
humaneval_block_sizes="128"
humaneval_top_ps="0.95"
humaneval_dtypes="bfloat16"

# Your models
base_model="GSAI-ML/LLaDA-8B-Instruct"
lora_models=(
    "/D2F-train/ckpt_llada_instruct_gt_threshold_sampling_1.2/llada_ddt_maskteacher/ddt_test/Decoder-llada_ddt_maskteacher-10k"  # UPDATE THIS PATH
)

# Parse arrays
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra MAX_LENGTH_ARRAY <<< "$max_lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra LIMITS_ARRAY <<< "$limits"
read -ra BLOCK_SIZES_ARRAY <<< "$block_sizes"
read -ra TOP_PS_ARRAY <<< "$top_ps"
read -ra DTYPES_ARRAY <<< "$dtypes"

read -ra HUMANEVAL_NSHOTS_ARRAY <<< "$humaneval_nshots"
read -ra HUMANEVAL_MAX_LENGTHS_ARRAY <<< "$humaneval_max_lengths"
read -ra HUMANEVAL_TEMP_ARRAY <<< "$humaneval_temperatures"
read -ra HUMANEVAL_LIMITS_ARRAY <<< "$humaneval_limits"
read -ra HUMANEVAL_BLOCK_SIZES_ARRAY <<< "$humaneval_block_sizes"
read -ra HUMANEVAL_TOP_PS_ARRAY <<< "$humaneval_top_ps"
read -ra HUMANEVAL_DTYPES_ARRAY <<< "$humaneval_dtypes"

# Validation
array_length=${#TASKS_ARRAY[@]}
if [[ ${#NSHOTS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#MAX_LENGTH_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TEMP_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#LIMITS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#BLOCK_SIZES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TOP_PS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#DTYPES_ARRAY[@]} -ne $array_length ]]; then
    echo "Error: All configuration arrays must have the same length!"
    exit 1
fi

humaneval_array_length=${#HUMANEVAL_NSHOTS_ARRAY[@]}
if [[ ${#HUMANEVAL_MAX_LENGTHS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TEMP_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_LIMITS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_BLOCK_SIZES_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_TOP_PS_ARRAY[@]} -ne $humaneval_array_length ]] || \
   [[ ${#HUMANEVAL_DTYPES_ARRAY[@]} -ne $humaneval_array_length ]]; then
    echo "Error: All HumanEval configuration arrays must have the same length!"
    exit 1
fi

export HF_ALLOW_CODE_EVAL=1

for lora_model in "${lora_models[@]}"; do
    lora_model_name=$(basename "$lora_model")
    echo "===================================================================="
    echo "Evaluating Bisection model: $lora_model_name"
    echo "===================================================================="
    
    # HumanEval evaluation
    for i in "${!HUMANEVAL_NSHOTS_ARRAY[@]}"; do
        output_path="eval_bisection_${lora_model_name}/humaneval-ns${HUMANEVAL_NSHOTS_ARRAY[$i]}-maxlen${HUMANEVAL_MAX_LENGTHS_ARRAY[$i]}-temp${HUMANEVAL_TEMP_ARRAY[$i]}-block${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]}-topp${HUMANEVAL_TOP_PS_ARRAY[$i]}-dtype${HUMANEVAL_DTYPES_ARRAY[$i]}"
        
        echo "Running HumanEval evaluation $((i+1))/${humaneval_array_length} for $lora_model_name..."
        echo "Config: Shots=${HUMANEVAL_NSHOTS_ARRAY[$i]}, MaxLength=${HUMANEVAL_MAX_LENGTHS_ARRAY[$i]}, Temp=${HUMANEVAL_TEMP_ARRAY[$i]}, Block=${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]}, TopP=${HUMANEVAL_TOP_PS_ARRAY[$i]}, Dtype=${HUMANEVAL_DTYPES_ARRAY[$i]}"
        
        # Build model args for bisection
        model_args="base_model_name_or_path=${base_model}"
        model_args="${model_args},peft_model_name_or_path=${lora_model}"
        model_args="${model_args},max_length=${HUMANEVAL_MAX_LENGTHS_ARRAY[$i]}"
        model_args="${model_args},temperature=${HUMANEVAL_TEMP_ARRAY[$i]}"
        model_args="${model_args},top_p=${HUMANEVAL_TOP_PS_ARRAY[$i]}"
        model_args="${model_args},block_size=${HUMANEVAL_BLOCK_SIZES_ARRAY[$i]}"
        model_args="${model_args},dtype=${HUMANEVAL_DTYPES_ARRAY[$i]}"
        model_args="${model_args},add_bos_token=true"
        model_args="${model_args},mask_token_id=126336"  # LLaDA mask token
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
            --main_process_port 29520 \
            --num_processes 8 \
            eval_llada_bisection.py \
            --model llada-bisection \
            --model_args "$model_args" \
            --tasks humaneval \
            --num_fewshot ${HUMANEVAL_NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path "$output_path" \
            --log_samples \
            --confirm_run_unsafe_code
    done
    
    # Other tasks evaluation
    for i in "${!TASKS_ARRAY[@]}"; do
        output_path="eval_bisection_${lora_model_name}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-maxlen${MAX_LENGTH_ARRAY[$i]}-temp${TEMP_ARRAY[$i]}-block${BLOCK_SIZES_ARRAY[$i]}-topp${TOP_PS_ARRAY[$i]}-dtype${DTYPES_ARRAY[$i]}"
        
        echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}, MaxLength: ${MAX_LENGTH_ARRAY[$i]}, Temp: ${TEMP_ARRAY[$i]}, Block: ${BLOCK_SIZES_ARRAY[$i]}, TopP: ${TOP_PS_ARRAY[$i]}, Dtype: ${DTYPES_ARRAY[$i]}"
        
        # Build model args for bisection
        model_args="base_model_name_or_path=${base_model}"
        model_args="${model_args},peft_model_name_or_path=${lora_model}"
        model_args="${model_args},max_length=${MAX_LENGTH_ARRAY[$i]}"
        model_args="${model_args},temperature=${TEMP_ARRAY[$i]}"
        model_args="${model_args},top_p=${TOP_PS_ARRAY[$i]}"
        model_args="${model_args},block_size=${BLOCK_SIZES_ARRAY[$i]}"
        model_args="${model_args},dtype=${DTYPES_ARRAY[$i]}"
        model_args="${model_args},add_bos_token=true"
        model_args="${model_args},mask_token_id=126336"
        
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
            --main_process_port 29520 \
            --num_processes 8 \
            eval_llada_bisection.py \
            --model llada-bisection \
            --model_args "$model_args" \
            --tasks ${TASKS_ARRAY[$i]} \
            --limit ${LIMITS_ARRAY[$i]} \
            --num_fewshot ${NSHOTS_ARRAY[$i]} \
            --batch_size 1 \
            --output_path "$output_path" \
            --log_samples \
            --confirm_run_unsafe_code \
            --apply_chat_template \
            --fewshot_as_multiturn
    done
done

echo "All bisection evaluations completed!"