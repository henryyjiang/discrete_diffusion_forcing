#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# GSM8K only - optimized for speed
tasks="gsm8k"
nshots="4"
max_lengths="1024"  # Total sequence length including prompt
temperatures="0"
limits="1519"  # Full GSM8K test set
block_sizes="32"
top_ps="0.95"
dtypes="bfloat16"
mc_nums="32"  # Monte Carlo samples for loglikelihood (32 for speed, 128 for quality)
diffusion_steps="32"  # For generation (32 for speed, 128 for quality)

# Base model and LoRA path
base_model="GSAI-ML/LLaDA-8B-Instruct"
lora_models=(
    "../D2F-train/ckpt_llada_instruct_gt_threshold_sampling_1.2/llada_ddt_maskteacher/ddt_test/Decoder-llada_ddt_maskteacher-10k"
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
read -ra MC_NUMS_ARRAY <<< "$mc_nums"
read -ra DIFFUSION_STEPS_ARRAY <<< "$diffusion_steps"

# Validation
array_length=${#TASKS_ARRAY[@]}
if [[ ${#NSHOTS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#MAX_LENGTH_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TEMP_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#LIMITS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#BLOCK_SIZES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#TOP_PS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#DTYPES_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#MC_NUMS_ARRAY[@]} -ne $array_length ]] || \
   [[ ${#DIFFUSION_STEPS_ARRAY[@]} -ne $array_length ]]; then
    echo "Error: All configuration arrays must have the same length!"
    exit 1
fi

for lora_model in "${lora_models[@]}"; do
    lora_model_name=$(basename "$lora_model")
    echo "===================================================================="
    echo "Evaluating Bisection model: $lora_model_name"
    echo "===================================================================="
    
    for i in "${!TASKS_ARRAY[@]}"; do
        output_path="eval_bisection_${lora_model_name}/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}-maxlen${MAX_LENGTH_ARRAY[$i]}-temp${TEMP_ARRAY[$i]}-block${BLOCK_SIZES_ARRAY[$i]}-topp${TOP_PS_ARRAY[$i]}-mc${MC_NUMS_ARRAY[$i]}-diffsteps${DIFFUSION_STEPS_ARRAY[$i]}-dtype${DTYPES_ARRAY[$i]}"
        
        echo "Running evaluation for ${TASKS_ARRAY[$i]}"
        echo "Config: Shots=${NSHOTS_ARRAY[$i]}, MaxLength=${MAX_LENGTH_ARRAY[$i]}, Temp=${TEMP_ARRAY[$i]}, Block=${BLOCK_SIZES_ARRAY[$i]}, TopP=${TOP_PS_ARRAY[$i]}, MC_Num=${MC_NUMS_ARRAY[$i]}, DiffSteps=${DIFFUSION_STEPS_ARRAY[$i]}, Dtype=${DTYPES_ARRAY[$i]}"
        
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
        model_args="${model_args},mc_num=${MC_NUMS_ARRAY[$i]}"
        model_args="${model_args},diffusion_steps=${DIFFUSION_STEPS_ARRAY[$i]}"
        
        accelerate launch \
            --main_process_port 29520 \
            --num_processes 8 \
            eval_bisection.py \
            --model bisection \
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