# export CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICES=0 accelerate launch   --config_file config/acc_config --num_processes 1 --main_process_port 29577 train.py --config config/bisection.yaml
