#!/usr/bin/env bash

CUDA_VISIBLE=2
logs_dir=./logs/
log_name=testlog.log
trial_name=test_log
output_file=/data2/whd/model_result/Samsum/test_modify/
config_file=./config/RCUPSconfig.json
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE} \
    nohup python run_summarization.py "${config_file}" "${output_file}" > "${logs_dir}${log_name}" 2>&1 &