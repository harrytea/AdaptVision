#!/bin/bash

# export NCCL_NET=Socket
# # export PATH=/opt/cuda/12.0.1_525.85.12/bin:/opt/cuda/12.0.1_525.85.12/lib64:/home/sist/wangyh/miniconda3/envs/llava2/bin:$PATH
# # export PATH=/opt/cuda/11.7.1_515.65.01/bin:/opt/cuda/11.7.1_515.65.01/nvvm:/home/sist/wangyh/miniconda3/envs/llava2/bin:$PATH
# export LD_LIBRARY_PATH=/home/sist/wangyh/miniconda3/envs/llava/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/lib64/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/lib64/stubs
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/targets/x86_64-linux/lib/stubs
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/lib64/
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/lib64/stubs
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/targets/x86_64-linux/lib/stubs

# OMP_NUM_THREADS=1 
# MKL_NUM_THREADS=1
echo $CHIEF_IP
# deepspeed --num_gpus 16 --num_nodes 2 \
deepspeed --master_addr=$CHIEF_IP \
    --hostfile /llm-cfs-nj/person/harryyhwang/mae_hostfile llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /llm-cfs-nj/person/harryyhwang/models/vicuna-7b-v1.5 \
    --data_path /llm-cfs-nj/person/harryyhwang/dataset/instruction \
    --image_folder /llm-cfs-nj/person/harryyhwang/dataset/images \
    --data_stage pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --report_to none
