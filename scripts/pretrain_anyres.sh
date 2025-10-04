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
# echo $CHIEF_IP
# deepspeed --num_gpus 16 --num_nodes 2 \
/opt/conda/envs/llava/bin/deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /data/oss_bucket_0/wangyh.ahui/models/lmsys/vicuna-7b-v1.5 \
    --data_path /data/oss_bucket_0/wangyh.ahui/datasets/AdaptVision_All_Data/instruct \
    --image_folder /data/oss_bucket_0/wangyh.ahui/datasets/AdaptVision_All_Data/images \
    --data_stage pretrain \
    --vision_tower /data/oss_bucket_0/wangyh.ahui/models/google/siglip2-so400m-patch16-512 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /data/oss_bucket_0/wangyh.ahui/models/llavanext/llava_7b_grid3_anyres_unpad_siglip2_512_pretrain \
    --use_pos_token True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "(1x1),...,(3x3)" \
    --max_grid_num 3 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to none  2>&1 | tee error.txt
