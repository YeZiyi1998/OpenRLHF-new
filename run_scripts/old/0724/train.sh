#!/bin/bash

set -x 
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# 1 96 2 s1 3 math
EXP_NAME=Qwen_pairwise_$1_$2_$3

# python /data2/rlhf/yzy/write_email.py --info $EXP_NAME

#USE YOUR TOKEN: REMEMBER!!!
WANDB_TOKENS=a77607626908409e45afa2ca225cf179e9a316fc
wandb login --relogin $WANDB_TOKENS

data_path="/data2/rlhf/yzy/data/sub_data/pairwise_critic_inference/$3"
if [ "$2" = "s1" ]; then
  add_data="None"
else
  add_data="/data2/rlhf/yzy/OpenRLHF-new/outputs/0724/inference/Qwen_Critic_machine_$3/Qwen_Critic_machine.jsonl"
fi

# model_path="/data2/rlhf/yzy/open_models/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"
model_path="/data2/rlhf/lixs/open_models/Qwen2-7B-Instruct/"
BATCH_SIZE_PER_GPU=16
LR=9e-6

#/eb_data
save_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/$EXP_NAME"
ckpt_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/$EXP_NAME/checkpoints"
hostfile="/data2/rlhf/yzy/OpenRLHF-new/run_scripts/0625/hostfile_$1"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

#TODO-be-check: bs large, adamw
GLOBAL_BATCH_SIZE=$(( ${BATCH_SIZE_PER_GPU} * ${WORLD_SIZE} ))

mkdir -p $save_path
mkdir -p $ckpt_path

read -r -d '' training_commands <<EOF
../../examples/train_2rm.py \
     --save_path $save_path \
     --logging_steps 10 \
     --micro_train_batch_size $BATCH_SIZE_PER_GPU \
     --train_batch_size $GLOBAL_BATCH_SIZE \
     --pretrain $model_path \
     --add_data $add_data \
     --quantized_type bf16 \
     --max_epochs 1 \
     --max_len 2560 \
     --zero_stage 3 \
     --l2 0.0001 \
     --eval_steps 250 \
     --save_steps 500 \
     --learning_rate $LR\
     --dataset $data_path \
     --dataset_probs 1 \
     --ckpt_path $ckpt_path \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB_TOKENS \
     --wandb_run_name $EXP_NAME \
     --wandb_project rl_critic_test \
     --mode $2
EOF

deepspeed $training_commands 2>&1 | tee $save_path/train_qwen.log
# python /data2/rlhf/yzy/write_email.py --info "$EXP_NAME done"
