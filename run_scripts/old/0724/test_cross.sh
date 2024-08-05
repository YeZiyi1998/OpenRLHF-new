#!/bin/bash

set -x 
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# 1 96 2 s1 3 math
EXP_NAME=Qwen_pairwise_$1_$2_$3

python /data2/rlhf/yzy/write_email.py --info $EXP_NAME

#USE YOUR TOKEN: REMEMBER!!!
WANDB_TOKENS=a77607626908409e45afa2ca225cf179e9a316fc
wandb login --relogin $WANDB_TOKENS

data_path="/data2/rlhf/yzy/data/sub_data/pairwise_critic_inference2/$3"
if [ "$2" = "s1" ]; then
  add_data="None"
else
  add_data="/data2/rlhf/yzy/OpenRLHF-new/outputs/inference/Qwen_Critic_machine_$3/Qwen_Critic_machine.jsonl"
fi

model_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/Qwen_pairwise_s1_96_math/s1"
model_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/Qwen_pairwise_96_s1_数学计算/s1"
BATCH_SIZE_PER_GPU=16
BATCH_SIZE_PER_GPU=1
LR=9e-6

#/eb_data
save_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/predictions/$EXP_NAME"
hostfile="/data2/rlhf/yzy/OpenRLHF-new/run_scripts/0625/hostfile_$1"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

#TODO-be-check: bs large, adamw
GLOBAL_BATCH_SIZE=$(( ${BATCH_SIZE_PER_GPU} * ${WORLD_SIZE} ))

mkdir -p $save_path

read -r -d '' training_commands <<EOF
../../examples/train_2rm.py \
     --save_path $save_path \
     --logging_steps 10 \
     --micro_train_batch_size $BATCH_SIZE_PER_GPU \
     --train_batch_size $GLOBAL_BATCH_SIZE \
     --pretrain $model_path \
     --quantized_type bf16 \
     --max_epochs 0 \
     --max_len 2048 \
     --zero_stage 3 \
     --l2 0.0001 \
     --eval_steps 250 \
     --save_steps 500 \
     --learning_rate $LR\
     --dataset $data_path \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB_TOKENS \
     --wandb_run_name $EXP_NAME \
     --wandb_project rl_critic_test \
     --mode $2 \
     --add_data $add_data
EOF

deepspeed --hostfile=$hostfile $training_commands 2>&1 | tee $save_path/train_qwen.log
python /data2/rlhf/yzy/write_email.py --info "$EXP_NAME done"
