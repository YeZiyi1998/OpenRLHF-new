set -x
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

#USE YOUR TOKEN: REMEMBER!!!
WANDB_TOKENS=a77607626908409e45afa2ca225cf179e9a316fc
wandb login --relogin $WANDB_TOKENS

EXP_NAME=Qwen_SFT_0625_reason

MODEL="/data2/rlhf/lixs/open_models/Qwen2-7B-Instruct/" 
DATA="/data2/rlhf/yzy/data/rm_train/train_20240425-v2-guide-300k/train_guide"
EVAL_DATA="/data2/rlhf/yzy/data/rm_train/train_20240425-v2-guide-300k/test_guide"
save_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/$EXP_NAME"
ckpt_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/$EXP_NAME/checkpoints"
BATCH_SIZE_PER_GPU=2
LR=9e-6

hostfile="./hostfile_tsp"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

#TODO-be-check: bs large, adamw
GLOBAL_BATCH_SIZE=$(( ${BATCH_SIZE_PER_GPU} * ${WORLD_SIZE} ))

mkdir -p $save_path

read -r -d '' training_commands <<EOF
../../examples/train_sft.py \
    --save_path $save_path \
    --logging_steps 10 \
    --micro_train_batch_size $BATCH_SIZE_PER_GPU \
    --micro_eval_batch_size 1 \
    --train_batch_size $GLOBAL_BATCH_SIZE \
    --pretrain $MODEL \
    --bf16 \
    --max_epochs 1 \
    --max_len 6400 \
    --zero_stage 3 \
    --eval_steps 500 \
    --save_steps 1500 \
    --learning_rate $LR\
    --dataset $DATA \
    --eval_dataset $EVAL_DATA \
    --dataset_probs 1.0 \
    --flash_attn \
    --lr_scheduler linear \
    --gradient_checkpointing \
    --ckpt_path $ckpt_path \
    --use_wandb $WANDB_TOKENS \
    --wandb_run_name $EXP_NAME \
    --wandb_project rl0624 
EOF

deepspeed --hostfile=$hostfile $training_commands 2>&1 | tee $save_path/train_qwen.log
# deepspeed --num_gpus 4 --master_addr 10.64.24.96 $training_commands 2>&1 | tee $save_path/train_qwen.log

