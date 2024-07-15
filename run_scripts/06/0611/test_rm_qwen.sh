set -x 
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

#USE YOUR TOKEN: REMEMBER!!!
WANDB_TOKENS=a77607626908409e45afa2ca225cf179e9a316fc
wandb login --relogin $WANDB_TOKENS

data_path="/data2/rlhf/lixs/data/rm_train/train_20240425-v2"

EXP_NAME=Qwen_MOE_RM_0611_2
# model_path="/data2/rlhf/lixs/open_models/DeepSeek-V2-Lite-Chat"
# model_path="/eb_data/rlhf/yezy/open_models/models--Qwen--Qwen1.5-MoE-A2.7B-Chat/snapshots/ec052fda178e241c7c443468d2fa1db6618996be"
model_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/Qwen_MOE_RM_0611_2"
BATCH_SIZE_PER_GPU=16
LR=9e-6

#/eb_data
save_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/$EXP_NAME"

hostfile="./hostfile_rm"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

#TODO-be-check: bs large, adamw
GLOBAL_BATCH_SIZE=$(( ${BATCH_SIZE_PER_GPU} * ${WORLD_SIZE} ))

mkdir -p $save_path

read -r -d '' training_commands <<EOF
../../examples/test_rm.py \
     --save_path $save_path \
     --logging_steps 10 \
     --micro_train_batch_size $BATCH_SIZE_PER_GPU \
     --train_batch_size $GLOBAL_BATCH_SIZE \
     --pretrain $model_path \
     --quantized_type bf16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --l2 0.0001 \
     --eval_steps 250 \
     --learning_rate $LR\
     --dataset $data_path \
     --dataset_probs 1 \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB_TOKENS \
     --wandb_run_name $EXP_NAME \
     --wandb_project rl 
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --train_batch_size 128 \
     #--flash_attn \
        #--compute_fp32_loss \
        #     --adam_offload \

deepspeed --hostfile=$hostfile $training_commands 2>&1 | tee $save_path/test_qwen.log
