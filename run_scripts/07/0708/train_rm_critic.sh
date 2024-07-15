set -x 
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python /data2/rlhf/yzy/write_email.py --info Qwen_RM_critic_96_baseline

#USE YOUR TOKEN: REMEMBER!!!
WANDB_TOKENS=a77607626908409e45afa2ca225cf179e9a316fc
wandb login --relogin $WANDB_TOKENS

data_path="/data2/rlhf/yzy/data/rm/critic_train_rm_baseline"

EXP_NAME=Qwen_RM_critic_96_baseline
# model_path="/data2/rlhf/lixs/open_models/DeepSeek-V2-Lite-Chat"
model_path="/data2/rlhf/lixs/open_models/Qwen2-7B-Instruct/"
BATCH_SIZE_PER_GPU=16
LR=9e-6

#/eb_data
save_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/$EXP_NAME"
ckpt_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/$EXP_NAME/checkpoints"
hostfile="/data2/rlhf/yzy/OpenRLHF-new/run_scripts/0625/hostfile_96"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

#TODO-be-check: bs large, adamw
GLOBAL_BATCH_SIZE=$(( ${BATCH_SIZE_PER_GPU} * ${WORLD_SIZE} ))

mkdir -p $save_path
mkdir -p $ckpt_path

read -r -d '' training_commands <<EOF
../../examples/train_rm.py \
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
     --save_steps 500 \
     --learning_rate $LR\
     --dataset $data_path \
     --dataset_probs 1 \
     --ckpt_path $ckpt_path \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB_TOKENS \
     --wandb_run_name $EXP_NAME \
     --wandb_project rl_critic 
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --train_batch_size 128 \
     #--flash_attn \
        #--compute_fp32_loss \
        #     --adam_offload \

deepspeed --hostfile=$hostfile $training_commands 2>&1 | tee $save_path/train_qwen.log
python /data2/rlhf/yzy/write_email.py --info Qwen_RM_critic_96_baseline_done
