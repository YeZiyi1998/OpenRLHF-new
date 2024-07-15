set -x 
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
#data_path="/baichuan/npc/lixs/data/rm_train/train_20240425"
#model_path="/es01/ssd/lixs/models/pretrained/baichuan3-33b-2T4-huggingface"
#save_path="/es01/ssd/lixs/outputs/33B_RM_0425/"

#model_path="/baichuan/npc/maluyao/model/pretrained/baichuan2-13b-enhance"
#save_path="/baichuan/npc/lixs/outputs/reward_models/open_test"

WANDB_TOKENS=37eebd70bd7510b86c29171886dbc690d655c0b3

data_path="/data2/rlhf/lixs/data/rm_train/train_20240425-v2"

EXP_NAME=33B_RM_0425-v2
model_path="/data2/rlhf/jialian/rm_model/model/bc_33_3t"
BATCH_SIZE_PER_GPU=1
LR=9e-6

#EXP_NAME=177B_RM_TEST
#model_path="/data2/rlhf/jialian/rm_model/model/bc_177_4t5/baichuan3-177b-4T5"
##model_path="/data2/rlhf/liufeng/checkpoint/pretrain_model/baichuan3-177b-4T85"
#BATCH_SIZE_PER_GPU=1
#LR=5e-7

save_path="/data2/rlhf/lixs/outputs/reward_models/$EXP_NAME"

hostfile="./hostfile_rm"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

#TODO-be-check: bs large, adamw
GLOBAL_BATCH_SIZE=$(( ${BATCH_SIZE_PER_GPU} * ${WORLD_SIZE} ))

mkdir -p $save_path

read -r -d '' training_commands <<EOF
../examples/train_rm.py \
     --save_path $save_path \
     --save_steps 5000 \
     --logging_steps 1 \
     --eval_steps 1000 \
     --micro_train_batch_size $BATCH_SIZE_PER_GPU \
     --train_batch_size $GLOBAL_BATCH_SIZE \
     --pretrain $model_path \
     --fp16 \
     --max_epochs 1 \
     --max_len 2048 \
     --zero_stage 3 \
     --compute_fp32_loss \
     --l2 0.0001 \
     --learning_rate $LR\
     --dataset $data_path \
     --dataset_probs 1 \
     --gradient_checkpointing \
     --use_wandb $WANDB_TOKENS \
     --wandb_run_name $EXP_NAME
EOF
     # --wandb [WANDB_TOKENS] or True (use wandb login command)
     # --train_batch_size 128 \
     #--flash_attn \
        #--compute_fp32_loss \
        #     --adam_offload \

deepspeed --hostfile=$hostfile $training_commands 2>&1 | tee $save_path/train.log
