set -x
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export NCCL_BLOCKING_WAIT=0

#USE YOUR TOKEN: REMEMBER!!!
WANDB_TOKENS=a77607626908409e45afa2ca225cf179e9a316fc
wandb login --relogin $WANDB_TOKENS

EXP_NAME=Qwen_INF_0625

MODEL="/data2/rlhf/lixs/open_models/Qwen2-7B-Instruct/" 
# MODEL="/data2/rlhf/yzy/open_models/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"
DATA="/data2/rlhf/yzy/data/rm_train/train_20240425-v2-guide/train"
save_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/$EXP_NAME"
BATCH_SIZE_PER_GPU=2
LR=1e-5

hostfile="./hostfile_rm"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

mkdir -p $save_path

read -r -d '' training_commands <<EOF
../../examples/batch_inference.py \
    --eval_task generate_vllm \
    --pretrain $MODEL \
    --bf16 \
    --max_len 6400 \
    --dataset $DATA \
    --dataset_probs 1.0 \
    --max_samples 100000000 \
    --zero_stage 0 \
    --tp_size 4 \
    --micro_batch_size $BATCH_SIZE_PER_GPU \
    --flash_attn \
    --output_path $save_path/output.jsonl
EOF

python $training_commands 2>&1 | tee $save_path/train_qwen.log
