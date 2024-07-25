set -x
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

EXP_NAME=Qwen_pairwise_qwen_s2_c_64

MODEL="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/Qwen_pairwise_qwen_s2_c_64/checkpoints/global_step$1/c"
# MODEL="/data2/rlhf/lixs/open_models/Qwen2-7B-Instruct/"
File_NAME="Qwen_Critic_machine.$1.jsonl" 
DATA="/data2/rlhf/yzy/data/rm/pairwise_critic_train_guide_qwen_300k/2/train"
DATA2="/data2/rlhf/yzy/data/rm/pairwise_critic_train/test"

save_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/inference/$EXP_NAME"
BATCH_SIZE_PER_GPU=8
LR=1e-5

hostfile="/data2/rlhf/yzy/OpenRLHF-new/run_scripts/0625/hostfile_tsp"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

mkdir -p $save_path

read -r -d '' training_commands <<EOF
../../examples/batch_inference.py \
    --eval_task generate \
    --pretrain $MODEL \
    --bf16 \
    --max_len 6400 \
    --dataset $DATA \
    --dataset2 $DATA2 \
    --dataset_probs 1.0 \
    --zero_stage 0 \
    --tp_size 4 \
    --greedy_sampling \
    --micro_batch_size $BATCH_SIZE_PER_GPU \
    --flash_attn \
    --max_new_tokens 64 \
    --input_template Human \
    --output_path $save_path/$File_NAME
EOF

deepspeed --hostfile $hostfile $training_commands 2>&1 | tee $save_path/train_qwen_inference_$1.log

python /data2/rlhf/yzy/write_email.py --info $1
