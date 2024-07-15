set -x
unset https_proxy
unset http_proxy
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
# export NCCL_BLOCKING_WAIT=0

#USE YOUR TOKEN: REMEMBER!!!
# WANDB_TOKENS=a77607626908409e45afa2ca225cf179e9a316fc
# wandb login --relogin $WANDB_TOKENS

EXP_NAME=Qwen_Critic_machine_pair_wise_96

# MODEL="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/Qwen_SFT_0625_baseline/checkpoints/global_step4500" 
# MODEL="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/Qwen_SFT_0625_baseline/checkpoints/global_step1500"
# MODEL="/data2/rlhf/yzy/OpenRLHF-new/outputs/reward_models/Qwen_SFT_0624_105_4"
MODEL="/data2/rlhf/lixs/open_models/Qwen2-7B-Instruct/"
File_NAME="Qwen_Critic_machine.jsonl" 
# MODEL="/data2/rlhf/yzy/open_models/models--Qwen--Qwen2-1.5B-Instruct/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"
# DATA="/data2/rlhf/yzy/data/rm_train/train_reason/test_critic"
DATA="/data2/rlhf/yzy/data/rm/pairwise_critic_inference/train/sample"
# DATA2="/data2/rlhf/yzy/data/rm/pairwise_critic_inference/test"

save_path="/data2/rlhf/yzy/OpenRLHF-new/outputs/inference/$EXP_NAME"
BATCH_SIZE_PER_GPU=8
LR=1e-5

hostfile="/data2/rlhf/yzy/OpenRLHF-new/run_scripts/0625/hostfile_96"
MACHINE_SIZE=$(wc -l < $hostfile)
WORLD_SIZE=$[MACHINE_SIZE * 8]

mkdir -p $save_path

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8049 --model $MODEL --served-model-name model_1 --tensor-parallel-size 4 > 8049.txt 2>&1 & 
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 8050 --model $MODEL --served-model-name model_2 --tensor-parallel-size 4 > 8050.txt 2>&1 & 

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
    --max_new_tokens 512 \
    --input_template Human \
    --output_path $save_path/$File_NAME
EOF

deepspeed $training_commands 2>&1 | tee $save_path/train_qwen.log
