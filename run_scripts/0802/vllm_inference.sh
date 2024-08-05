export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0,1,2,3

EXP_NAME=Qwen_Critic_machine_llm_知识问答

MODEL="../../../open_models/Qwen2-7B-Instruct/"
File_NAME="Qwen_Critic_machine.jsonl" 
DATA="../../../data/sub_data/pairwise_critic_inference2_get_answer/知识问答"
exist_prompt="/cpfs/29f69eb5e2e60f26/user/rlhf/yzy/OpenRLHF-new/outputs/07/0727/inference/Qwen_Critic_machine_数学计算_get_answer/Qwen_Critic_machine.filter.jsonl"

save_path="../../outputs/inference/${EXP_NAME}"
BATCH_SIZE_PER_GPU=3
LR=1e-5

MACHINE_SIZE=1
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
    --zero_stage 0 \
    --tp_size 4 \
    --micro_batch_size $BATCH_SIZE_PER_GPU \
    --flash_attn \
    --max_new_tokens 512 \
    --input_template Human \
    --tp_size 4 \
    --exist_prompt $exist_prompt \
    --output_path $save_path/$File_NAME
EOF

python $training_commands 2>&1 | tee $save_path/train_qwen.log
# python /data2/rlhf/yzy/write_email.py --info Qwen_RM_critic_104_machine_done

