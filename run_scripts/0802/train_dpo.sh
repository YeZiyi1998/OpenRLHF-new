export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

#USE YOUR TOKEN: REMEMBER!!!
WANDB_TOKENS=a77607626908409e45afa2ca225cf179e9a316fc
wandb login --relogin $WANDB_TOKENS

data_path="../../../data/sub_data/pairwise_critic_inference2_get_answer/数学计算"

EXP_NAME=pairwise_get_answer_数学计算_s2
model_path="../../../open_models/Qwen2-7B-Instruct/"
BATCH_SIZE_PER_GPU=16
LR=9e-6
add_data="../../outputs/inference/Qwen_Critic_machine_llm_数学计算/Qwen_Critic_machine.jsonl"

#/eb_data
save_path="../../outputs/reward_models/$EXP_NAME"
ckpt_path="../../outputs/reward_models/$EXP_NAME/checkpoints"
MACHINE_SIZE=1
WORLD_SIZE=$[MACHINE_SIZE * 8]

#TODO-be-check: bs large, adamw
GLOBAL_BATCH_SIZE=$(( ${BATCH_SIZE_PER_GPU} * ${WORLD_SIZE} ))

mkdir -p $save_path
mkdir -p $ckpt_path

read -r -d '' training_commands <<EOF
../../examples/train_dpo.py \
     --save_path $save_path \
     --logging_steps 10 \
     --micro_train_batch_size $BATCH_SIZE_PER_GPU \
     --train_batch_size $GLOBAL_BATCH_SIZE \
     --pretrain $model_path \
     --bf16 \
     --max_epochs 1 \
     --max_len 2560 \
     --zero_stage 3 \
     --l2 0.0001 \
     --eval_steps 250 \
     --save_steps 500 \
     --beta 0.1 \
     --learning_rate $LR\
     --dataset $data_path \
     --dataset_probs 1 \
     --ckpt_path $ckpt_path \
     --flash_attn \
     --gradient_checkpointing \
     --use_wandb $WANDB_TOKENS \
     --wandb_run_name $EXP_NAME \
     --wandb_project rl_critic_test \
     --mode s2
EOF

deepspeed $training_commands 2>&1 | tee $save_path/train_qwen.log
# python /data2/rlhf/yzy/write_email.py --info Qwen_RM_critic_104_machine_done

     
     --learning_rate 5e-7 \
     --dataset Anthropic/hh-rlhf,tasksource/oasst1_pairwise_rlhf_reward,lmsys/chatbot_arena_conversations,openai/webgpt_comparisons \
     --dataset_probs 0.72,0.08,0.12,0.08 \
     --flash_attn \
     --gradient_checkpointing \
     --ref_offload