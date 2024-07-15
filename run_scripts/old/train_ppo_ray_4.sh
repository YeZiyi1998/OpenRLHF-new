set -x
#export PATH=$HOME/.local/bin/:$PATH

unset https_proxy
unset http_proxy
WANDB_TOKENS=37eebd70bd7510b86c29171886dbc690d655c0b3

ptx_data="/data2/rlhf/lixs/data/ppo_train/ptx.jsonl"
prompt_data="/data2/rlhf/lixs/data/ppo_train/ppo_train_prompt_240520.jsonl"

reward_model_path="/data2/rlhf/lixs/outputs/reward_models/33B_RM_TEST"
sft_model_path="/data2/rlhf/jialian/rm_model/model/bc_33_3t"
work_dir=$(dirname $(dirname $(readlink -f "${BASH_SOURCE[0]}")))

echo $work_dir

EXP_NAME=33B_PPO_wo_PTX
save_path="/eb_data/rlhf/lixs/outputs/ppo_models/$EXP_NAME"
mkdir -p $save_path

run_env_json='{"working_dir": "PATH"}'
run_env_json=$(sed "s|PATH|${work_dir}|g" <<< $run_env_json)

ray stop
hostfile=./hostfile_rm4
head_node=$(head -1 $hostfile | awk -F " " '{print $1}')

ray start --head --port=6379 --include-dashboard=true --dashboard-host=0.0.0.0 --dashboard-port=8265

tail -n +2 $hostfile | while read line
do
  host=`echo $line | awk -F " " '{print $1}'`
  echo $host
  ssh -n root@$host 'source /data2/rlhf/lixs/l.sh && ray stop'
done

tail -n +2 $hostfile | while read line
do
  host=`echo $line | awk -F " " '{print $1}'`
  echo $host
  ssh -n root@$host "source /data2/rlhf/lixs/l.sh && ray start --address=$head_node:6379"
done

# 启动使用 vLLM 的 Ray PPO，默认配置需要 16 个 GPU
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"working_dir": "/data2/rlhf/lixs/code/OpenRLHF-new","excludes": ["*.log","*.wandb","*.pt"] }' \
    --no-wait \
    -- python3 examples/train_ppo_ray.py \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 4 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 8 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 8 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 4 \
    --pretrain $sft_model_path \
    --reward_pretrain $reward_model_path \
    --save_path $save_path \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 1024 \
    --generate_max_len 1024 \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --prompt_data $prompt_data \
    --max_samples 80000 \
    --normalize_reward \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --use_wandb $WANDB_TOKENS \
    --wandb_run_name $EXP_NAME
    #2>&1 | tee $save_path/train.log

#sleep 360000
#--pretrain_data $ptx_data \
#--flash_attn \