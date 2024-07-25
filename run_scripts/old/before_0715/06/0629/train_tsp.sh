export NCCL_DEBUG=INFO
python /data2/rlhf/yzy/write_email.py --info "cd ../0625 sh train_sft_qwen_tsp2.sh"
cd ../0625
sh train_sft_qwen_tsp2.sh
python /data2/rlhf/yzy/write_email.py --info "cd ../00701 sh train_sft_qwen_tsp2.sh"
cd ../0701
sh train_tsp.sh
