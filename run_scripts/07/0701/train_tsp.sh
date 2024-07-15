sh train_sft_gen2_tsp.sh 1e-6
python write_email.py --info '0701 sh train_tsp.sh 1e-6'
sh train_sft_gen2_tsp.sh 1e-7
python write_email.py --info '0701 sh train_tsp.sh 1e-7'
sh train_sft_gen2_tsp.sh 1e-5
python write_email.py --info '0701 sh train_tsp.sh 1e-5'
sh train_sft_gen2_tsp.sh 1e-4
python write_email.py --info '0701 sh train_tsp.sh 1e-4'
cd ../0625
conda activate yzy_torch_2.2
sh train_sft_qwen_tsp2.sh
