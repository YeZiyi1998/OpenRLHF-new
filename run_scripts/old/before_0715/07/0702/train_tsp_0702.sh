cd ../0701
sh train_sft_gen2_tsp.sh 1e-1
python write_email.py --info '0701 sh train_tsp.sh 1e-1'
sh train_sft_gen2_tsp.sh 1e-2
python write_email.py --info '0701 sh train_tsp.sh 1e-2'
