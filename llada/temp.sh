bash utility_profile.sh C2F --num_processes 4 --main_process_port 29500 --task gsm8k --length 128
sleep 5
bash utility_profile.sh Dual --num_processes 4 --main_process_port 29500 --task gsm8k --length 128
sleep 5
bash utility_profile.sh None --num_processes 4 --main_process_port 29500 --task gsm8k --length 128