bash utility_profile.sh C2F --num_processes 4 --main_process_port 29500 --task humaneval --length 512
sleep 5
bash utility_profile.sh Dual --num_processes 4 --main_process_port 29500 --task humaneval --length 512
sleep 5
bash utility_profile.sh None --num_processes 4 --main_process_port 29500 --task humaneval --length 512