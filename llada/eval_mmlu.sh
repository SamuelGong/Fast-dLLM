# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

task=mmlu
length=128
block_length=2
steps=64

# c2f cache
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_kv_cache='C2F',show_speed=True \
--output_path evals_results/C2F/${length}/${block_length}/${steps}/${task} --log_samples

# dual cache
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},use_kv_cache='Dual',show_speed=True \
--output_path evals_results/Dual/${length}/${block_length}/${steps}/${task} --log_samples

# no cache
accelerate launch eval_llada.py --tasks ${task} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${length},block_length=${block_length},use_kv_cache='None',show_speed=True \
--output_path evals_results/None/${length}/${block_length}/${steps}/${task} --log_samples
