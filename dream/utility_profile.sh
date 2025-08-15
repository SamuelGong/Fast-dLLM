#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_simple.sh <method> [--length N] [--task NAME]
# Example: ./run_simple.sh C2F --length 256 --task mmlu_pro

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <method> [--length N] [--task NAME]"
  exit 1
fi

method="$1"
shift || true

# Defaults
length=128
task="mmlu"
model_path="Dream-org/Dream-v0-Instruct-7B"
model="dream"
script="eval_dream.py"
output_root="evals_results"
num_processes=1
main_process_port=29500
limit=''

# Optional flags: --length, --task
while [[ $# -gt 0 ]]; do
  case "$1" in
    --length|-l)
      length="${2:-}"; shift 2 ;;
    --task|-t)
      task="${2:-}"; shift 2 ;;
    --num_processes|-n)
      num_processes="${2:-}"; shift 2 ;;
    --main_process_port|-p)
      main_process_port="${2:-}"; shift 2 ;;
    --limit|-L)
      limit="${2:-}"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

# if the user omits --limit, $limit stays empty, no --limit flag is sent
limit_arg=()
if [[ -n "$limit" ]]; then
  limit_arg=(--limit "$limit")
fi

# Environment (as in your snippet)
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

sys_inst_arg=()
if [[ "$task" == "humaneval" ]]; then
  sys_inst_arg=(--system_instruction "Complete the following function by directly appending your answer to the original code. Do not leave any comment before or after the code.")
fi

# Loops: block_length (outer), steps (inner) over powers of two up to length
bl=1
while (( bl <= length )); do
  blocks=$(( length / bl ))   # number of blocks for this block_length
  st=1
  while (( st <= length )); do
    # Skip rule: skip if steps < #blocks
    if (( st < blocks )); then
#      echo "Skip: method=${method} task=${task} length=${length} bl=${bl} steps=${st} (< blocks=${blocks})"
      st=$(( st * 2 ))
      continue
    fi

    # If output directory already exists, skip this run
    out_dir="${output_root}/${task}/${method}/${length}/${bl}/${st}"
    if [[ -d "${out_dir}" ]]; then
      echo "Skip (exists): method=${method} task=${task} length=${length} bl=${bl} steps=${st} (blocks=${blocks})"
      st=$(( st * 2 ))
      continue
    fi

    echo "Run: method=${method} task=${task} length=${length} bl=${bl} steps=${st} (blocks=${blocks})"
    mkdir -p "${out_dir}"

    accelerate launch --num_processes "${num_processes}" --main_process_port "${main_process_port}" "${script}" --tasks "${task}" \
      --confirm_run_unsafe_code --model "${model}" \
      --model_args "pretrained=${model_path},max_new_tokens=${length},diffusion_steps=${st},block_length=${bl},use_kv_cache=${method},show_speed=True" \
      --output_path "${out_dir}" --log_samples \
      "${limit_arg[@]}" "${sys_inst_arg[@]}"

    st=$(( st * 2 ))
  done
  bl=$(( bl * 2 ))
done
