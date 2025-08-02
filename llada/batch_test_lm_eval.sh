#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run_simple.sh <method> [--length N] [--task NAME]
# Example: ./run_simple.sh C2F --length 256 --task mmlu_pro

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <method> [--length N] [--task NAME]"
  exit 1
fi

method="$1"
shift || true

# Defaults
length=128
task="mmlu"
model_path="GSAI-ML/LLaDA-8B-Instruct"
model="llada_dist"
script="eval_llada.py"
output_root="evals_results"

# Optional flags: --length, --task
while [[ $# -gt 0 ]]; do
  case "$1" in
    --length|-l)
      length="${2:-}"; shift 2 ;;
    --task|-t)
      task="${2:-}"; shift 2 ;;
    *)
      echo "Unknown argument: $1"
      exit 1 ;;
  esac
done

# Environment (as in your snippet)
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

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

    echo "Run: method=${method} task=${task} length=${length} bl=${bl} steps=${st} (blocks=${blocks})"
    out_dir="${output_root}/${task}/${method}/${length}/${bl}/${st}"
    mkdir -p "${out_dir}"

    accelerate launch "${script}" --tasks "${task}" \
      --confirm_run_unsafe_code --model "${model}" \
      --model_args "model_path='${model_path}',gen_length=${length},steps=${st},block_length=${bl},use_kv_cache='${method}',show_speed=True" \
      --output_path "${out_dir}" --log_samples

    st=$(( st * 2 ))
  done
  bl=$(( bl * 2 ))
done
