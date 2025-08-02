import os
import argparse
import subprocess
from pathlib import Path
from itertools import product
from tqdm.auto import tqdm

# --- Environment ---
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "true"

POWERS = [1, 2, 4, 8, 16, 32, 64, 128]


def run_eval(method: str, task: str, length: int, block_length: int,
             steps_for_arg: int, steps_for_folder: int):
    """
    method: 'C2F', 'Dual', or 'None'
    steps_for_arg:    value passed to --model_args steps=...
    steps_for_folder: value used in the output_path folder name
    """
    model_args = [
        "model_path='GSAI-ML/LLaDA-8B-Instruct'",
        f"gen_length={length}",
        f"steps={steps_for_arg}",
        f"block_length={block_length}",
        f"use_kv_cache='{method}'"
        "show_speed=True",
    ]
    out_dir = Path(f"evals_results/{method}/{length}/{block_length}/{steps_for_folder}/{task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "accelerate", "launch", "eval_llada.py",
        "--tasks", task,
        "--confirm_run_unsafe_code",
        "--model", "llada_dist",
        "--model_args", ",".join(model_args),
        "--output_path", str(out_dir),
        "--log_samples",
    ]
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description="Grid runner with configurable methods and tqdm progress.")
    p.add_argument("--task", default="mmlu")
    p.add_argument("--length", type=int, default=128)
    p.add_argument(
        "-m", "--methods",
        nargs="+",
        choices=["C2F", "Dual", "None"],
        default=["C2F", "Dual", "None"],
        help="One or more methods to run."
    )
    p.add_argument(
        "--block",
        nargs="*",
        type=int,
        default=POWERS,
        help="Block lengths to try (default: powers of two up to 128)."
    )
    p.add_argument(
        "--steps",
        nargs="*",
        type=int,
        default=POWERS,
        help="Steps values to try (default: powers of two up to 128)."
    )
    args = p.parse_args()

    # Build valid (block_length, steps) pairs based on skip rule: keep if (length // bl) <= st
    valid_pairs = [
        (bl, st)
        for bl, st in product(args.block_lengths, args.steps)
        if (args.length // bl) <= st
    ]
    if not valid_pairs:
        print("No (block_length, steps) pairs satisfy the skip rule. Nothing to run.")
        return

    total_cmds = len(valid_pairs) * len(args.methods)

    with tqdm(total=total_cmds, desc="Total runs") as pbar:
        for bl, st in tqdm(valid_pairs, desc="Experiment pairs", leave=False):
            for method in args.methods:
                # Determine effective steps passed to model
                if method == "None":
                    steps_for_arg = args.length if args.none_steps_arg == "length" else st
                else:
                    steps_for_arg = st

                # Folder name: keep the loop's `steps` for consistency with your original layout
                run_eval(method, args.task, args.length, bl, steps_for_arg, steps_for_folder=st)
                pbar.update(1)


if __name__ == "__main__":
    main()
