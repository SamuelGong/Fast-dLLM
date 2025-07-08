"""qcache_baseline.py – vanilla vs. naive Q‑cache benchmark

Adds **output text printing (outside timing)** so you can eyeball whether the two
modes generate identical/acceptable answers. Still keeps model‑load outside the
measured section and avoids double‑loading into memory.

Run example:
    python qcache_baseline.py --question "What is diffusion?" --steps 128 --gen 128
"""

import argparse
from contextlib import contextmanager
import torch
from transformers import AutoTokenizer, AutoModel

from generate import (generate, generate_with_prefix_cache,
                      generate_with_dual_cache, generate_with_finegrained_cache,
                      generate_with_dual_cache_and_q_cache, generate_coarse_to_fine)
from torch.profiler import profile, ProfilerActivity
from model.modeling_llada import LLaDAModelLM

# ───────────────────────────────────────── config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"


# ─────────────────────────────────── timing helper
@contextmanager
def cuda_timer(label="Elapsed"):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    yield lambda: start.elapsed_time(end) / 1000  # returns seconds
    end.record(); torch.cuda.synchronize()
    print(f"{label}: {start.elapsed_time(end)/1000:.3f}s")

# ─────────────────────────────────── benchmark helper


def benchmark(prompt, tokenizer, *, steps, gen_len, block_len, use_kv_cache):
    tag = use_kv_cache
    print(f"\nLoading model for {tag} …")
    # model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE).eval()
    model = LLaDAModelLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE).eval()

    # warm‑up
    with torch.inference_mode():
        _ = model(prompt[:, :1]); torch.cuda.synchronize()

    # seq_len = prompt.shape[1] + gen_len
    # attach_qcache_monkey(model, prompt.shape[1] + gen_len) if use_qcache else None
    with cuda_timer(f"{tag}") as get_elapsed:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            if use_kv_cache == "None":
                out, nfe = generate(model, prompt, steps=steps, gen_length=gen_len,
                               block_length=block_len, temperature=0.,
                               remasking='low_confidence')
            elif use_kv_cache == "Prefix":
                out, nfe = generate_with_prefix_cache(model, prompt, steps=steps, gen_length=gen_len,
                               block_length=block_len, temperature=0.,
                               remasking='low_confidence')
            elif use_kv_cache == "Dual":
                out, nfe = generate_with_dual_cache(model, prompt, steps=steps, gen_length=gen_len,
                               block_length=block_len, temperature=0.,
                               remasking='low_confidence')
            elif use_kv_cache == "C2F":
                out, nfe = generate_coarse_to_fine(model, prompt, steps=steps, gen_length=gen_len,
                                                   block_length=block_len, temperature=0.,
                                                   remasking='low_confidence')
            elif use_kv_cache == "Fine":
                out, nfe = generate_with_finegrained_cache(model, prompt, steps=steps, gen_length=gen_len,
                               block_length=block_len, temperature=0.,
                               remasking='low_confidence')
            elif use_kv_cache == "DualAndQuery":
                out, nfe = generate_with_dual_cache_and_q_cache(model, prompt, steps=steps, gen_length=gen_len,
                               block_length=block_len, temperature=0.,
                               remasking='low_confidence')
    # decode and show (outside timing)
    answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
    print(f"{tag} output → {answer}\n")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))

    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    # print(prof.key_averages().table(row_limit=20))

    # free memory
    del model; torch.cuda.empty_cache()

# ─────────────────────────────────────── main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Explain diffusion models briefly.")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen", type=int, default=128)
    # ap.add_argument("--steps", type=int, default=512)
    # ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--block", type=int, default=4)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    prompt_txt = tokenizer.apply_chat_template([{"role": "user", "content": args.question}], add_generation_prompt=True, tokenize=False)
    prompt = torch.tensor(tokenizer(prompt_txt)["input_ids"], device=DEVICE).unsqueeze(0)

    # benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="None")
    # benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="Prefix")
    benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="Dual")
    # benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="Fine")
    # benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="DualAndQuery")
    benchmark(prompt, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="C2F")


if __name__ == "__main__":
    main()
