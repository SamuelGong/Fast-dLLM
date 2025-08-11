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
from torch.profiler import profile, ProfilerActivity
from model.modeling_dream import DreamModel
from model.generation_utils_block import DreamGenerationMixin
import types

import sys
sys.path.append('../')
from llada.kvcache_baseline import evaluate_qa

# ───────────────────────────────────────── config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_NAME = "Dream-org/Dream-v0-Instruct-7B"


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


def benchmark(inputs, tokenizer, *, steps, gen_len, block_len, use_kv_cache, debug=False):
    tag = use_kv_cache
    print(f"\nLoading model for {tag} …")
    # model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE).eval()
    model = DreamModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=DTYPE).to(DEVICE).eval()
    model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
    model._sample = types.MethodType(DreamGenerationMixin._sample, model)

    # warm‑up
    with torch.inference_mode():
        _ = model(
            inputs.input_ids[, :1],
            attention_mask=inputs.attention_mask[, :1]
        ); torch.cuda.synchronize()

    # seq_len = prompt.shape[1] + gen_len
    # attach_qcache_monkey(model, prompt.shape[1] + gen_len) if use_qcache else None
    with cuda_timer(f"{tag}") as get_elapsed:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            if use_kv_cache == "None":
                # out, nfe = generate(model, prompt, steps=steps, gen_length=gen_len,
                #                block_length=block_len, temperature=0.,
                #                remasking='low_confidence', tokenizer=tokenizer)
                output = model.diffusion_generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=gen_len,
                    output_history=True,
                    return_dict_in_generate=True,
                    steps=steps,
                    temperature=0.,
                    block_length=block_len,
                    # generation_tokens_hook_func=generation_tokens_hook_func
                )
            elif use_kv_cache == "Prefix":
                pass
                # out, nfe = generate_with_prefix_cache(model, prompt, steps=steps, gen_length=gen_len,
                #                block_length=block_len, temperature=0.,
                #                remasking='low_confidence', tokenizer=tokenizer)
            elif use_kv_cache == "Dual":
                pass
                # out, nfe = generate_with_dual_cache(model, prompt, steps=steps, gen_length=gen_len,
                #                block_length=block_len, temperature=0.,
                #                remasking='low_confidence', tokenizer=tokenizer)
            elif use_kv_cache == "C2F":
                pass
                # out, nfe = generate_coarse_to_fine(model, prompt, steps=steps, gen_length=gen_len,
                #                                    block_length=block_len, temperature=0.,
                #                                    remasking='low_confidence', tokenizer=tokenizer, debug=debug)

    # decode and show (outside timing)
    elapsed_seconds = get_elapsed()
    answer = tokenizer.decode(output.sequences[0][len(inputs.input_ids[0]):].tolist())
    answer = answer.split(tokenizer.eos_token)[0].strip()
    print(f"{tag} output → {answer}\n")
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))

    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    # print(prof.ke
    # _averages().table(row_limit=20))

    # free memory
    del model; torch.cuda.empty_cache()
    return elapsed_seconds, answer

# ─────────────────────────────────────── main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="Explain diffusion models briefly.")
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--gen", type=int, default=128)
    # ap.add_argument("--steps", type=int, default=512)
    # ap.add_argument("--gen", type=int, default=512)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": args.question}], add_generation_prompt=True)

    lat, answer = benchmark(inputs, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="None")
    # lat, answer = benchmark(inputs, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="Prefix")
    # lat, answer = benchmark(inputs, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block, use_kv_cache="Dual")
    # lat, answer = benchmark(inputs, tokenizer, steps=args.steps, gen_len=args.gen, block_len=args.block,
    #                         use_kv_cache="C2F", debug=args.debug)
    evaluation = evaluate_qa(args.question, answer)
    print(lat, evaluation)


if __name__ == "__main__":
    main()
