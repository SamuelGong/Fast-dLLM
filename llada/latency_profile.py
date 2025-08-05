import os
import json
import torch
from tqdm.auto import tqdm
from kvcache_baseline import benchmark, evaluate_qa
from transformers import AutoTokenizer

# gen = 128
# block_len_list = [2 ** n for n in range(0, 8)]  # how many tokens per block
# steps_list = [2 ** n for n in range(0, 8)]  # how many generation steps in total

gen = 512  # how many tokens to generate
block_len_list = [2 ** n for n in range(0, 10)]  # how many tokens per block
steps_list = [2 ** n for n in range(0, 10)]  # how many generation steps in total

method_list = ["C2F", "Dual", "None"]
question_list = [
    "Explain diffusion models briefly.",
    "What is Retrieval-Augmented Generation?",
    "When will human realize Artificial General Intelligence?"
]
result_dict = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"


def _cells_per_method(gen, block_len_list, steps_list, num_questions: int) -> int:
    """How many (block_len, steps) cells will be processed for one method?"""
    per_question = sum(len([s for s in steps_list if s >= gen // bl]) for bl in block_len_list)
    return num_questions * per_question


def main():
    experiment_name = "profile"
    output_file = f"{experiment_name}.json"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # --- progress bars: totals ---
    total_per_method = _cells_per_method(gen, block_len_list, steps_list, len(question_list))
    overall_total = total_per_method * len(method_list)
    overall_pbar = tqdm(total=overall_total, desc="Total", unit="cell", position=0)

    if os.path.isfile(output_file):
        with open(output_file, "r", encoding="utf-8") as fin:
            result_dict = json.load(fin)
    else:
        result_dict = {}
    for method in method_list:
        method_pbar = tqdm(total=total_per_method, desc=method, unit="cell", position=1, leave=False)

        if method not in result_dict:
            result_dict[method] = []
        for qid, question in enumerate(question_list):
            prompt_txt = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                add_generation_prompt=True,
                tokenize=False
            )
            prompt = torch.tensor(tokenizer(prompt_txt)["input_ids"], device=DEVICE).unsqueeze(0)
            if len(result_dict[method]) == qid:
                result_dict[method].append({
                    "question": question,
                    "result": {}
                })

            for block_len in block_len_list:
                num_blocks = gen // block_len

                if block_len not in result_dict[method][qid]["result"]:
                    result_dict[method][qid]["result"][block_len] = {}
                for steps in steps_list:
                    if steps < num_blocks:
                        continue
                    if steps in result_dict[method][qid]["result"][block_len]:
                        print(f"Skipping steps {steps} for block_len {block_len}, qid {qid}, method {method}")
                        continue

                    lat, answer = benchmark(
                        prompt=prompt,
                        tokenizer=tokenizer,
                        steps=steps,
                        gen_len=gen,
                        block_len=block_len,
                        use_kv_cache=method,
                    )

                    evaluation = evaluate_qa(
                        question=question,
                        answer=answer,
                    )
                    result_dict[method][qid]["result"][block_len][steps] = {
                        "latency": lat,
                        "answer": answer,
                        "perplexity": evaluation["perplexity"]
                    }
                    with open(output_file, "w", encoding="utf-8") as fout:
                        json.dump(result_dict, fout, ensure_ascii=False, indent=4)
                        fout.flush()

                    # update progress bars
                    method_pbar.update(1)
                    overall_pbar.update(1)
                    method_pbar.set_postfix(blk=block_len, steps=steps, lat=f"{lat:.3f}",
                                            ppl=f"{evaluation['perplexity']:.1f}")
        method_pbar.close()
    overall_pbar.close()


if __name__ == '__main__':
    main()
