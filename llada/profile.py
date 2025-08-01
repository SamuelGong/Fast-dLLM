import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

gen = 128  # how many tokens to generate
method_list = ["C2F", "Dual"]
question_list = [
    "Explain diffusion models briefly."
]
block_list = [2 ** n for n in range(1, 8)]  # how many tokens per block
steps_list = [2 ** n for n in range(1, 8)]  # how many generation steps in total
result_dict = {}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MODEL_NAME = "GSAI-ML/LLaDA-8B-Instruct"


def get_evaluation(prompt, answer, model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device=DEVICE, dtype=DTYPE)
    model.eval()

    enc = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt'
    )
    input_ids = enc.to(DEVICE)

    prompt_enc = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors='pt'
    )
    prompt_len = prompt_enc.shape[1]

    labels = input_ids.clone()
    labels[:, :prompt_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
    ppl = torch.exp(loss).item()
    return {
        "perplexity": ppl
    }


def main():
    from kvcache_baseline import benchmark

    experiment_name = "profile"
    output_file = f"{experiment_name}.json"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    result_dict = {}
    with open(output_file, "w", encoding="utf-8") as fout:
        for method in method_list:
            result_dict[method] = []
            for question in question_list:
                prompt_txt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": question}],
                    add_generation_prompt=True,
                    tokenize=False
                )
                prompt = torch.tensor(tokenizer(prompt_txt)["input_ids"], device=DEVICE).unsqueeze(0)
                result_dict[method].append({
                    "question": question,
                    "result": {}
                })

                for block in block_list:
                    result_dict[method][-1]["result"][block] = {}
                    for steps in steps_list:
                        if steps < block:
                            continue

                        lat, ans = benchmark(
                            prompt=prompt,
                            tokenizer=tokenizer,
                            steps=steps,
                            gen_len=gen,
                            block_len=block,
                            use_kv_cache=method,
                        )
                        result_dict[method][-1]["result"][block][steps] = {
                            "latency": lat,
                            "answer": ans
                        }
                        json.dump(result_dict[method], fout, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    main()
