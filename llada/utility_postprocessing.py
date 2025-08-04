import os
import glob
import json

task = "humaneval"
gen = 128
block_len_list = [2 ** n for n in range(0, 8)]  # how many tokens per block
steps_list = [2 ** n for n in range(0, 8)]  # how many generation steps in total
method_list = ["C2F", "Dual", "None"]
input_dir = "evals_results"
output_file = f"utility_profile_{task}.json"
model_name = "GSAI-ML/LLaDA-8B-Instruct"
model_name = model_name.replace("/", "__")

result_dict = {}
for method in method_list:
    result_dict[method] = {}
    for block_len in block_len_list:
        result_dict[method][block_len] = {}
        for steps in steps_list:
            result_prefix = os.path.join(
                input_dir,
                task,
                method,
                f"{gen}",
                f"{block_len}",
                f"{steps}",
                f"{model_name}"
            )

            pattern = os.path.join(result_prefix, "results_*.json")
            matches = glob.glob(pattern)
            if not matches:
                print(f"No results_*.json found under {result_prefix}")
                continue
            else:
                print(f"Processing {result_prefix}")

            # If multiple, choose the most recent; or use matches[0] if you want the first.
            input_path = max(matches, key=os.path.getmtime)
            with open(input_path, "r", encoding="utf-8") as f:
                input_json = json.load(f)
            extracted_result = input_json["results"][task]
            result_dict[method][block_len][steps] = extracted_result

with open(output_file, 'w', encoding="utf-8") as f:
    json.dump(result_dict, f, ensure_ascii=False, indent=4)
