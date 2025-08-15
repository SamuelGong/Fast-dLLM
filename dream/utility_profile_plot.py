import json
from tqdm.auto import tqdm
from collections import defaultdict
from latency_profile_plot import plot_methods_heatmap


task = "humaneval"
input_file = f"utility_profile_{task}.json"
task_metric_map = {
    "humaneval": "pass@1,create_test"
}


def main():
    with open(input_file, 'r') as fin:
        result_dict = json.load(fin)

    mean_metric_per_method = {}
    for method, entries in result_dict.items():
        # Step A: collect lists of latencies
        metric_accum = defaultdict(lambda: defaultdict(list))
        for block_len, steps_map in entries.items():
            for steps, stats in steps_map.items():
                metric_accum[block_len][steps].append(stats[f"{task_metric_map[task]}"])

        # Step B: compute means
        mean_map = {}
        for block_len, steps_map in metric_accum.items():
            mean_map[block_len] = {
                steps: sum(e) / len(e)
                for steps, e in steps_map.items()
            }
        mean_metric_per_method[method] = mean_map

    plot_methods_heatmap(
        mean_metric_per_method,
        title=f"Accuracy on {task}",
        xlabel="Block Length",
        ylabel="Steps",
        output_path=f"utility_profile_{task}.png"
    )


if __name__ == "__main__":
    main()
