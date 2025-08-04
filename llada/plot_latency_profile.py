import json
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


input_file = "latency-profile.json"

def plot_methods_heatmap(
    method_data: dict,
    *,
    title: str = "",
    xlabel: str = "Steps",
    ylabel: str = "Block Length",
    cmap: str | None = None,
    output_path: str | None = None,
    dpi: int = 300,
    figsize_per_plot: tuple = (4, 4),
    annotate: bool = True,
    fmt: str = ".2f",
    annotation_fontsize: int = 8,
):
    """
    method_data: {method_name: {block_len: {steps: value}}}
    """

    # --- coerce numeric-like keys to ints (JSON turns them into strings) ---
    def _to_int_keys(nested):
        out = {}
        for r, cols in nested.items():
            try: r_i = int(r)
            except Exception: r_i = r
            out[r_i] = {}
            for c, v in cols.items():
                try: c_i = int(c)
                except Exception: c_i = c
                out[r_i][c_i] = v
        return out

    method_data = {m: _to_int_keys(n) for m, n in method_data.items()}
    methods = list(method_data.keys())
    n = len(methods)

    # union grid
    cols = sorted({c for nested in method_data.values() for c in nested})  # outer keys
    rows = sorted({r for nested in method_data.values() for rc in nested.values() for r in rc}, reverse=True)

    # aligned DataFrames + global color scale
    dfs = {}
    vmin, vmax = np.inf, -np.inf
    for m, nested in method_data.items():
        df = pd.DataFrame(nested).reindex(index=rows, columns=cols)
        dfs[m] = df
        if df.size and not df.isna().all().all():
            vmin = min(vmin, np.nanmin(df.values))
            vmax = max(vmax, np.nanmax(df.values))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("All heatmaps are empty (NaNs only). Check your data aggregation.")

    cmap_obj = plt.get_cmap(cmap or "viridis")
    denom = (vmax - vmin) if vmax > vmin else 1.0

    # luminance helper for contrast-aware text color
    def luminance(rgba):
        r, g, b = rgba[:3]
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    # plot
    fig, axes = plt.subplots(
        1, n,
        figsize=(figsize_per_plot[0] * n, figsize_per_plot[1]),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    im = None
    for idx, (ax, m) in enumerate(zip(axes, methods)):
        data = dfs[m].values.astype(float)
        data_masked = np.ma.masked_invalid(data)

        im = ax.imshow(data_masked, aspect='auto', cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_title(m)
        ax.set_xlabel(xlabel)
        if idx == 0:
            ax.set_ylabel(ylabel)
        else:
            # hide y-axis labels on all but the first subplot
            ax.set_ylabel(None)
            ax.tick_params(left=False, labelleft=False)

        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols)
        ax.set_yticks(range(len(rows)))
        if idx == 0:
            ax.set_yticklabels(rows)

        # annotate cells with contrast-aware text
        if annotate:
            for i in range(len(rows)):
                for j in range(len(cols)):
                    val = data[i, j]
                    if np.isfinite(val):
                        norm = (val - vmin) / denom
                        bg = cmap_obj(norm)
                        txt_color = "white" if luminance(bg) < 0.5 else "black"
                        ax.text(
                            j, i, format(val, fmt),
                            ha="center", va="center",
                            fontsize=annotation_fontsize,
                            color=txt_color,
                        )

    # shared colorbar + suptitle
    fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.03, pad=0.04)
    if title:
        fig.suptitle(title)

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved composite heatmap to {output_path}")
    else:
        plt.show()


def main():
    with open(input_file, 'r') as fin:
        result_dict = json.load(fin)

    # Only for visualization
    # how many (block_len, steps) cells per method?
    method_totals = {
        method: sum(len(steps_map) for entry in entries
                    for steps_map in entry["result"].values())
        for method, entries in result_dict.items()
    }
    overall_total = sum(method_totals.values())
    overall_pbar = tqdm(total=overall_total, desc="Evaluations", unit="cell", position=0)

    mean_latency_per_method = {}
    mean_perplexity_per_method = {}
    for method, entries in result_dict.items():
        # Step A: collect lists of latencies
        lat_accum = defaultdict(lambda: defaultdict(list))
        ppl_accum = defaultdict(lambda: defaultdict(list))
        method_pbar = tqdm(total=method_totals[method], desc=method, unit="cell",
                           leave=False, position=1)

        for entry in entries:
            for block_len, steps_map in entry["result"].items():
                for steps, stats in steps_map.items():
                    lat_accum[block_len][steps].append(stats["latency"])
                    ppl_accum[block_len][steps].append(stats["perplexity"])
        method_pbar.close()

        # Step B: compute means
        mean_lat_map = {}
        for block_len, steps_map in lat_accum.items():
            mean_lat_map[block_len] = {
                steps: sum(latencies) / len(latencies)
                for steps, latencies in steps_map.items()
            }
        mean_latency_per_method[method] = mean_lat_map

        mean_ppl_map = {}
        for block_len, steps_map in ppl_accum.items():
            mean_ppl_map[block_len] = {
                steps: sum(perplexities) / len(perplexities)
                for steps, perplexities in steps_map.items()
            }
        mean_perplexity_per_method[method] = mean_ppl_map
    overall_pbar.close()

    plot_methods_heatmap(
        mean_latency_per_method,
        title="All Methods Latency (s)",
        xlabel="Block Length",
        ylabel="Steps",
        output_path="latency-profile.png"
    )
    plot_methods_heatmap(
        mean_perplexity_per_method,
        title="All Methods Perplexity",
        xlabel="Block Length",
        ylabel="Steps",
        output_path="latency-profile-2.png"
    )


if __name__ == "__main__":
    main()
