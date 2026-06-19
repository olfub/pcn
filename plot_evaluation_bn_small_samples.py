import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, ScalarFormatter


DEFAULT_GRAPHS = ["asia", "child", "insurance", "alarm"]
DEFAULT_SAMPLE_SIZES = [50, 100, 250, 500, 1000, 10000]

TITLE_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 17
TICK_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14


def load_seed_average(seed_dir: Path, file_name: str) -> float:
    file_path = seed_dir / file_name
    if not file_path.exists():
        return np.nan

    values = np.load(file_path)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan

    return float(np.nanmean(values))


def collect_sample_stats(sample_dir: Path, file_name: str) -> tuple[float, float]:
    if not sample_dir.exists():
        return np.nan, np.nan

    seed_dirs = sorted(
        p for p in sample_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )
    if not seed_dirs:
        return np.nan, np.nan

    seed_averages = np.asarray(
        [load_seed_average(seed_dir, file_name) for seed_dir in seed_dirs], dtype=float
    )
    seed_averages = seed_averages[np.isfinite(seed_averages)]
    if seed_averages.size == 0:
        return np.nan, np.nan

    return float(np.mean(seed_averages)), float(np.std(seed_averages))


def plot_graph(
    ax: plt.Axes,
    results_root: Path,
    graph: str,
    sample_sizes: list[int],
    pcn_file: str,
    pc_file: str,
    pcn_color: str,
    pc_color: str,
    show_ylabel: bool = True,
) -> None:
    pcn_means = []
    pcn_stds = []
    pc_means = []
    pc_stds = []

    for n in sample_sizes:
        sample_dir = results_root / graph / f"num_samples_{n}"

        pcn_mean, pcn_std = collect_sample_stats(sample_dir, pcn_file)
        pc_mean, pc_std = collect_sample_stats(sample_dir, pc_file)

        pcn_means.append(pcn_mean)
        pcn_stds.append(pcn_std)
        pc_means.append(pc_mean)
        pc_stds.append(pc_std)

    x = np.asarray(sample_sizes, dtype=float)
    pcn_means_arr = np.asarray(pcn_means, dtype=float)
    pcn_stds_arr = np.asarray(pcn_stds, dtype=float)
    pc_means_arr = np.asarray(pc_means, dtype=float)
    pc_stds_arr = np.asarray(pc_stds, dtype=float)

    ax.plot(
        x,
        pcn_means_arr,
        marker="o",
        linewidth=3.6,
        markersize=9,
        markeredgewidth=1.6,
        color=pcn_color,
        label="PCN",
    )
    ax.plot(
        x,
        pc_means_arr,
        marker="s",
        linewidth=3.6,
        markersize=9,
        markeredgewidth=1.6,
        color=pc_color,
        label="PC",
    )

    pcn_valid = np.isfinite(pcn_means_arr) & np.isfinite(pcn_stds_arr)
    if np.any(pcn_valid):
        ax.fill_between(
            x[pcn_valid],
            pcn_means_arr[pcn_valid] - pcn_stds_arr[pcn_valid],
            pcn_means_arr[pcn_valid] + pcn_stds_arr[pcn_valid],
            color=pcn_color,
            alpha=0.15,
        )

    pc_valid = np.isfinite(pc_means_arr) & np.isfinite(pc_stds_arr)
    if np.any(pc_valid):
        ax.fill_between(
            x[pc_valid],
            pc_means_arr[pc_valid] - pc_stds_arr[pc_valid],
            pc_means_arr[pc_valid] + pc_stds_arr[pc_valid],
            color=pc_color,
            alpha=0.15,
        )

    ax.set_title(graph, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("# samples", fontsize=AXIS_LABEL_FONT_SIZE)
    if show_ylabel:
        ax.set_ylabel("L1 error", fontsize=AXIS_LABEL_FONT_SIZE)
    else:
        ax.set_ylabel("")
    ax.grid(alpha=0.3)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", which="major", labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE, frameon=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot BN small-sample L1 errors in a 1x4 grid, comparing PCN and PC."
        )
    )
    parser.add_argument(
        "--experimental_series",
        type=str,
        default="bn_for_workshop_samples",
        help="Name of experiment subfolder inside experiments/.",
    )
    parser.add_argument(
        "--experiments_dir",
        type=Path,
        default=Path("experiments"),
        help="Root folder containing experimental series folders.",
    )
    parser.add_argument(
        "--graphs",
        nargs="+",
        default=DEFAULT_GRAPHS,
        help="Exactly 4 graph identifiers to plot in a 1x4 grid.",
    )
    parser.add_argument(
        "--sample_sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLE_SIZES,
        help="Sample sizes to show on the x-axis.",
    )
    parser.add_argument(
        "--pcn_file",
        type=str,
        default="pcn_l1.npy",
        help="Filename (within each seed folder) for PCN L1 errors.",
    )
    parser.add_argument(
        "--pc_file",
        type=str,
        default="einet_l1.npy",
        help="Filename (within each seed folder) for PC baseline L1 errors.",
    )
    parser.add_argument(
        "--pcn_color",
        type=str,
        default="tab:blue",
        help="Matplotlib color for PCN line.",
    )
    parser.add_argument(
        "--pc_color",
        type=str,
        default="tab:orange",
        help="Matplotlib color for PC line.",
    )
    parser.add_argument(
        "--figure_path",
        type=Path,
        default=Path("plots") / "bn_small_samples_l1_grid.pdf",
        help="Where to save the generated figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window after saving.",
    )
    args = parser.parse_args()

    if len(args.graphs) != 4:
        raise ValueError("Please provide exactly 4 graphs for the 1x4 subplot grid.")

    results_root = args.experiments_dir / args.experimental_series
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=False)
    fig.subplots_adjust(wspace=0.24)
    for idx, graph in enumerate(args.graphs):
        ax = axes[idx]
        plot_graph(
            ax=ax,
            results_root=results_root,
            graph=graph,
            sample_sizes=args.sample_sizes,
            pcn_file=args.pcn_file,
            pc_file=args.pc_file,
            pcn_color=args.pcn_color,
            pc_color=args.pc_color,
            show_ylabel=(idx == 0),
        )

    args.figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {args.figure_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()