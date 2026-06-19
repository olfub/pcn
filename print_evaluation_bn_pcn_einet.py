import argparse
from pathlib import Path

import numpy as np


DEFAULT_GRAPHS = [
    "asia",
    "child",
    "insurance",
    "alarm",
]

DEFAULT_SAMPLE_SIZES = [50, 100, 250, 500, 1000, 10000]

MODEL_FILES = [
    ("pcn", "pcn_l1.npy"),
    ("einet", "einet_l1.npy"),
]

INFERENCE_TIME_FILES = {
    "pcn": "pcn_inference_times.npy",
    "einet": "einet_inference_times.npy",
    "true bn": "true_bn_inference_times.npy",
}

TRAINING_TIME_FILES = {
    "pcn": "pcn_training_time.npy",
    "einet": "einet_training_time.npy",
}


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def load_seed_score(seed_dir: Path, file_name: str) -> float:
    file_path = seed_dir / file_name
    if not file_path.exists():
        return np.nan

    values = np.load(file_path)
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan

    return float(np.nanmean(values))


def collect_seed_scores(experiment_dir: Path, file_name: str) -> np.ndarray:
    if not experiment_dir.exists():
        return np.array([], dtype=float)

    seed_dirs = sorted(
        p for p in experiment_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )
    if not seed_dirs:
        return np.array([], dtype=float)

    scores = [load_seed_score(seed_dir, file_name) for seed_dir in seed_dirs]
    scores = np.asarray(scores, dtype=float)
    return scores[np.isfinite(scores)]


def collect_experiment_stats(experiment_dir: Path, file_name: str) -> tuple[float, float]:
    scores = collect_seed_scores(experiment_dir, file_name)
    if scores.size == 0:
        return np.nan, np.nan

    return float(np.mean(scores)), float(np.std(scores))


def format_cell(mean: float, std: float, precision: int, scale_power: int) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "--"

    scale = 10**scale_power
    mean_scaled = mean * scale
    std_scaled = std * scale
    return f"{mean_scaled:.{precision}f} $\\pm$ {std_scaled:.{precision}f}"


def format_single_value(value: float, precision: int, scale_power: int) -> str:
    if not np.isfinite(value):
        return "--"

    value_scaled = value * (10**scale_power)
    return f"{value_scaled:.{precision}f}"


def format_cell_no_scaling(mean: float, std: float, precision: int) -> str:
    if not np.isfinite(mean) or not np.isfinite(std):
        return "--"

    return f"{mean:.{precision}f} $\\pm$ {std:.{precision}f}"


def auto_discover_graphs(results_root: Path) -> list[str]:
    return sorted([p.name for p in results_root.iterdir() if p.is_dir()])


def auto_discover_sample_sizes(results_root: Path, graphs: list[str]) -> list[int]:
    sample_sizes: set[int] = set()

    for graph in graphs:
        graph_dir = results_root / graph
        if not graph_dir.exists():
            continue

        for path in graph_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("num_samples_"):
                continue
            suffix = path.name[len("num_samples_") :]
            if suffix.isdigit():
                sample_sizes.add(int(suffix))

    return sorted(sample_sizes)


def print_latex_table(
    results_root: Path,
    graphs: list[str],
    sample_sizes: list[int],
    precision: int,
    scale_power: int,
    add_median_columns: bool,
    add_inference_time_columns: bool,
    add_training_time_columns: bool,
) -> None:
    header_columns = ["Graph", "Samples"]
    header_columns.extend(model_label for model_label, _ in MODEL_FILES)

    if add_median_columns:
        header_columns.extend(f"{model_label} median" for model_label, _ in MODEL_FILES)

    if add_inference_time_columns:
        header_columns.extend(
            f"{model_label} inf. time" for model_label in INFERENCE_TIME_FILES
        )

    if add_training_time_columns:
        header_columns.extend(f"{model_label} train time" for model_label in TRAINING_TIME_FILES)

    column_spec = "ll" + "c" * (len(header_columns) - 2)

    print(r"\begin{tabular}{" + column_spec + "}")
    print(r"\toprule")
    print(" & ".join(latex_escape(column) for column in header_columns) + r" \\")
    print(r"\midrule")

    for graph in graphs:
        for sample_size in sample_sizes:
            experiment_dir = results_root / graph / f"num_samples_{sample_size}"
            cells = [latex_escape(graph), str(sample_size)]

            for _, file_name in MODEL_FILES:
                mean, std = collect_experiment_stats(experiment_dir, file_name)
                cells.append(format_cell(mean, std, precision, scale_power))

            if add_median_columns:
                for _, file_name in MODEL_FILES:
                    scores = collect_seed_scores(experiment_dir, file_name)
                    median = float(np.median(scores)) if scores.size > 0 else np.nan
                    cells.append(format_single_value(median, precision, scale_power))

            if add_inference_time_columns:
                for time_file in INFERENCE_TIME_FILES.values():
                    time_mean, time_std = collect_experiment_stats(experiment_dir, time_file)
                    cells.append(format_cell_no_scaling(time_mean, time_std, precision))

            if add_training_time_columns:
                for time_file in TRAINING_TIME_FILES.values():
                    time_mean, time_std = collect_experiment_stats(experiment_dir, time_file)
                    cells.append(format_cell_no_scaling(time_mean, time_std, precision))

            print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX table from BN small-sample PCN vs EInet results."
    )
    parser.add_argument(
        "--experimental_series",
        type=str,
        default="bn_for_workshop_samples",
        help="Name of experiment subfolder inside experiments/",
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
        help="Graph identifiers to include as rows.",
    )
    parser.add_argument(
        "--sample_sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SAMPLE_SIZES,
        help="Sample sizes to include as rows.",
    )
    parser.add_argument(
        "--auto_discover_graphs",
        action="store_true",
        help="Discover graph folders from results_root.",
    )
    parser.add_argument(
        "--auto_discover_sample_sizes",
        action="store_true",
        help="Discover sample sizes across selected graphs from results_root.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Number of decimal places in mean/std values.",
    )
    parser.add_argument(
        "--scale_power",
        type=int,
        default=4,
        help=(
            "Display error values scaled by 10^scale_power. "
            "Use caption legend like: values are in units of 10^{-4}."
        ),
    )
    parser.add_argument(
        "--add_median_columns",
        action="store_true",
        help="Append median columns (one per model) for L1 values.",
    )
    parser.add_argument(
        "--add_inference_time_columns",
        action="store_true",
        help="Append inference-time columns (seconds), unscaled.",
    )
    parser.add_argument(
        "--add_training_time_columns",
        action="store_true",
        help="Append training-time columns (seconds), unscaled.",
    )
    args = parser.parse_args()

    results_root = args.experiments_dir / args.experimental_series
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    graphs = args.graphs
    if args.auto_discover_graphs:
        discovered_graphs = auto_discover_graphs(results_root)
        if discovered_graphs:
            graphs = discovered_graphs

    sample_sizes = args.sample_sizes
    if args.auto_discover_sample_sizes:
        discovered_sample_sizes = auto_discover_sample_sizes(results_root, graphs)
        if discovered_sample_sizes:
            sample_sizes = discovered_sample_sizes

    print_latex_table(
        results_root,
        graphs,
        sample_sizes,
        args.precision,
        args.scale_power,
        args.add_median_columns,
        args.add_inference_time_columns,
        args.add_training_time_columns,
    )


if __name__ == "__main__":
    main()
