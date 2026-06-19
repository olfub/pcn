import argparse
from pathlib import Path

import numpy as np


DEFAULT_GRAPH_PARAMS = [4, 5, 6, 7, 8]

MODEL_FILES = [
    ("bn learned", "bn_learned_l1.npy"),
    ("pcn", "pcn_l1.npy"),
]

INFERENCE_TIME_FILES = {
    "bn learned": "bn_learned_inference_times.npy",
    "pcn": "pcn_inference_times.npy",
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
    seed_dirs = sorted(
        p for p in experiment_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )
    if not seed_dirs:
        return np.array([], dtype=float)

    scores = [load_seed_score(seed_dir, file_name) for seed_dir in seed_dirs]
    scores = np.asarray(scores, dtype=float)
    return scores[np.isfinite(scores)]


def collect_experiment_stats(experiment_dir: Path, file_name: str) -> tuple[float, float]:
    if not experiment_dir.exists():
        return np.nan, np.nan

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


def auto_discover_graph_params(results_root: Path, graph_type: str) -> list[int]:
    prefix = f"{graph_type}_"
    graph_params: list[int] = []
    for path in sorted(results_root.iterdir()):
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        suffix = path.name[len(prefix) :]
        if suffix.isdigit():
            graph_params.append(int(suffix))
    return sorted(set(graph_params))


def print_latex_table(
    results_root: Path,
    graph_type: str,
    graph_params: list[int],
    precision: int,
    scale_power: int,
    add_median_columns: bool,
    add_inference_time_columns: bool,
) -> None:
    header_columns = ["Graph param"]
    header_columns.extend(model_label for model_label, _ in MODEL_FILES)

    if add_median_columns:
        header_columns.extend(f"{model_label} median" for model_label, _ in MODEL_FILES)

    if add_inference_time_columns:
        header_columns.extend(f"{model_label} inf. time" for model_label, _ in MODEL_FILES)

    column_spec = "l" + "c" * (len(header_columns) - 1)

    print(r"\begin{tabular}{" + column_spec + "}")
    print(r"\toprule")
    header = " & ".join(latex_escape(column) for column in header_columns) + r" \\"
    print(header)
    print(r"\midrule")

    for graph_param in graph_params:
        cells = [str(graph_param)]
        experiment_dir = results_root / f"{graph_type}_{graph_param}"

        for _, file_name in MODEL_FILES:
            mean, std = collect_experiment_stats(experiment_dir, file_name)
            cells.append(format_cell(mean, std, precision, scale_power))

        if add_median_columns:
            for _, file_name in MODEL_FILES:
                scores = collect_seed_scores(experiment_dir, file_name) if experiment_dir.exists() else np.array([], dtype=float)
                median = float(np.median(scores)) if scores.size > 0 else np.nan
                cells.append(format_single_value(median, precision, scale_power))

        if add_inference_time_columns:
            for model_label, _ in MODEL_FILES:
                time_file = INFERENCE_TIME_FILES[model_label]
                time_mean, time_std = collect_experiment_stats(experiment_dir, time_file)
                cells.append(format_cell_no_scaling(time_mean, time_std, precision))

        row = " & ".join(cells) + r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX table from worst-case graph evaluation results."
    )
    parser.add_argument(
        "--experimental_series",
        type=str,
        default="bn_for_workshop_grids",
        help="Name of experiment subfolder inside experiments/",
    )
    parser.add_argument(
        "--experiments_dir",
        type=Path,
        default=Path("experiments"),
        help="Root folder containing experimental series folders.",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        default="grid",
        choices=["grid", "fully"],
        help="Graph family prefix used in results folders (e.g., grid_4).",
    )
    parser.add_argument(
        "--graph_params",
        nargs="+",
        type=int,
        default=DEFAULT_GRAPH_PARAMS,
        help="Graph parameters to include as table rows.",
    )
    parser.add_argument(
        "--auto_discover_params",
        action="store_true",
        help="Discover graph parameters from existing result folders.",
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
        default=2,
        help=(
            "Display error values scaled by 10^scale_power. "
            "Use caption legend like: values are in units of 10^{-4}."
        ),
    )
    parser.add_argument(
        "--add_median_columns",
        action="store_true",
        help="Append two median columns (one per model) for L1 values.",
    )
    parser.add_argument(
        "--add_inference_time_columns",
        action="store_true",
        help="Append inference-time columns (seconds), unscaled.",
    )
    args = parser.parse_args()

    results_root = args.experiments_dir / args.experimental_series
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    graph_params = args.graph_params
    if args.auto_discover_params:
        discovered = auto_discover_graph_params(results_root, args.graph_type)
        if discovered:
            graph_params = discovered

    print_latex_table(
        results_root,
        args.graph_type,
        graph_params,
        args.precision,
        args.scale_power,
        args.add_median_columns,
        args.add_inference_time_columns,
    )


if __name__ == "__main__":
    main()