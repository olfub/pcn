import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np


DEFAULT_SIZES = [2, 4, 8, 16, 32, 64, 128]
RESULT_CSV_NAME = "ipf_random_results.csv"
RESULT_JSON_NAME = "ipf_random_results.json"


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def load_seed_result(seed_dir: Path) -> dict[str, float] | None:
    csv_path = seed_dir / RESULT_CSV_NAME
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader, None)
            if row is None:
                return None
            iteration_errors_raw = row.get("ipf_iteration_errors", "")
            iteration_errors: list[float] = []
            if iteration_errors_raw:
                try:
                    parsed = ast.literal_eval(iteration_errors_raw)
                except (SyntaxError, ValueError):
                    parsed = []
                if isinstance(parsed, list):
                    iteration_errors = [float(v) for v in parsed if np.isfinite(float(v))]

            second_last_iteration_error = np.nan
            if len(iteration_errors) >= 2:
                second_last_iteration_error = float(iteration_errors[-2])

            return {
                "initial_average_error": float(row["initial_average_error"]),
                "average_error": float(row["average_error"]),
                "runtime_seconds": float(row["runtime_seconds"]),
                "ipf_second_last_iteration_error": second_last_iteration_error,
                "ipf_num_iterations": float(row.get("ipf_num_iterations", np.nan)),
            }

    json_path = seed_dir / RESULT_JSON_NAME
    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            row = json.load(f)
            iteration_errors = row.get("ipf_iteration_errors", [])
            second_last_iteration_error = np.nan
            if isinstance(iteration_errors, list) and len(iteration_errors) >= 2:
                second_last_iteration_error = float(iteration_errors[-2])

            return {
                "initial_average_error": float(row["initial_average_error"]),
                "average_error": float(row["average_error"]),
                "runtime_seconds": float(row["runtime_seconds"]),
                "ipf_second_last_iteration_error": second_last_iteration_error,
                "ipf_num_iterations": float(row.get("ipf_num_iterations", np.nan)),
            }

    return None


def collect_seed_values(size_dir: Path, key: str) -> np.ndarray:
    if not size_dir.exists():
        return np.array([], dtype=float)

    seed_dirs = sorted(
        p for p in size_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")
    )
    if not seed_dirs:
        return np.array([], dtype=float)

    values: list[float] = []
    for seed_dir in seed_dirs:
        result = load_seed_result(seed_dir)
        if result is None:
            continue
        value = result.get(key)
        if value is None:
            continue
        if np.isfinite(value):
            values.append(float(value))

    return np.asarray(values, dtype=float)


def collect_stats(size_dir: Path, key: str) -> tuple[float, float]:
    values = collect_seed_values(size_dir, key)
    if values.size == 0:
        return np.nan, np.nan

    return float(np.mean(values)), float(np.std(values))


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


def auto_discover_sizes(results_root: Path) -> list[int]:
    sizes: list[int] = []
    for path in sorted(results_root.iterdir()):
        if not path.is_dir() or not path.name.startswith("size_"):
            continue
        suffix = path.name[len("size_") :]
        if suffix.isdigit():
            sizes.append(int(suffix))
    return sorted(set(sizes))


def print_latex_table(
    results_root: Path,
    sizes: list[int],
    precision: int,
    initial_error_scale_power: int,
    final_error_scale_power: int,
    runtime_scale_power: int,
    add_median_columns: bool,
) -> None:
    header_columns = [
        "Size",
        "#iterations",
        "final avg error",
        "runtime (s)",
    ]

    if add_median_columns:
        header_columns.extend(
            [
                "#iterations median",
                "final avg error median",
                "runtime median",
            ]
        )

    column_spec = "l" + "c" * (len(header_columns) - 1)

    print(r"\begin{tabular}{" + column_spec + "}")
    print(r"\toprule")
    print(" & ".join(latex_escape(column) for column in header_columns) + r" \\")
    print(r"\midrule")

    for size in sizes:
        size_dir = results_root / f"size_{size}"
        cells = [str(size)]
        final_mean, final_std = collect_stats(size_dir, "average_error")
        num_iter_mean, num_iter_std = collect_stats(size_dir, "ipf_num_iterations")
        runtime_mean, runtime_std = collect_stats(size_dir, "runtime_seconds")

        cells.append(format_cell(num_iter_mean, num_iter_std, precision, 0))
        cells.append(format_cell(final_mean, final_std, precision, final_error_scale_power))
        cells.append(format_cell(runtime_mean, runtime_std, precision, runtime_scale_power))

        if add_median_columns:
            final_values = collect_seed_values(size_dir, "average_error")
            num_iter_values = collect_seed_values(size_dir, "ipf_num_iterations")
            runtime_values = collect_seed_values(size_dir, "runtime_seconds")

            final_median = float(np.median(final_values)) if final_values.size > 0 else np.nan
            num_iter_median = (
                float(np.median(num_iter_values)) if num_iter_values.size > 0 else np.nan
            )
            runtime_median = float(np.median(runtime_values)) if runtime_values.size > 0 else np.nan

            cells.append(format_single_value(num_iter_median, precision, 0))
            cells.append(
                format_single_value(final_median, precision, final_error_scale_power)
            )
            cells.append(format_single_value(runtime_median, precision, runtime_scale_power))

        print(" & ".join(cells) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print a LaTeX table from random IPF benchmark results."
    )
    parser.add_argument(
        "--experimental_series",
        type=str,
        default="ipf_random",
        help="Name of experiment subfolder inside experiments/",
    )
    parser.add_argument(
        "--experiments_dir",
        type=Path,
        default=Path("experiments"),
        help="Root folder containing experimental series folders.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=DEFAULT_SIZES,
        help="Problem sizes to include as table rows.",
    )
    parser.add_argument(
        "--auto_discover_sizes",
        action="store_true",
        help="Discover sizes from existing result folders.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=2,
        help="Number of decimal places in mean/std values.",
    )
    parser.add_argument(
        "--initial_error_scale_power",
        type=int,
        default=0,
        help=(
            "Display initial average errors scaled by 10^initial_error_scale_power."
        ),
    )
    parser.add_argument(
        "--final_error_scale_power",
        type=int,
        default=8,
        help=(
            "Display final average errors scaled by 10^final_error_scale_power. "
            "Use caption legend like: final error values are in units of 10^{-8}."
        ),
    )
    parser.add_argument(
        "--runtime_scale_power",
        type=int,
        default=0,
        help="Display runtime values scaled by 10^runtime_scale_power.",
    )
    parser.add_argument(
        "--add_median_columns",
        action="store_true",
        help=(
            "Append median columns for initial/final error, 2nd last iteration error, "
            "iteration count, and runtime."
        ),
    )
    args = parser.parse_args()

    results_root = args.experiments_dir / args.experimental_series
    if not results_root.exists():
        raise FileNotFoundError(f"Results directory not found: {results_root}")

    sizes = args.sizes
    if args.auto_discover_sizes:
        discovered = auto_discover_sizes(results_root)
        if discovered:
            sizes = discovered

    print_latex_table(
        results_root=results_root,
        sizes=sizes,
        precision=args.precision,
        initial_error_scale_power=args.initial_error_scale_power,
        final_error_scale_power=args.final_error_scale_power,
        runtime_scale_power=args.runtime_scale_power,
        add_median_columns=args.add_median_columns,
    )


if __name__ == "__main__":
    main()
