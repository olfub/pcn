import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATASETS = [
	"asia",
	"child",
	"alarm",
	"insurance",
	"win95pts",
	"hepar2",
	"hailfinder",
	"water",
	"barley",
	"mildew",
]

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


def collect_seed_scores(dataset_dir: Path, file_name: str) -> np.ndarray:
	seed_dirs = sorted(
		p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")
	)
	if not seed_dirs:
		return np.array([], dtype=float)

	scores = [load_seed_score(seed_dir, file_name) for seed_dir in seed_dirs]
	scores = np.asarray(scores, dtype=float)
	return scores[np.isfinite(scores)]


def collect_dataset_stats(
	dataset_dir: Path, file_name: str
) -> tuple[float, float]:
	scores = collect_seed_scores(dataset_dir, file_name)

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


def print_latex_table(
	results_root: Path,
	datasets: list[str],
	precision: int,
	scale_power: int,
	add_median_columns: bool,
	add_inference_time_columns: bool,
) -> None:
	header_columns = ["Dataset"]
	header_columns.extend(model_label for model_label, _ in MODEL_FILES)

	if add_median_columns:
		header_columns.extend(
			f"{model_label} median" for model_label, _ in MODEL_FILES
		)

	if add_inference_time_columns:
		header_columns.extend(
			f"{model_label} inf. time" for model_label, _ in MODEL_FILES
		)

	column_spec = "l" + "c" * (len(header_columns) - 1)

	print(r"\begin{tabular}{" + column_spec + "}")
	print(r"\toprule")
	header = " & ".join(latex_escape(column) for column in header_columns) + " \\\\"
	print(header)
	print(r"\midrule")

	for dataset in datasets:
		cells = [latex_escape(dataset)]
		dataset_dir = results_root / dataset

		for model_label, file_name in MODEL_FILES:
			mean, std = collect_dataset_stats(dataset_dir, file_name)
			cells.append(format_cell(mean, std, precision, scale_power))

		if add_median_columns:
			for _, file_name in MODEL_FILES:
				scores = collect_seed_scores(dataset_dir, file_name)
				median = float(np.median(scores)) if scores.size > 0 else np.nan
				cells.append(format_single_value(median, precision, scale_power))

		if add_inference_time_columns:
			for model_label, _ in MODEL_FILES:
				time_file = INFERENCE_TIME_FILES[model_label]
				time_mean, time_std = collect_dataset_stats(dataset_dir, time_file)
				cells.append(format_cell_no_scaling(time_mean, time_std, precision))

		row = " & ".join(cells) + " \\\\"
		print(row)

	print(r"\bottomrule")
	print(r"\end{tabular}")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Print a LaTeX table from BN evaluation results."
	)
	parser.add_argument(
		"--experimental_series",
		type=str,
		default="bn_for_workshop",
		help="Name of experiment subfolder inside experiments/",
	)
	parser.add_argument(
		"--experiments_dir",
		type=Path,
		default=Path("experiments"),
		help="Root folder containing experimental series folders.",
	)
	parser.add_argument(
		"--datasets",
		nargs="+",
		default=DEFAULT_DATASETS,
		help="Datasets to include as table columns.",
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
			"Display values scaled by 10^scale_power. "
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
		help=(
			"Append two inference-time columns (one per model). "
			"Inference times are not scaled."
		),
	)
	args = parser.parse_args()

	results_root = args.experiments_dir / args.experimental_series
	if not results_root.exists():
		raise FileNotFoundError(f"Results directory not found: {results_root}")

	print_latex_table(
		results_root,
		args.datasets,
		args.precision,
		args.scale_power,
		args.add_median_columns,
		args.add_inference_time_columns,
	)


if __name__ == "__main__":
	main()
