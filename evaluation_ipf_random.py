import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from simple_einet.einet import Einet

from pcn.methods.ipf import IPF
from utils import make_deterministic, set_smart_einet_config


def _to_padded_target_tensor(
    target_marginals: list[torch.Tensor], cardinality: int
) -> torch.Tensor:
    device = target_marginals[0].device
    target_tensor = torch.zeros((len(target_marginals), cardinality), device=device)
    for i, marginal in enumerate(target_marginals):
        target_tensor[i, : marginal.shape[0]] = marginal
    return target_tensor


def compute_average_error(
    einet: Einet,
    target_marginals: list[torch.Tensor],
    target_marginal_scopes: list[int],
    scaling_factors: torch.Tensor | None = None,
) -> float:
    cardinality = einet.config.leaf_kwargs["num_bins"]
    num_features = einet.config.num_features
    target_tensor = _to_padded_target_tensor(target_marginals, cardinality)
    current_marginals = torch.zeros_like(target_tensor)

    for i, scope in enumerate(target_marginal_scopes):
        marginalized_scopes = [k for k in range(num_features) if k != scope]
        for value in range(cardinality):
            input_tensor = torch.full(
                (1, num_features), float(value), dtype=torch.float32, device=target_tensor.device
            )
            current_marginals[i, value] = torch.exp(
                einet.forward(
                    input_tensor,
                    marginalized_scopes,
                    scaling_factors=scaling_factors,
                )
            ).item()

    return torch.mean(torch.abs(current_marginals - target_tensor)).item()


def generate_random_targets(
    size: int, cardinality: int, device: torch.device
) -> tuple[list[torch.Tensor], list[int]]:
    target_marginals = []
    for _ in range(size):
        marginal = torch.rand(cardinality, device=device)
        marginal = marginal / marginal.sum()
        target_marginals.append(marginal)
    target_scopes = list(range(size))
    return target_marginals, target_scopes


def save_results(output_dir: Path, result: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "ipf_random_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    csv_path = output_dir / "ipf_random_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate random IPF inputs of a given size and store average error and runtime."
    )
    parser.add_argument(
        "--size",
        type=int,
        required=True,
        help="Number of variables/features used for random IPF input generation.",
    )
    parser.add_argument(
        "--cardinality",
        type=int,
        default=4,
        help="Cardinality (number of discrete values) per variable.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="IPF convergence threshold.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum number of IPF iterations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device to run on.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/ipf_random"),
        help="Directory where results are stored.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size < 2:
        raise ValueError("--size must be >= 2.")
    if args.cardinality < 2:
        raise ValueError("--cardinality must be >= 2.")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available on this machine.")

    make_deterministic(args.seed, deterministic_cudnn=False)
    device = torch.device(args.device)

    einet_config = set_smart_einet_config(
        num_features=args.size,
        max_cardinality=args.cardinality,
    )
    einet = Einet(einet_config).to(device)

    target_marginals, target_marginal_scopes = generate_random_targets(
        size=args.size,
        cardinality=args.cardinality,
        device=device,
    )

    ipf = IPF(
        default_epsilon=args.epsilon,
        default_max_iterations=args.max_iterations,
        default_verbose=False,
    )

    initial_average_error = compute_average_error(
        einet=einet,
        target_marginals=target_marginals,
        target_marginal_scopes=target_marginal_scopes,
        scaling_factors=None,
    )

    start_time = time.perf_counter()
    scaling_factors, ipf_stats = ipf.compute_scaling_factors(
        einet=einet,
        target_marginals=target_marginals,
        target_marginal_scopes=target_marginal_scopes,
        return_stats=True,
    )
    runtime_seconds = time.perf_counter() - start_time

    average_error = compute_average_error(
        einet=einet,
        target_marginals=target_marginals,
        target_marginal_scopes=target_marginal_scopes,
        scaling_factors=scaling_factors,
    )

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "size": args.size,
        "cardinality": args.cardinality,
        "seed": args.seed,
        "epsilon": args.epsilon,
        "max_iterations": args.max_iterations,
        "initial_average_error": initial_average_error,
        "average_error": average_error,
        "ipf_num_iterations": ipf_stats["num_iterations"],
        "ipf_final_error": ipf_stats["final_error"],
        "ipf_iteration_errors": ipf_stats["iteration_errors"],
        "runtime_seconds": runtime_seconds,
    }

    save_results(args.output_dir, result)

    print("IPF random benchmark complete.")
    print(f"Result directory: {args.output_dir}")
    print(f"Average error: {average_error:.8f}")
    print(f"IPF iterations: {ipf_stats['num_iterations']}")
    print(f"IPF final error: {ipf_stats['final_error']:.8f}")
    print(f"Runtime (seconds): {runtime_seconds:.6f}")


if __name__ == "__main__":
    main()