import argparse
import random
import time
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty

import numpy as np
import torch
from pgmpy.inference import VariableElimination
from rtpt import RTPT
from simple_einet.einet import Einet

import wandb
from datasets.bn_models import get_problem
from pcn.methods.ipf import IPF
from pcn.pcn import PCN
from utils import make_deterministic, set_smart_einet_config


def evaluate_model(
    identifier,
    seed,
    num_samples,
    pcn_inference_pruning,
    experimental_series,
    max_epochs,
):
    eval_start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    folder_path = (
        Path("experiments")
        / experimental_series
        / identifier
        / f"num_samples_{num_samples}"
        / f"seed_{seed}"
    )
    folder_path.mkdir(parents=True, exist_ok=True)
    short_exp_str = f"{experimental_series}_{identifier}_n{num_samples}_seed{seed}"

    wandb.init(
        project="PCN",
        name=f"{short_exp_str}_pcn_vs_einet",
        config={
            "experimental_series": experimental_series,
            "identifier": identifier,
            "seed": seed,
            "num_samples": num_samples,
            "pcn_inference_pruning": pcn_inference_pruning,
            "baseline": "single_einet",
        },
    )

    make_deterministic(seed, False)
    print(f"Running evaluation for {short_exp_str} on {device}")

    supported_identifiers = [
        "asia",
        "child",
        "alarm",
        "win95pts",
        "insurance",
        "hepar2",
        "hailfinder",
        "water",
        "barley",
        "mildew",
    ]

    if identifier not in supported_identifiers:
        raise ValueError(
            f"Unknown identifier {identifier}. Supported: {', '.join(supported_identifiers)}."
        )

    problem = get_problem(identifier, n_samples=num_samples, seed=seed)
    bnh = problem["model"]
    names = problem["names"]
    samples = problem["samples"]
    samples_tensor = torch.tensor(samples).to(device)

    node_cardinalities = {
        name: int(np.max(samples[:, i]) + 1) for i, name in enumerate(names)
    }

    # Train/load PCN.
    pcn = PCN(
        bnh.graph, samples_tensor, names, node_cardinalities=node_cardinalities
    ).to(device)
    pcn_model_path = folder_path / "pcn_model.pth"

    if pcn_model_path.exists():
        print(f"Loading PCN model from {pcn_model_path}")
        pcn.load_state_dict(torch.load(pcn_model_path, map_location=device))
        pcn.set_marginals(samples_tensor, names)
        pcn_training_time = 0.0
    else:
        data_dict = pcn.prepare_data(samples_tensor, names)
        optimizer = torch.optim.Adam(pcn.parameters(), lr=0.001)
        rtpt = RTPT(
            name_initials="FB",
            experiment_name=f"PCN {short_exp_str}",
            max_iterations=max_epochs,
        )
        rtpt.start()
        start = time.time()
        best_loss = float("inf")
        patience = 100
        patience_counter = 0

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = -pcn.forward(samples_tensor, names, data_dict=data_dict).sum()
            loss.backward()
            optimizer.step()
            wandb.log({"pcn_train_loss": loss.item()}, step=epoch)
            if (epoch + 1) % 100 == 0:
                print(f"PCN Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.4f}")
            rtpt.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"PCN early stopping at epoch {epoch+1}")
                    break

        pcn_training_time = time.time() - start
        print(f"PCN training time: {pcn_training_time:.2f} seconds")
        torch.save(pcn.state_dict(), pcn_model_path)
        pcn.set_marginals(samples_tensor, names)

    wandb.log({"pcn_training_time": pcn_training_time})

    # Train/load single EInet baseline over all variables.
    max_cardinality = max(node_cardinalities.values())
    einet = Einet(
        set_smart_einet_config(len(names), max_cardinality=max_cardinality)
    ).to(device)
    einet_model_path = folder_path / "single_einet_model.pth"

    if einet_model_path.exists():
        print(f"Loading single EInet model from {einet_model_path}")
        einet.load_state_dict(torch.load(einet_model_path, map_location=device))
        einet_training_time = 0.0
    else:
        optimizer = torch.optim.Adam(einet.parameters(), lr=0.001)
        rtpt = RTPT(
            name_initials="FB",
            experiment_name=f"SingleEInet {short_exp_str}",
            max_iterations=max_epochs,
        )
        rtpt.start()
        start = time.time()
        best_loss = float("inf")
        patience = 100
        patience_counter = 0

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            loss = -einet.forward(samples_tensor).sum()
            loss.backward()
            optimizer.step()
            wandb.log({"einet_train_loss": loss.item()}, step=epoch)
            if (epoch + 1) % 100 == 0:
                print(
                    f"Single EInet Epoch [{epoch+1}/{max_epochs}], Loss: {loss.item():.4f}"
                )
            rtpt.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Single EInet early stopping at epoch {epoch+1}")
                    break

        einet_training_time = time.time() - start
        print(f"Single EInet training time: {einet_training_time:.2f} seconds")
        torch.save(einet.state_dict(), einet_model_path)

    wandb.log({"einet_training_time": einet_training_time})

    infer_true = VariableElimination(bnh.get_model())

    timeout_indices = {
        "true_bn": [],
    }
    predictions = {
        "pcn": [],
        "einet": [],
        "true_bn": [],
    }
    errors = {
        "pcn_l1": [],
        "einet_l1": [],
        "pcn_l2": [],
        "einet_l2": [],
    }
    inference_times = {
        "pcn": [],
        "einet": [],
        "true_bn": [],
    }

    num_queries = 100
    evidence_prob = 1 / 3
    all_queries = []
    query_samples = bnh.sample(num_queries, names)

    for i in range(num_queries):
        query_variables = {}
        valid = False
        while not valid:
            for name in names:
                if np.random.rand() < evidence_prob:
                    query_variables[name] = int(query_samples[i, names.index(name)])
            # skip marginalizing everything (probability is always 1)
            if len(query_variables) > 1:
                valid = True

        target_variable = random.choice(list(query_variables.keys()))
        target_value = query_variables[target_variable]
        del query_variables[target_variable]
        all_queries.append(({target_variable: target_value}, query_variables))

    for count, (target_query, query_variables) in enumerate(all_queries):
        print(
            f"Evaluating query {count} for: {target_query} with evidence {query_variables}"
        )

        # True BN
        q = Queue()
        start = time.time()
        p = Process(target=run_ve, args=(q, infer_true, query_variables, target_query))
        p.start()
        p.join(timeout=3600)
        if p.is_alive():
            print("Inference for true_bn took too long and was terminated.")
            p.terminate()
            p.join()
            true_bn_prob = float("nan")
            timeout_indices["true_bn"].append(True)
        else:
            try:
                true_bn_prob = q.get(timeout=10)
                timeout_indices["true_bn"].append(False)
            except Empty:
                print("Failed to get result from process for true_bn.")
                true_bn_prob = float("nan")
                timeout_indices["true_bn"].append(True)
        end = time.time()
        inference_times["true_bn"].append(end - start)
        predictions["true_bn"].append(true_bn_prob)

        # PCN
        start = time.time()
        pcn_first_query_variables = query_variables.copy()
        pcn_first_query_variables.update(target_query)
        marginalized = [
            i for i in range(len(names)) if names[i] not in pcn_first_query_variables
        ]
        input_tensor = (
            torch.tensor(
                [
                    [
                        (
                            0
                            if name not in pcn_first_query_variables
                            else pcn_first_query_variables[name]
                        )
                        for name in names
                    ]
                ]
            )
            .float()
            .to(device)
        )
        pcn_prob_log = pcn.predict(
            input_tensor,
            names,
            IPF(),
            marginalized,
            optimize_inference=pcn_inference_pruning,
            return_log_prob=True,
        ).item()

        if len(query_variables) > 0:
            marginalized = [i for i in range(len(names)) if names[i] not in query_variables]
            input_tensor = (
                torch.tensor(
                    [
                        [
                            0 if name not in query_variables else query_variables[name]
                            for name in names
                        ]
                    ]
                )
                .float()
                .to(device)
            )
            pcn_evidence_prob_log = pcn.predict(
                input_tensor,
                names,
                IPF(),
                marginalized,
                optimize_inference=pcn_inference_pruning,
                return_log_prob=True,
            ).item()
            if pcn_evidence_prob_log == -float("inf") or pcn_evidence_prob_log < -1e20:
                print("PCN evidence probability is zero, setting query probability to 0.")
                pcn_prob = 0.0
            else:
                pcn_prob = torch.exp(
                    torch.Tensor([pcn_prob_log - pcn_evidence_prob_log])
                )
                pcn_prob = torch.clamp(pcn_prob, 0.0, 1.0).item()
        else:
            pcn_prob = torch.exp(torch.Tensor([pcn_prob_log]))
            pcn_prob = torch.clamp(pcn_prob, 0.0, 1.0).item()

        end = time.time()
        inference_times["pcn"].append(end - start)
        predictions["pcn"].append(pcn_prob)

        # Single EInet baseline
        start = time.time()
        with torch.no_grad():
            einet_joint_query = query_variables.copy()
            einet_joint_query.update(target_query)
            marginalized = [
                i for i in range(len(names)) if names[i] not in einet_joint_query
            ]
            einet_joint_input = (
                torch.tensor(
                    [
                        [
                            0 if name not in einet_joint_query else einet_joint_query[name]
                            for name in names
                        ]
                    ]
                )
                .float()
                .to(device)
            )
            einet_joint_log = einet.forward(
                einet_joint_input,
                marginalized if len(marginalized) > 0 else None,
            ).item()

            if len(query_variables) > 0:
                marginalized = [
                    i for i in range(len(names)) if names[i] not in query_variables
                ]
                einet_evidence_input = (
                    torch.tensor(
                        [
                            [
                                0 if name not in query_variables else query_variables[name]
                                for name in names
                            ]
                        ]
                    )
                    .float()
                    .to(device)
                )
                einet_evidence_log = einet.forward(
                    einet_evidence_input,
                    marginalized if len(marginalized) > 0 else None,
                ).item()
                if einet_evidence_log == -float("inf") or einet_evidence_log < -1e20:
                    print("EInet evidence probability is zero, setting query probability to 0.")
                    einet_prob = 0.0
                else:
                    einet_prob = torch.exp(
                        torch.Tensor([einet_joint_log - einet_evidence_log])
                    )
                    einet_prob = torch.clamp(einet_prob, 0.0, 1.0).item()
            else:
                einet_prob = torch.exp(torch.Tensor([einet_joint_log]))
                einet_prob = torch.clamp(einet_prob, 0.0, 1.0).item()

        end = time.time()
        inference_times["einet"].append(end - start)
        predictions["einet"].append(einet_prob)

        true_prob = predictions["true_bn"][-1]
        for model_key in ["pcn", "einet"]:
            pred_prob = predictions[model_key][-1]
            errors[f"{model_key}_l1"].append(abs(true_prob - pred_prob))
            errors[f"{model_key}_l2"].append((true_prob - pred_prob) ** 2)

        wandb.log(
            {
                "pcn_query_prob": predictions["pcn"][-1],
                "einet_query_prob": predictions["einet"][-1],
                "true_bn_query_prob": predictions["true_bn"][-1],
                "pcn_query_error_l1": errors["pcn_l1"][-1],
                "einet_query_error_l1": errors["einet_l1"][-1],
                "pcn_query_error_l2": errors["pcn_l2"][-1],
                "einet_query_error_l2": errors["einet_l2"][-1],
                "pcn_query_inference_time": inference_times["pcn"][-1],
                "einet_query_inference_time": inference_times["einet"][-1],
                "true_bn_query_inference_time": inference_times["true_bn"][-1],
            }
        )

    np.save(folder_path / "pcn_training_time.npy", np.array([pcn_training_time]))
    np.save(folder_path / "einet_training_time.npy", np.array([einet_training_time]))

    for model_key in predictions:
        np.save(
            folder_path / f"{model_key}_predictions.npy",
            np.array(predictions[model_key]),
        )
    for error_key in errors:
        np.save(
            folder_path / f"{error_key}.npy",
            np.array(errors[error_key]),
        )
    for time_key in inference_times:
        np.save(
            folder_path / f"{time_key}_inference_times.npy",
            np.array(inference_times[time_key]),
        )

    print("Evaluation Summary:")
    print(
        "True BN average inference time: " + str(np.nanmean(inference_times["true_bn"]))
    )
    print("PCN average inference time: " + str(np.nanmean(inference_times["pcn"])))
    print(
        "Single EInet average inference time: "
        + str(np.nanmean(inference_times["einet"]))
    )
    print(
        "PCN average L1 error: "
        + str(np.nanmean(errors["pcn_l1"]))
        + ", PCN average L2 error: "
        + str(np.nanmean(errors["pcn_l2"]))
    )
    print(
        "Single EInet average L1 error: "
        + str(np.nanmean(errors["einet_l1"]))
        + ", Single EInet average L2 error: "
        + str(np.nanmean(errors["einet_l2"]))
    )

    total_eval_time = time.time() - eval_start_time
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")

    wandb.log(
        {
            "true_bn_avg_inference_time": np.nanmean(inference_times["true_bn"]),
            "pcn_avg_inference_time": np.nanmean(inference_times["pcn"]),
            "einet_avg_inference_time": np.nanmean(inference_times["einet"]),
            "pcn_avg_l1_error": np.nanmean(errors["pcn_l1"]),
            "pcn_avg_l2_error": np.nanmean(errors["pcn_l2"]),
            "einet_avg_l1_error": np.nanmean(errors["einet_l1"]),
            "einet_avg_l2_error": np.nanmean(errors["einet_l2"]),
            "total_evaluation_time": total_eval_time,
        }
    )

    wandb.finish()


def run_ve(queue, infer, query_variables, target_query):
    evidence = query_variables
    bn_probabilities = infer.query(
        variables=list(target_query.keys()),
        evidence=evidence,
        elimination_order="MinFill",
    )
    bn_prob = bn_probabilities.get_value(**target_query).item()
    queue.put(bn_prob)


def main():
    parser = argparse.ArgumentParser(
        description="Run a full evaluation comparing PCN and single EInet to true BN."
    )
    parser.add_argument(
        "--experimental_series",
        type=str,
        default="testing",
        help="A category for grouping related experiments together.",
    )
    parser.add_argument(
        "--identifier",
        type=str,
        choices=[
            "asia",
            "child",
            "alarm",
            "win95pts",
            "insurance",
            "hepar2",
            "hailfinder",
            "water",
            "barley",
            "mildew",
        ],
        default="asia",
        help="The type of graph structure to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Number of samples to generate for evaluation.",
    )
    parser.add_argument(
        "--pcn_inference_pruning",
        action="store_true",
        help="Whether to use pruning during PCN inference.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10000,
        help="Maximum number of training epochs for both PCN and single EInet.",
    )
    args = parser.parse_args()

    evaluate_model(
        args.identifier,
        args.seed,
        args.num_samples,
        args.pcn_inference_pruning,
        args.experimental_series,
        args.max_epochs,
    )


if __name__ == "__main__":
    main()