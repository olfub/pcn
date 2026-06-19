import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from rtpt import RTPT

import wandb
from datasets.binary_bn import BinaryBayesianNetwork
from datasets.generate_graph import get_grid_graph, get_fully_connected_graph
from pcn.methods.ipf import IPF
from pcn.pcn import PCN
from utils import make_deterministic


def evaluate_model(
    graph_type,
    graph_param,
    seed,
    num_samples,
    pcn_inference_pruning,
    experimental_series,
    max_epochs,
):
    ################################################
    ############# Setup Experiment #################
    ################################################

    eval_start_time = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    folder_path = (
        Path("experiments") / experimental_series / f"{graph_type}_{graph_param}" / f"seed_{seed}"
    )
    folder_path.mkdir(parents=True, exist_ok=True)
    short_exp_str = f"{experimental_series}_{graph_type}_{graph_param}_seed{seed}"

    # initialize wandb
    wandb.init(
        project="PCN",
        name=short_exp_str,
        config={
            "experimental_series": experimental_series,
            "graph_type": graph_type,
            "graph_param": graph_param,
            "seed": seed,
            "num_samples": num_samples,
            "pcn_inference_pruning": pcn_inference_pruning,
        },
    )

    make_deterministic(seed, False)
    print(f"Running evaluation for {short_exp_str} on {device}")

    # create graph and model
    if graph_type == "fully":
        graph = get_fully_connected_graph(num_nodes=graph_param)
    elif graph_type == "grid":
        graph = get_grid_graph(grid_width=graph_param)
    else:
        raise ValueError("Invalid graph type. Choose 'grid' or 'fully'.")
    bvm = BinaryBayesianNetwork(graph, seed=seed)
    bvm.add_binary_cpds()
    names = [str(node) for node in graph.nodes()]
    samples = bvm.sample(num_samples, names)
    samples_tensor = torch.tensor(samples).to(device)

    ##############################################
    ############# Train Models ###################
    ##############################################

    # fit PCN and save model weights
    node_cardinalities = {}
    for i, name in enumerate(names):
        node_cardinalities[name] = int(np.max(samples[:, i]) + 1)
    pcn = PCN(
        bvm.graph, samples_tensor, names, node_cardinalities=node_cardinalities
    ).to(device)
    data_dict = pcn.prepare_data(samples_tensor, names)
    optimizer = torch.optim.Adam(pcn.parameters(), lr=0.001)
    num_epochs = max_epochs
    rtpt = RTPT(
        name_initials="FB",
        experiment_name=f"PCN {short_exp_str}",
        max_iterations=num_epochs,
    )
    rtpt.start()
    start = time.time()
    best_loss = float("inf")
    patience = 100
    patience_counter = 0

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass to compute loss
        loss = -pcn.forward(samples_tensor, names, data_dict=data_dict).sum()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        wandb.log({"pcn_train_loss": loss.item()}, step=epoch)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        rtpt.step()

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    end = time.time()
    pcn_training_time = end - start
    print(f"PCN training time: {pcn_training_time:.2f} seconds")
    wandb.log({"pcn_training_time": pcn_training_time})
    model_path = folder_path / "pcn_model.pth"
    torch.save(pcn.state_dict(), model_path)
    pcn.set_marginals(samples_tensor, names)

    # set up inference on true Bayesian network
    infer_true = VariableElimination(bvm.get_model())

    # set up Bayesian network fitted from data
    bm = DiscreteBayesianNetwork()
    bm.add_nodes_from(bvm.graph.nodes)
    bm.add_edges_from(bvm.graph.edges)
    data_df = pd.DataFrame(samples, columns=names)
    start = time.time()
    bm.fit(data_df, estimator=MaximumLikelihoodEstimator)
    end = time.time()
    bn_fitting_time = end - start
    print(f"BN fitting time: {bn_fitting_time:.2f} seconds")
    wandb.log({"bn_fitting_time": bn_fitting_time})
    infer_learned = VariableElimination(bm)

    ##############################################
    ############# Evaluate Models ################
    ##############################################

    predictions = {
        "pcn": [],
        "bn_learned": [],
        "true_bn": [],
    }
    errors = {
        "pcn_l1": [],
        "bn_learned_l1": [],
        "pcn_l2": [],
        "bn_learned_l2": [],
    }
    inference_times = {
        "pcn": [],
        "bn_learned": [],
        "true_bn": [],
    }
    all_queries = [{names[0]: 0, names[-1]: 1},
                   {names[0]: 1, names[-1]: 0},
                   {names[0]: 0, names[-1]: 0},
                   {names[0]: 1, names[-1]: 1}]

    # evaluate on all queries
    for query_variables in all_queries:
        print(f"Evaluating query for: {query_variables}")

        # BNs
        for infer in [infer_true, infer_learned]:
            if infer == infer_true:
                model_key = "true_bn"
            else:
                model_key = "bn_learned"
            start = time.time()
            bn_probabilities = infer.query(
                variables=list(query_variables.keys()),
                evidence={},
                elimination_order="MinFill",
            )
            bn_prob = bn_probabilities.get_value(**query_variables).item()
            end = time.time()
            inference_times[model_key].append(end - start)
            predictions[model_key].append(bn_prob)

        # PCN
        start = time.time()
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
        pcn_prob = pcn.predict(
            input_tensor,
            names,
            IPF(),
            marginalized,
            optimize_inference=pcn_inference_pruning,
        ).item()
        end = time.time()
        inference_times["pcn"].append(end - start)
        predictions["pcn"].append(pcn_prob)

        # compute errors
        true_prob = predictions["true_bn"][-1]
        for model_key in ["pcn", "bn_learned"]:
            pred_prob = predictions[model_key][-1]
            error_l1 = abs(true_prob - pred_prob)
            error_l2 = (true_prob - pred_prob) ** 2
            errors[f"{model_key}_l1"].append(error_l1)
            errors[f"{model_key}_l2"].append(error_l2)

        # log results to wandb
        wandb.log(
            {
                "pcn_query_prob": predictions["pcn"][-1],
                "bn_learned_query_prob": predictions["bn_learned"][-1],
                "true_bn_query_prob": predictions["true_bn"][-1],
                "pcn_query_error_l1": errors["pcn_l1"][-1],
                "bn_learned_query_error_l1": errors["bn_learned_l1"][-1],
                "pcn_query_error_l2": errors["pcn_l2"][-1],
                "bn_learned_query_error_l2": errors["bn_learned_l2"][-1],
                "pcn_query_inference_time": inference_times["pcn"][-1],
                "bn_learned_query_inference_time": inference_times["bn_learned"][-1],
                "true_bn_query_inference_time": inference_times["true_bn"][-1],
            }
        )

    # save results to file
    # training times
    np.save(folder_path / "pcn_training_time.npy", np.array([pcn_training_time]))
    np.save(folder_path / "bn_fitting_time.npy", np.array([bn_fitting_time]))
    # predictions
    for model_key in predictions:
        np.save(
            folder_path / f"{model_key}_predictions.npy",
            np.array(predictions[model_key]),
        )
    # errors
    for error_key in errors:
        np.save(
            folder_path / f"{error_key}.npy",
            np.array(errors[error_key]),
        )
    # inference times
    for time_key in inference_times:
        np.save(
            folder_path / f"{time_key}_inference_times.npy",
            np.array(inference_times[time_key]),
        )

    # print summary
    print("Evaluation Summary:")
    print("True BN average inference time: " + str(np.mean(inference_times["true_bn"])))
    print(
        "Learned BN average inference time: "
        + str(np.mean(inference_times["bn_learned"]))
    )
    print("PCN average inference time: " + str(np.mean(inference_times["pcn"])))
    print(
        "Learned BN average L1 error: "
        + str(np.mean(errors["bn_learned_l1"]))
        + ", Learned BN average L2 error: "
        + str(np.mean(errors["bn_learned_l2"]))
    )
    print(
        "PCN average L1 error: "
        + str(np.mean(errors["pcn_l1"]))
        + ", PCN average L2 error: "
        + str(np.mean(errors["pcn_l2"]))
    )

    eval_end_time = time.time()
    total_eval_time = eval_end_time - eval_start_time
    print(f"Total evaluation time: {total_eval_time:.2f} seconds")

    wandb.log(
        {
            "true_bn_avg_inference_time": np.mean(inference_times["true_bn"]),
            "bn_learned_avg_inference_time": np.mean(inference_times["bn_learned"]),
            "pcn_avg_inference_time": np.mean(inference_times["pcn"]),
            "bn_learned_avg_l1_error": np.mean(errors["bn_learned_l1"]),
            "bn_learned_avg_l2_error": np.mean(errors["bn_learned_l2"]),
            "pcn_avg_l1_error": np.mean(errors["pcn_l1"]),
            "pcn_avg_l2_error": np.mean(errors["pcn_l2"]),
            "total_evaluation_time": total_eval_time,
        }
    )

    wandb.finish()


def main():
    parser = argparse.ArgumentParser(
        description="Run a full evaluation comparing PCNs to BNs."
    )
    parser.add_argument(
        "--experimental_series",
        type=str,
        default="testing",
        help="A category for grouping related experiments together.",
    )
    parser.add_argument(
        "--graph_type",
        type=str,
        choices=["grid", "fully"],
        default="fully",
        help="Type of graph to generate (grid or fully connected).",    
    )
    parser.add_argument(
        "--graph_param",
        type=int,
        default=20,
        help="Parameter for the graph (e.g., width for grid or number of nodes for fully connected).",
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
        default=10000,
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
        help="Maximum number of training epochs for the PCN.",
    )
    args = parser.parse_args()

    experimental_series = args.experimental_series
    graph_type = args.graph_type
    graph_param = args.graph_param
    seed = args.seed
    num_samples = args.num_samples
    pcn_inference_pruning = args.pcn_inference_pruning
    max_epochs = args.max_epochs
    evaluate_model(
        graph_type,
        graph_param,
        seed,
        num_samples,
        pcn_inference_pruning,
        experimental_series,
        max_epochs,
    )


if __name__ == "__main__":
    main()
