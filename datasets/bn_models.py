import networkx as nx

from pgmpy.readwrite import BIFReader

from datasets.discrete_bn import BayesianNetworkHandler


DATASET_IDENTIFIERS = [
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


def get_problem(identifier, n_samples=10000, seed=42):
    """
    Generates a Bayesian Network problem instance based on the specified identifier.

    Parameters:
        identifier (str): The identifier of the Bayesian Network to generate.
        n_samples (int): The number of samples to generate from the Bayesian Network. Defaults to 10,000.

    Returns:
        dict: A dictionary containing the Bayesian Network model, node names, and generated samples.
    """
    reader = BIFReader(f"datasets/bnlearn_files/{identifier}.bif")
    model = reader.get_model()

    # set graph
    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes())
    graph.add_edges_from(model.edges())

    # set the Bayesian Network Handler and the cpds
    bnh = BayesianNetworkHandler(graph, seed=seed)
    cpds = model.get_cpds()
    bnh.add_predefined_cpds(cpds)
    bnh.check_model()

    # variable names
    names = list(bnh.get_model().nodes())

    # generate samples
    samples = bnh.sample(n_samples, names)
    return {
        "model": bnh,
        "names": names,
        "samples": samples,
    }


# import numpy as np
# ident = DATASET_IDENTIFIERS[8]  # default identifier
# result = get_problem(ident)
# bayesian_network_model = result["model"]
# node_names = result["names"]
# generated_samples = result["samples"]
# full_min = 1
# full_max = 0
# for cpd in bayesian_network_model.get_model().get_cpds():
#     min = np.min(cpd.values)
#     max = np.max(cpd.values)
#     print(f"CPD for {cpd.variable}: min={min}, max={max}")
#     if min < full_min:
#         full_min = min
#     if max > full_max:
#         full_max = max
# print(f"Overall min CPD value: {full_min}")
# print(f"Overall max CPD value: {full_max}")