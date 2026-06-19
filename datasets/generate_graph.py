import random
import warnings

import networkx as nx
import numpy as np


def generate_random_graph(
    seed, num_nodes, edge_prob, option="erdos-renyi", force_weakly_connected=True
):
    """
    Generates a random acyclic graph based on the given parameters.

    Parameters:
        seed (int): Random seed for reproducibility.
        num_nodes (int): Number of nodes in the graph.
        edge_prob (float): Probability of edge creation between nodes.
        option (str): Graph generation option ("erdos-renyi").
        force_weakly_connected (bool): If True, ensures no isolated nodes in the graph.

    Returns:
        graph (networkx.DiGraph): A directed acyclic graph (DAG).
    """
    # rough estimate on the probability of the graph being connected
    node_without_edges_prob = (1 - edge_prob) ** (num_nodes - 1)
    expected_isolated_nodes = num_nodes * node_without_edges_prob
    if force_weakly_connected and expected_isolated_nodes >= 1:
        warnings.warn(
            f"With the given parameters, the expected number of isolated nodes is {expected_isolated_nodes:.2f}. "
            "Consider increasing the edge probability or decreasing the number of nodes to ensure connectivity."
        )

    random.seed(seed)
    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()

    attempt_limit = 100
    count = 0
    while count < attempt_limit:
        if option == "erdos-renyi":
            # Generate an Erdos-Renyi graph
            new_seed = int(rng.integers(0, 1e6))
            er_graph = nx.erdos_renyi_graph(
                num_nodes, edge_prob, seed=new_seed, directed=True
            )
            graph = nx.DiGraph()
            graph.add_nodes_from(er_graph.nodes())
            graph.add_edges_from([(u, v) for u, v in er_graph.edges() if u < v])
            new_labels = [int(label) for label in rng.permutation(num_nodes)]
        else:
            raise ValueError("Invalid option. Choose 'erdos-renyi'.")
        if not force_weakly_connected or nx.is_weakly_connected(graph):
            break
        count += 1
        if count == attempt_limit:
            raise RuntimeError(
                "Failed to generate a connected graph after multiple attempts."
            )

    # Ensure the graph is acyclic
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Generated graph is not acyclic. Try different parameters.")

    # the following two steps are to ensure that the graph representation does not inform about node ordering
    # relabel nodes
    graph = nx.relabel_nodes(
        graph, mapping=dict(zip(graph.nodes(), new_labels)), copy=True
    )
    # and shuffle internal node order
    nodes = list(graph.nodes())
    rng.shuffle(nodes)
    new_graph = nx.DiGraph()
    new_graph.add_nodes_from(nodes)
    new_graph.add_edges_from(graph.edges())
    graph = new_graph

    return graph


def get_graph(problem_type):
    """
    Generates a predefined graph structure based on the specified problem type.

    Parameters:
        problem_type (str): The type of graph structure to generate. Options are "chain",
                            "collider", "fork", "backdoor", and "diamond".

    Returns:
        graph (networkx.DiGraph): A directed acyclic graph (DAG) representing the specified structure.
    """
    graph = nx.DiGraph()

    if problem_type == "chain":
        graph.add_nodes_from(["0", "1", "2"])
        graph.add_edges_from([("0", "1"), ("1", "2")])
    elif problem_type == "collider":
        graph.add_nodes_from(["0", "1", "2"])
        graph.add_edges_from([("0", "2"), ("1", "2")])
    elif problem_type == "fork":
        graph.add_nodes_from(["0", "1", "2"])
        graph.add_edges_from([("0", "1"), ("0", "2")])
    elif problem_type == "backdoor":
        graph.add_nodes_from(["0", "1", "2"])
        graph.add_edges_from([("0", "1"), ("0", "2"), ("1", "2")])
    elif problem_type == "diamond":
        graph.add_nodes_from(["0", "1", "2", "3"])
        graph.add_edges_from([("0", "1"), ("0", "2"), ("1", "3"), ("2", "3")])
    else:
        raise ValueError(
            "Invalid problem type. Choose from 'chain', 'collider', 'fork', 'backdoor', or 'diamond'."
        )

    return graph


def get_grid_graph(grid_width=2):
    """
    Generates a grid graph of specified width.

    Parameters:
        grid_width (int): The width of the grid (number of nodes along one side).

    Returns:
        graph (networkx.DiGraph): A directed acyclic graph (DAG) representing the grid structure.
    """
    graph = nx.DiGraph()
    nodes = [str(i) for i in range(grid_width * grid_width)]
    graph.add_nodes_from(nodes)
    for i in range(grid_width):
        for j in range(grid_width):
            node = str(i * grid_width + j)
            if j < grid_width - 1:
                right_node = str(i * grid_width + (j + 1))
                graph.add_edge(node, right_node)
            if i < grid_width - 1:
                down_node = str((i + 1) * grid_width + j)
                graph.add_edge(node, down_node)
    return graph


def get_fully_connected_graph(num_nodes):
    """
    Generates a fully connected directed acyclic graph (DAG) with the specified number of nodes.

    Parameters:
        num_nodes (int): The number of nodes in the graph.

    Returns:
        graph (networkx.DiGraph): A fully connected directed acyclic graph (DAG).
    """
    graph = nx.DiGraph()
    nodes = [str(i) for i in range(num_nodes)]
    graph.add_nodes_from(nodes)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            graph.add_edge(str(i), str(j))
    return graph
