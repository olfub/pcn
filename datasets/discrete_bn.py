import networkx as nx
import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork


class BayesianNetworkHandler:
    def __init__(self, graph, seed=None):
        """
        Initializes the BayesianNetworkHandler with the given graph.

        Parameters:
            graph (networkx.DiGraph): A directed acyclic graph (DAG) representing the structure of the Bayesian Network.
            random_seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.graph = graph
        self.random_seed = seed
        self.model = DiscreteBayesianNetwork()
        self._validate_graph()
        self._create_network()
        if self.random_seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def _validate_graph(self):
        """
        Validates that the input graph is a DAG.
        """
        if not self.graph.is_directed():
            raise ValueError("The input graph must be directed.")
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("The input graph must be a directed acyclic graph (DAG).")

    def _create_network(self):
        """
        Creates the Bayesian Network structure from the input graph.
        """
        # Add nodes and edges to the Bayesian Network
        self.model.add_nodes_from(self.graph.nodes())
        self.model.add_edges_from(self.graph.edges())

    def generate_cpds(self, cardinality=2):
        """
        Adds Conditional Probability Distributions (CPDs) to the Bayesian Network.
        Currently only supports common cardinality for all variables.
        Parameters:
            cardinality (int): The cardinality of the variables. Defaults to 2 for binary variables.
        """
        for node in self.model.nodes():
            parents = list(self.model.predecessors(node))
            if not parents:
                # If the node has no parents, add a prior probability
                values = self.rng.random(size=(cardinality, 1))
                values /= values.sum()  # Normalize probabilities
                cpd = TabularCPD(
                    variable=node,
                    variable_card=cardinality,
                    values=values.tolist(),  # Convert to list for TabularCPD
                )
            else:
                # If the node has parents, add a conditional probability table
                parent_card = [cardinality] * len(parents)
                num_rows = cardinality
                num_cols = np.prod(parent_card)
                values = self.rng.random(size=(num_rows, num_cols))
                values /= values.sum(axis=0)  # Normalize probabilities
                cpd = TabularCPD(
                    variable=node,
                    variable_card=cardinality,
                    values=values,
                    evidence=parents,
                    evidence_card=parent_card,
                )
            self.model.add_cpds(cpd)

    def add_predefined_cpds(self, cpds):
        """
        Adds predefined CPDs to the Bayesian Network.

        Parameters:
            cpds (list): A list of pgmpy.factors.discrete.TabularCPD objects.
        """
        for cpd in cpds:
            self.model.add_cpds(cpd)

    def check_model(self):
        """
        Checks if the Bayesian Network is valid.
        """
        if not self.model.check_model():
            raise ValueError("The Bayesian Network is not valid.")

    def get_model(self):
        """
        Returns the constructed Bayesian Network.

        Returns:
            pgmpy.models.DiscreteBayesianNetwork: The Bayesian Network model.
        """
        return self.model

    def sample(self, n_samples, nodes):
        """
        Generates samples from the Bayesian Network.

        Parameters:
            nodes (list): The list of nodes to include in the samples.
            n_samples (int): The number of samples to generate. Defaults to 1.

        Returns:
            np.ndarray: A 2D array where each row represents a sample.
        """
        samples = self.model.simulate(
            n_samples=n_samples, seed=self.rng.integers(0, 1e9)
        )
        samples_np = np.zeros((n_samples, len(nodes)), dtype=np.float32)
        for i, node in enumerate(nodes):
            map_func = self.model.get_cpds(node).name_to_no[node]
            samples[node] = samples[node].map(map_func)
            samples_np[:, i] = samples[node].to_numpy().astype(np.float32)
        return samples_np
