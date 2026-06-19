import warnings
from typing import Optional, Union

import networkx as nx
import torch
from torch import nn

from pcn.component import init_component
from pcn.methods.scaling_methods import ScalingMethod


class PCN(nn.Module):
    """
    Probabilistic Circuit Network (PCN) class.
    An PCN consists of several PCs, each of which representing a single variable but modeled together with its causal parents.
    """

    # TODO most commands do not allow batch dimension > 1, add this functionality

    def __init__(
        self,
        graph: nx.DiGraph,
        data: torch.Tensor,
        node_names: list[Union[str]],
        node_cardinalities: dict[str, int],
    ) -> None:
        """
        Create the PCN object.

        Args:
            graph (nx.DiGraph): The directed graph representing the PCN structure.
            data (torch.Tensor): The input data as a PyTorch tensor.
            node_names (list[str]): The names of the columns in the data.
        """
        super().__init__()
        self.components = nn.ModuleDict({})
        self.graph = graph
        self.interventions = {}
        self.node_names = list(self.graph.nodes())
        self.node_cardinalities = node_cardinalities
        self.marginal_probabilities = None
        self._init_model(data, node_names)

    def _init_model(self, data: torch.Tensor, node_names: list[str]) -> None:
        """
        Create the PCN from the given data.

        Args:
            data (torch.Tensor): The input data as a PyTorch tensor.
            node_names (list[str]): The names of the columns in the data.
        """
        assert sorted(node_names) == sorted(
            self.node_names
        ), "node names do not match the node names in the graph."

        # intialize all the components
        for var in node_names:
            # create the component for the variable
            var_parents = list(self.graph.predecessors(var))
            # use as cardinality the maximum cardinality of the variable and its parents (for PCNs)
            max_cardinality = 0
            for v in [var] + var_parents:
                if self.node_cardinalities[v] > max_cardinality:
                    max_cardinality = self.node_cardinalities[v]
            component = init_component(
                var, var_parents, max_cardinality=max_cardinality
            )
            if var_parents == []:
                # set the probabilities for root nodes (ConstantComponent)
                component.set_probs(data[:, node_names.index(var)])
            self.components[var] = component

    def prepare_data(self, x: torch.Tensor, node_names: list[str]) -> dict:
        """
        Transform the input data into a dictionary format, such that this does not need to be done every time in later calls.

        Args:
            x (torch.Tensor): The input tensor (data samples).
            node_names (list[str]): The names of the columns in the input tensor.

        Returns:
            dict: A dictionary where keys are variable names and values are tuples of (component, data).
        """
        data_dict = {}
        for var, component in self.components.items():
            # get index of current variable and parents
            var_index = node_names.index(var)
            var_parents = list(self.graph.predecessors(var))
            # get the corresponding dataset columns
            var_parents_indices = [node_names.index(p) for p in var_parents]
            all_indices = [var_index] + var_parents_indices
            var_data = x[:, all_indices]
            data_dict[var] = (component, var_data)
        return data_dict

    def variables_to_indices(self, variables: list[Union[str]]) -> list[int]:
        """
        Convert a list of variable names or indices to a list of indices.

        Args:
            variables (list[Union[str]]): List of variable names (str).

        Returns:
            list[int]: List of variable indices.
        """
        return torch.tensor([self.node_names.index(var) for var in variables])

    def do_intervention(
        self, interventions: dict[Union[str], int], suppress_warning: bool = False
    ) -> None:
        """
        Intervene on the causal model. This will be taken into account during inference.

        Args:
            interventions (dict[Union[str], int]): A dictionary where keys are node indices (str) and values are the intervened values.
            suppress_warning (bool, optional): If True, suppress the warning about existing interventions. Defaults to False.
        """
        if len(self.interventions) > 0 and not suppress_warning:
            warnings.warn(
                "Some interventions were already applied; the new interventions will be added to the current graph. To reset, use reset_graph()."
            )
        self.interventions = interventions

    def reset_graph(self) -> None:
        """
        Reset the PCN to its original state, removing all interventions.
        """
        self.interventions = {}

    def forward(
        self, x: torch.Tensor, node_names: list[str], data_dict: dict = None
    ) -> torch.Tensor:
        """
        Simple forward pass through the PCN. Only returns the log probabilities for
        the given input x. Intended for training the PCN. Should not be used for
        inference, as it doesn't take marginalizations or scaling factors into account.
        If data_dict is provided, it will be used instead of computing the data slices again.

        Args:
            x (torch.Tensor): The input tensor (data samples).
            node_names (list[str]): The names of the columns in the input tensor.
            data_dict (dict, optional): A precomputed data dictionary for faster computation. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        if len(self.interventions) > 0:
            warnings.warn(
                "Interventions are currently not taken into account during the forward pass. Use predict() for inference with interventions."
            )
        assert sorted(node_names) == sorted(
            self.node_names
        ), "Column names do not match the node names in the graph."
        # TODO figure out if the streams stuff works correctly; then either use it in both cases (depending on data_dict) or not at all or as an option (parameter)

        # if a data_dict is provided, use it
        if data_dict is not None:
            # I did not find a great benefit of parallelization here
            # Either it doesn't work as intended or it will only help for very large batch sizes
            streams = [torch.cuda.Stream() for _ in range(len(self.components))]
            outputs = []
            for i, (component, data) in enumerate(data_dict.values()):
                with torch.cuda.stream(streams[i]):
                    log_prob = component(data)
                    outputs.append(log_prob)
            torch.cuda.synchronize()
            log_prob = torch.stack(outputs).sum()
            return log_prob

        # if no data_dict is provided, all components are computed sequentially
        # (of course, this could be parallelized as well, but since the benefit is
        # questionable anyhow, I did not implement it here)
        log_prob = torch.zeros((x.shape[0], 1), dtype=torch.float32).to(x.device)
        for var, component in self.components.items():
            # get index of current variable and parents
            var_index = node_names.index(var)
            var_parents = list(self.graph.predecessors(var))
            # get the corresponding dataset columns
            var_parents_indices = [node_names.index(p) for p in var_parents]
            all_indices = [var_index] + var_parents_indices
            var_data = x[:, all_indices]
            log_prob += component.forward(var_data)

        return log_prob

    def predict(
        self,
        x: torch.Tensor,
        node_names: list[str],
        scaling_method: ScalingMethod,
        marginalized_scopes: Optional[torch.Tensor] = None,
        optimize_inference: bool = True,
        return_log_prob: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass through the PCN to compute the log probabilities of the input tensor under possible marginalizations.

        Args:
            x (torch.Tensor): The input tensor (evidence).
            node_names (list[str]): The names of the columns in the input tensor.
            scaling_method (ScalingMethod): The scaling method to use for iterative proportional fitting.
            marginalized_scopes (torch.Tensor, optional): The scopes for marginalization. Defaults to None.
            optimize_inference (bool, optional): Whether to optimize inference by skipping certain nodes. Defaults to True.
            return_log_prob (bool, optional): Whether to return log probabilities instead of probabilities. Defaults

        Returns:
            torch.Tensor: The computed probabilities for the probabilistic query.
        """
        assert sorted(node_names) == sorted(
            self.node_names
        ), "Column names do not match the node names in the graph."
        # even with interventions, the causal order remains correct
        causal_order = list(nx.topological_sort(self.graph))

        # start with a probability of 1 (log_prob = 0)
        log_prob = torch.zeros((x.shape[0], 1), dtype=torch.float32).to(x.device)
        current_marginal_probabilities = {
            key: torch.zeros_like(value)
            for key, value in self.marginal_probabilities.items()
        }

        if optimize_inference:
            # several nodes do not need to be computed: any marginalized node that either
            # 1. has no ancestors that are not marginalized (i.e., their probability
            # distribution is the default one), or 2. has no descendants that are not
            # marginalized (i.e., they do not influence any non-marginalized variable)
            marginalized_nodes = []
            to_compute = []
            # check all nodes
            for node in causal_order:
                if node in marginalized_scopes:
                    # check if any ancestor is not marginalized
                    ancestors = nx.ancestors(self.graph, node)
                    any_ancestor_not_marginalized = any(
                        anc not in marginalized_scopes for anc in ancestors
                    )
                    # check if any descendant is not marginalized
                    descendants = nx.descendants(self.graph, node)
                    any_descendant_not_marginalized = any(
                        desc not in marginalized_scopes for desc in descendants
                    )
                    if (
                        not any_ancestor_not_marginalized
                        or not any_descendant_not_marginalized
                    ):
                        # if no ancestor or no descendant is not marginalized, we can ignore this node
                        marginalized_nodes.append(node)
                    else:
                        # otherwise, we need to compute it
                        to_compute.append(node)
                else:
                    to_compute.append(node)

            # remove all marginalized nodes from causal order (we can ignore those, their probability will be 1) but keep causal order
            causal_order = [n for n in causal_order if n not in marginalized_nodes]
        else:
            marginalized_nodes = []

        # iterate over nodes in causal order
        for node in causal_order:
            # get the index and component
            node_index = self.node_names.index(node)
            component = self.components[node]

            if node in self.interventions:
                # apply the intervention by setting the marginal probability accordingly
                intervention_value = self.interventions[node]
                current_marginal_probabilities[node_index][intervention_value] = 1.0
                # the log probability of an intervened node is 0 (100%) and we can cotinue to the next node
                continue

            if component.parents != []:
                # if the node has parents, we need to match the marginal probabilities of the parents
                parent_indices = [
                    self.node_names.index(p)
                    for p in component.parents
                    if p not in marginalized_nodes
                ]
                # the marginal scopes are the indices of the parents in the input tensor
                target_scopes = [
                    i + 1
                    for i in range(len(component.parents))
                    if component.parents[i] not in marginalized_nodes
                ]
                # use a scaling method (e.g., iterative proportional fitting) to find the scaling factors
                scaling_factors = scaling_method.compute_scaling_factors(
                    component.einet,
                    [current_marginal_probabilities[i] for i in parent_indices],
                    target_scopes
                )

            # select the right columns of the data
            all_vars = [component.var_name] + component.parents
            node_indices = [node_names.index(var) for var in all_vars]
            current_input = x[:, node_indices]

            # select the right marginalization scopes
            if marginalized_scopes is not None:
                # marginalization indices must be relative to the current component
                current_marginalized_scopes = [
                    i
                    for i, index in enumerate(node_indices)
                    if index in marginalized_scopes
                ]
            else:
                current_marginalized_scopes = []

            # forward pass through the component
            this_prob = component.predict(
                current_input,
                marginalized_scopes=current_marginalized_scopes,
                scaling_factors=scaling_factors if component.parents != [] else None,
            )
            current_marginal_probabilities[node_index] = component.target_marginals(
                current_input,
                marginalized_scopes=current_marginalized_scopes,
                cardinality=self.node_cardinalities[node],
                scaling_factors=scaling_factors if component.parents != [] else None,
            )
            assert current_marginal_probabilities[node_index].shape[0] == self.node_cardinalities[
                node
            ], f"Marginal probabilities for variable {node} have incorrect cardinality."
            log_prob += this_prob

        if return_log_prob:
            return log_prob
        else:
            # return final probabilities in non-log form
            return torch.exp(log_prob)

    def set_marginals(
        self,
        x: torch.Tensor,
        node_names: list[str],
    ):
        """
        Compute and set the marginal probabilities for each variable in the PCN.

        Args:
            x (torch.Tensor): The input tensor (data samples).
            node_names (list[str]): The names of the columns in the input tensor.
        """
        if len(self.interventions) > 0:
            warnings.warn(
                "Interventions are currently not taken into account when computing marginal probabilities using set_marginals()."
            )
        # compute marginal probabilities for each variable
        self.marginal_probabilities = {}
        # TODO parallelize
        for i, var in enumerate(self.node_names):
            var_parents = list(self.graph.predecessors(var))
            var_parents_indices = [node_names.index(p) for p in var_parents]
            all_indices = [node_names.index(var)] + var_parents_indices
            var_data = x[:, all_indices]
            assert var_data.shape[1] == len(
                [var] + var_parents
            ), "Data slice has incorrect number of columns."
            self.marginal_probabilities[i] = (
                self.components[var]
                .target_marginals(
                    var_data[:1],
                    marginalized_scopes=[0],
                    cardinality=self.node_cardinalities[var],
                )
                .to(x.device)
            )
