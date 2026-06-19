from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.categorical import Categorical
from torch import nn

from utils import set_smart_einet_config


class Component(ABC, nn.Module):
    """
    Abstract base class for components in a PCN.
    This class defines the interface for all components, including PCNComponent and ConstantComponent.
    """

    def __init__(self, var_name: str, parents: List[str], max_cardinality: int = 2) -> None:
        """
        Initialize the component with a variable name and its parents.

        Args:
            var_name (str): The name of the variable represented by this component.
            parents (List[str]): List of parent variable names.
            max_cardinality (int): The maximum cardinality of the variable. Defaults to 2 for binary variables.
        """
        super().__init__()
        self.var_name = var_name
        self.parents = parents
        self.default_marginals = None
        self.max_cardinality = max_cardinality

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple forward pass through the component. Only returns the log probabilities for the given input x. Intended for training the component. Should not be used for inference, as it doesn't take marginalizations or scaling factors into account.

        Args:
            x (torch.Tensor): The input tensor (data samples).

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        pass

    @abstractmethod
    def predict(
        self,
        x: torch.Tensor,
        marginalized_scopes: Optional[List[int]] = None,
        scaling_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for making predictions to answer probabilistic queries.

        Args:
            x (torch.Tensor): The input tensor (evidence).
            marginalized_scopes (List[int], optional): List of indices of variables to marginalize over. Defaults to None.
            scaling_factors (torch.Tensor, optional): Scaling factors for the computation. Defaults to None.

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        pass

    @abstractmethod
    def target_marginals(
        self,
        x: Optional[torch.Tensor] = None,
        marginalized_scopes: Optional[List[int]] = None,
        cardinality: Optional[int] = None,
        scaling_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the target marginals for the component. If evidence (x) is provided, the marginals correspond to full probability mass on the evidence. If no evidence is provided, the marginals correspond to the marginal probabilities (possibly scaled).

        Args:
            x (torch.Tensor, optional): The input tensor. Defaults to None.
            marginalized_scopes (List[int], optional): List of indices of variables to marginalize over. Defaults to None.
            cardinality (int, optional): The expected cardinality of the variable. Defaults to None.
            scaling_factors (torch.Tensor, optional): Scaling factors for the computation. Defaults to None.

        Returns:
            torch.Tensor: The target marginals for the component.
        """
        pass


class PCNComponent(Component):
    """
    Class that represents a PC in a PCN.
    """

    def __init__(
        self,
        var_name: str,
        parents: List[str],
        einet_config: Optional[EinetConfig] = None,
        max_cardinality: int = 2,
    ) -> None:
        """
        Initialize the PCNComponent with a variable name and its parents.

        Args:
            var_name (str): The name of the variable represented by this PC.
            parents (List[str]): List of parent variable names.
            einet_config (EinetConfig, optional): Configuration for the Einet. If None, a smart default configuration is set.
            max_cardinality (int): The maximum cardinality of the variable. Defaults to 2 for binary variables.
        """
        super().__init__(var_name, parents, max_cardinality=max_cardinality)
        if einet_config is None:
            einet_config = set_smart_einet_config(len(self.parents) + 1, max_cardinality=max_cardinality)
        self.einet = Einet(einet_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple forward pass through the PCNComponent. Only returns the log probabilities for the given input x. Intended for training the component. Should not be used for inference, as it doesn't take marginalizations or scaling factors into account.

        Args:
            x (torch.Tensor): The input tensor (data samples).

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        return self.einet.forward(x)

    def predict(
        self,
        x: torch.Tensor,
        marginalized_scopes: Optional[List[int]] = None,
        scaling_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the PCNComponent for making predictions to answer probabilistic queries.

        Args:
            x (torch.Tensor): The input tensor (evidence).
            marginalized_scopes (torch.Tensor, optional): List of indices of variables to marginalize over. Defaults to None.
            scaling_factors (torch.Tensor, optional): Scaling factors for the computation. Defaults to None.

        Returns:
            torch.Tensor: The computed probabilities for the probabilistic query.
        """
        if 0 in marginalized_scopes:
            # if the target variable is marginalized, we return a probability of 1
            return torch.zeros((x.shape[0], 1), dtype=torch.float32).to(x.device)
        # compute joint probability
        if marginalized_scopes == []:
            marginalized_scopes = None
        joint_prob = self.einet.forward(
            x, marginalized_scopes, scaling_factors=scaling_factors
        )

        # since the goal of the PCN is to compute the overall probability by
        # multiplying the probabilities of the components, we might need to compute the
        # conditional probability, which means we need to divide by the probability
        # where the target variable is marginalized
        if marginalized_scopes is not None:
            marginalized_scopes = [0] + [scope for scope in marginalized_scopes]
        else:
            marginalized_scopes = [0]
        conditioned_prob = self.einet.forward(
            x, marginalized_scopes, scaling_factors=scaling_factors
        )

        # subtraction because of log probabilities
        return joint_prob - conditioned_prob

    def target_marginals(
        self,
        x: Optional[torch.Tensor] = None,
        marginalized_scopes: Optional[List[int]] = None,
        cardinality: Optional[int] = None,
        scaling_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the target marginals for the PCNComponent. If evidence (x) is provided, the marginals correspond to full probability mass on the evidence. If no evidence is provided, the marginals correspond to the marginal probabilities (possibly scaled).

        Args:
            x (torch.Tensor, optional): The input tensor. Defaults to None.
            marginalized_scopes (List[int], optional): List of indices of variables to marginalize over. Defaults to None.
            scaling_factors (torch.Tensor, optional): Scaling factors for the computation. Defaults to None.
            cardinality (int, optional): The expected cardinality of the variable. Defaults to None.

        Returns:
            torch.Tensor: The target marginals for the component.
        """
        assert x.shape[0] == 1  # TODO marginals for batches (but low priority)
        if 0 not in marginalized_scopes:
            # if the target variable is not marginalized, there is evidence for it,
            # so we remove the probability mass not corresponding to the evidence
            # and set the probability of the evidence to 1
            result = torch.zeros((self.max_cardinality,), dtype=torch.float32).to(x.device)
            result[x[0, 0].long()] = 1.0
        else:
            result = torch.zeros((self.max_cardinality,), dtype=torch.float32).to(x.device)
            # marginalize everything but the target variable
            marginalized_scopes = list(range(1, len(self.parents) + 1))

            for i in range(0, self.max_cardinality):
                # marginal probability for i
                input_tensor = torch.full(
                    (1, len(self.parents) + 1), float(i), dtype=torch.float32
                ).to(x.device)
                result[i] = self.einet.forward(
                    input_tensor, marginalized_scopes, scaling_factors=scaling_factors
                )
            result = torch.exp(result)
        
        if cardinality is not None and result.shape[0] > cardinality:
            # trim and renormalize
            result = result[:cardinality]
            result /= result.sum()
        elif cardinality is not None and result.shape[0] < cardinality:
            raise ValueError(
                f"Computed marginal probabilities for variable {self.var_name} has smaller cardinality than expected."
            )
        return result


class ConstantComponent(Component, nn.Module):
    """
    Class that represents a constant component in a PCN.
    This is used for variables that do not have parents or are not modeled by a PC.
    """

    def __init__(self, var_name: str, parents: Optional[List[str]] = None, max_cardinality: int = 2) -> None:
        """
        Initialize the ConstantComponent with a variable name and its constant value.

        Args:
            var_name (str): The name of the variable represented by this constant component.
            parents (List[str], optional): List of parent variable names. Defaults to None.
            max_cardinality (int): The maximum cardinality of the variable. Defaults to 2 for binary variables.
        """
        assert (
            parents is None or parents == []
        ), "ConstantComponent should not have parents."
        super().__init__(var_name, parents, max_cardinality=max_cardinality)
        self.probs = None

    def set_probs(self, data: torch.Tensor) -> None:
        """
        Train the constant component with the given data.

        Args:
            data (torch.Tensor): The input data tensor.
        """
        # compute probabilities from data
        self.probs = torch.zeros((self.max_cardinality,)).to(data.device)
        for i in range(self.max_cardinality):
            self.probs[i] = (data == i).sum().item() / data.numel()
        self.probs = torch.log(
            self.probs
        )  # TODO do I need to add an epsilon here for 0 probabilities? on the other
        # hand, especially for binary variables, probabilities should not be 0 and 1,
        # this would be pointless

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple forward pass through the constant component. Since we do not train this component using backpropagation, this method simply returns the optimal log probabilities for the given input x (i.e., 0 for all samples).

        Args:
            x (torch.Tensor): The input tensor (data samples).

        Returns:
            torch.Tensor: The output tensor after the forward pass.
        """
        # while setting the probabilities would not be necessary for this method,
        # this ensures that the probabilities have been set before PCN training
        assert self.probs is not None, "Probabilities have not been set yet."
        return torch.zeros((x.shape[0], 1)).to(x.device)

    def predict(
        self,
        x: torch.Tensor,
        marginalized_scopes: torch.Tensor = None,
        scaling_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the ConstantComponent for making predictions to answer probabilistic queries.

        Args:
            x (torch.Tensor): The input tensor (evidence).
            marginalized_scopes (torch.Tensor, optional): List of indices of variables to marginalize over. Defaults to None.
            scaling_factors (torch.Tensor, optional): Scaling factors for the computation. Defaults to None.

        Returns:
            torch.Tensor: The computed probabilities for the probabilistic query.
        """
        assert x.shape[1] == 1
        if 0 in marginalized_scopes:
            # probability of 1 means 0 in log space
            return torch.zeros((x.shape[0], 1)).to(x.device)
        probs = self.probs[x.long()]  # get the probabilities for the values in x

        if scaling_factors is not None:
            # apply scaling factors if provided
            probs = probs + torch.log(scaling_factors[x.long()])

        return probs

    def target_marginals(
        self,
        x: Optional[torch.Tensor] = None,
        marginalized_scopes: Optional[List[int]] = None,
        cardinality: Optional[int] = None,
        scaling_factors: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the target marginals for the ConstantComponent. If evidence (x) is provided, the marginals correspond to full probability mass on the evidence. If no evidence is provided, the marginals correspond to the marginal probabilities (possibly scaled).

        Args:
            x (torch.Tensor, optional): The input tensor. Defaults to None.
            marginalized_scopes (List[int], optional): List of indices of variables to marginalize over. Defaults to None.
            cardinality (int, optional): The expected cardinality of the variable. Defaults to None.
            scaling_factors (torch.Tensor, optional): Scaling factors for the computation. Defaults to None.

        Returns:
            torch.Tensor: The target marginals for the component.
        """
        assert x.shape[0] == 1  # TODO marginals for batches (but low priority)
        if 0 in marginalized_scopes:
            # if the target variable is marginalized, we return the marginal
            # probabilities
            result = self.probs.clone()
            if scaling_factors is not None:
                result += torch.log(scaling_factors)

            result = torch.exp(result)
        else:
            # otherwise, there is evidence for the target variable, so we remove the
            # probability mass not corresponding to the evidence and set the
            # probability of the evidence to 1
            result = torch.zeros((self.max_cardinality,), dtype=torch.float32).to(x.device)
            result[x[0, 0].long()] = 1.0
        
        if cardinality is not None and result.shape[0] != cardinality:
            raise ValueError(
                f"Computed marginal probabilities for variable {self.var_name} has different cardinality than expected."
            )
        return result


def init_component(var_name: str, parents: List[str], max_cardinality: int = 2, **kwargs) -> Component:
    """
    Initialize a component based on the variable name and its parents.
    If the variable has no parents, it returns a ConstantComponent.
    Otherwise, it returns a PCNComponent.

    Args:
        var_name (str): The name of the variable represented by this component.
        parents (List[str]): List of parent variable names.
        max_cardinality (int): The maximum cardinality of the variable. Defaults to 2 for binary variables.
        **kwargs: Additional keyword arguments for PCNComponent initialization.

    Returns:
        Component: An instance of either PCNComponent or ConstantComponent.
    """
    if not parents:
        return ConstantComponent(var_name, [], max_cardinality=max_cardinality)
    else:
        return PCNComponent(var_name, parents, max_cardinality=max_cardinality, **kwargs)