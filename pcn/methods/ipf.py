from typing import Any, Dict, List, Tuple

import torch
from simple_einet.einet import Einet

from pcn.methods.scaling_methods import ScalingMethod


def iterative_proportional_fitting(
    einet: Einet,
    target_marginals: List[torch.Tensor],
    target_marginal_scopes: List[int],
    epsilon: float,
    max_iterations: int,
    verbose: bool,
    return_stats: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Compute scaling factors for the given target marginals using iterative proportional fitting.

    Args:
        einet (Einet): The Einet model to adjust.
        target_marginals (List[torch.Tensor]): The target marginal distributions to fit.
        target_marginal_scopes (List[int]): The scopes of the target marginals.
        epsilon (float): The convergence threshold for the average error.
        max_iterations (int): The maximum number of iterations to perform.
        verbose (bool): If True, print detailed information during the process.
        return_stats (bool): If True, additionally return information on the convergence process.

    Returns:
        torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]: The computed scaling factors,
            and optionally information on the convergence process if return_stats is True.
    """
    einet_num_features = einet.config.num_features

    # transform target marginals list to tensor and set scaling factors corresponding to "dead" values to 0
    einet_cardinality = einet.config.leaf_kwargs["num_bins"]
    target_marginals_tensor = torch.zeros(
        (len(target_marginals), einet_cardinality)
    ).to(target_marginals[0].device)
    scaling_factors = torch.ones((einet_num_features, einet_cardinality)).to(
        target_marginals[0].device
    )
    for i in range(len(target_marginals)):
        target_marginals_tensor[i, : target_marginals[i].shape[0]] = target_marginals[i]
        if target_marginals[i].shape[0] < einet_cardinality:
            scaling_factors[
                target_marginal_scopes[i], target_marginals[i].shape[0] :
            ] = 0.0
    target_marginals = target_marginals_tensor

    if not isinstance(einet, Einet):
        # Currently, this code is written to specifically compute scaling factors for an Einet model.
        raise TypeError("The 'einet' parameter must be of type 'Einet'.")
    if target_marginals.shape[0] != len(target_marginal_scopes):
        # The number of target marginals must match the number of target marginal scopes.
        raise ValueError(
            "The shape of target_marginals must match the length of target_marginal_scopes."
        )

    # compute current marginals
    # TODO vectorize
    current_marginals = torch.zeros_like(target_marginals)
    # iterate over all variables to be marginalized
    for i in range(target_marginals.shape[0]):
        # iterate over all values of the variable to be marginalized
        for j in range(target_marginals.shape[1]):
            input_tensor = (
                torch.Tensor([[j] * einet_num_features])
                .float()
                .to(target_marginals.device)
            )
            # the index to not marginalize is given by the target marginal scopes
            einet_feature_index = target_marginal_scopes[i]
            marginalized_scopes = [
                k for k in range(einet_num_features) if k != einet_feature_index
            ]
            # compute the marginal probability for the current variable and value
            current_marginals[i, j] = torch.exp(
                einet.forward(
                    input_tensor, marginalized_scopes, scaling_factors=scaling_factors
                )
            ).item()

    if verbose:
        print(f"Target Marginal Probabilities: \n{target_marginals}")
        print(f"Initial Marginal Probabilities: \n{current_marginals}")

    # the intial average error is calculated and all scaling factors are initialized with constant 1
    # (constant 1 means no scaling, i.e., the current marginals are already equal to the target marginals)
    average_error = torch.mean(torch.abs(current_marginals - target_marginals))
    iteration = 0
    iteration_errors: List[float] = []
    scaling_factors = torch.ones((einet_num_features, target_marginals.shape[1])).to(
        target_marginals.device
    )

    # iterative proportional fitting loop
    while average_error > epsilon and iteration < max_iterations:
        # iterate over all features where the marginal probabilities should be adjusted
        for i in range(target_marginals.shape[0]):
            # compute the scaling factor for the current feature
            einet_index = target_marginal_scopes[i]
            scaling_factors[einet_index] *= target_marginals[i] / current_marginals[i]

            # to save compute, now only compute the marginal probabilities for the next feature
            # this does not affect the correctness of the algorithm, as the next index is sufficient
            next_index = (i + 1) % target_marginals.shape[0]

            # the index to not marginalize is given by the target marginal scopes
            einet_feature_index = target_marginal_scopes[next_index]
            marginalized_scopes = [
                k for k in range(einet_num_features) if k != einet_feature_index
            ]

            # iterate over all values of the variable to be marginalized
            # TODO vectorize
            for j in range(target_marginals.shape[1]):
                input_tensor = (
                    torch.Tensor([[j] * einet_num_features])
                    .float()
                    .to(target_marginals.device)
                )
                # compute the marginal probability for the next variable and value
                current_marginals[next_index, j] = torch.exp(
                    einet.forward(
                        input_tensor,
                        marginalized_scopes,
                        scaling_factors=scaling_factors,
                    )
                ).item()

        # note, that this error is a bit larger than it would actually be, as the current_marginals are not
        # fully updated; the true error is smaller (therefore, the functionality is correct and a solution with
        # an error smaller than epsilon is guaranteed)
        average_error = torch.mean(torch.abs(current_marginals - target_marginals))
        iteration_errors.append(average_error.item())
        iteration += 1
        if verbose:
            print(f"Iteration {iteration}: Average Error: {average_error.item()}")

    if iteration == max_iterations:
        # raise RuntimeError(
        #     "Iterative Proportional Fitting did not converge within the maximum number of iterations."
        # )
        print(
            f"Warning: Iterative Proportional Fitting did not converge within the maximum number of iterations. "
            f"Final average error: {average_error.item()}"
        )

    if verbose:
        print(f"Final Marginal Probabilities: \n{current_marginals}")
        print(f"Scaling Factors: \n{scaling_factors}")

    if return_stats:
        stats = {
            "iteration_errors": iteration_errors,
            "num_iterations": iteration,
            "final_error": average_error.item(),
        }
        return scaling_factors, stats

    return scaling_factors


class IPF(ScalingMethod):
    """
    Iterative Proportional Fitting (IPF) method for adjusting marginal distributions of PCs.
    """

    def __init__(
        self,
        default_epsilon: float = 1e-6,
        default_max_iterations: int = 1000,
        default_verbose: bool = False,
        default_return_stats: bool = False,
    ) -> None:
        """
        Create the IPF method object.

        Args:
            default_epsilon (float): The convergence threshold for the average error.
            default_max_iterations (int): The maximum number of iterations to perform.
            default_verbose (bool): If True, print detailed information during the process.
            default_return_stats (bool): If True, return additional information on the convergence process.
        """
        super().__init__()
        self.default_epsilon = default_epsilon
        self.default_max_iterations = default_max_iterations
        self.default_verbose = default_verbose
        self.default_return_stats = default_return_stats
        
    def _compute_scaling_factors(
        self,
        einet: Einet,
        target_marginals: torch.Tensor,
        target_marginal_scopes: List[int],
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute scaling factors for the given target marginals using iterative proportional fitting.

        Args:
            einet (Einet): The Einet model used to compute the marginal probabilities after each step.
            target_marginals (torch.Tensor): The target marginal distributions to fit.
            target_marginal_scopes (List[int]): The scopes of the target marginals.
            **kwargs: Additional arguments for the iterative proportional fitting method.

        Returns:
            torch.Tensor: The computed scaling factors.
        """
        if "epsilon" in kwargs:
            epsilon = kwargs["epsilon"]
        else:
            epsilon = self.default_epsilon
        if "max_iterations" in kwargs:
            max_iterations = kwargs["max_iterations"]
        else:
            max_iterations = self.default_max_iterations
        if "verbose" in kwargs:
            verbose = kwargs["verbose"]
        else:
            verbose = self.default_verbose
        if "return_stats" in kwargs:
            return_stats = kwargs["return_stats"]
        else:
            return_stats = self.default_return_stats

        return iterative_proportional_fitting(
            einet,
            target_marginals,
            target_marginal_scopes,
            epsilon=epsilon,
            max_iterations=max_iterations,
            verbose=verbose,
            return_stats=return_stats,
        )
