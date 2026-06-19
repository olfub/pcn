import time
from abc import ABC, abstractmethod
from typing import List

import torch
from simple_einet.einet import Einet


class ScalingMethod(ABC):
    """
    Abstract method class to determin scaling factor for adjusting marginal distriutions of PCs.
    """

    def __init__(self) -> None:
        """
        Create the method class object.
        """
        super().__init__()
        self.time = 0

    @abstractmethod
    def _compute_scaling_factors(
        self,
        einet: Einet,
        target_marginals: torch.Tensor,
        target_marginal_scopes: List[int],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute scaling factors for the given target marginals.

        Args:
            einet (Einet): The Einet model to adjust.
            target_marginals (torch.Tensor): The target marginal distributions to fit.
            target_marginal_scopes (List[int]): The scopes of the target marginals.
            **kwargs: Additional keyword arguments for specific implementations.

        Returns:
            torch.Tensor: The computed scaling factors.
        """
        pass

    def compute_scaling_factors(
        self,
        einet: Einet,
        target_marginals: torch.Tensor,
        target_marginal_scopes: List[int],
        **kwargs
    ) -> torch.Tensor:
        """
        Compute scaling factors for the given target marginals and record time.

        Args:
            einet (Einet): The Einet model to adjust.
            target_marginals (torch.Tensor): The target marginal distributions to fit.
            target_marginal_scopes (List[int]): The scopes of the target marginals.
            **kwargs: Additional keyword arguments for specific implementations.

        Returns:
            torch.Tensor: The computed scaling factors.
        """
        start_time = time.time()
        scaling_factors = self._compute_scaling_factors(
            einet, target_marginals, target_marginal_scopes, **kwargs
        )
        self.time += time.time() - start_time
        return scaling_factors

    def get_time(self) -> float:
        """
        Get the time taken for the last computations.

        Returns:
            float: The time taken in seconds.
        """
        return self.time

    def reset_time(self) -> None:
        """
        Reset the time.
        """
        self.time = 0
