import random

import numpy as np
import torch
from simple_einet.einet import EinetConfig
from simple_einet.layers.distributions.categorical import Categorical


def make_deterministic(seed, deterministic_cudnn=True):
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def set_smart_einet_config(num_features, max_cardinality=2) -> EinetConfig:
    """
    Set a smart (default) configuration for the Einet.

    Parameters:
        num_features (int): The number of features (variables) in the Einet.
        max_cardinality (int): The maximum cardinality of the variables. Defaults to 2

    Returns:
        EinetConfig: A configuration object for the Einet.
    """
    num_channels = 1
    num_sums = 3
    num_leaves = 5
    num_repetitions = 3
    num_classes = 1
    depth = int(
        torch.log2(torch.tensor(num_features)).floor().item()
    )  # use largest possible depth
    dropout = 0.0
    leaf_type = Categorical
    leaf_kwargs = {"num_bins": max_cardinality}
    layer_type = "einsum"
    structure = "top-down"

    return EinetConfig(
        num_features=num_features,
        num_channels=num_channels,
        num_sums=num_sums,
        num_leaves=num_leaves,
        num_repetitions=num_repetitions,
        num_classes=num_classes,
        depth=depth,
        dropout=dropout,
        leaf_type=leaf_type,
        leaf_kwargs=leaf_kwargs,
        layer_type=layer_type,
        structure=structure,
    )
