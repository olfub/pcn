import numpy as np


def random_cpt(rng: np.random.Generator, nr_vars: int) -> np.ndarray:
    """
    Generate a random conditional probability table (CPT) for a given number of variables.

    Args:
        rng (np.random.Generator): Random number generator instance.
        nr_vars (int): Number of variables for which to generate the CPT.

    Returns:
        np.ndarray: A random conditional probability table with shape (2**nr_vars,).

    """
    cpt = rng.random(2**nr_vars)
    return cpt


def sample_from_cpt(
    rng: np.random.Generator, cpt: np.ndarray, var_settings: int | np.ndarray
) -> np.ndarray:
    """
    Sample from a conditional probability table (CPT) based on the given variable settings.

    Args:
        rng (np.random.Generator): Random number generator instance.
        cpt (np.ndarray): Conditional probability table from which to sample.
        var_settings (np.ndarray): Array of variable settings for the samples.

    Returns:
        np.ndarray: Binary samples generated based on the CPT and variable settings.
    """
    if len(cpt) > 1:
        # sample from a random conditional probability table given var_settings (parents)
        base_factor = 2 ** np.arange(var_settings.shape[1])
        cpt_indices = np.sum(var_settings * base_factor, axis=1)
    else:
        # if cpt is a single value, use it directly (no parents)
        cpt_indices = np.zeros(var_settings, dtype=np.int8)

    # select the correct probabilities from the cpt based on the indices
    probs = cpt[cpt_indices]

    # sample from the probabilities
    sample_probs = rng.random(probs.shape[0])

    # return as binary samples
    binary_samples = sample_probs < probs
    return binary_samples.astype(dtype=np.int8)
