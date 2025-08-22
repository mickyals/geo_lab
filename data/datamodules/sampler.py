import numpy as np


######
## these should probably me moved to under data module for dataset logic not processing logic.
####

def sampler(seed: int = None) -> np.random.Generator:
    """
    Create a numpy random Generator object with the given seed.

    Parameters
    ----------
    seed : int
        The seed value to use to genera te the random numbers.

    Returns
    -------
    np.random.Generator
        A numpy random Generator object with the given seed.
    """
    return np.random.default_rng(seed)



def shuffler(sampler, total_samples, num_sensors_per_batch):
    """
    Create a random sampler that shuffles the data to sample from.

    Parameters
    ----------
    sampler : np.random.Generator
        A numpy random Generator object to use to generate the random numbers.
    total_samples : int
        The total number of samples to sample from.
    num_sensors : int
        The number of sensors to sample from.

    Returns
    -------
    np.ndarray
        An array of indices of shape (num_sensors,) where each index is into the
        total_samples array.
    """
    # Create a random number generator with the given seed
    rng = sampler
    # Return a shuffled array of indices of shape (num_sensors,)
    # where each index is into the total_samples array
    return rng.choice(total_samples, size=num_sensors, replace=False)








