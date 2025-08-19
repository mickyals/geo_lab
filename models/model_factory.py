# Universal factory: takes architecture config and returns model instance
# Supports:
# - single networks (SIREN, FourierNet)
# - multi-network systems (Neural DMD)
# - hypernetworks
# Handles nested configs and ensures modularity

from .coordinate_models.builders import build_model
from .coordinate_models.core.embeddings import EMBEDDINGS
from .coordinate_models.core.initializations import INITIALIZERS
from .coordinate_models.core.activations import ACTIVATIONS
from .coordinate_models.core.layers import LAYERS


def create_model(config):
    """
    Create a model instance based on the provided configuration.

    Args:
        config (dict): The configuration for the model.

    Returns:
        torch.nn.Module: The created model instance.

    Raises:
        ValueError: If the configuration is missing a required key, or if there
        is an error creating the model.
    """
    try:
        # Build the model using the provided configuration
        model = build_model(config)
    except KeyError as e:
        # Raise a ValueError if a required key is missing from the configuration
        raise ValueError(f"Missing key in config: {e}")
    except Exception as e:
        # Raise a ValueError if there is an error creating the model
        raise ValueError(f"Error creating model: {e}")

    return model



def list_models():
    """
    List all available models.

    Returns:
        list: A list of strings representing the names of the available models.
    """
    # Import the model registry from the builders module
    from .coordinate_models.builders import MODEL_REGISTRY

    # Return the keys of the model registry, which represent the available models
    return list(MODEL_REGISTRY.keys())



def list_embeddings():
    """
    Return a list of all available embeddings.

    Returns:
        list: A list of strings representing the names of the available embeddings.
    """
    # Import the embeddings registry from the core module
    from .coordinate_models.core.embeddings import EMBEDDINGS

    # Return the keys of the embeddings registry, which represent the available embeddings
    return list(EMBEDDINGS.keys())



def list_initializers():
    """
    Return a list of all available initializers.

    Returns:
        list: A list of strings representing the names of the available initializers.
    """
    # Return the keys of the initializers registry, which represent the available initializers
    return list(INITIALIZERS.keys())



def list_activations():
    """
    Return a list of all available activations.

    Returns:
        list: A list of strings representing the names of the available activations.
    """
    # Return the keys of the activations registry, which represent the available activations
    return list(ACTIVATIONS.keys())



def list_layers():
    """
    Return a list of all available layers.

    Returns:
        list: A list of strings representing the names of the available layers.
    """
    # Return the keys of the layers registry, which represent the available layers
    return list(LAYERS.keys())
