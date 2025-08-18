# Build networks from config files: can build single or multi-network architectures
# Example: Neural DMD can use builders to combine encoder + dynamics + decoder networks'

from .architectures.siren import SIREN_REGISTRY
from .core.embeddings import get_embedding

MODEL_REGISTRY = {
    **SIREN_REGISTRY,

}


def build_model(config):
    """
    Builds a model based on the provided configuration.

    Args:
        config (dict): The configuration for the model.

    Returns:
        torch.nn.Module: The built model.

    Raises:
        ValueError: If the model with the given name does not exist.
    """
    # Extract the model configuration from the overall configuration
    model_config = config.get("model")
    model_name = model_config.get("name")

    # Check if the model exists in the registry
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model with name {model_name} does not exist.")

    # Get the class of the model from the registry
    model_cls = MODEL_REGISTRY[model_name]

    # Extract the keyword arguments for the model
    model_kwargs = model_config.get("model_kwargs", {})

    # Check if an embedding should be used and update the model kwargs accordingly
    if config.get("use_embedding", False):
        embedding_type = config.get("embedding_name", "GAUSSIAN_POSITIONAL")
        embedding_kwargs = config.get("embedding_kwargs", {})
        model_kwargs.update({
            "use_embedding": True,
            "embedding_type": embedding_type,
            "embedding_kwargs": embedding_kwargs
        })

    # Instantiate and return the model
    return model_cls(**model_kwargs)
