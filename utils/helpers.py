#simple stuff like setting seed and getting device and setting device and building a model summary
import random
import numpy as np
import torch
from torchinfo import summary


# ==============================================================#
###                    MODEL SUMMARY                          ###
# ==============================================================#
def model_summary(model, input_size, device=None, verbose=1):
    """
    Prints a summary of the model architecture.

    Args:
        model (torch.nn.Module): Model to summarize.
        input_size (tuple): Example input shape (excluding batch size).
        device (torch.device, optional): Device to use for summary.
        verbose (int): Verbosity level for summary.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    summary(model, input_size=input_size, device=str(device), verbose=verbose)


# ==============================================================#
###                        SEED SETTER                        ###
# ==============================================================#
def set_seed(seed: int):
    """
    Fix random seeds across libraries for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==============================================================#
###                        DEVICE HELPERS                     ###
# ==============================================================#
def set_device():
    """
    Returns the best available device (GPU if available, else CPU).
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_count():
    """
    Returns the number of available CUDA devices.
    """
    return torch.cuda.device_count()


def get_device_name(index=0):
    """
    Get the name of a specific GPU (if available).

    Args:
        index (int): GPU index.
    """
    if torch.cuda.is_available() and index < torch.cuda.device_count():
        return torch.cuda.get_device_name(index)
    return "CPU"
