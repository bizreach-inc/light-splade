import torch


def contiguous(model: torch.nn.Module) -> torch.nn.Module:
    """Ensure module parameters are contiguous in memory.

    Safe serialization formats (e.g., safetensors) require parameter tensors to be contiguous. This helper iterates all
    parameters and replaces their data with a contiguous copy when necessary.

    Args:
        model (torch.nn.Module): Module whose parameters will be checked.

    Returns:
        torch.nn.Module: The same module instance with contiguous parameter data.
    """
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()
    return model
