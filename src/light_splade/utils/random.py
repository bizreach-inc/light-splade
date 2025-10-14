import os
import random

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy and PyTorch to improve reproducibility.

    Args:
        seed (int): Seed value to use for all RNGs. Defaults to 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
