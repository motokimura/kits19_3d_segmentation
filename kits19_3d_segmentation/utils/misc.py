import random

import numpy as np
import torch


def set_seed(seed):
    """Set seed for randome number generator.
    Args:
        seed (int): seed for randome number generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def configure_cudnn(deterministic=True, benchmark=False):
    """configure cuDNN.
    Args:
        deterministic (bool) : make cuDNN behavior deterministic if True.
        benchmark (bool) : use cuDNN benchmark function if True.
    """
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic
