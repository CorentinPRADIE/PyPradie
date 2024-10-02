# pypradie/__init__.py

import numpy as np

from .autograd.tensor import Tensor
from .autograd.no_grad_ctx import no_grad
from .nn import (
    layer,
    Linear,
    ReLU,
    Sigmoid,
    Tanh,
    Sequential,
    MSELoss,
    CrossEntropyLoss,
)
from .optim import SGD, Adam
from .utils import data
from .utils.max_fn import max

# Define common data types
float32 = np.float32
long = np.int64

def tensor(data, requires_grad=False, dtype=None):
    """Factory function to create a Tensor object.

    Args:
        data (array-like): The input data to create the tensor from.
        requires_grad (bool, optional): If True, gradients will be computed for this tensor during backpropagation. Defaults to False.
        dtype (numpy.dtype, optional): The desired data type of the tensor. Defaults to None.

    Returns:
        Tensor: A new Tensor object.
    """
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)

__all__ = [
    "Tensor",
    "Linear",
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Sequential",
    "MSELoss",
    "CrossEntropyLoss",
    "SGD",
    "Adam",
    "max",
    "tensor",
]
