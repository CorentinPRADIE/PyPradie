# pypradie/nn/functional.py

import numpy as np
from pypradie.autograd.tensor import Tensor

def tanh(input):
    """Applies the hyperbolic tangent function element-wise.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor with tanh applied.
    """
    if not isinstance(input, Tensor):
        raise TypeError("Input must be a Tensor.")
    return input.apply(np.tanh)

def sigmoid(input):
    """Applies the sigmoid function element-wise.

    Args:
        input (Tensor): Input tensor.

    Returns:
        Tensor: Output tensor with sigmoid applied.
    """
    if not isinstance(input, Tensor):
        raise TypeError("Input must be a Tensor.")
    return input.apply(lambda x: 1 / (1 + np.exp(-x)))
