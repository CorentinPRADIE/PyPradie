# pypradie/nn/layers/softmax.py

import numpy as np
from pypradie.autograd.tensor import Tensor
from pypradie.nn.layer import Layer

class Softmax(Layer):
    """Applies the Softmax function to an input Tensor."""

    def __init__(self, dim=-1):
        """Initializes the Softmax layer.

        Args:
            dim (int, optional): The dimension over which Softmax is computed. Defaults to -1.
        """
        super().__init__()
        self.dim = dim

    def forward(self, input):
        """Forward pass for Softmax.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying Softmax.
        """
        # Subtract the max value for numerical stability
        shifted_input = input.data - np.max(input.data, axis=self.dim, keepdims=True)
        exp_x = np.exp(shifted_input)
        softmax_x = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        return Tensor(softmax_x, requires_grad=input.requires_grad)

    def __repr__(self):
        return f"Softmax(dim={self.dim})"
