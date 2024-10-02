# pypradie/nn/activations.py

import numpy as np
from pypradie.nn.layer import Layer
from pypradie.autograd.tensor import Tensor

class Tanh(Layer):
    """Applies the hyperbolic tangent activation function."""

    def forward(self, input):
        """Forward pass for Tanh activation.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying tanh.
        """
        return input.apply(np.tanh)

    def __repr__(self):
        return "Tanh()"

class Sigmoid(Layer):
    """Applies the sigmoid activation function."""

    def forward(self, input):
        """Forward pass for Sigmoid activation.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying sigmoid.
        """
        return input.apply(lambda x: 1 / (1 + np.exp(-x)))

    def __repr__(self):
        return "Sigmoid()"

class ReLU(Layer):
    """Applies the rectified linear unit activation function."""

    def forward(self, input):
        """Forward pass for ReLU activation.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying ReLU.
        """
        relu_result = np.maximum(0, input.data)
        out = Tensor(relu_result, requires_grad=input.requires_grad, dtype=input.data.dtype)
        if input.requires_grad:
            out.creators = [input]
            out.creation_op = "relu"
        return out

    def __repr__(self):
        return "ReLU()"
