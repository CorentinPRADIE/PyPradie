# pypradie/nn/layers/linear.py

import numpy as np
from pypradie.autograd.tensor import Tensor
from pypradie.nn.layer import Layer

class Linear(Layer):
    """Applies a linear transformation to the incoming data: y = xW^T + b."""

    def __init__(self, in_features, out_features):
        """Initializes the linear layer.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
        """
        super().__init__()
        # Initialize weights and biases
        W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.weight = Tensor(W, requires_grad=True)
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
        self.params = [self.weight, self.bias]

    def forward(self, input):
        """Forward pass for the linear layer.

        Args:
            input (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        output = input.mm(self.weight)
        # Expand bias to match batch size
        batch_size = input.size(0)
        bias_expanded = self.bias.expand(batch_size, self.bias.size(0))
        return output + bias_expanded

    def __repr__(self):
        return f"Linear(in_features={self.weight.data.shape[0]}, out_features={self.weight.data.shape[1]})"
