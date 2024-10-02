# pypradie/nn/layers/rnn_cell.py

import numpy as np
from pypradie.nn.layers.linear import Linear
from pypradie.nn.activations import Sigmoid, Tanh
from pypradie.autograd.tensor import Tensor
from pypradie.nn.layer import Layer

class RNNCell(Layer):
    """An Elman RNN cell."""

    def __init__(self, input_size, hidden_size, output_size, activation="tanh"):
        """Initializes the RNN cell.

        Args:
            input_size (int): Number of features in the input.
            hidden_size (int): Number of features in the hidden state.
            output_size (int): Number of features in the output.
            activation (str, optional): Activation function ('tanh' or 'sigmoid'). Defaults to 'tanh'.
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if activation == "sigmoid":
            self.activation = Sigmoid()
        elif activation == "tanh":
            self.activation = Tanh()
        else:
            raise ValueError("Unsupported activation function")

        # Weight matrices
        self.w_ih = Linear(input_size, hidden_size)
        self.w_hh = Linear(hidden_size, hidden_size)
        self.w_ho = Linear(hidden_size, output_size)

        # Collect parameters
        self.params = []
        self.params += self.w_ih.get_parameters()
        self.params += self.w_hh.get_parameters()
        self.params += self.w_ho.get_parameters()

    def forward(self, input, hidden):
        """Forward pass for the RNN cell.

        Args:
            input (Tensor): Input tensor at the current time step.
            hidden (Tensor): Hidden state tensor from the previous time step.

        Returns:
            tuple: A tuple containing:
                - output (Tensor): Output tensor at the current time step.
                - new_hidden (Tensor): Updated hidden state tensor.
        """
        combined = self.w_ih(input) + self.w_hh(hidden)
        new_hidden = self.activation(combined)
        output = self.w_ho(new_hidden)
        return output, new_hidden

    def init_hidden(self, batch_size=1):
        """Initializes the hidden state to zeros.

        Args:
            batch_size (int, optional): Batch size. Defaults to 1.

        Returns:
            Tensor: Zero-initialized hidden state tensor.
        """
        return Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=True)

    def __repr__(self):
        return f"RNNCell(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, activation={self.activation})"
