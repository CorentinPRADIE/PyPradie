# pypradie/nn/layers/embedding.py

import numpy as np
from pypradie.autograd.tensor import Tensor
from pypradie.nn.layer import Layer

class Embedding(Layer):
    """Turns indices into dense vectors of fixed size."""

    def __init__(self, vocab_size, embedding_dim):
        """Initializes the embedding layer.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
        """
        super().__init__()
        # Random initialization of weights
        self.weight = Tensor(
            (np.random.rand(vocab_size, embedding_dim) - 0.5) / embedding_dim,
            requires_grad=True
        )
        self.params = [self.weight]

    def forward(self, input):
        """Forward pass for the embedding layer.

        Args:
            input (Tensor): Input tensor containing indices.

        Returns:
            Tensor: Output tensor with embeddings.
        """
        return self.weight[input.data]

    def __repr__(self):
        return f"Embedding(vocab_size={self.weight.data.shape[0]}, embedding_dim={self.weight.data.shape[1]})"
