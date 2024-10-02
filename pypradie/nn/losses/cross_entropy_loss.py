# pypradie/nn/losses/cross_entropy_loss.py

import numpy as np
from pypradie.autograd.tensor import Tensor
from pypradie.nn.layer import Layer

class CrossEntropyLoss(Layer):
    """Computes the Cross-Entropy loss between input and target."""

    def forward(self, input, target):
        """Forward pass for Cross-Entropy loss.

        Args:
            input (Tensor): Input tensor (logits) of shape (batch_size, num_classes).
            target (Tensor): Target tensor (labels) of shape (batch_size,).

        Returns:
            Tensor: Scalar tensor representing the loss.
        """
        # Compute log softmax
        max_logits = np.max(input.data, axis=1, keepdims=True)
        shifted_logits = input.data - max_logits  # For numerical stability
        exp_logits = np.exp(shifted_logits)
        sum_exp_logits = np.sum(exp_logits, axis=1, keepdims=True)
        log_probs = shifted_logits - np.log(sum_exp_logits)

        # Negative log likelihood
        batch_size = target.data.shape[0]
        indices = (np.arange(batch_size), target.data.astype(int))
        nll = -log_probs[indices]

        # Compute mean loss
        loss_value = np.sum(nll) / batch_size
        loss = Tensor(loss_value, requires_grad=True)
        if loss.requires_grad:
            loss.creators = [input]
            loss.creation_op = "cross_entropy_loss"
        return loss
