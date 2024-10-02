# pypradie/nn/losses/mse_loss.py

from pypradie.nn.layer import Layer
from pypradie.autograd.tensor import Tensor

class MSELoss(Layer):
    """Computes the Mean Squared Error (MSE) loss."""

    def forward(self, pred, target):
        """Forward pass for MSE loss.

        Args:
            pred (Tensor): Predicted values.
            target (Tensor): Ground truth values.

        Returns:
            Tensor: Scalar tensor representing the loss.
        """
        diff = pred - target
        loss = (diff * diff).sum() / diff.data.size
        return loss

    def __repr__(self):
        return f"{self.__class__.__name__}()"
