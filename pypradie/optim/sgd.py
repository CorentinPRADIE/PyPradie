# pypradie/optim/sgd.py

class SGD:
    """Implements stochastic gradient descent (optionally with momentum)."""

    def __init__(self, parameters, lr=0.1):
        """Initializes the SGD optimizer.

        Args:
            parameters (list): List of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 0.1.
        """
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        """Sets the gradients of all optimized parameters to zero."""
        for p in self.parameters:
            if p.grad is not None:
                p.grad.data *= 0

    def step(self):
        """Performs a single optimization step."""
        for p in self.parameters:
            if p.grad is not None:
                p.data -= self.lr * p.grad.data
