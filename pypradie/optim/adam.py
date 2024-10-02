# pypradie/optim/adam.py

import numpy as np

class Adam:
    """Implements the Adam optimization algorithm."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """Initializes the Adam optimizer.

        Args:
            params (list): List of parameters to optimize.
            lr (float, optional): Learning rate. Defaults to 0.001.
            betas (tuple, optional): Coefficients used for computing running averages of gradient and its square. Defaults to (0.9, 0.999).
            eps (float, optional): Term added to the denominator to improve numerical stability. Defaults to 1e-8.
            weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.
        """
        if not isinstance(params, list):
            params = list(params)

        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize first and second moment vectors
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
        self.t = 0  # Time step

    def step(self):
        """Performs a single optimization step."""
        self.t += 1  # Increment time step

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            # Get gradient and apply weight decay if applicable
            grad = param.grad.data
            if self.weight_decay != 0:
                grad += self.weight_decay * param.data

            # Update biased first moment estimate
            self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad

            # Update biased second raw moment estimate
            self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad ** 2)

            # Compute bias-corrected first and second moment estimates
            m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
            v_hat = self.v[i] / (1 - self.betas[1] ** self.t)

            # Update parameters
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Sets the gradients of all optimized parameters to zero."""
        for param in self.params:
            param.grad = None
