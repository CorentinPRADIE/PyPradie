# pypradie/nn/layer.py

class Layer:
    """Base class for all neural network layers."""

    def __init__(self):
        """Initializes the layer."""
        self.params = []

    def get_parameters(self):
        """Returns the parameters of the layer.

        Returns:
            list: A list of parameters (tensors) in the layer.
        """
        return self.params

    def forward(self, *args):
        """Defines the computation performed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def __call__(self, *args):
        """Makes the layer callable.

        Args:
            *args: Input arguments.

        Returns:
            Tensor: The result of the forward pass.
        """
        return self.forward(*args)
