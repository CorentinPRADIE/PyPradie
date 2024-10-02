# pypradie/nn/models/sequential.py

from pypradie.nn.layer import Layer

class Sequential(Layer):
    """A sequential container of layers."""

    def __init__(self, *layers):
        """Initializes the Sequential model.

        Args:
            *layers: Variable number of layer instances.
        """
        super().__init__()
        self.layers = list(layers)

    def forward(self, input):
        """Forward pass through all layers.

        Args:
            input (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after passing through all layers.
        """
        for layer in self.layers:
            input = layer(input)
        return input

    def get_parameters(self):
        """Collects parameters from all layers.

        Returns:
            list: A list of all parameters in the sequential model.
        """
        params = []
        for layer in self.layers:
            params.extend(layer.get_parameters())
        return params

    def __repr__(self):
        layer_str = '\n  '.join(str(layer) for layer in self.layers)
        return f"{self.__class__.__name__}(\n  {layer_str}\n)"
