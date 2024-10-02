# pypradie/utils/max_fn.py

import numpy as np

def max(tensor, dim=None):
    """Mimics PyTorch's max function.

    Args:
        tensor (numpy.ndarray): The input tensor.
        dim (int, optional): The dimension to reduce. If None, reduces all dimensions. Defaults to None.

    Returns:
        tuple:
            - values (numpy.ndarray): The maximum values.
            - indices (numpy.ndarray or None): The indices of the maximum values along the specified dimension.
    """
    if dim is None:
        max_value = np.max(tensor)
        return max_value, None
    else:
        max_values = np.max(tensor, axis=dim)
        max_indices = np.argmax(tensor, axis=dim)
        return max_values, max_indices
