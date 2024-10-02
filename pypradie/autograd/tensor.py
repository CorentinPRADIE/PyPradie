# pypradie/autograd/tensor.py

import numpy as np

class Tensor:
    """A class representing a tensor with automatic differentiation capabilities."""

    def __init__(self, data, requires_grad=False, dtype=None):
        """Initialize a Tensor.

        Args:
            data (array-like): The data for the tensor.
            requires_grad (bool, optional): Whether to track operations for gradient computation. Defaults to False.
            dtype (numpy.dtype, optional): The desired data type of the tensor. Defaults to None.
        """
        self.data = np.array(data, dtype=dtype)
        if self.data.ndim == 0:
            self.data = self.data.reshape(1)
        self.requires_grad = requires_grad
        self.grad = None
        self.creators = None  # Tensors that created this tensor
        self.creation_op = None  # Operation that created this tensor
        self.index = None  # For indexing operations
        self.original_shape = None  # For operations like view, transpose, and expand
        self.softmax_output = None  # For log_softmax operation
        self.constant = None  # For operations involving constants

    @property
    def shape(self):
        """tuple: The shape of the tensor."""
        return self.data.shape

    def size(self, dim=None):
        """Get the size of the tensor.

        Args:
            dim (int, optional): The dimension to get the size of. If None, returns the shape. Defaults to None.

        Returns:
            int or tuple: Size of the specified dimension or the entire shape.
        """
        if dim is None:
            return self.data.shape
        else:
            return self.data.shape[dim]

    def dim(self):
        """Get the number of dimensions of the tensor.

        Returns:
            int: The number of dimensions.
        """
        return self.data.ndim

    def backward(self, grad=None):
        """Compute the gradients of the tensor with respect to its creators.

        Args:
            grad (Tensor, optional): The gradient of the loss with respect to this tensor. Defaults to None.
        """
        if not self.requires_grad:
            return

        if grad is None:
            grad = Tensor(np.ones_like(self.data), dtype=self.data.dtype)

        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), dtype=self.data.dtype)
        self.grad.data += grad.data  # Accumulate gradient data

        if self.creators is not None:
            if self.creation_op == "add":
                self.creators[0].backward(grad)
                self.creators[1].backward(grad)

            elif self.creation_op == "sub":
                self.creators[0].backward(grad)
                neg_grad = Tensor(-grad.data, dtype=self.data.dtype)
                self.creators[1].backward(neg_grad)

            elif self.creation_op == "mul":
                self.creators[0].backward(grad * self.creators[1])
                self.creators[1].backward(grad * self.creators[0])

            elif self.creation_op == "div":
                lhs, rhs = self.creators
                grad_lhs = grad / rhs
                grad_rhs = -grad * self.data / (rhs.data ** 2)
                lhs.backward(grad_lhs)
                rhs.backward(grad_rhs)

            elif self.creation_op == "div_const":
                grad_input = grad / self.constant
                self.creators[0].backward(grad_input)

            elif self.creation_op == "rdiv_const":
                grad_input = -grad * self.constant / (self.data ** 2)
                self.creators[0].backward(Tensor(grad_input, dtype=self.data.dtype))

            elif self.creation_op == "mm":
                x = self.creators[0]
                W = self.creators[1]

                grad_data = grad.data
                if grad_data.ndim == 1:
                    grad_data = grad_data.reshape(1, -1)

                # Compute gradient w.r.t input x
                W_T = W.data.T
                grad_x_data = np.dot(grad_data, W_T)
                grad_x = Tensor(grad_x_data.squeeze(), dtype=self.data.dtype)
                x.backward(grad_x)

                # Compute gradient w.r.t weights W
                x_data = x.data
                if x_data.ndim == 1:
                    x_data = x_data.reshape(-1, 1)
                x_T = x_data.T
                grad_w_data = np.dot(x_T, grad_data)
                grad_W = Tensor(grad_w_data, dtype=self.data.dtype)
                W.backward(grad_W)

            elif self.creation_op == "neg":
                neg_grad = Tensor(-grad.data, dtype=self.data.dtype)
                self.creators[0].backward(neg_grad)

            elif self.creation_op == "sum":
                grad_data = grad.data
                if np.isscalar(grad_data):
                    grad_data = np.array(grad_data)
                expanded_grad = grad_data * np.ones_like(self.creators[0].data) / np.prod(self.data.shape)
                self.creators[0].backward(Tensor(expanded_grad, dtype=self.data.dtype))

            elif self.creation_op == "view":
                grad_input = grad.data.reshape(self.creators[0].data.shape)
                self.creators[0].backward(Tensor(grad_input, dtype=self.data.dtype))

            elif self.creation_op == "transpose":
                grad_input = grad.data.T
                self.creators[0].backward(Tensor(grad_input, dtype=self.data.dtype))

            elif self.creation_op == "index":
                grad_input = np.zeros_like(self.creators[0].data)
                grad_input[self.index] = grad.data
                self.creators[0].backward(Tensor(grad_input, dtype=self.data.dtype))

            elif self.creation_op == "expand":
                grad_data = grad.data

                original_shape = self.creators[0].data.shape
                expanded_shape = self.data.shape

                # Pad original_shape with ones at the front to match the number of dimensions
                num_missing_dims = len(expanded_shape) - len(original_shape)
                padded_original_shape = (1,) * num_missing_dims + original_shape

                # Identify axes where original size is 1 and expanded size > 1
                sum_axes = tuple(
                    i for i, (orig_size, expanded_size) in enumerate(zip(padded_original_shape, expanded_shape))
                    if orig_size == 1 and expanded_size > 1
                )

                grad_input = grad_data
                if sum_axes:
                    grad_input = grad_input.sum(axis=sum_axes, keepdims=True)

                # Reshape grad_input to match the original shape
                grad_input = grad_input.reshape(original_shape)
                self.creators[0].backward(Tensor(grad_input, dtype=self.data.dtype))

            elif self.creation_op == "relu":
                grad_input = grad.data.copy()
                grad_input[self.data <= 0] = 0
                self.creators[0].backward(Tensor(grad_input, dtype=self.data.dtype))

            elif self.creation_op == "log_softmax":
                grad_input = grad.data.copy()
                softmax_output = self.softmax_output  # Already computed during forward pass

                # Compute the gradient w.r.t. the input of log_softmax
                grad_softmax = grad_input - (grad_input.sum(axis=1, keepdims=True) * softmax_output)

                self.creators[0].backward(Tensor(grad_softmax, dtype=self.data.dtype))

    def __add__(self, other):
        """Element-wise addition of tensors.

        Args:
            other (Tensor or scalar): The tensor or scalar to add.

        Returns:
            Tensor: Resulting tensor after addition.
        """
        if isinstance(other, Tensor):
            data = self.data + other.data
            requires_grad = self.requires_grad or other.requires_grad
            out = Tensor(data, requires_grad=requires_grad, dtype=self.data.dtype)
            if out.requires_grad:
                out.creators = [self, other]
                out.creation_op = "add"
            return out
        else:
            data = self.data + other
            out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
            if out.requires_grad:
                out.creators = [self]
                out.creation_op = "add_const"
                out.constant = other
            return out

    def __radd__(self, other):
        """Right-side addition."""
        return self.__add__(other)

    def __sub__(self, other):
        """Element-wise subtraction of tensors.

        Args:
            other (Tensor or scalar): The tensor or scalar to subtract.

        Returns:
            Tensor: Resulting tensor after subtraction.
        """
        if isinstance(other, Tensor):
            data = self.data - other.data
            requires_grad = self.requires_grad or other.requires_grad
            out = Tensor(data, requires_grad=requires_grad, dtype=self.data.dtype)
            if out.requires_grad:
                out.creators = [self, other]
                out.creation_op = "sub"
            return out
        else:
            data = self.data - other
            out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
            if out.requires_grad:
                out.creators = [self]
                out.creation_op = "sub_const"
                out.constant = other
            return out

    def __rsub__(self, other):
        """Right-side subtraction."""
        return (-self) + other

    def __mul__(self, other):
        """Element-wise multiplication of tensors.

        Args:
            other (Tensor or scalar): The tensor or scalar to multiply.

        Returns:
            Tensor: Resulting tensor after multiplication.
        """
        if isinstance(other, Tensor):
            data = self.data * other.data
            requires_grad = self.requires_grad or other.requires_grad
            out = Tensor(data, requires_grad=requires_grad, dtype=self.data.dtype)
            if out.requires_grad:
                out.creators = [self, other]
                out.creation_op = "mul"
            return out
        else:
            data = self.data * other
            out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
            if out.requires_grad:
                out.creators = [self]
                out.creation_op = "mul_const"
                out.constant = other
            return out

    def __rmul__(self, other):
        """Right-side multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other):
        """Element-wise division of tensors.

        Args:
            other (Tensor or scalar): The tensor or scalar to divide by.

        Returns:
            Tensor: Resulting tensor after division.
        """
        if isinstance(other, Tensor):
            data = self.data / other.data
            requires_grad = self.requires_grad or other.requires_grad
            out = Tensor(data, requires_grad=requires_grad, dtype=self.data.dtype)
            if out.requires_grad:
                out.creators = [self, other]
                out.creation_op = "div"
            return out
        else:
            data = self.data / other
            out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
            if out.requires_grad:
                out.creators = [self]
                out.creation_op = "div_const"
                out.constant = other
            return out

    def __rtruediv__(self, other):
        """Right-side division."""
        data = other / self.data
        out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
        if out.requires_grad:
            out.creators = [self]
            out.creation_op = "rdiv_const"
            out.constant = other
        return out

    def __neg__(self):
        """Negation of the tensor.

        Returns:
            Tensor: Negated tensor.
        """
        data = -self.data
        out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
        if self.requires_grad:
            out.creators = [self]
            out.creation_op = "neg"
        return out

    def mm(self, other):
        """Matrix multiplication of tensors.

        Args:
            other (Tensor): The tensor to multiply with.

        Returns:
            Tensor: Resulting tensor after matrix multiplication.
        """
        data = np.dot(self.data, other.data)
        requires_grad = self.requires_grad or other.requires_grad
        out = Tensor(data, requires_grad=requires_grad, dtype=self.data.dtype)
        if out.requires_grad:
            out.creators = [self, other]
            out.creation_op = "mm"
        return out

    def sum(self):
        """Compute the sum of all elements in the tensor.

        Returns:
            Tensor: A tensor containing the sum.
        """
        data = self.data.sum()
        out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
        if self.requires_grad:
            out.creators = [self]
            out.creation_op = "sum"
        return out

    def view(self, *shape):
        """Reshape the tensor.

        Args:
            *shape: The desired shape.

        Returns:
            Tensor: Reshaped tensor.
        """
        data = self.data.reshape(shape)
        out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
        if self.requires_grad:
            out.creators = [self]
            out.creation_op = 'view'
            out.original_shape = self.data.shape
        return out

    def transpose(self):
        """Transpose the tensor.

        Returns:
            Tensor: Transposed tensor.
        """
        data = self.data.T
        out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
        if self.requires_grad:
            out.creators = [self]
            out.creation_op = 'transpose'
        return out

    def expand(self, *sizes):
        """Expand the tensor to a new shape.

        Args:
            *sizes: The desired expanded sizes.

        Returns:
            Tensor: Expanded tensor.
        """
        expanded_data = np.broadcast_to(self.data, sizes)
        out = Tensor(expanded_data, requires_grad=self.requires_grad, dtype=self.data.dtype)
        if self.requires_grad:
            out.creators = [self]
            out.creation_op = 'expand'
            out.original_shape = self.data.shape
        return out

    def __getitem__(self, index):
        """Get items using indexing.

        Args:
            index (int, slice, or tuple): The indices to select.

        Returns:
            Tensor: The selected elements as a new tensor.
        """
        data = self.data[index]
        out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
        if self.requires_grad:
            out.creators = [self]
            out.creation_op = 'index'
            out.index = index
        return out

    def __repr__(self):
        """String representation of the tensor.

        Returns:
            str: The string representation.
        """
        return f"Tensor({self.data})"

    # Additional methods for compatibility and utility
    def item(self):
        """Get a Python scalar from a single-element tensor.

        Returns:
            scalar: The scalar value.
        """
        return self.data.item()

    def numpy(self):
        """Convert the tensor data to a NumPy array.

        Returns:
            numpy.ndarray: The tensor data as a NumPy array.
        """
        return self.data

    def zero_(self):
        """Set the tensor data to zero in-place."""
        self.data.fill(0)

    def detach(self):
        """Return a new tensor detached from the current computation graph.

        Returns:
            Tensor: A new tensor with the same data but no gradient tracking.
        """
        return Tensor(self.data.copy(), requires_grad=False, dtype=self.data.dtype)

    def clone(self):
        """Create a copy of the tensor.

        Returns:
            Tensor: A new tensor with the same data and requires_grad flag.
        """
        return Tensor(self.data.copy(), requires_grad=self.requires_grad, dtype=self.data.dtype)

    def requires_grad_(self, requires_grad=True):
        """Set the requires_grad flag in-place.

        Args:
            requires_grad (bool, optional): The new value for requires_grad. Defaults to True.
        """
        self.requires_grad = requires_grad

    def apply(self, func):
        """Apply a function element-wise to the tensor.

        Args:
            func (callable): The function to apply.

        Returns:
            Tensor: A new tensor with the function applied.
        """
        data = func(self.data)
        out = Tensor(data, requires_grad=self.requires_grad, dtype=self.data.dtype)
        return out
