# pypradie/autograd/no_grad_ctx.py

from contextlib import contextmanager

# Global variable to track the state of gradient computation
_grad_enabled = True

@contextmanager
def no_grad():
    """Context manager to disable gradient computation temporarily.

    Yields:
        None: Allows code execution within the context where gradients are not computed.
    """
    global _grad_enabled
    # Store the current state
    prev_grad_enabled = _grad_enabled
    # Disable gradient tracking
    _grad_enabled = False
    try:
        yield
    finally:
        # Restore the previous state after the context ends
        _grad_enabled = prev_grad_enabled

def is_grad_enabled():
    """Check if gradient computation is currently enabled.

    Returns:
        bool: True if gradients are enabled, False otherwise.
    """
    return _grad_enabled
