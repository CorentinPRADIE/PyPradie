# pypradie/nn/losses/__init__.py

from .mse_loss import MSELoss
from .cross_entropy_loss import CrossEntropyLoss

__all__ = [
    'MSELoss',
    'CrossEntropyLoss'
]
