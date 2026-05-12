import torch

from .logic import Logic


class BooleanLogic(Logic):
    """Boolean logic implementation for constraint evaluation.

    Provides standard Boolean operations (AND, OR, NOT, LEQ) using
    PyTorch's logical operations for crisp true/false evaluations.
    """

    def __init__(self):
        super().__init__("bool")

    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        """Boolean logical negation.

        Args:
            x: Tensor to negate.

        Returns:
            Boolean tensor with negated values.
        """
        return torch.logical_not(x)

    def NEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Boolean inequality comparison.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Boolean tensor with True where x != y.
        """
        return x != y

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Boolean less than or equal comparison.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Boolean tensor with True where x <= y.
        """
        return x <= y
    
    def LT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Boolean less than comparison.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Boolean tensor with True where x < y.
        """
        return x < y
    
    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        """Boolean logical conjunction of multiple tensors.

        Args:
            *xs: Variable number of tensors to combine with AND.

        Returns:
            Boolean tensor with True where all tensors are True.
        """
        return torch.stack(xs, dim=0).all(dim=0)

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        """Boolean logical disjunction of multiple tensors.

        Args:
            *xs: Variable number of tensors to combine with OR.

        Returns:
            Boolean tensor with True where any tensors are True.
        """
        return torch.stack(xs, dim=0).any(dim=0)
