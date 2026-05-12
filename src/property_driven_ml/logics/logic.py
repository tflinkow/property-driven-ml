import torch

from abc import ABC, abstractmethod
from functools import reduce


# TODO: possibly not very clean, but we expect deriving classes to override
# either EQ or NEQ, LEQ or GEQ, LT or GT, (whatever suits them) but don't really enforce it
# (could lead to infinite recursion during development)
class Logic(ABC):
    """Abstract base class for logical frameworks used in property-driven learning.

    Provides a common interface for different logical systems (Boolean, fuzzy, STL)
    used to evaluate constraints and properties in neural network training.

    Args:
        name: Human-readable name for this logic framework.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        """Logical negation in this logic framework.

        Args:
            x: Tensor to negate.

        Returns:
            Tensor representing NOT x in this logic.
        """
        pass

    def NEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Inequality in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x != y in this logic.
        """
        return self.NOT(self.EQ(x, y))

    def EQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Equality in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x == y in this logic.
        """
        return self.NOT(self.NEQ(x, y))

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Less than or equal comparison in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x <= y in this logic.
        """
        return self.GEQ(y, x)

    def GEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Greater than or equal comparison in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x >= y in this logic.
        """
        return self.LEQ(y, x)

    def LT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Less than comparison in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x < y in this logic.
        """
        return self.GT(y, x)

    def GT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Greater than comparison in this logic framework.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x > y in this logic.
        """
        return self.LT(y, x)

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        """Logical conjunction of multiple tensors.

        Args:
            *xs: Variable number of tensors to combine with AND.

        Returns:
            Tensor representing the conjunction of all inputs.

        Raises:
            ValueError: If fewer than 2 arguments provided.
        """
        if len(xs) < 2:
            raise ValueError(
                "AND requires at least 2 arguments. If you have a list xs, make sure to unpack it with *xs."
            )

        return reduce(self.AND2, xs)

    def AND2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Binary logical conjunction.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Tensor representing x AND y.
        """
        return self.AND(x, y)

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        """Logical disjunction of multiple tensors.

        Args:
            *xs: Variable number of tensors to combine with OR.

        Returns:
            Tensor representing the disjunction of all inputs.

        Raises:
            ValueError: If fewer than 2 arguments provided.
        """
        if len(xs) < 2:
            raise ValueError(
                "OR requires at least 2 arguments. If you have a list xs, make sure to unpack it with *xs."
            )

        return reduce(self.OR2, xs)

    def OR2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Binary logical disjunction.

        Args:
            x: First tensor.
            y: Second tensor.

        Returns:
            Tensor representing x OR y.
        """
        return self.OR(x, y)

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logical implication in this logic.

        Args:
            x: Antecedent tensor.
            y: Consequent tensor.

        Returns:
            Tensor representing x ==> y.
        """
        return self.OR(self.NOT(x), y)

    def EQUIV(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Logical equivalence in this logic.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Tensor representing x <==> y.
        """
        return self.AND(self.IMPL(x, y), self.IMPL(y, x))
