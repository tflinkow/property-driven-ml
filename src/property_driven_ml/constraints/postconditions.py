import torch
import torch.nn.functional as F
import torch.linalg as LA

import itertools

import numpy as np

from abc import ABC, abstractmethod
from typing import Callable

from ..logics.logic import Logic
from ..logics.fuzzy_logics import FuzzyLogic


class Postcondition(ABC):
    """
    Abstract base class for postconditions/ output properties.
    """

    @abstractmethod
    def build_postcondition(self, *args, **kwargs) -> Callable[[Logic], torch.Tensor]:
        """
        Get the postcondition function for this property.

        This method should be implemented by subclasses with their specific signature.
        Common parameters include:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor.
            y_target: Target output tensor.

        Additional parameters may be specific to the postcondition implementation.

        Args:
            *args: Positional arguments specific to the postcondition.
            **kwargs: Keyword arguments specific to the postcondition.

        Returns:
            Function that takes a Logic instance and returns postcondition tensor.
        """
        pass


class StandardRobustnessPostcondition(Postcondition):
    """
    Postcondition ensuring model robustness to adversarial perturbations.

    Enforces that the change in output probabilities between original and
    adversarial inputs remains within a specified threshold delta.

    Args:
        device: PyTorch device for tensor computations.
        delta: Maximum allowed change in output probabilities.
    """

    def __init__(self, device: torch.device, delta: float | torch.Tensor):
        self.device = device

        assert 0.0 <= delta <= 1.0, (  # nosec
            "delta is a probability and should be within the range [0, 1]"
        )
        self.delta = torch.as_tensor(delta, device=self.device)

    def build_postcondition(
        self,
        N: torch.nn.Module,
        x: torch.Tensor,
        x_adv: torch.Tensor,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get robustness postcondition for probability difference bounds.

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Adversarial input tensor.

        Returns:
            Function that constrains infinity norm of probability differences.
        """
        y = N(x)
        y_adv = N(x_adv)

        diff = F.softmax(y_adv, dim=1) - F.softmax(y, dim=1)

        return lambda logic: logic.LEQ(
            LA.vector_norm(diff, ord=float("inf"), dim=1), self.delta
        )


class ClassificationRobustnessPostcondition(Postcondition):
    def __init__(self, device: torch.device):
        self.device = device

    def build_postcondition(
        self, N: torch.nn.Module, x_adv: torch.Tensor, y_target: torch.Tensor
    ) -> Callable[[Logic], torch.Tensor]:
        logits = N(x_adv)
        batch_size = logits.size(0)
        device = logits.device

        true_logits = logits[torch.arange(batch_size, device=device), y_target]

        def _(logic: Logic) -> torch.Tensor:
            if isinstance(logic, FuzzyLogic):
                probs = torch.sigmoid(logits)
                true_probs = probs[torch.arange(batch_size, device=device), y_target]

                return logic.AND(
                    *[
                        logic.GEQ(true_probs, probs[:, j])
                        for j in range(logits.shape[1])
                    ]
                )

            return logic.AND(
                *[logic.GEQ(true_logits, logits[:, j]) for j in range(logits.shape[1])]
            )

        return _


class StrongClassificationRobustnessPostcondition(Postcondition):
    """
    Postcondition ensuring model robustness to adversarial perturbations.

    Enforces that the model predicts the true label for the adversarial inputs.

    Args:
        device: PyTorch device for tensor computations.
    """

    def __init__(self, device: torch.device, delta: float | torch.Tensor):
        self.device = device

        self.delta = torch.as_tensor(delta, device=self.device)  # [0, 1]
        self.delta_logit = torch.log(self.delta / (1.0 - self.delta)).to(
            self.device
        )  # real

        print(f"delta={self.delta} delta_logit={self.delta_logit}")

    def build_postcondition(
        self,
        N: torch.nn.Module,
        x_adv: torch.Tensor,
        y_target: torch.Tensor,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get robustness postcondition for strong classification robustness.

        Args:
            N: Neural network model.
            x_adv: Adversarial input tensor.
            y_target: Target output tensor.

        Returns:
            Function that expresses strong classification robustness.
        """
        logits = N(x_adv)
        batch_size = logits.size(0)
        device = logits.device

        true_logits = logits[torch.arange(batch_size, device=device), y_target]

        # for fuzzy logics, need monotone transform into [0, 1]
        T = 1.0
        true_probs = torch.sigmoid(true_logits / T)

        def _(logic: Logic) -> torch.Tensor:
            if isinstance(logic, FuzzyLogic):
                return logic.GEQ(true_probs, self.delta)
            else:
                return logic.GEQ(true_logits, self.delta_logit)

        return _


class LipschitzRobustnessPostcondition(Postcondition):
    """
    Postcondition enforcing Lipschitz continuity for model robustness.

    Ensures that the rate of change in model outputs is bounded by the
    Lipschitz constant L relative to input perturbations.

    Args:
        device: PyTorch device for tensor computations.
        L: Lipschitz constant bounding the rate of output change.
    """

    def __init__(self, device: torch.device, L: float | torch.Tensor):
        self.device = device
        self.L = torch.as_tensor(L, device=device)

    def build_postcondition(
        self, N: torch.nn.Module, x: torch.Tensor, x_adv: torch.Tensor
    ) -> Callable[[Logic], torch.Tensor]:
        """Get Lipschitz postcondition relating input and output changes.

        Args:
            N: Neural network model.
            x: Original input tensor.
            x_adv: Perturbed input tensor.

        Returns:
            Function that constrains output change by L times input change.
        """
        y = N(x)
        y_adv = N(x_adv)

        diff_x = LA.vector_norm(x_adv - x, ord=2, dim=1)
        diff_y = LA.vector_norm(y_adv - y, ord=2, dim=1)

        return lambda logic: logic.LEQ(diff_y, self.L * diff_x)


class NotBothPostcondition(Postcondition):
    """
    Postcondition ensuring a physical-world inspired constraint on dice images.

    Enforces that the network may not predict faces at the same time that are
    on opposite sides of the die (e.g. faces 1 and 6).

    Args:
        device: PyTorch device for tensor computations.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.zero = torch.tensor(0.0, device=self.device)
        self.half = torch.tensor(0.5, device=self.device)

        self.opposite_pairs = [(0, 5), (1, 4), (2, 3)]

        self.all_triples = [tuple(c) for c in itertools.combinations(range(6), 3)]
        valid_triples = [
            tuple(sorted(c)) for c in itertools.product(*self.opposite_pairs)
        ]

        self.invalid_triples = [t for t in self.all_triples if t not in valid_triples]

        self.invalid_quadruples = list(itertools.combinations(range(6), 4))

    def build_postcondition(
        self, N: torch.nn.Module, x_adv: torch.Tensor
    ) -> Callable[[Logic], torch.Tensor]:
        """Get postcondition for opposite faces.

        Args:
            N: Neural network model.
            x_adv: Adversarial input tensor.

        Returns:
            Function that ensures network predictions align with real-world knowledge.
        """
        logits = N(x_adv)

        # for each pair (i,j), do not predict both
        def _(logic: Logic) -> torch.Tensor:
            if isinstance(logic, FuzzyLogic):
                return logic.AND(
                    *[
                        logic.OR(
                            logic.LEQ(torch.sigmoid(logits[:, i]), self.half),
                            logic.LEQ(torch.sigmoid(logits[:, j]), self.half),
                        )
                        for i, j in self.opposite_pairs
                    ]
                )
            else:
                return logic.AND(
                    *[
                        logic.OR(
                            logic.LEQ(logits[:, i], self.zero),
                            logic.LEQ(logits[:, j], self.zero),
                        )
                        for i, j in self.opposite_pairs
                    ]
                )

        return _


class ExactlyOnePerPairPostcondition(Postcondition):
    """
    Postcondition ensuring a physical-world inspired constraint on dice images.

    Enforces that the network may not predict faces at the same time that are
    on opposite sides of the die (e.g. faces 1 and 6).

    Args:
        device: PyTorch device for tensor computations.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.zero = torch.tensor(0.0, device=self.device)

        self.opposite_pairs = [(0, 5), (1, 4), (2, 3)]

        self.all_triples = [tuple(c) for c in itertools.combinations(range(6), 3)]
        valid_triples = [
            tuple(sorted(c)) for c in itertools.product(*self.opposite_pairs)
        ]

        self.invalid_triples = [t for t in self.all_triples if t not in valid_triples]

        self.invalid_quadruples = list(itertools.combinations(range(6), 4))

    def build_postcondition(
        self,
        N: torch.nn.Module,
        x_adv: torch.Tensor,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get postcondition for opposite faces.

        Args:
            N: Neural network model.
            x_adv: Adversarial input tensor.

        Returns:
            Function that ensures network predictions align with real-world knowledge.
        """
        logits = N(x_adv)

        def _(logic: Logic) -> torch.Tensor:
            margin = 0.1
            constraints = []

            if isinstance(logic, FuzzyLogic):
                probs = torch.sigmoid(logits)

                low = torch.sigmoid(torch.tensor(-margin, device=logits.device))
                high = torch.sigmoid(torch.tensor(+margin, device=logits.device))

                for i, j in self.opposite_pairs:
                    at_least_one = logic.OR(
                        logic.GEQ(probs[:, i], high),
                        logic.GEQ(probs[:, j], high),
                    )

                    at_most_one = logic.OR(
                        logic.LEQ(probs[:, i], low),
                        logic.LEQ(probs[:, j], low),
                    )

                    constraints.append(logic.AND(at_least_one, at_most_one))
            else:
                for i, j in self.opposite_pairs:
                    at_least_one = logic.OR(
                        logic.GEQ(logits[:, i], +margin),
                        logic.GEQ(logits[:, j], +margin),
                    )

                    at_most_one = logic.OR(
                        logic.LEQ(logits[:, i], -margin),
                        logic.LEQ(logits[:, j], -margin),
                    )

                    constraints.append(logic.AND(at_least_one, at_most_one))

            return logic.AND(*constraints)

        return _


class ClothingFootwearPostcondition(Postcondition):
    def __init__(self, device: torch.device):
        self.device = device
        self.clothing = torch.tensor(
            [0, 2, 3, 4, 6], device=device
        )  # t-shirt/top, pullover, dress, coat, shirt
        self.footwear = torch.tensor(
            [5, 7, 9], device=device
        )  # sandal, sneaker, ankle boot
        self.margin = 0.0

    def build_postcondition(self, N, x_adv, y_target):
        logits = N(x_adv)
        batch_size = logits.size(0)
        device = logits.device

        true_logits = logits[torch.arange(batch_size, device=device), y_target]

        clothing_mask = torch.isin(y_target, self.clothing)
        footwear_mask = torch.isin(y_target, self.footwear)

        def _(logic: Logic) -> torch.Tensor:
            constraints = []

            for j in self.footwear.tolist():
                # for clothing samples: compare true logit against footwear logit j.
                # for non-clothing samples: compare true logit against itself.
                other_logit = torch.where(
                    clothing_mask,
                    logits[:, j],
                    true_logits,
                )

                constraints.append(logic.GEQ(true_logits, other_logit + self.margin))

            for j in self.clothing.tolist():
                # for footwear samples: compare true logit against clothing logit j.
                # for non-footwear samples: compare true logit against itself.
                other_logit = torch.where(
                    footwear_mask,
                    logits[:, j],
                    true_logits,
                )

                constraints.append(logic.GEQ(true_logits, other_logit + self.margin))

            return logic.AND(*constraints)

        return _


class AlsomitraOutputPostcondition(Postcondition):
    """
    Postcondition ensuring model outputs fall within specified bounds.

    Enforces that neural network outputs remain within lower and upper bounds.

    Args:
        device: PyTorch device for tensor computations.
        lo: Lower bound for outputs (nan means no lower bound).
        hi: Upper bound for outputs (nan means no upper bound).
    """

    def __init__(
        self,
        device: torch.device,
        lo: float | torch.Tensor = np.nan,
        hi: float | torch.Tensor = np.nan,
    ):
        self.device = device
        self.min = 0.181
        self.max = 0.193
        self.lo = self.normalize(torch.as_tensor(lo, device=device))
        self.hi = self.normalize(torch.as_tensor(hi, device=device))

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Normalize the output tensor y based on the min and max bounds.

        Args:
            y: Output tensor to normalize.

        Returns:
            Normalized tensor.
        """
        return (y - self.min) / (self.max - self.min)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        """
        Denormalize the output tensor y based on the min and max bounds.

        Args:
            y: Input tensor to denormalize.

        Returns:
            Denormalized tensor.
        """
        return y * (self.max - self.min) + self.min

    def build_postcondition(
        self,
        N: torch.nn.Module,
        x_adv: torch.Tensor,
    ) -> Callable[[Logic], torch.Tensor]:
        """
        Get output bounds postcondition for adversarial inputs.

        This implementation uses a specialized signature that only includes
        the parameters actually needed by this postcondition type.

        Args:
            N: Neural network model.
            x_adv: Adversarial input tensor.
            scale: Optional scaling factor for normalization.
            centre: Optional centre point for normalization.

        Returns:
            Function that constrains outputs to specified bounds.
        """
        y_adv = N(x_adv).squeeze()

        if torch.isnan(self.lo) and not torch.isnan(
            self.hi
        ):  # no lower bound, but upper bound
            return lambda logic: logic.LEQ(y_adv, self.hi)
        elif not torch.isnan(self.lo) and not torch.isnan(
            self.hi
        ):  # both lower and upper bound
            return lambda logic: logic.AND(
                logic.GEQ(y_adv, self.lo), logic.LEQ(y_adv, self.hi)
            )
        elif not torch.isnan(self.lo) and torch.isnan(
            self.hi
        ):  # lower bound, no upper bound
            return lambda logic: logic.GEQ(y_adv, self.lo)
        else:  # no lower and no upper bound
            raise ValueError(
                "need to specify either lower or upper (or both) bounds for e_x"
            )


class GroupPostcondition(Postcondition):
    """
    Postcondition ensuring similar outputs for grouped inputs.

    Enforces that neural network outputs remain within delta for inputs
    that belong to the same group, promoting consistency within groups.

    Args:
        device: PyTorch device for tensor computations.
        indices: List of lists, each inner list contains indices of inputs in a group.
        delta: Maximum allowed difference between outputs within each group.
    """

    def __init__(self, device: torch.device, indices: list[list[int]], delta: float):
        self.device = device

        self.indices = indices

        assert 0.0 <= delta <= 1.0, (  # nosec
            "delta is a probability and should be within the range [0, 1]"
        )
        self.delta = torch.as_tensor(delta, device=self.device)

    def build_postcondition(
        self,
        N: torch.nn.Module,
        x_adv: torch.Tensor | None,
    ) -> Callable[[Logic], torch.Tensor]:
        """Get group consistency postcondition for adversarial inputs.

        This implementation demonstrates another specialized signature,
        using only the model and adversarial inputs (no original input needed).

        Args:
            N: Neural network model.
            x_adv: Adversarial input tensor.

        Returns:
            Function that constrains grouped outputs to be within delta bounds.
        """
        y_adv = F.softmax(N(x_adv), dim=1)
        sums = [torch.sum(y_adv[:, i], dim=1) for i in self.indices]

        return lambda logic: logic.AND(
            *[
                logic.OR(logic.LEQ(s, self.delta), logic.GEQ(s, 1.0 - self.delta))
                for s in sums
            ]
        )
