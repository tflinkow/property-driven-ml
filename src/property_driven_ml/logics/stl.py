import torch

from .logic import Logic


class STL(Logic):
    """Signal Temporal Logic implementation for real-valued constraints.

    Provides smooth approximations of logical operations using
    exponential functions, enabling gradient-based optimization
    while preserving logical semantics.

    Args:
        k: Smoothness parameter (higher values give sharper approximations).
    """

    def __init__(self, k: float = 5.0):
        super().__init__(f"STL_{k}")
        self.k = k

    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        """STL logical negation.

        Args:
            x: Tensor to negate.

        Returns:
            Negated tensor -x.
        """
        return -x

    def EQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -torch.abs(x - y)

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """STL less than or equal operation.

        Args:
            x: Left-hand side tensor.
            y: Right-hand side tensor.

        Returns:
            Real-valued difference y - x (positive when x <= y).
        """
        return y - x

    def LT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.AND(self.LEQ(x, y), self.NEQ(x, y))

    def AND(self, *xs) -> torch.Tensor:
        xs = torch.stack(xs, dim=0)
        x_min = torch.min(xs, dim=0, keepdim=True).values

        eps = 1e-12  # TODO: make param!
        near_zero = torch.abs(x_min) <= eps

        # sign-preserving safe denom
        s = torch.sign(x_min)
        s = torch.where(s == 0, torch.ones_like(s), s)
        denom = s * torch.clamp(torch.abs(x_min), min=eps)

        # tilde x_i = (x_i - x_min) / x_min
        t = (xs - x_min) / denom

        # case 1: x_min < 0
        w_neg = torch.softmax(self.k * t, dim=0)
        exp_t = torch.exp(torch.clamp(t, max=0.0))  # avoid explosion
        out_neg = x_min * torch.sum(exp_t * w_neg, dim=0, keepdim=True)

        # case 2: x_min > 0
        w_pos = torch.softmax(-self.k * t, dim=0)
        out_pos = torch.sum(xs * w_pos, dim=0, keepdim=True)

        neg = x_min < -eps
        pos = x_min > eps

        out = x_min * 0.0
        out = torch.where(neg, out_neg, out)
        out = torch.where(pos, out_pos, out)
        out = torch.where(near_zero, x_min * 0.0, out)

        return out.squeeze(0)

    def OR(self, *xs):
        return self.NOT(self.AND(*(self.NOT(x) for x in xs)))
