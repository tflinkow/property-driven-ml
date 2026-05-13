import torch

from .logic import Logic


class QLL(Logic):
    def __init__(self, p: float = 5.0):
        super().__init__(f"QLL_{p}")

        assert p > 0, "p must be positive"  # nosec
        self.p = p

    def NOT(self, x: torch.Tensor) -> torch.tensor:
        return -x

    def EQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.abs(x - y)

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x - y

    def IMPL(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x - y

    def LT(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.AND(self.LEQ(x, y), self.NEQ(x, y))

    def AND(self, *xs: torch.Tensor):
        xs = torch.stack(xs, dim=0)
        lse = torch.logsumexp(self.p * xs, dim=0)

        return lse / self.p

    def OR(self, *xs: torch.Tensor):
        xs = torch.stack(xs, dim=0)
        lse = torch.logsumexp(-self.p * xs, dim=0)

        return -lse / self.p
