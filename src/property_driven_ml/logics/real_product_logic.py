import torch

from .logic import Logic

    
class RealProductLogic(Logic):
    def __init__(self, name="RealProduct"):
        super().__init__(name)

    def AND(self, *xs: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.stack(xs, dim=0), dim=0)

    def OR(self, *xs: torch.Tensor) -> torch.Tensor:
        zs = torch.stack(xs, dim=0)

        # t_i = exp(-z_i)
        ts = torch.exp(-zs)

        return -torch.log(
            1.0 - torch.prod(1.0 - ts, dim=0)
        )

    def NOT(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.log(1.0 - torch.exp(-x))

    def LEQ(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.relu(x - y)