from __future__ import annotations

import torch


# GradNorm (https://arxiv.org/abs/1711.02257) based on https://github.com/brianlan/pytorch-grad-norm/blob/master/train.py
class GradNorm:
    """Gradient normalization for multi-task learning.

    Implements the GradNorm algorithm to automatically balance multiple
    loss terms during training by adjusting their relative weights.

    Args:
        N: Neural network model.
        device: PyTorch device for computations.
        optimizer: Optimizer for model parameters.
        lr: Learning rate for weight optimization.
        alpha: Restoring force strength for loss balancing.
        initial_dl_weight: Initial weight for the second loss term.
    """

    def __init__(
        self,
        N: torch.nn.Module,
        device: torch.device,
        optimizer,
        lr: float,
        alpha: float,
        initial_dl_weight: float = 1.0,
    ):
        self.initial_loss = None
        self.weights = torch.nn.Parameter(
            torch.tensor([2.0 - initial_dl_weight, initial_dl_weight], device=device)
        )
        self.N = N
        self.device = device
        self.optimizer_train = optimizer
        self.optimizer_weights = torch.optim.Adam([self.weights], lr=lr)
        self.alpha = alpha

    def _grad_norm(self, loss: torch.Tensor) -> torch.Tensor:
        grads = torch.autograd.grad(
            loss,
            list(self.N.fc1.parameters()),
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )

        grads = [g.reshape(-1) for g in grads if g is not None]

        if not grads:
            return torch.zeros((), device=self.device)

        return torch.norm(torch.cat(grads), p=2)

    def balance(self, pred_loss: torch.Tensor, dl_loss: torch.Tensor):
        """Balance the weights between prediction and constraint losses.

        Uses gradient magnitudes to automatically adjust loss weights
        for more stable multi-task training.

        Args:
            pred_loss: Prediction loss tensor (e.g. cross-entropy).
            dl_loss: Differentiable logic loss tensor.

        Returns:
            Tuple of (weighted_pred_loss, weighted_dl_loss).
        """
        tasks = torch.stack([pred_loss, dl_loss])

        if self.initial_loss is None:
            self.initial_loss = tasks.detach().clone()

        # keep weights positive and normalized
        with torch.no_grad():
            self.weights.clamp_(min=1e-8)
            self.weights.mul_(2.0 / self.weights.sum())

        weighted_tasks = self.weights * tasks

        # compute per-task gradient norms G_i = ||grad_W (w_i L_i)||_2
        G = torch.stack(
            [
                self._grad_norm(weighted_tasks[0]),
                self._grad_norm(weighted_tasks[1]),
            ]
        )

        # relative inverse training rates
        loss_ratio = tasks.detach() / self.initial_loss
        inv_rate = loss_ratio / loss_ratio.mean()

        # target is treated as constant w.r.t. weights
        target = G.detach().mean() * (inv_rate**self.alpha)

        gradnorm_loss = torch.abs(G - target).sum()

        # update weights
        self.optimizer_weights.zero_grad(set_to_none=True)
        gradnorm_loss.backward(retain_graph=True)
        self.optimizer_weights.step()

        # normalise weights to be >= 0 and add up to 2
        with torch.no_grad():
            self.weights.clamp_(min=1e-8)
            self.weights.mul_(2.0 / self.weights.sum())

        # update model
        self.optimizer_train.zero_grad(set_to_none=True)
        total_loss = (self.weights.detach() * tasks).sum()
        total_loss.backward()
        self.optimizer_train.step()
