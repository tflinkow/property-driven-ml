import torch

from contextlib import contextmanager


@contextmanager
def maybe(context_manager, flag: bool):
    """Conditionally apply a context manager based on a flag.

    Args:
        context_manager: Context manager to apply if flag is True.
        flag: Whether to apply the context manager.

    Yields:
        Context manager result if flag is True, otherwise None.
    """
    if flag:
        with context_manager as cm:
            yield cm
    else:
        yield None


def safe_div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Safely divide tensors, avoiding division by zero.

    Args:
        x: Numerator tensor.
        y: Denominator tensor.

    Returns:
        Division result with zeros replaced by epsilon.
    """
    sign = torch.sign(y) # return 1 for +x where x > 0, 0 for x == 0, -1 for x < 0
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    return x / (sign * y.abs().clamp(min=torch.finfo(y.dtype).eps))


def safe_zero(x: torch.Tensor) -> torch.Tensor:
    """Replace zeros in tensor with epsilon to avoid numerical issues.

    Args:
        x: Input tensor.

    Returns:
        Tensor with zeros replaced by epsilon.
    """
    return x.clamp(min=torch.finfo(x.dtype).eps)


def safe_pow(x: torch.Tensor, y: torch.Tensor | float | int) -> torch.Tensor:
    """Safely compute power, avoiding issues with zero base.

    Args:
        x: Base tensor.
        y: Exponent (tensor, float, or int).

    Returns:
        Power result with zero base replaced by epsilon.
    """
    return torch.pow(safe_zero(x), y)


def safe_exp(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(torch.clamp(x, max=60.0)) # TODO: explain!