"""
Training data structures for property-driven machine learning.

This module defines data structures used to track training progress
and statistics during property-driven learning.
"""

import torch

from dataclasses import dataclass
from typing import Optional, Literal
from abc import ABC, abstractmethod


@dataclass(slots=True)
class EpochInfo(ABC):
    """Base class for epoch information."""

    pred_metric: float
    pred_loss: float

    constr_acc: Optional[float] = None
    random_loss: Optional[float] = None

    input_img: Optional[torch.Tensor] = None
    random_img: Optional[torch.Tensor] = None
    adv_img: Optional[torch.Tensor] = None

    @property
    @abstractmethod
    def phase(self) -> Literal["train", "test"]:
        """Phase identifier."""


@dataclass(slots=True)
class EpochInfoTrain(EpochInfo):
    constr_sec: Optional[float] = None
    constr_loss: Optional[float] = None

    pred_grad_norm: Optional[float] = None
    constr_grad_norm: Optional[float] = None
    constr_loss_weight: Optional[float] = None
    weighted_constr_grad_norm: Optional[float] = None
    grad_ratio: Optional[float] = None

    @property
    def phase(self) -> Literal["train"]:
        return "train"


@dataclass(slots=True)
class EpochInfoTest(EpochInfo):
    constr_sec_self: Optional[float] = None
    constr_loss_self: Optional[float] = None

    constr_sec_common: Optional[float] = None
    constr_loss_common: Optional[float] = None

    @property
    def phase(self) -> Literal["test"]:
        return "test"
