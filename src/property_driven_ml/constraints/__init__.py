"""
Constraint definitions for property-driven machine learning.

This module provides constraint classes that define properties that
machine learning models should satisfy.

The new unified constraint architecture combines input regions and output
constraints in a single class hierarchy, eliminating the need for separate
BoundedDataset classes.
"""

from .constraints import (
    Constraint,
    StandardRobustnessConstraint,
    StrongClassificationRobustnessConstraint,
    ClassificationRobustnessConstraint,
    ExactlyOnePerPairConstraint,
    NotBothConstraint,
    ClothingFootwearConstraint,
    AlsomitraProperty1Constraint,
    AlsomitraProperty2Constraint,
    AlsomitraProperty3Constraint,
)
from .preconditions import EpsilonBall
from .postconditions import (
    StandardRobustnessPostcondition,
    StrongClassificationRobustnessPostcondition,
    ClassificationRobustnessPostcondition,
    ExactlyOnePerPairPostcondition,
    NotBothPostcondition,
    ClothingFootwearPostcondition,
    LipschitzRobustnessPostcondition,
    GroupPostcondition,
    AlsomitraOutputPostcondition,
)

__all__ = [
    # Constraints
    "Constraint",
    "StandardRobustnessConstraint",
    "StrongClassificationRobustnessConstraint",
    "ClassificationRobustnessConstraint",
    "PrimeEvenOnlyIfTwoConstraint",
    "ExactlyOnePerPairConstraint",
    "NotBothConstraint",
    "ClothingFootwearConstraint",
    "AlsomitraProperty1Constraint",
    "AlsomitraProperty2Constraint",
    "AlsomitraProperty3Constraint",
    # Preconditions
    "EpsilonBall",
    # Postconditions
    "StandardRobustnessPostcondition",
    "StrongClassificationRobustnessPostcondition",
    "ClassificationRobustnessPostcondition",
    "ExactlyOnePerPairPostcondition",
    "NotBothPostcondition",
    "ClothingFootwearPostcondition",
    "LipschitzRobustnessPostcondition",
    "GroupPostcondition",
    "AlsomitraOutputPostcondition",
]
