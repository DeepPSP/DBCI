"""
"""

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


__all__ = ["ConfidenceInterval"]


class ConfidenceIntervalSides(Enum):
    TwoSided = "two-sided"
    LeftSided = "left-sided"
    RightSided = "right-sided"


_SIDE_NAME_MAP = {
    "two-sided": ConfidenceIntervalSides.TwoSided.value,
    "2-sided": ConfidenceIntervalSides.TwoSided.value,
    "two_sided": ConfidenceIntervalSides.TwoSided.value,
    "2_sided": ConfidenceIntervalSides.TwoSided.value,
    "ts": ConfidenceIntervalSides.TwoSided.value,
    "t": ConfidenceIntervalSides.TwoSided.value,
    "2-sides": ConfidenceIntervalSides.TwoSided.value,
    "two_sides": ConfidenceIntervalSides.TwoSided.value,
    "two-sides": ConfidenceIntervalSides.TwoSided.value,
    "2_sides": ConfidenceIntervalSides.TwoSided.value,
    "two": ConfidenceIntervalSides.TwoSided.value,
    "2": ConfidenceIntervalSides.TwoSided.value,
    2: ConfidenceIntervalSides.TwoSided.value,
    "left-sided": ConfidenceIntervalSides.LeftSided.value,
    "left_sided": ConfidenceIntervalSides.LeftSided.value,
    "left": ConfidenceIntervalSides.LeftSided.value,
    "ls": ConfidenceIntervalSides.LeftSided.value,
    "l": ConfidenceIntervalSides.LeftSided.value,
    "right-sided": ConfidenceIntervalSides.RightSided.value,
    "right": ConfidenceIntervalSides.RightSided.value,
    "rs": ConfidenceIntervalSides.RightSided.value,
    "r": ConfidenceIntervalSides.RightSided.value,
}


@dataclass
class ConfidenceInterval:
    """
    Attributes
    ----------
    lower_bound: float,
        the lower bound of the confidence interval.
    upper_bound: float,
        the upper bound of the confidence interval.
    estimate: float,
        estimate of (the difference of) the binomial proportion.
    level: float,
        confidence level, should be inside the interval (0, 1).
    method: str,
        computation method of the confidence interval.
    sides: str, default "two-sided",
        the sides of the confidence interval, should be one of
        "two-sided" (aliases "2-sided", "two_sided", "2_sided", "2-sides", "two_sides", "two-sides", "2_sides", "ts", "t", "two", "2"),
        "left-sided" (aliases "left_sided", "left", "ls", "l"),
        "right-sided" (aliases "right_sided", "right", "rs", "r"),
        case insensitive.

    """

    lower_bound: float
    upper_bound: float
    estimate: float
    level: float
    method: str
    sides: str = "two-sided"

    def __post_init__(self) -> None:
        assert 0 < self.level < 1
        assert self.sides.lower() in _SIDE_NAME_MAP
        self.sides = _SIDE_NAME_MAP[self.sides.lower()]

    def astuple(self) -> Tuple[float, float]:
        return (self.lower_bound, self.upper_bound)

    def __repr__(self) -> str:
        return f"({round(self.lower_bound, 5):.5f}, {round(self.upper_bound, 5):.5f})"

    __str__ = __repr__
