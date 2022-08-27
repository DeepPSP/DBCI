"""
"""

from dataclasses import dataclass
from typing import NoReturn, Tuple


__all__ = ["ConfidenceInterval"]


@dataclass
class ConfidenceInterval:
    """
    Attributes
    ----------
    lower_bound: float,
        the lower bound of the confidence interval
    upper_bound: float,
        the upper bound of the confidence interval
    level: float,
        confidence level, should be inside the interval (0, 1)
    type: str,
        type (computation method) of the confidence interval

    """

    lower_bound: float
    upper_bound: float
    level: float
    type: str

    def __post_init__(self) -> NoReturn:
        assert 0 < self.level < 1

    def astuple(self) -> Tuple[float, float]:
        return (self.lower_bound, self.upper_bound)

    def __repr__(self) -> str:
        return f"({self.lower_bound}, {self.upper_bound})"

    __str__ = __repr__
