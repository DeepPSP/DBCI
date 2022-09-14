"""
"""

from dataclasses import dataclass
from typing import NoReturn, Tuple

from deprecate_kwargs import deprecate_kwargs


__all__ = ["ConfidenceInterval"]


@deprecate_kwargs([["method", "type"]])
@dataclass
class ConfidenceInterval:
    """
    Attributes
    ----------
    lower_bound: float,
        the lower bound of the confidence interval
    upper_bound: float,
        the upper bound of the confidence interval
    estimate: float,
        estimate of (the difference of) the binomial proportion
    level: float,
        confidence level, should be inside the interval (0, 1)
    type: str,
        type (computation method) of the confidence interval

    """

    lower_bound: float
    upper_bound: float
    estimate: float
    level: float
    type: str

    def __post_init__(self) -> NoReturn:
        assert 0 < self.level < 1
        # replace field `type` with `method`
        self.method = self.type
        del self.type
        method_fld = self.__dataclass_fields__.pop("type")
        method_fld.name = "method"
        self.__dataclass_fields__["method"] = method_fld

    def astuple(self) -> Tuple[float, float]:
        return (self.lower_bound, self.upper_bound)

    def __repr__(self) -> str:
        return f"({self.lower_bound}, {self.upper_bound})"

    __str__ = __repr__
