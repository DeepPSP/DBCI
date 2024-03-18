"""
"""

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional, Tuple, Union

import pandas as pd

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
    """Dataclass for holding meta information of a confidence interval.

    Attributes
    ----------
    lower_bound : float
        The lower bound of the confidence interval.
    upper_bound : float
        The upper bound of the confidence interval.
    estimate : float
        Estimate of (the difference of) the binomial proportion.
    level : float
        Confidence level, should be inside the interval (0, 1).
    method : str
        Computation method of the confidence interval.
    sides : str, default "two-sided"
        Sides of the confidence interval, should be one of

        - "two-sided" (aliases "2-sided", "two_sided", "2_sided",
          "2-sides", "two_sides", "two-sides", "2_sides", "ts", "t", "two", "2"),
        - "left-sided" (aliases "left_sided", "left", "ls", "l"),
        - "right-sided" (aliases "right_sided", "right", "rs", "r"),

        case insensitive.
    digits : int, default 7
        Number of digits to round the confidence interval to in the string representation.

    """

    lower_bound: float
    upper_bound: float
    estimate: float
    level: float
    method: str
    sides: str = "two-sided"
    digits: int = 7

    def __post_init__(self) -> None:
        assert 0 < self.level < 1
        assert self.sides.lower() in _SIDE_NAME_MAP
        self.sides = _SIDE_NAME_MAP[self.sides.lower()]

    def astuple(self) -> Tuple[float, float]:
        return (self.lower_bound, self.upper_bound)

    def asdict(self) -> dict:
        d = asdict(self)
        d.pop("digits")
        return d

    def astable(self, to: Optional[str] = None, digits: Optional[Union[int, bool]] = None) -> Union[str, pd.DataFrame]:
        """Return the confidence interval as a table (dataframe).

        Parameters
        ----------
        to : str, default None
            Format of the table. Supported formats are "latex", "latex_raw", "html", "markdown", "md", "string", "json".
        digits : int or bool, default None
            Number of digits to round the confidence interval to in the string representation.
            If ``True``, use the default digits. If ``False`` or ``None``, use the float data type.

        Returns
        -------
        str or pandas.DataFrame
            The confidence interval as a table (dataframe) in the specified format.

        """
        if (digits is None) or (isinstance(digits, bool) and digits is False):
            # float data type
            lower_bound, upper_bound = self.lower_bound, self.upper_bound
        elif isinstance(digits, bool) and digits is True:
            # string data type with default digits
            lower_bound = f"{self.lower_bound:.{self.digits}f}"
            upper_bound = f"{self.upper_bound:.{self.digits}f}"
        elif isinstance(digits, int):
            # string data type with specified digits
            lower_bound = f"{self.lower_bound:.{digits}f}"
            upper_bound = f"{self.upper_bound:.{digits}f}"
        else:
            raise ValueError(f"Unsupported digits type {repr(digits)}")
        table = pd.DataFrame(
            {
                "Estimate": [self.estimate],
                "Lower Bound": [lower_bound],
                "Upper Bound": [upper_bound],
                "Confidence Level": [self.level],
                "Method": [self.method],
                "Sides": [self.sides],
            }
        )
        if to is None:
            return table
        elif to in ["latex", "latex_raw"]:
            return table.to_latex(index=False, escape=False)
        elif to in ["html"]:
            return table.to_html(index=False, escape=False)
        elif to in ["markdown", "md"]:
            return table.to_markdown(index=False)
        elif to in ["string"]:
            return table.to_string(index=False)
        elif to in ["json"]:
            return table.to_json(orient="records")
        else:
            raise ValueError(f"Unsupported format {repr(to)}")

    def __repr__(self) -> str:
        # return f"({round(self.lower_bound, 5):.5f}, {round(self.upper_bound, 5):.5f})"
        lower_bound = round(self.lower_bound, self.digits)
        upper_bound = round(self.upper_bound, self.digits)
        return f"({lower_bound:.{self.digits}f}, {upper_bound:.{self.digits}f})"

    __str__ = __repr__
