"""
"""

import numpy as np
from scipy.stats import norm

from ._confint import ConfidenceInterval


__all__ = ["compute_confidence_interval"]


def compute_confidence_interval(
    n_positive: int,
    n_negative: int,
    conf_level: float = 0.95,
    confint_type: str = "wilson",
) -> ConfidenceInterval:
    """
    Parameters
    ----------
    n_positive: int,
        number of positive samples
    n_negative: int,
        number of negative samples
    conf_level: float, default 0.95,
        confidence level, should be inside the interval (0, 1)
    confint_type: str, default "wilson",
        type (computation method) of the confidence interval

    Returns
    -------
    an instance of `ConfidenceInterval`

    """
    z = norm.ppf((1 + conf_level) / 2)
    tot = n_positive + n_negative
    if confint_type.lower() == "wilson":
        rate = n_positive / tot
        return ConfidenceInterval(
            (
                (
                    rate
                    + z * z / (2 * tot)
                    - z * np.sqrt((rate * (1 - rate) + z * z / (4 * tot)) / tot)
                )
                / (1 + z * z / tot)
            ),
            (
                (
                    rate
                    + z * z / (2 * tot)
                    + z * np.sqrt((rate * (1 - rate) + z * z / (4 * tot)) / tot)
                )
                / (1 + z * z / tot)
            ),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() == "wald":
        rate = n_positive / tot
        return ConfidenceInterval(
            rate - z * np.sqrt(rate * (1 - rate) / tot),
            rate + z * np.sqrt(rate * (1 - rate) / tot),
            conf_level,
            confint_type.lower(),
        )
    else:
        raise ValueError(f"{confint_type} is not supported")
