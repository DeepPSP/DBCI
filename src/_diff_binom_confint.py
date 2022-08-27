"""
"""

import numpy as np
from scipy.stats import norm

from ._confint import ConfidenceInterval


__all__ = ["compute_difference_confidence_interval"]


def compute_difference_confidence_interval(
    n_positive: int,
    n_negative: int,
    ref_positive: int,
    ref_negative: int,
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
    ref_positive: int,
        number of positive samples of the reference
    ref_negative: int,
        number of negative samples of the reference
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
    ref_tot = ref_positive + ref_negative
    r1 = n_positive / tot
    r2 = ref_positive / ref_tot
    if confint_type.lower() in ["wilson", "newcombe"]:
        lower1 = (
            (2 * tot * r1 + z**2 - z * np.sqrt(4 * tot * r1 * (1 - r1) + z**2))
            / 2
            / (tot + z**2)
        )
        upper1 = (
            (2 * tot * r1 + z**2 + z * np.sqrt(4 * tot * r1 * (1 - r1) + z**2))
            / 2
            / (tot + z**2)
        )
        lower2 = (
            (
                2 * ref_tot * r2
                + z**2
                - z * np.sqrt(4 * ref_tot * r2 * (1 - r2) + z**2)
            )
            / 2
            / (ref_tot + z**2)
        )
        upper2 = (
            (
                2 * ref_tot * r2
                + z**2
                + z * np.sqrt(4 * ref_tot * r2 * (1 - r2) + z**2)
            )
            / 2
            / (ref_tot + z**2)
        )
        return ConfidenceInterval(
            r1 - r2 - np.sqrt((r1 - lower1) ** 2 + (upper2 - r2) ** 2),
            r1 - r2 + np.sqrt((r2 - lower2) ** 2 + (upper1 - r1) ** 2),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["wilson_cc", "newcombe_cc"]:
        raise NotImplementedError
    elif "wald" in confint_type.lower():
        item = z * np.sqrt(r1 * (1 - r1) / tot + r2 * (1 - r2) / ref_tot)
        if confint_type.lower() == "wald_cc":
            return ConfidenceInterval(
                r1 - r2 - item - 0.5 / tot - 0.5 / ref_tot,
                r1 - r2 + item + 0.5 / tot + 0.5 / ref_tot,
                conf_level,
                confint_type.lower(),
            )
        return ConfidenceInterval(
            r1 - r2 - item, r1 - r2 + item, conf_level, confint_type.lower()
        )
    elif confint_type.lower() in ["haldane", "jeffreys-perks"]:
        v = 0.25 / tot - 0.25 / ref_tot
        u = v + 0.5 / ref_tot
        if confint_type.lower() == "haldane":
            psi = 0.5 * (r1 + r2)
        else:
            psi = 0.5 * (
                (n_positive + 0.5) / (tot + 1) + (ref_positive + 0.5) / (ref_tot + 1)
            )
        w = (
            np.sqrt(
                u
                * (
                    4 * psi * (1 - psi)
                    - (r1 - r2) ** 2
                    + 2 * v * (1 - 2 * psi) * (r1 - r2)
                    + 4 * ((z * u) ** 2) * psi * (1 - psi)
                    + (z * v * (1 - 2 * psi)) ** 2
                )
            )
            * z
            / (1 + u * z**2)
        )
        theta_star = ((r1 - r2) + v * (1 - 2 * psi) * z**2) / (1 + u * z**2)
        return ConfidenceInterval(
            theta_star - w, theta_star + w, conf_level, confint_type.lower()
        )
    else:
        raise ValueError(f"{confint_type} is not supported")
