"""
"""

import numpy as np
from scipy.stats import norm

from ._confint import ConfidenceInterval


__all__ = [
    "compute_difference_confidence_interval",
    "list_difference_confidence_interval_types",
]


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
    ratio = n_positive / tot
    ref_ratio = ref_positive / ref_tot
    delta_ratio = ratio - ref_ratio
    if confint_type.lower() in ["wilson", "newcombe"]:
        item1 = z * np.sqrt(4 * tot * ratio * (1 - ratio) + z**2)
        lower1 = (2 * tot * ratio + z**2 - item1) / 2 / (tot + z**2)
        upper1 = (2 * tot * ratio + z**2 + item1) / 2 / (tot + z**2)
        item2 = z * np.sqrt(4 * ref_tot * ref_ratio * (1 - ref_ratio) + z**2)
        lower2 = (2 * ref_tot * ref_ratio + z**2 - item2) / 2 / (ref_tot + z**2)
        upper2 = (2 * ref_tot * ref_ratio + z**2 + item2) / 2 / (ref_tot + z**2)
        return ConfidenceInterval(
            delta_ratio - np.sqrt((ratio - lower1) ** 2 + (upper2 - ref_ratio) ** 2),
            delta_ratio + np.sqrt((ref_ratio - lower2) ** 2 + (upper1 - ratio) ** 2),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["wilson_cc", "newcombe_cc"]:
        # has error
        raise NotADirectoryError
        item1 = 1 + z * np.sqrt(
            z**2 - 2 - 1 / tot + 4 * ratio * (tot * (1 - ratio) + 1)
        )
        lower1 = (2 * tot * ratio + z**2 - item1) / 2 / (tot + z**2)
        upper1 = (2 * tot * ratio + z**2 + item1) / 2 / (tot + z**2)
        item2 = 1 + z * np.sqrt(
            z**2 - 2 - 1 / ref_tot + 4 * ref_ratio * (ref_tot * (1 - ref_ratio) + 1)
        )
        lower2 = (2 * ref_tot * ref_ratio + z**2 - item2) / 2 / (ref_tot + z**2)
        upper2 = (2 * ref_tot * ref_ratio + z**2 + item2) / 2 / (ref_tot + z**2)
        return ConfidenceInterval(
            delta_ratio - np.sqrt((ratio - lower1) ** 2 + (upper2 - ref_ratio) ** 2),
            delta_ratio + np.sqrt((ref_ratio - lower2) ** 2 + (upper1 - ratio) ** 2),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["wald", "wald_cc"]:
        item = z * np.sqrt(
            ratio * (1 - ratio) / tot + ref_ratio * (1 - ref_ratio) / ref_tot
        )
        if confint_type.lower() == "wald_cc":
            return ConfidenceInterval(
                delta_ratio - item - 0.5 / tot - 0.5 / ref_tot,
                delta_ratio + item + 0.5 / tot + 0.5 / ref_tot,
                conf_level,
                confint_type.lower(),
            )
        return ConfidenceInterval(
            delta_ratio - item,
            delta_ratio + item,
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["haldane", "jeffreys-perks"]:
        v = 0.25 / tot - 0.25 / ref_tot
        u = v + 0.5 / ref_tot
        if confint_type.lower() == "haldane":
            psi = 0.5 * (ratio + ref_ratio)
        else:  # "jeffreys-perks"
            psi = 0.5 * (
                (n_positive + 0.5) / (tot + 1) + (ref_positive + 0.5) / (ref_tot + 1)
            )
        w = (
            np.sqrt(
                u * (4 * psi * (1 - psi) - delta_ratio**2)
                + 2 * v * (1 - 2 * psi) * delta_ratio
                + 4 * ((z * u) ** 2) * psi * (1 - psi)
                + (z * v * (1 - 2 * psi)) ** 2
            )
            * z
            / (1 + u * z**2)
        )
        theta_star = (delta_ratio + v * (1 - 2 * psi) * z**2) / (1 + u * z**2)
        return ConfidenceInterval(
            theta_star - w, theta_star + w, conf_level, confint_type.lower()
        )
    elif confint_type.lower() in ["mee", "miettinen-nurminen"]:
        # impelementation has error
        raise NotImplementedError
        # theta = ref_tot / tot
        # a = 1 + theta
        # increment = 1e-5
        # for j in np.arange(-1, 1, increment):
        #     b = -(1 + theta + ratio + theta * ref_ratio + j * (theta + 2))
        #     c = j * (j + 2 * ratio + theta + 1) + ratio + theta * ref_ratio
        #     d = -ratio * j * (1 + j)
        #     tmp = b / 3 / a
        #     v = tmp**3 - b * c / 6 / a**2 + d / 2 / a
        #     u = np.sign(v) * np.sqrt(tmp**2 + c / 3 / a)
        #     w = (np.pi + np.arccos(v / u**3)) / 3
        #     ratio_mle = 2 * u * np.cos(w) - tmp
        #     ref_ratio_mle = ratio_mle + j
        #     if confint_type.lower() == "mee":
        #         lamb = 1
        #     else:  # "miettinen-nurminen"
        #         lamb = (tot + ref_tot) / (tot + ref_tot + 1)
        #     item = z * np.sqrt(
        #         lamb
        #         * (
        #             ratio_mle * (1 - ratio_mle) / tot
        #             + ref_ratio_mle * (1 - ref_ratio_mle) / ref_tot
        #         )
        #     )
        # return ConfidenceInterval(
        #     delta_ratio - item, delta_ratio + item, conf_level, confint_type.lower()
        # )
    elif confint_type.lower() == "true-profile":
        raise NotImplementedError
    elif confint_type.lower() == "exact":
        raise NotImplementedError
    elif confint_type.lower() == "mid-p":
        raise NotImplementedError
    elif confint_type.lower() == "hauck-anderson":
        raise NotImplementedError
    elif confint_type.lower() == "agresti-caffo":
        raise NotImplementedError
    elif confint_type.lower() == "santner-snell":
        raise NotImplementedError
    elif confint_type.lower() == "chan-zhang":
        raise NotImplementedError
    elif confint_type.lower() == "brown-li":
        raise NotImplementedError
    elif confint_type.lower() == "miettinen-nurminen-brown-li":
        raise NotImplementedError
    elif confint_type.lower() == "agresti-min":
        raise NotImplementedError
    elif confint_type.lower() == "wang":
        raise NotImplementedError
    elif confint_type.lower() == "pradhan-banerjee":
        raise NotImplementedError
    else:
        raise ValueError(f"{confint_type} is not supported")


_supported_types = [
    "wilson",
    "newcombe",
    # "wilson_cc",
    # "newcombe_cc",
    "wald",
    "wald_cc",
    "haldane",
    "jeffreys-perks",
    # "mee",
    # "miettinen-nurminen",
    # "true-profile",
    # "exact",
    # "mid-p",
    # "hauck-anderson",
    # "agresti-caffo",
    # "santner-snell",
    # "chan-zhang",
    # "brown-li",
    # "miettinen-nurminen-brown-li",
]


def list_difference_confidence_interval_types() -> None:
    """ """
    
    print("\n".join(_supported_types))
