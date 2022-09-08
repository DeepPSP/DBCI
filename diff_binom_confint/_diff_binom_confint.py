"""
"""

import warnings

import numpy as np
from scipy.stats import norm

from ._confint import ConfidenceInterval


__all__ = [
    "compute_difference_confidence_interval",
    "list_difference_confidence_interval_types",
]


def compute_difference_confidence_interval(
    n_positive: int,
    n_total: int,
    ref_positive: int,
    ref_total: int,
    conf_level: float = 0.95,
    confint_type: str = "wilson",
) -> ConfidenceInterval:
    """
    Compute the confidence interval of the difference between two binomial proportions.

    Parameters
    ----------
    n_positive: int,
        number of positive samples
    n_total: int,
        total number of samples
    ref_positive: int,
        number of positive samples of the reference
    ref_total: int,
        total number of samples of the reference
    conf_level: float, default 0.95,
        confidence level, should be inside the interval (0, 1)
    confint_type: str, default "wilson",
        type (computation method) of the confidence interval

    Returns
    -------
    an instance of `ConfidenceInterval`

    """
    qnorm = norm.ppf
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    z = qnorm((1 + conf_level) / 2)
    n_negative = n_total - n_positive
    ref_negative = ref_total - ref_positive
    ratio = n_positive / n_total
    ref_ratio = ref_positive / ref_total
    delta_ratio = ratio - ref_ratio
    if confint_type.lower() in ["wilson", "newcombe", "score"]:
        item1 = z * np.sqrt(4 * n_total * ratio * (1 - ratio) + z**2)
        lower1 = (2 * n_total * ratio + z**2 - item1) / 2 / (n_total + z**2)
        upper1 = (2 * n_total * ratio + z**2 + item1) / 2 / (n_total + z**2)
        item2 = z * np.sqrt(4 * ref_total * ref_ratio * (1 - ref_ratio) + z**2)
        lower2 = (2 * ref_total * ref_ratio + z**2 - item2) / 2 / (ref_total + z**2)
        upper2 = (2 * ref_total * ref_ratio + z**2 + item2) / 2 / (ref_total + z**2)
        return ConfidenceInterval(
            delta_ratio - np.sqrt((ratio - lower1) ** 2 + (upper2 - ref_ratio) ** 2),
            delta_ratio + np.sqrt((ref_ratio - lower2) ** 2 + (upper1 - ratio) ** 2),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["wilson-cc", "newcombe-cc", "score-cc"]:
        # https://corplingstats.wordpress.com/2019/04/27/correcting-for-continuity/
        # equation (6) and (6')
        e = 2 * n_total * ratio + z**2
        f = z**2 - 1 / n_total + 4 * n_total * ratio * (1 - ratio)
        g = 4 * ratio - 2
        h = 2 * (n_total + z**2)
        lower1 = (e - (z * np.sqrt(f + g) + 1)) / h
        upper1 = (e + (z * np.sqrt(f - g) + 1)) / h
        e = 2 * ref_total * ref_ratio + z**2
        f = z**2 - 1 / ref_total + 4 * ref_total * ref_ratio * (1 - ref_ratio)
        g = 4 * ref_ratio - 2
        h = 2 * (ref_total + z**2)
        lower2 = (e - (z * np.sqrt(f + g) + 1)) / h
        upper2 = (e + (z * np.sqrt(f - g) + 1)) / h
        return ConfidenceInterval(
            delta_ratio - np.sqrt((ratio - lower1) ** 2 + (upper2 - ref_ratio) ** 2),
            delta_ratio + np.sqrt((ref_ratio - lower2) ** 2 + (upper1 - ratio) ** 2),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["wald", "wald-cc"]:
        item = z * np.sqrt(
            ratio * (1 - ratio) / n_total + ref_ratio * (1 - ref_ratio) / ref_total
        )
        if confint_type.lower() == "wald-cc":
            return ConfidenceInterval(
                delta_ratio - item - 0.5 / n_total - 0.5 / ref_total,
                delta_ratio + item + 0.5 / n_total + 0.5 / ref_total,
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
        v = 0.25 / n_total - 0.25 / ref_total
        u = v + 0.5 / ref_total
        if confint_type.lower() == "haldane":
            psi = 0.5 * (ratio + ref_ratio)
        else:  # "jeffreys-perks"
            psi = 0.5 * (
                (n_positive + 0.5) / (n_total + 1)
                + (ref_positive + 0.5) / (ref_total + 1)
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
        theta = ref_total / n_total
        a = 1 + theta
        increment = 1e-5
        itv = []
        flag = None
        for j in np.arange(-1, 1 + increment, increment):
            b = -(1 + theta + ratio + theta * ref_ratio + j * (theta + 2))
            c = j * (j + 2 * ratio + theta + 1) + ratio + theta * ref_ratio
            d = -ratio * j * (1 + j)
            tmp_b = b / 3 / a
            tmp_c = c / 3 / a
            v = tmp_b**3 - tmp_b * tmp_c * 3 / 2 + d / 2 / a
            u = np.sign(v) * np.sqrt(tmp_b**2 - tmp_c)
            w = (np.pi + np.arccos(v / u**3)) / 3
            ratio_mle = 2 * u * np.cos(w) - tmp_b
            ref_ratio_mle = ratio_mle - j
            if confint_type.lower() == "mee":
                lamb = 1
            else:  # "miettinen-nurminen"
                lamb = (n_total + ref_total) / (n_total + ref_total - 1)
            var = np.sqrt(
                lamb
                * (
                    ratio_mle * (1 - ratio_mle) / n_total
                    + ref_ratio_mle * (1 - ref_ratio_mle) / ref_total
                )
            )
            var = (delta_ratio - j) / var
            if -z < var < z:
                flag = True
                itv.append(j)
            elif flag:
                break
        return ConfidenceInterval(
            np.min(itv), np.max(itv), conf_level, confint_type.lower()
        )
    elif confint_type.lower() == "true-profile":
        theta = ref_total / n_total
        a = 1 + theta
        increment = 1e-5
        itv = []
        flag = None
        for j in np.arange(-1, 1 + increment, increment):
            b = -(1 + theta + ratio + theta * ref_ratio + j * (theta + 2))
            c = j * (j + 2 * ratio + theta + 1) + ratio + theta * ref_ratio
            d = -ratio * j * (1 + j)
            tmp_b = b / 3 / a
            tmp_c = c / 3 / a
            v = tmp_b**3 - tmp_b * tmp_c * 3 / 2 + d / 2 / a
            u = np.sign(v) * np.sqrt(tmp_b**2 - tmp_c)
            w = (np.pi + np.arccos(v / u**3)) / 3
            ratio_mle = 2 * u * np.cos(w) - tmp_b
            ref_ratio_mle = ratio_mle - j
            var = (
                n_positive * np.log(ratio_mle / ratio)
                + ref_positive * np.log(ref_ratio_mle / ref_ratio)
                + n_negative * np.log((1 - ratio_mle) / (1 - ratio))
                + ref_negative * np.log((1 - ref_ratio_mle) / (1 - ref_ratio))
            )
            if var >= -(z**2) / 2:
                flag = True
                itv.append(j)
            elif flag:
                break
        return ConfidenceInterval(
            np.min(itv), np.max(itv), conf_level, confint_type.lower()
        )
    elif confint_type.lower() == "hauck-anderson":
        item = 1 / 2 / min(n_total, ref_total) + z * np.sqrt(
            ratio * (1 - ratio) / (n_total - 1)
            + ref_ratio * (1 - ref_ratio) / (ref_total - 1)
        )
        return ConfidenceInterval(
            delta_ratio - item, delta_ratio + item, conf_level, confint_type.lower()
        )
    elif confint_type.lower() == "agresti-caffo":
        ratio_1 = (n_positive + 1) / (n_total + 2)
        ratio_2 = (ref_positive + 1) / (ref_total + 2)
        item = z * np.sqrt(
            (ratio_1 * (1 - ratio_1) / (n_total + 2))
            + (ratio_2 * (1 - ratio_2) / (ref_total + 2))
        )
        return ConfidenceInterval(
            ratio_1 - ratio_2 - item,
            ratio_1 - ratio_2 + item,
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["brown-li", "brown-li-jeffrey"]:
        ratio_1 = (n_positive + 0.5) / (n_total + 1)
        ratio_2 = (ref_positive + 0.5) / (ref_total + 1)
        item = z * np.sqrt(
            ratio_1 * (1 - ratio_1) / n_total + ratio_2 * (1 - ratio_2) / ref_total
        )
        return ConfidenceInterval(
            ratio_1 - ratio_2 - item,
            ratio_1 - ratio_2 + item,
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() == "miettinen-nurminen-brown-li":
        weight = 2 / 3
        lower_mn, upper_mn = compute_difference_confidence_interval(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            conf_level,
            "miettinen-nurminen",
        ).astuple()
        lower_bl, upper_bl = compute_difference_confidence_interval(
            n_positive, n_total, ref_positive, ref_total, conf_level, "brown-li"
        ).astuple()
        lower = weight * lower_mn + (1 - weight) * lower_bl
        upper = weight * upper_mn + (1 - weight) * upper_bl
        return ConfidenceInterval(lower, upper, conf_level, confint_type.lower())
    elif confint_type.lower() == "exact":
        raise NotImplementedError
    elif confint_type.lower() == "mid-p":
        raise NotImplementedError
    elif confint_type.lower() == "santner-snell":
        raise NotImplementedError
    elif confint_type.lower() == "chan-zhang":
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
    "score",
    "wilson-cc",
    "newcombe-cc",
    "score-cc",
    "wald",
    "wald-cc",
    "haldane",
    "jeffreys-perks",
    "mee",
    "miettinen-nurminen",
    "true-profile",
    "hauck-anderson",
    "agresti-caffo",
    "brown-li",
    "brown-li-jeffrey",
    "miettinen-nurminen-brown-li",
    # "exact",
    # "mid-p",
    # "santner-snell",
    # "chan-zhang",
    # "agresti-min",
    # "wang",
    # "pradhan-banerjee",
]

_type_aliases = {
    "wilson": "newcombe",
    "wilson-cc": "newcombe-cc",
    "score": "newcombe",
    "score-cc": "newcombe-cc",
    "brown-li-jeffrey": "brown-li",
}


def list_difference_confidence_interval_types() -> None:
    """ """

    print("\n".join(_supported_types))
