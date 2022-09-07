"""
"""

from typing import NoReturn

import numpy as np
from scipy.stats import norm, beta

from ._confint import ConfidenceInterval


__all__ = [
    "compute_confidence_interval",
    "list_confidence_interval_types",
]


def compute_confidence_interval(
    n_positive: int,
    n_total: int,
    conf_level: float = 0.95,
    confint_type: str = "wilson",
) -> ConfidenceInterval:
    """
    Compute the confidence interval for a binomial proportion.

    Parameters
    ----------
    n_positive: int,
        number of positive samples
    n_total: int,
        total number of samples
    conf_level: float, default 0.95,
        confidence level, should be inside the interval (0, 1)
    confint_type: str, default "wilson",
        type (computation method) of the confidence interval

    Returns
    -------
    an instance of `ConfidenceInterval`

    """
    qnorm = norm.ppf
    qbeta = beta.ppf

    z = qnorm((1 + conf_level) / 2)
    n_negative = n_total - n_positive
    ratio = n_positive / n_total
    if confint_type.lower() in ["wilson", "newcombe"]:
        item = z * np.sqrt((ratio * (1 - ratio) + z * z / (4 * n_total)) / n_total)
        return ConfidenceInterval(
            ((ratio + z * z / (2 * n_total) - item) / (1 + z * z / n_total)),
            ((ratio + z * z / (2 * n_total) + item) / (1 + z * z / n_total)),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["wilson-cc", "newcombe-cc"]:
        e = 2 * n_total * ratio + z**2
        f = z**2 - 1 / n_total + 4 * n_total * ratio * (1 - ratio)
        g = 4 * ratio - 2
        h = 2 * (n_total + z**2)
        return ConfidenceInterval(
            (e - (z * np.sqrt(f + g) + 1)) / h,
            (e + (z * np.sqrt(f - g) + 1)) / h,
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() in ["wald", "wald-cc"]:
        item = z * np.sqrt(ratio * (1 - ratio) / n_total)
        if confint_type.lower() == "wald-cc":
            item += 0.5 / n_total
        return ConfidenceInterval(
            ratio - item,
            ratio + item,
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() == "agresti-coull":
        ratio_tilde = (n_positive + 0.5 * z**2) / (n_total + z**2)
        item = z * np.sqrt(ratio_tilde * (1 - ratio_tilde) / (n_total + z**2))
        return ConfidenceInterval(
            ratio_tilde - item,
            ratio_tilde + item,
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() == "jeffreys":
        return ConfidenceInterval(
            qbeta(0.5 * (1 - conf_level), n_positive + 0.5, n_negative + 0.5)
            if n_positive > 0
            else 0,
            qbeta(0.5 * (1 + conf_level), n_positive + 0.5, n_negative + 0.5)
            if n_negative > 0
            else 1,
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() == "clopper-pearson":
        return ConfidenceInterval(
            qbeta(0.5 * (1 - conf_level), n_positive, n_negative + 1),
            qbeta(0.5 * (1 + conf_level), n_positive + 1, n_negative),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() == "arcsine":
        ratio_tilde = (n_positive + 0.375) / (n_total + 0.75)
        return ConfidenceInterval(
            np.sin(np.arcsin(np.sqrt(ratio_tilde)) - 0.5 * z / np.sqrt(n_total)) ** 2,
            np.sin(np.arcsin(np.sqrt(ratio_tilde)) + 0.5 * z / np.sqrt(n_total)) ** 2,
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() == "logit":
        lambda_hat = np.log(ratio / (1 - ratio))
        V_hat = 1 / ratio / n_negative
        lambda_lower = lambda_hat - z * np.sqrt(V_hat)
        lambda_upper = lambda_hat + z * np.sqrt(V_hat)
        return ConfidenceInterval(
            np.exp(lambda_lower) / (1 + np.exp(lambda_lower)),
            np.exp(lambda_upper) / (1 + np.exp(lambda_upper)),
            conf_level,
            confint_type.lower(),
        )
    elif confint_type.lower() == "pratt":
        if n_positive == 0:
            return ConfidenceInterval(
                0,
                1 - np.power(1 - conf_level, 1 / n_total),
                conf_level,
                confint_type.lower(),
            )
        elif n_positive == 1:
            return ConfidenceInterval(
                1 - np.power(0.5 * (1 + conf_level), 1 / n_total),
                1 - np.power(0.5 * (1 - conf_level), 1 / n_total),
                conf_level,
                confint_type.lower(),
            )
        elif n_negative == 1:
            return ConfidenceInterval(
                np.power(0.5 * (1 - conf_level), 1 / n_total),
                np.power(0.5 * (1 + conf_level), 1 / n_total),
                conf_level,
                confint_type.lower(),
            )
        elif n_negative == 0:
            return ConfidenceInterval(
                np.power(1 - conf_level, 1 / n_total),
                1,
                conf_level,
                confint_type.lower(),
            )
        else:
            a = ((n_positive + 1) / n_negative) ** 2
            b = 81 * (n_positive + 1) * n_negative - 9 * n_total - 8
            c = (
                -3
                * z
                * np.sqrt(
                    9 * (n_positive + 1) * n_negative * (9 * n_total + 5 - z**2)
                    + n_total
                    + 1
                )
            )
            d = 81 * (n_positive + 1) ** 2 - 9 * (n_positive + 1) * (2 + z**2) + 1
            upper = 1 / (1 + a * ((b + c) / d) ** 3)
            a = (n_positive / (n_negative - 1)) ** 2
            b = 81 * n_positive * (n_negative - 1) - 9 * n_total - 8
            c = (
                3
                * z
                * np.sqrt(
                    9 * n_positive * (n_negative - 1) * (9 * n_total + 5 - z**2)
                    + n_total
                    + 1
                )
            )
            d = 81 * n_positive**2 - 9 * n_positive * (2 + z**2) + 1
            lower = 1 / (1 + a * ((b + c) / d) ** 3)
            return ConfidenceInterval(lower, upper, conf_level, confint_type.lower())
    elif confint_type.lower() == "witting":
        raise NotImplementedError
    elif confint_type.lower() == "midp":
        raise NotImplementedError
    elif confint_type.lower() == "lik":
        raise NotImplementedError
    elif confint_type.lower() == "blaker":
        raise NotImplementedError
    elif confint_type.lower() in ["modified-wilson", "modified-newcombe"]:
        raise NotImplementedError
    elif confint_type.lower() == "modified-jeffreys":
        raise NotImplementedError
    else:
        raise ValueError(f"{confint_type} is not supported")


_supported_types = [
    "wilson",
    "newcombe",
    "wilson-cc",
    "newcombe-cc",
    "wald",
    "wald-cc",
    "agresti-coull",
    "jeffreys",
    "clopper-pearson",
    "arcsine",
    "logit",
    "pratt",
    # "witting",
    # "midp",
    # "lik",
    # "blaker",
    # "modified-wilson",
    # "modified-newcombe",
    # "modified-jeffreys",
]


_type_aliases = {
    "wilson": "newcombe",
    "wilson-cc": "newcombe-cc",
    "modified-wilson": "modified-newcombe",
}


def list_confidence_interval_types() -> NoReturn:
    """ """

    print("\n".join(_supported_types))
