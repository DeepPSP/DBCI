"""
"""

from typing import NoReturn

import numpy as np
from scipy.stats import norm, beta, binom
from scipy.optimize import brentq

from ._confint import ConfidenceInterval
from ._utils import add_docstring, remove_parameters_returns_from_docstring


__all__ = [
    "compute_confidence_interval",
    "list_confidence_interval_types",
]


RNG = np.random.default_rng()  # random number generator


# aliases of statistical functions
pbinom = binom.cdf
qbinom = binom.ppf
dbinom = binom.pmf
dbinom_log = binom.logpmf
qnorm = norm.ppf
qbeta = beta.ppf
uniroot = brentq


def compute_confidence_interval(
    n_positive: int,
    n_total: int,
    conf_level: float = 0.95,
    confint_type: str = "wilson",
    clip: bool = True,
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
    clip: bool, default True,
        whether to clip the confidence interval to the interval (0, 1)

    Returns
    -------
    confint: ConfidenceInterval,
        the confidence interval

    """

    if confint_type not in _supported_types:
        raise ValueError(
            f"confint_type should be one of {_supported_types}, "
            f"but got {confint_type}"
        )

    if conf_level <= 0 or conf_level >= 1:
        raise ValueError(
            f"conf_level should be inside the interval (0, 1), " f"but got {conf_level}"
        )

    if n_positive > n_total:
        raise ValueError(
            f"n_positive should be less than or equal to n_total, "
            f"but got n_positive={n_positive} and n_total={n_total}"
        )

    if n_positive < 0:
        raise ValueError(
            f"n_positive should be non-negative, but got n_positive={n_positive}"
        )

    if n_total <= 0:
        raise ValueError(f"n_total should be positive, but got n_total={n_total}")

    confint = _compute_confidence_interval(
        n_positive, n_total, conf_level, confint_type
    )
    if clip:
        confint.lower_bound = max(0, confint.lower_bound)
        confint.upper_bound = min(1, confint.upper_bound)
    return confint


@add_docstring(
    remove_parameters_returns_from_docstring(
        compute_confidence_interval.__doc__, parameters="clip"
    )
)
def _compute_confidence_interval(
    n_positive: int,
    n_total: int,
    conf_level: float = 0.95,
    confint_type: str = "wilson",
) -> ConfidenceInterval:
    """ """

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
        # n_pos_tilde = n_positive + RNG.uniform(0, 1)
        raise NotImplementedError
    elif confint_type.lower() == "mid-p":
        raise NotImplementedError
    elif confint_type.lower() == "lik":
        tol = 1e-5
        lower, upper = 0, 1
        if (
            n_positive != 0
            and tol < ratio
            and _bin_dev(tol, n_positive, ratio, n_total, -z, tol) <= 0
        ):
            lower = uniroot(
                lambda y: _bin_dev(y, n_positive, ratio, n_total, -z, tol),
                tol,
                1 - tol if (ratio < tol or ratio == 1) else ratio,
                full_output=False,
            )
        if n_negative != 0 and ratio < 1 - tol:
            if (
                _bin_dev(
                    1 - tol,
                    n_positive,
                    tol if ratio > 1 - tol else ratio,
                    n_total,
                    z,
                    tol,
                )
                < 0
                and _bin_dev(
                    tol,
                    n_positive,
                    1 - tol if (ratio < tol or ratio == 1) else ratio,
                    n_total,
                    -z,
                    tol,
                )
                <= 0
            ):
                upper = lower = uniroot(
                    lambda y: _bin_dev(y, n_positive, ratio, n_total, -z, tol),
                    tol,
                    ratio,
                    full_output=False,
                )
            else:
                upper = uniroot(
                    lambda y: _bin_dev(y, n_positive, ratio, n_total, z, tol),
                    tol if ratio > 1 - tol else ratio,
                    1 - tol,
                    full_output=False,
                )
        return ConfidenceInterval(lower, upper, conf_level, confint_type.lower())
    elif confint_type.lower() == "blaker":
        # tol = np.sqrt(np.finfo(float).eps)
        tol = 1e-5
        lower, upper = 0, 1
        if n_positive > 0:
            lower = qbeta(0.5 * (1 - conf_level), n_positive, n_negative + 1)
            while _acceptbin(n_positive, n_total, lower + tol) < 1 - conf_level:
                lower += tol
        if n_negative > 0:
            upper = qbeta(0.5 * (1 + conf_level), n_positive + 1, n_negative)
            while _acceptbin(n_positive, n_total, upper - tol) < 1 - conf_level:
                upper -= tol
        return ConfidenceInterval(lower, upper, conf_level, confint_type.lower())
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
    # "mid-p",
    "lik",
    "blaker",
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


def _acceptbin(n_positive: int, n_total: int, prob: int) -> float:
    """ """

    p1 = 1 - pbinom(n_positive - 1, n_total, prob)
    p2 = pbinom(n_positive, n_total, prob)

    a1 = p1 + pbinom(qbinom(p1, n_total, prob) - 1, n_total, prob)
    a2 = p2 + 1 - pbinom(qbinom(1 - p2, n_total, prob), n_total, prob)

    return min(a1, a2)


def _bin_dev(y, x, mu, wt, bound=0, tol=1e-5) -> float:
    """binomial deviance for y, x, wt"""
    ll_y = 0 if y in [0, 1] else dbinom_log(x, wt, y)
    ll_mu = 0 if mu in [0, 1] else dbinom_log(x, wt, mu)
    res = 0 if np.abs(y - mu) < tol else np.sign(y - mu) * np.sqrt(-2 * (ll_y - ll_mu))
    return res - bound
