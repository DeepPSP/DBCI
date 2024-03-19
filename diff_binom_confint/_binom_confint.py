"""
"""

from typing import Optional, Union

import numpy as np
from deprecate_kwargs import deprecate_kwargs
from deprecated import deprecated
from scipy.optimize import brentq
from scipy.stats import beta, binom, ncx2, norm

from ._confint import _SIDE_NAME_MAP, ConfidenceInterval, ConfidenceIntervalSides
from ._utils import add_docstring, remove_parameters_returns_from_docstring

__all__ = [
    "compute_confidence_interval",
    "list_confidence_interval_methods",
]


RNG = np.random.default_rng()  # random number generator
DEFAULT_METHOD = "wilson"


# aliases of statistical functions
pbinom = binom.cdf
qbinom = binom.ppf
dbinom = binom.pmf
dbinom_log = binom.logpmf
qnorm = norm.ppf
qbeta = beta.ppf
qchisq = ncx2.ppf
uniroot = brentq


@deprecate_kwargs([["method", "confint_type"]])
def compute_confidence_interval(
    n_positive: int,
    n_total: int,
    conf_level: float = 0.95,
    confint_type: Optional[str] = None,
    clip: bool = True,
    sides: Union[str, int] = "two-sided",
    digits: int = 7,
) -> ConfidenceInterval:
    """Compute the confidence interval for a binomial proportion.

    Parameters
    ----------
    n_positive : int
        Number of positive samples.
    n_total : int
        Total number of samples.
    conf_level : float, default 0.95
        Confidence level, should be inside the interval ``(0, 1)``.
    confint_type : str, optional
        Type (computation method) of the confidence interval.
        Default is "wilson".
    clip : bool, default True
        Whether to clip the confidence interval to the interval ``(0, 1)``.
    sides : str or int, default "two-sided"
        Sides of the confidence interval, should be one of

        - "two-sided" (aliases "2-sided", "two_sided", "2_sided", "2-sides",
          "two_sides", "two-sides", "2_sides", "ts", "t", "two", "2", 2),
        - "left-sided" (aliases "left_sided", "left", "ls", "l"),
        - "right-sided" (aliases "right_sided", "right", "rs", "r"),

        case insensitive.
    digits : int, default 7
        Number of digits to round the confidence interval to in the string representation.

    Returns
    -------
    confint : ConfidenceInterval
        The confidence interval.

    """
    if confint_type is None:
        confint_type = DEFAULT_METHOD
    if confint_type not in _supported_types:
        raise ValueError(f"method should be one of {repr(_supported_types)}, but got {repr(confint_type)}")

    if conf_level <= 0 or conf_level >= 1:
        raise ValueError(f"conf_level should be inside the interval (0, 1), but got {repr(conf_level)}")

    if n_positive > n_total:
        raise ValueError(
            f"n_positive should be less than or equal to n_total, " f"but got n_positive={n_positive} and n_total={n_total}"
        )

    if n_positive < 0:
        raise ValueError(f"n_positive should be non-negative, but got n_positive={n_positive}")

    if n_total <= 0:
        raise ValueError(f"n_total should be positive, but got n_total={n_total}")

    sides = str(sides).lower()
    if sides not in _SIDE_NAME_MAP:
        raise ValueError(f"sides should be one of {repr(list(_SIDE_NAME_MAP))}, but got {repr(sides)}")
    else:
        sides = _SIDE_NAME_MAP[sides]

    confint = _compute_confidence_interval(n_positive, n_total, conf_level, confint_type, sides, digits)

    if clip:
        confint.lower_bound = max(0, confint.lower_bound)
        confint.upper_bound = min(1, confint.upper_bound)

    if confint.sides == ConfidenceIntervalSides.LeftSided.value:
        confint.upper_bound = 1
    elif confint.sides == ConfidenceIntervalSides.RightSided.value:
        confint.lower_bound = 0

    return confint


@add_docstring(
    """
    NOTE
    ----
    The lower bound and upper bound are not adjusted w.r.t. `sides`.

    """,
    "append",
)
@add_docstring(remove_parameters_returns_from_docstring(compute_confidence_interval.__doc__, parameters="clip"))
def _compute_confidence_interval(
    n_positive: int,
    n_total: int,
    conf_level: float = 0.95,
    confint_type: str = "wilson",
    sides: Union[str, int] = "two-sided",
    digits: int = 7,
) -> ConfidenceInterval:
    if sides != "two-sided":
        zeta = qnorm(conf_level)
        margin = 1 - conf_level
        _conf_level = 1 - 2 * margin
    else:
        zeta = qnorm((1 + conf_level) / 2)
        margin = 0.5 * (1 - conf_level)
        _conf_level = conf_level
    n_negative = n_total - n_positive
    ratio = n_positive / n_total
    neg_ratio = 1 - ratio
    if confint_type.lower() in ["wilson", "newcombe"]:
        item = zeta * np.sqrt((ratio * neg_ratio + zeta * zeta / (4 * n_total)) / n_total)
        return ConfidenceInterval(
            ((ratio + zeta * zeta / (2 * n_total) - item) / (1 + zeta * zeta / n_total)),
            ((ratio + zeta * zeta / (2 * n_total) + item) / (1 + zeta * zeta / n_total)),
            ratio,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() in ["wilson-cc", "newcombe-cc"]:
        # https://corplingstats.wordpress.com/2019/04/27/correcting-for-continuity/
        # equation (6) and (6')
        e = 2 * n_total * ratio + zeta**2
        f = zeta**2 - 1 / n_total + 4 * n_total * ratio * neg_ratio
        g = 4 * ratio - 2
        h = 2 * (n_total + zeta**2)
        return ConfidenceInterval(
            (e - (zeta * np.sqrt(f + g) + 1)) / h,
            (e + (zeta * np.sqrt(f - g) + 1)) / h,
            ratio,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() in ["wald", "wald-cc"]:
        item = zeta * np.sqrt(ratio * neg_ratio / n_total)
        if confint_type.lower() == "wald-cc":
            item += 0.5 / n_total
        return ConfidenceInterval(
            ratio - item,
            ratio + item,
            ratio,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() == "agresti-coull":
        ratio_tilde = (n_positive + 0.5 * zeta**2) / (n_total + zeta**2)
        item = zeta * np.sqrt(ratio_tilde * (1 - ratio_tilde) / (n_total + zeta**2))
        return ConfidenceInterval(
            ratio_tilde - item,
            ratio_tilde + item,
            ratio_tilde,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() == "jeffreys":
        # print(margin, sides)
        return ConfidenceInterval(
            qbeta(margin, n_positive + 0.5, n_negative + 0.5) if n_positive > 0 else 0,
            qbeta(1 - margin, n_positive + 0.5, n_negative + 0.5) if n_negative > 0 else 1,
            ratio,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() == "clopper-pearson":
        return ConfidenceInterval(
            qbeta(margin, n_positive, n_negative + 1),
            qbeta(1 - margin, n_positive + 1, n_negative),
            ratio,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() == "arcsine":
        ratio_tilde = (n_positive + 0.375) / (n_total + 0.75)
        return ConfidenceInterval(
            np.sin(np.arcsin(np.sqrt(ratio_tilde)) - 0.5 * zeta / np.sqrt(n_total)) ** 2,
            np.sin(np.arcsin(np.sqrt(ratio_tilde)) + 0.5 * zeta / np.sqrt(n_total)) ** 2,
            ratio_tilde,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() == "logit":
        # lambda_hat = np.log(ratio / neg_ratio)
        # V_hat = 1 / ratio / n_negative  # derivative of `lambda_hat` w.r.t. `n_positive`
        # lambda_lower = lambda_hat - zeta * np.sqrt(V_hat)
        # lambda_upper = lambda_hat + zeta * np.sqrt(V_hat)
        # note that for x = n_positive, one has (apply exp() and use L'Hopital's rule to get the limits)
        # :math:`\lim_{x \to 0+} lambda\_lower = -\infty`, hence :math:`\lim_{x \to 0+} \exp(lambda\_lower) / (1 + \exp(lambda\_lower)) = 0`
        # :math:`\lim_{x \to 0+} lambda\_upper = -\infty`, hence :math:`\lim_{x \to 0+} \exp(lambda\_upper) / (1 + \exp(lambda\_upper)) = 0`
        # :math:`\lim_{x \to n\_total-} lambda\_lower = +\infty`, hence :math:`\lim_{x \to n\_total-} \exp(lambda\_lower) / (1 + \exp(lambda\_lower)) = 1`
        # :math:`\lim_{x \to n\_total-} lambda\_upper = +\infty`, hence :math:`\lim_{x \to n\_total-} \exp(lambda\_upper) / (1 + \exp(lambda\_upper)) = 1`
        # therefore one can set
        # confidence interval = [0, 0] if n_positive = 0
        # confidence interval = [1, 1] if n_positive = n_total
        if n_positive == 0:
            return ConfidenceInterval(0, 0, ratio, conf_level, confint_type.lower(), str(sides), digits)
        elif n_positive == n_total:
            return ConfidenceInterval(1, 1, ratio, conf_level, confint_type.lower(), str(sides), digits)

        lambda_hat = np.log(ratio / neg_ratio)
        V_hat = 1 / ratio / n_negative  # derivative of `lambda_hat` w.r.t. `n_positive`
        lambda_lower = lambda_hat - zeta * np.sqrt(V_hat)
        lambda_upper = lambda_hat + zeta * np.sqrt(V_hat)
        return ConfidenceInterval(
            np.exp(lambda_lower) / (1 + np.exp(lambda_lower)),
            np.exp(lambda_upper) / (1 + np.exp(lambda_upper)),
            ratio,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() == "pratt":
        if n_positive == 0:
            return ConfidenceInterval(
                0,
                1 - np.power(1 - _conf_level, 1 / n_total),
                ratio,
                conf_level,
                confint_type.lower(),
                str(sides),
                digits,
            )
        elif n_positive == 1:
            return ConfidenceInterval(
                1 - np.power(1 - margin, 1 / n_total),
                1 - np.power(margin, 1 / n_total),
                ratio,
                conf_level,
                confint_type.lower(),
                str(sides),
                digits,
            )
        elif n_negative == 1:
            return ConfidenceInterval(
                np.power(margin, 1 / n_total),
                np.power(1 - margin, 1 / n_total),
                ratio,
                conf_level,
                confint_type.lower(),
                str(sides),
                digits,
            )
        elif n_negative == 0:
            return ConfidenceInterval(
                np.power(1 - _conf_level, 1 / n_total),
                1,
                ratio,
                conf_level,
                confint_type.lower(),
                str(sides),
                digits,
            )
        else:
            a = ((n_positive + 1) / n_negative) ** 2
            b = 81 * (n_positive + 1) * n_negative - 9 * n_total - 8
            c = -3 * zeta * np.sqrt(9 * (n_positive + 1) * n_negative * (9 * n_total + 5 - zeta**2) + n_total + 1)
            d = 81 * (n_positive + 1) ** 2 - 9 * (n_positive + 1) * (2 + zeta**2) + 1
            upper = 1 / (1 + a * ((b + c) / d) ** 3)
            a = (n_positive / (n_negative - 1)) ** 2
            b = 81 * n_positive * (n_negative - 1) - 9 * n_total - 8
            c = 3 * zeta * np.sqrt(9 * n_positive * (n_negative - 1) * (9 * n_total + 5 - zeta**2) + n_total + 1)
            d = 81 * n_positive**2 - 9 * n_positive * (2 + zeta**2) + 1
            lower = 1 / (1 + a * ((b + c) / d) ** 3)
            return ConfidenceInterval(lower, upper, ratio, conf_level, confint_type.lower(), str(sides), digits)
    elif confint_type.lower() == "witting":
        # stochastic, checked by seeting n_pos_tilde = n_positive
        # n_pos_tilde = n_positive
        n_pos_tilde = n_positive + RNG.uniform(0, 1)
        return ConfidenceInterval(
            _qbinom_abscont(_conf_level, n_total, n_pos_tilde) if n_positive != 0 else 0.0,
            _qbinom_abscont(1 - _conf_level, n_total, n_pos_tilde) if n_positive != n_total else 1.0,
            ratio,
            conf_level,
            confint_type.lower(),
            str(sides),
            digits,
        )
    elif confint_type.lower() == "mid-p":
        if n_positive == 0:
            lower = 0
        else:
            lower = uniroot(
                lambda pi: _f_low(pi, n_positive, n_total, _conf_level),
                0,
                ratio,
                full_output=False,
            )
        if n_negative == 0:
            upper = 1
        else:
            upper = uniroot(
                lambda pi: _f_up(pi, n_positive, n_total, _conf_level),
                ratio,
                1,
                full_output=False,
            )
        return ConfidenceInterval(lower, upper, ratio, conf_level, confint_type.lower(), str(sides), digits)
    elif confint_type.lower() == "lik":
        tol = 1e-5
        lower, upper = 0, 1
        if n_positive != 0 and tol < ratio and _bin_dev(tol, n_positive, ratio, n_total, -zeta, tol) <= 0:
            lower = uniroot(
                lambda y: _bin_dev(y, n_positive, ratio, n_total, -zeta, tol),
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
                    zeta,
                    tol,
                )
                < 0
                and _bin_dev(
                    tol,
                    n_positive,
                    1 - tol if (ratio < tol or ratio == 1) else ratio,
                    n_total,
                    -zeta,
                    tol,
                )
                <= 0
            ):
                upper = lower = uniroot(
                    lambda y: _bin_dev(y, n_positive, ratio, n_total, -zeta, tol),
                    tol,
                    ratio,
                    full_output=False,
                )
            else:
                upper = uniroot(
                    lambda y: _bin_dev(y, n_positive, ratio, n_total, zeta, tol),
                    tol if ratio > 1 - tol else ratio,
                    1 - tol,
                    full_output=False,
                )
        return ConfidenceInterval(lower, upper, ratio, conf_level, confint_type.lower(), str(sides), digits)
    elif confint_type.lower() == "blaker":
        # tol = np.sqrt(np.finfo(float).eps)
        tol = 1e-5
        lower, upper = 0, 1
        if n_positive > 0:
            lower = qbeta(margin, n_positive, n_negative + 1)
            while _acceptbin(n_positive, n_total, lower + tol) < 1 - _conf_level:
                lower += tol
        if n_negative > 0:
            upper = qbeta(1 - margin, n_positive + 1, n_negative)
            while _acceptbin(n_positive, n_total, upper - tol) < 1 - _conf_level:
                upper -= tol
        return ConfidenceInterval(lower, upper, ratio, conf_level, confint_type.lower(), str(sides), digits)
    elif confint_type.lower() in ["modified-wilson", "modified-newcombe"]:
        term1 = (n_positive + 0.5 * zeta**2) / (n_total + zeta**2)
        term2 = zeta * np.sqrt(n_total) * np.sqrt(ratio * neg_ratio + zeta**2 / (4 * n_total)) / (n_total + zeta**2)
        if (n_total <= 50 and n_positive in [1, 2]) or (n_total >= 51 and n_positive in [1, 2, 3]):
            lower = 0.5 * qchisq(margin, 2 * n_positive, 0) / n_total
        else:
            lower = max(0, term1 - term2)
        if (n_total <= 50 and n_negative in [1, 2]) or (n_total >= 51 and n_negative in [1, 2, 3]):
            upper = 0.5 * qchisq(margin, 2 * n_negative, 0) / n_total
        else:
            upper = min(1, term1 + term2)
        return ConfidenceInterval(lower, upper, ratio, conf_level, confint_type.lower(), str(sides), digits)
    elif confint_type.lower() == "modified-jeffreys":
        if n_negative == 0:
            lower = np.power(margin, 1 / n_total)
        elif n_positive <= 1:
            lower = 0
        else:
            lower = qbeta(margin, n_positive + 0.5, n_negative + 0.5)
        if n_positive == 0:
            upper = 1 - np.power(margin, 1 / n_total)
        elif n_negative <= 1:
            upper = 1
        else:
            upper = qbeta(1 - margin, n_positive + 0.5, n_negative + 0.5)
        return ConfidenceInterval(lower, upper, ratio, conf_level, confint_type.lower(), str(sides), digits)
    else:
        newline = "\n"
        raise ValueError(
            f"""method {repr(confint_type)} is not supported, choose one from {newline}{newline.join(_supported_types)}"""
        )


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
    "witting",
    "mid-p",
    "lik",
    "blaker",
    "modified-wilson",
    "modified-newcombe",
    "modified-jeffreys",
]
_supported_methods = _supported_types


_stochastic_types = [
    "witting",
]
_stochastic_methods = _stochastic_types


_type_aliases = {
    "wilson": "newcombe",
    "wilson-cc": "newcombe-cc",
    "modified-wilson": "modified-newcombe",
}
_method_aliases = _type_aliases


@deprecated(version="0.0.4", reason="Use `list_confidence_interval_methods` instead.")
def list_confidence_interval_types() -> None:
    """List all supported confidence interval types."""
    print("\n".join(_supported_types))


def list_confidence_interval_methods() -> None:
    """List all supported confidence interval types."""
    print("\n".join(_supported_types))


def _acceptbin(n_positive: int, n_total: int, prob: int) -> float:
    p1 = 1 - pbinom(n_positive - 1, n_total, prob)
    p2 = pbinom(n_positive, n_total, prob)

    a1 = p1 + pbinom(qbinom(p1, n_total, prob) - 1, n_total, prob)
    a2 = p2 + 1 - pbinom(qbinom(1 - p2, n_total, prob), n_total, prob)

    return min(a1, a2)


def _bin_dev(y: float, x: int, mu: float, wt: int, bound: float = 0.0, tol: float = 1e-5) -> float:
    """Binomial deviance for `y`, `x`, `wt`."""
    ll_y = 0 if y in [0, 1] else dbinom_log(x, wt, y)
    ll_mu = 0 if mu in [0, 1] else dbinom_log(x, wt, mu)
    res = 0 if np.abs(y - mu) < tol else np.sign(y - mu) * np.sqrt(-2 * (ll_y - ll_mu))
    return res - bound


def _f_low(pi: float, x: int, n: int, conf_level: float) -> float:
    """Function to find root of for the lower bound of the CI."""
    return 0.5 * dbinom(x, n, pi) + 1 - pbinom(x, n, pi) - 0.5 * (1 - conf_level)


def _f_up(pi: float, x: int, n: int, conf_level: float) -> float:
    """Function to find root of for the upper bound of the CI."""
    return 0.5 * dbinom(x, n, pi) + pbinom(x - 1, n, pi) - 0.5 * (1 - conf_level)


def _pbinom_abscont(q: float, size: int, prob: float) -> float:
    v = np.trunc(q)
    term1 = pbinom(v - 1, size, prob)
    term2 = (q - v) * dbinom(v, size, prob)
    return term1 + term2


def _qbinom_abscont(p: float, size: int, x: int) -> float:
    return uniroot(lambda prob: _pbinom_abscont(x, size, prob) - p, 0, 1, full_output=False)
