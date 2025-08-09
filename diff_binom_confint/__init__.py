"""
diff_binom_confint
====================

API documentation for the diff_binom_confint package.

.. contents::
    :depth: 2
    :local:
    :backlinks: top

.. currentmodule:: diff_binom_confint

compute_confidence_interval
---------------------------------------------

Compute confidence interval for binomial proportion.

.. autosummary::
    :toctree: generated/

    compute_confidence_interval
    list_confidence_interval_methods

compute_difference_confidence_interval
---------------------------------------------

Compute confidence interval for difference of binomial proportions.

.. autosummary::
    :toctree: generated/

    compute_difference_confidence_interval
    list_difference_confidence_interval_methods

ConfidenceInterval
---------------------------------------------

Dataclass for holding meta information of a confidence interval.

.. autosummary::
    :toctree: generated/

    ConfidenceInterval

High-level Functions
---------------------------------------------

High-level functions.

.. autosummary::
    :toctree: generated/

    make_risk_report

"""

from ._applications import make_risk_report
from ._binom_confint import compute_confidence_interval, list_confidence_interval_methods
from ._confint import ConfidenceInterval
from ._diff_binom_confint import compute_difference_confidence_interval, list_difference_confidence_interval_methods
from .version import __version__

__all__ = [
    "compute_confidence_interval",
    "list_confidence_interval_methods",
    "compute_difference_confidence_interval",
    "list_difference_confidence_interval_methods",
    "ConfidenceInterval",
    "make_risk_report",
    "__version__",
]
