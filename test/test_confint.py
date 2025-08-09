"""
"""

import pandas as pd
import pytest

from diff_binom_confint._confint import ConfidenceInterval


def test_confidence_interval():
    confint_1 = ConfidenceInterval(
        lower_bound=0.8060068421523548,
        upper_bound=0.9822620934650205,
        estimate=0.8941344678086877,
        level=0.95,
        method="wilson",
        sides="two-sided",
    )

    confint_2 = ConfidenceInterval(
        lower_bound=0.8060068421523548,
        upper_bound=0.9822620934650205,
        estimate=0.8941344678086877,
        level=0.95,
        method="miettinen-nurminen-brown-li",
        sides="left_sided",
    )

    assert str(confint_1) == str(confint_2) == "(0.8060068, 0.9822621)"
    assert confint_1 != confint_2

    assert confint_1.asdict() == {
        "lower_bound": 0.8060068421523548,
        "upper_bound": 0.9822620934650205,
        "estimate": 0.8941344678086877,
        "level": 0.95,
        "method": "wilson",
        "sides": "two-sided",
    }

    table = confint_1.astable()
    assert isinstance(table, pd.DataFrame)
    assert len(table) == 1
    assert set(table.columns) == {"Estimate", "Lower Bound", "Upper Bound", "Confidence Level", "Method", "Sides"}

    for fmt in ["html", "latex", "latex_raw", "markdown", "md", "string", "json"]:
        assert isinstance(confint_1.astable(to=fmt), str)

    for digits in [None, True, False, 3]:
        assert isinstance(confint_1.astable(to="html", digits=digits), str)

    with pytest.raises(ValueError, match="Unsupported digits type"):
        confint_1.astable(digits="xxx")
