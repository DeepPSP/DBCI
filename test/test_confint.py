"""
"""

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

    assert str(confint_1) == str(confint_2) == "(0.80601, 0.98226)"
    assert confint_1 != confint_2
