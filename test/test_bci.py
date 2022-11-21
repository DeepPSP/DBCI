"""
"""

import re
import warnings
from pathlib import Path
from typing import List

import pandas as pd
from pytest import approx, raises

from diff_binom_confint import compute_confidence_interval
from diff_binom_confint._binom_confint import (
    _supported_methods,
    _method_aliases,
    _stochastic_methods,
)


_TEST_DATA_DIR = Path(__file__).parent / "test-data"


def load_test_data() -> List[pd.DataFrame]:
    test_file_pattern = "example-(?P<n_positive>[\\d]+)-(?P<n_total>[\\d]+)\\.csv"
    test_files = Path(_TEST_DATA_DIR).glob("*.csv")
    test_data = []
    for file in test_files:
        match = re.match(test_file_pattern, file.name)
        if match:
            n_positive = int(match.group("n_positive"))
            n_total = int(match.group("n_total"))
            df_data = pd.read_csv(file)
            for t1, t2 in _method_aliases.items():
                if t1 in df_data["method"].values:
                    df_data = pd.concat(
                        [df_data, df_data[df_data["method"] == t1].assign(method=t2)]
                    )
                elif t2 in df_data["method"].values:
                    df_data = pd.concat(
                        [df_data, df_data[df_data["method"] == t2].assign(method=t1)]
                    )
            test_data.append(
                {
                    "n_positive": n_positive,
                    "n_total": n_total,
                    "data": df_data,
                }
            )
    print(f"Totally {len(test_data)} test data loaded")
    return test_data


def test_confidence_interval():
    """ """
    n_positive, n_total = 84, 101
    df_data = pd.read_csv(_TEST_DATA_DIR / "example-84-101.csv")
    for t1, t2 in _method_aliases.items():
        if t1 in df_data["method"].values:
            df_data = pd.concat(
                [df_data, df_data[df_data["method"] == t1].assign(method=t2)]
            )
        elif t2 in df_data["method"].values:
            df_data = pd.concat(
                [df_data, df_data[df_data["method"] == t2].assign(method=t1)]
            )
    # assert set(_supported_methods) <= set(
    #     df_data["method"].values
    # ), f"""methods {set(_supported_methods) - set(df_data["method"].values)} has no test data"""
    no_test_data_methods = list(set(_supported_methods) - set(df_data["method"].values))
    if len(no_test_data_methods) > 0:
        warnings.warn(
            f"""methods `{no_test_data_methods}` has no test data""",
            RuntimeWarning,
        )

    max_length = max([len(x) for x in _supported_methods]) + 1
    error_bound = 1e-4

    print("Testing 2-sided confidence interval")
    for confint_method in _supported_methods:
        if confint_method in _stochastic_methods + no_test_data_methods:
            continue
        lower, upper = compute_confidence_interval(
            n_positive,
            n_total,
            method=confint_method,
        ).astuple()
        print(f"{confint_method.ljust(max_length)}: [{lower:.2%}, {upper:.2%}]")
        row = df_data[df_data["method"] == confint_method].iloc[0]
        assert lower == approx(
            row["lower_bound"], abs=error_bound
        ), f"for {confint_method}, lower bound should be {row['lower_bound']:.2%}, but got {lower:.2%}"
        assert upper == approx(
            row["upper_bound"], abs=error_bound
        ), f"for {confint_method}, upper bound should be {row['upper_bound']:.2%}, but got {upper:.2%}"

    print("Testing left-sided confidence interval")
    for confint_method in _supported_methods:
        if confint_method in _stochastic_methods:
            continue
        lower, upper = compute_confidence_interval(
            n_positive,
            n_total,
            conf_level=0.975,
            method=confint_method,
            sides="left",
        ).astuple()
        print(f"{confint_method.ljust(max_length)}: [{lower:.2%}, {upper:.2%}]")
        row = df_data[df_data["method"] == confint_method].iloc[0]
        assert lower == approx(
            row["lower_bound"], abs=error_bound
        ), f"for {confint_method}, lower bound should be {row['lower_bound']:.2%}, but got {lower:.2%}"
        assert (
            upper == 1
        ), f"for {confint_method}, upper bound should be {1.0:.2%}, but got {upper:.2%}"

    print("Testing right-sided confidence interval")
    for confint_method in _supported_methods:
        if confint_method in _stochastic_methods:
            continue
        lower, upper = compute_confidence_interval(
            n_positive,
            n_total,
            conf_level=0.975,
            method=confint_method,
            sides="right",
        ).astuple()
        print(f"{confint_method.ljust(max_length)}: [{lower:.2%}, {upper:.2%}]")
        row = df_data[df_data["method"] == confint_method].iloc[0]
        assert (
            lower == 0
        ), f"for {confint_method}, lower bound should be {0.0:.2%}, but got {lower:.2%}"
        assert upper == approx(
            row["upper_bound"], abs=error_bound
        ), f"for {confint_method}, upper bound should be {row['upper_bound']:.2%}, but got {upper:.2%}"

    print("test_confidence_interval passed")


def test_errors():
    with raises(ValueError, match="confint_type should be one of"):
        compute_confidence_interval(1, 2, method="not-supported")
    with raises(
        ValueError, match="conf_level should be inside the interval \\(0, 1\\)"
    ):
        compute_confidence_interval(1, 2, conf_level=0)
    with raises(ValueError, match="n_positive should be less than or equal to n_total"):
        compute_confidence_interval(2, 1)
    with raises(ValueError, match="n_positive should be non-negative"):
        compute_confidence_interval(-1, 1)
    with raises(ValueError, match="n_total should be positive"):
        compute_confidence_interval(0, 0)
    with raises(ValueError, match="sides should be one of"):
        compute_confidence_interval(1, 2, sides="3-sided")
