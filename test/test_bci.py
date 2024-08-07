"""
"""

import re
from pathlib import Path

import pandas as pd
from pytest import approx, raises, warns

from diff_binom_confint import compute_confidence_interval
from diff_binom_confint._binom_confint import (
    _compute_confidence_interval,
    _method_aliases,
    _stochastic_methods,
    _supported_methods,
    list_confidence_interval_methods,
    list_confidence_interval_types,
)

_TEST_DATA_DIR = Path(__file__).parent / "test-data"


def test_load_data() -> None:
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
                    df_data = pd.concat([df_data, df_data[df_data["method"] == t1].assign(method=t2)])
                elif t2 in df_data["method"].values:
                    df_data = pd.concat([df_data, df_data[df_data["method"] == t2].assign(method=t1)])
            test_data.append(
                {
                    "n_positive": n_positive,
                    "n_total": n_total,
                    "data": df_data,
                }
            )
    print(f"Totally {len(test_data)} test data loaded")
    # return test_data
    assert len(test_data) > 0, "No test data loaded"


def test_confidence_interval():
    n_positive, n_total = 84, 101
    df_data = pd.read_csv(_TEST_DATA_DIR / "example-84-101.csv")
    for t1, t2 in _method_aliases.items():
        if t1 in df_data["method"].values:
            df_data = pd.concat([df_data, df_data[df_data["method"] == t1].assign(method=t2)])
        elif t2 in df_data["method"].values:
            df_data = pd.concat([df_data, df_data[df_data["method"] == t2].assign(method=t1)])
    # assert set(_supported_methods) <= set(
    #     df_data["method"].values
    # ), f"""methods {set(_supported_methods) - set(df_data["method"].values)} has no test data"""
    no_test_data_methods = list(set(_supported_methods) - set(df_data["method"].values))
    # if len(no_test_data_methods) > 0:
    #     warnings.warn(
    #         f"""methods `{no_test_data_methods}` has no test data""",
    #         RuntimeWarning,
    #     )

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
        ), f"for {repr(confint_method)}, lower bound should be {row['lower_bound']:.2%}, but got {lower:.2%}"
        assert upper == approx(
            row["upper_bound"], abs=error_bound
        ), f"for {repr(confint_method)}, upper bound should be {row['upper_bound']:.2%}, but got {upper:.2%}"

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
        ), f"for {repr(confint_method)}, lower bound should be {row['lower_bound']:.2%}, but got {lower:.2%}"
        assert upper == 1, f"for {repr(confint_method)}, upper bound should be {1.0:.2%}, but got {upper:.2%}"

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
        assert lower == 0, f"for {repr(confint_method)}, lower bound should be {0.0:.2%}, but got {lower:.2%}"
        assert upper == approx(
            row["upper_bound"], abs=error_bound
        ), f"for {repr(confint_method)}, upper bound should be {row['upper_bound']:.2%}, but got {upper:.2%}"

    # "witting" is not tested in the above, since no test data is available
    lower, upper = compute_confidence_interval(
        n_positive,
        n_total,
        conf_level=0.975,
        method="witting",
        sides="left",
    ).astuple()
    assert upper == 1, f"for method witting, upper bound should be {1.0:.2%}, but got {upper:.2%}"
    lower, upper = compute_confidence_interval(
        n_positive,
        n_total,
        conf_level=0.975,
        method="witting",
        sides="right",
    ).astuple()
    assert lower == 0, f"for method witting, lower bound should be {0.0:.2%}, but got {lower:.2%}"

    # several edge cases are not covered in the loop
    # TODO add assertions for these edge cases
    for n_positive, n_total in [[0, 1], [1, 1], [2, 2], [2, 3]]:
        compute_confidence_interval(
            n_positive=n_positive,
            n_total=n_total,
            method="pratt",
        )
    for n_positive, n_total in [[0, 1], [1, 1]]:
        compute_confidence_interval(
            n_positive=n_positive,
            n_total=n_total,
            method="mid-p",
        )
    for n_positive, n_total in [[1, 50], [49, 50]]:
        compute_confidence_interval(
            n_positive=n_positive,
            n_total=n_total,
            method="modified-wilson",
        )
    for n_positive, n_total in [[1, 1], [1, 2], [0, 1]]:
        compute_confidence_interval(
            n_positive=n_positive,
            n_total=n_total,
            method="modified-jeffreys",
        )

    print("test_confidence_interval passed")


def test_list_confidence_interval_methods():
    assert list_confidence_interval_methods() is None  # print to stdout

    with warns(DeprecationWarning):
        assert list_confidence_interval_types() is None  # print to stdout


def test_errors():
    with raises(ValueError, match="method should be one of"):
        compute_confidence_interval(1, 2, method="not-supported")
    with raises(ValueError, match="conf_level should be inside the interval \\(0, 1\\)"):
        compute_confidence_interval(1, 2, conf_level=0)
    with raises(ValueError, match="n_positive should be less than or equal to n_total"):
        compute_confidence_interval(2, 1)
    with raises(ValueError, match="n_positive should be non-negative"):
        compute_confidence_interval(-1, 1)
    with raises(ValueError, match="n_total should be positive"):
        compute_confidence_interval(0, 0)
    with raises(ValueError, match="sides should be one of"):
        compute_confidence_interval(1, 2, sides="3-sided")
    with raises(ValueError, match=f"method {repr('not-supported')} is not supported"):
        _compute_confidence_interval(1, 2, confint_type="not-supported")


def test_reported_error_cases():
    """Test the error cases reported to the issue tracker
    to check if the error is still present.
    """
    error_cases = [
        {"n_positive": 2, "n_total": 2, "method": "witting", "conf_level": 0.975},  # issue #5
        {"n_positive": 0, "n_total": 850, "method": "witting", "conf_level": 0.975},  # issue #5
    ]

    for case in error_cases:
        compute_confidence_interval(**case)
