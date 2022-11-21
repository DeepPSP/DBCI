"""
"""

import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pytest import approx, raises

from diff_binom_confint import compute_difference_confidence_interval
from diff_binom_confint._diff_binom_confint import (
    _supported_methods,
    _method_aliases,
    _stochastic_methods,
)
from diff_binom_confint._confint import ConfidenceInterval


_TEST_DATA_DIR = Path(__file__).parent / "test-data"


def test_load_data() -> List[pd.DataFrame]:
    test_file_pattern = "example-(?P<n_positive>[\\d]+)-(?P<n_total>[\\d]+)-vs-(?P<ref_positive>[\\d]+)-(?P<ref_total>[\\d]+)\\.csv"
    test_files = Path(_TEST_DATA_DIR).glob("*.csv")
    test_data = []
    for file in test_files:
        match = re.match(test_file_pattern, file.name)
        if match:
            n_positive = int(match.group("n_positive"))
            n_total = int(match.group("n_total"))
            ref_positive = int(match.group("ref_positive"))
            ref_total = int(match.group("ref_total"))
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
                    "ref_positive": ref_positive,
                    "ref_total": ref_total,
                    "data": df_data,
                }
            )
    print(f"Totally {len(test_data)} test data loaded")
    return test_data


def load_newcombee_data() -> dict:
    """ """
    df_newcombee_data = pd.read_csv(
        _TEST_DATA_DIR / "newcombee.csv", index_col=0, header=[0, 1]
    )
    examples = defaultdict(dict)
    for k, v in df_newcombee_data.to_dict().items():
        example, bound = k
        for method, value in v.items():
            if method in examples[example]:
                examples[example][method][bound] = value
            else:
                examples[example][method] = {bound: value}
    examples = dict(examples)
    example_name_pattern = "(?P<n_positive>[\\d]+)/(?P<n_total>[\\d]+)-(?P<ref_positive>[\\d]+)/(?P<ref_total>[\\d]+)"

    for k, v in examples.items():
        matched = re.match(example_name_pattern, k)
        ratio_diff = int(matched["n_positive"]) / int(matched["n_total"]) - int(
            matched["ref_positive"]
        ) / int(matched["ref_total"])
        examples[k] = {
            method: ConfidenceInterval(
                bounds["lower_bound"],
                bounds["upper_bound"],
                ratio_diff,
                0.95,
                method,
            )
            for method, bounds in v.items()
        }
    return examples


def test_newcombee_data():
    """ """
    error_bound = 1e-4
    newcombee_data = load_newcombee_data()
    example_name_pattern = "(?P<n_positive>[\\d]+)/(?P<n_total>[\\d]+)-(?P<ref_positive>[\\d]+)/(?P<ref_total>[\\d]+)"
    for k, v in newcombee_data.items():
        print(f"Testing Newcombee data, example -- {k}")
        max_length = max([len(x) for x in v]) + 1
        matched = re.match(example_name_pattern, k)
        n_positive = int(matched["n_positive"])
        n_total = int(matched["n_total"])
        # n_negative = n_total - n_positive
        ref_positive = int(matched["ref_positive"])
        ref_total = int(matched["ref_total"])
        # ref_negative = ref_total - ref_positive
        for method, confint in v.items():
            if method not in _supported_methods or method in _stochastic_methods:
                continue
            lower, upper = compute_difference_confidence_interval(
                n_positive, n_total, ref_positive, ref_total, method=method, clip=False
            ).astuple()
            print(f"{method.ljust(max_length)}: [{lower:.2%}, {upper:.2%}]")
            if np.isnan(confint.lower_bound):
                assert (
                    lower < -1
                ), f"For method {method}, unclipped lower bound should be < -1, but got {lower}"
            else:
                assert lower == approx(
                    confint.lower_bound, abs=error_bound
                ), f"For method {method}, lower bound should be {confint.lower_bound}, but got {lower}"
            if np.isnan(confint.upper_bound):
                assert (
                    upper > 1
                ), f"For method {method}, unclipped upper bound should be > 1, but got {upper}"
            else:
                assert upper == approx(
                    confint.upper_bound, abs=error_bound
                ), f"For method {method}, upper bound should be {confint.upper_bound}, but got {upper}"


def test_difference_confidence_interval():
    """
    A CASE STUDY: EXAMPLE FROM AN HIV CLINICAL TRIAL

    Consider Week 48 results of the PROGRESS trial as presented by Reynes (2010).
    There, the proportion of patients who were responders on a new treatment was 84/101 (83.2%)
    while the proportion of patients who were responders on the standard of care was 89/105 (84.8%).
    Non-inferiority (NI) testing was to be performed by constructing a confidence interval for the difference in proportions.
    If the lower bound of the confidence interval is above a pre-defined margin (eg, -12%), then NI can be concluded.
    The table below shows what the confidence intervals from the various methods above (excluding Methods 8 and 9) would be.

    Reference
    ---------
    Reynes JLA, Pulido F, et al., “Lopinavir/ritonavir combined with raltegravir demonstrated similar antiviral efficacy and safety as lopinavir/ritonavir combined with tenofovir disoproxil fumarate/emtricitabine in treatment-naïve HIV-1 infected subjects.” Program and abstracts of the XVIII International AIDS Conference; July 18-23, 2010; Vienna, Austria. Abstract MOAB0101.

    """
    n_positive, n_total = 84, 101
    ref_positive, ref_total = 89, 105
    df_data = pd.read_csv(_TEST_DATA_DIR / "example-84-101-vs-89-105.csv")
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
            f"""methods {no_test_data_methods} has no test data""",
            RuntimeWarning,
        )

    max_length = max([len(x) for x in _supported_methods]) + 1
    error_bound = 1e-4

    print("Testing 2-sided confidence interval")
    for confint_method in _supported_methods:
        if confint_method in _stochastic_methods + no_test_data_methods:
            continue
        lower, upper = compute_difference_confidence_interval(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            method=confint_method,
        ).astuple()
        print(f"{confint_method.ljust(max_length)}: [{lower:.2%}, {upper:.2%}]")
        row = df_data[df_data["method"] == confint_method].iloc[0]
        assert lower == approx(
            row["lower_bound"], abs=error_bound
        ), f"For method {confint_method}, lower bound should be {row['lower_bound']:.2%}, but got {lower:.2%}"
        assert upper == approx(
            row["upper_bound"], abs=error_bound
        ), f"For method {confint_method}, upper bound should be {row['upper_bound']:.2%}, but got {upper:.2%}"

    print("Testing left-sided confidence interval")
    for confint_method in _supported_methods:
        if confint_method in _stochastic_methods + no_test_data_methods:
            continue
        lower, upper = compute_difference_confidence_interval(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            conf_level=0.975,
            method=confint_method,
            sides="left",
        ).astuple()
        print(f"{confint_method.ljust(max_length)}: [{lower:.2%}, {upper:.2%}]")
        row = df_data[df_data["method"] == confint_method].iloc[0]
        assert lower == approx(
            row["lower_bound"], abs=error_bound
        ), f"For method {confint_method}, lower bound should be {row['lower_bound']:.2%}, but got {lower:.2%}"
        assert (
            upper == 1
        ), f"For method {confint_method}, upper bound should be {1.0:.2%}, but got {upper:.2%}"

    print("Testing right-sided confidence interval")
    for confint_method in _supported_methods:
        if confint_method in _stochastic_methods + no_test_data_methods:
            continue
        lower, upper = compute_difference_confidence_interval(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            conf_level=0.975,
            method=confint_method,
            sides="right",
        ).astuple()
        print(f"{confint_method.ljust(max_length)}: [{lower:.2%}, {upper:.2%}]")
        row = df_data[df_data["method"] == confint_method].iloc[0]
        assert (
            lower == -1
        ), f"For method {confint_method}, lower bound should be {-1.0:.2%}, but got {lower:.2%}"
        assert upper == approx(
            row["upper_bound"], abs=error_bound
        ), f"For method {confint_method}, upper bound should be {row['upper_bound']:.2%}, but got {upper:.2%}"

    print("test_difference_confidence_interval passed")


def test_difference_confidence_interval_edge_case():
    """ """
    n_positive, n_total = 10, 10
    ref_positive, ref_total = 0, 20
    df_data = pd.read_csv(_TEST_DATA_DIR / "example-10-10-vs-0-20.csv")
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
            f"""methods {no_test_data_methods} has no test data""",
            RuntimeWarning,
        )

    max_length = max([len(x) for x in _supported_methods]) + 1 + len("[clipped] ")
    error_bound = 1e-4

    for confint_method in _supported_methods:
        if confint_method in _stochastic_methods + no_test_data_methods:
            continue
        lower_clip, upper_clip = compute_difference_confidence_interval(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            method=confint_method,
            clip=True,
        ).astuple()
        l_just_len = max_length - len("[clipped] ")
        print(
            f"[clipped] {confint_method.ljust(l_just_len)}: [{lower_clip:.2%}, {upper_clip:.2%}]"
        )

        lower_noclip, upper_noclip = compute_difference_confidence_interval(
            n_positive,
            n_total,
            ref_positive,
            ref_total,
            method=confint_method,
            clip=False,
        ).astuple()
        l_just_len = max_length
        print(
            f"{confint_method.ljust(l_just_len)}: [{lower_noclip:.2%}, {upper_noclip:.2%}]"
        )

        row = df_data[df_data["method"] == confint_method].iloc[0]
        assert (
            lower_clip == lower_noclip == approx(row["lower_bound"], abs=error_bound)
        ), f"For method {confint_method}, lower bound should be {row['lower_bound']:.2%}, but got {lower_clip:.2%}"
        if np.isnan(row["upper_bound"]):
            assert (
                upper_noclip > 1
            ), f"For method {confint_method}, non-clipped upper bound should be > 1, but got {upper_noclip:.2%}"
            assert upper_clip == approx(
                1
            ), f"For method {confint_method}, clipped upper bound should be 1, but got {upper_clip:.2%}"
        else:
            assert (
                upper_noclip
                == upper_clip
                == approx(row["upper_bound"], abs=error_bound)
            ), f"For method {confint_method}, upper bound should be {row['upper_bound']:.2%}, but got {upper_noclip:.2%}"

    print("test_difference_confidence_interval_edge_case passed")


def test_errors():
    with raises(ValueError, match="confint_type should be one of"):
        compute_difference_confidence_interval(1, 2, 1, 2, method="not-supported")
    with raises(
        ValueError, match="conf_level should be inside the interval \\(0, 1\\)"
    ):
        compute_difference_confidence_interval(1, 2, 1, 2, conf_level=0)
    with raises(ValueError, match="n_positive should be less than or equal to n_total"):
        compute_difference_confidence_interval(2, 1, 1, 2)
    with raises(
        ValueError, match="ref_positive should be less than or equal to ref_total"
    ):
        compute_difference_confidence_interval(1, 2, 2, 1)
    with raises(ValueError, match="n_positive should be non-negative"):
        compute_difference_confidence_interval(-1, 2, 1, 2)
    with raises(ValueError, match="n_total should be positive"):
        compute_difference_confidence_interval(0, 0, 1, 2)
    with raises(ValueError, match="ref_positive should be non-negative"):
        compute_difference_confidence_interval(1, 2, -1, 2)
    with raises(ValueError, match="ref_total should be positive"):
        compute_difference_confidence_interval(1, 2, 0, 0)
    with raises(ValueError, match="sides should be one of"):
        compute_difference_confidence_interval(1, 2, 1, 2, sides="3-sided")
