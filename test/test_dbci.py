"""
"""

import re
from pathlib import Path
from typing import List

import pandas as pd
from pytest import approx

try:
    from diff_binom_confint import compute_difference_confidence_interval
    from diff_binom_confint._diff_binom_confint import (
        _supported_methods,
        _method_aliases,
        _stochastic_methods,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from diff_binom_confint import compute_difference_confidence_interval
    from diff_binom_confint._diff_binom_confint import (
        _supported_methods,
        _method_aliases,
        _stochastic_methods,
    )


_TEST_DATA_DIR = Path(__file__).parent / "test-data"


def load_test_data() -> List[pd.DataFrame]:
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

    max_length = max([len(x) for x in _supported_methods]) + 1
    error_bound = 1e-4

    for confint_method in _supported_methods:
        if confint_method in _stochastic_methods:
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
        ), f"for {confint_method}, lower bound should be {row['lower_bound']:.2%}, but got {lower:.2%}"
        assert upper == approx(
            row["upper_bound"], abs=error_bound
        ), f"for {confint_method}, upper bound should be {row['upper_bound']:.2%}, but got {upper:.2%}"

    print("test_difference_confidence_interval passed")


if __name__ == "__main__":
    load_test_data()
    test_difference_confidence_interval()
