"""
"""

import re
from pathlib import Path
from typing import List

import pandas as pd
from pytest import approx

try:
    from diff_binom_confint import compute_confidence_interval
    from diff_binom_confint._binom_confint import (
        _supported_types,
        _type_aliases,
        _stochastic_types,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from diff_binom_confint import compute_confidence_interval
    from diff_binom_confint._binom_confint import (
        _supported_types,
        _type_aliases,
        _stochastic_types,
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
            for t1, t2 in _type_aliases.items():
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
    for t1, t2 in _type_aliases.items():
        if t1 in df_data["method"].values:
            df_data = pd.concat(
                [df_data, df_data[df_data["method"] == t1].assign(method=t2)]
            )
        elif t2 in df_data["method"].values:
            df_data = pd.concat(
                [df_data, df_data[df_data["method"] == t2].assign(method=t1)]
            )

    max_length = max([len(x) for x in _supported_types]) + 1
    error_bound = 1e-4

    for confint_type in _supported_types:
        if confint_type in _stochastic_types:
            continue
        lower, upper = compute_confidence_interval(
            n_positive,
            n_total,
            confint_type=confint_type,
        ).astuple()
        print(f"{confint_type.ljust(max_length)}: [{lower:.2%}, {upper:.2%}]")
        row = df_data[df_data["method"] == confint_type].iloc[0]
        assert lower == approx(
            row["lower_bound"], abs=error_bound
        ), f"for {confint_type}, lower bound should be {row['lower_bound']:.2%}, but got {lower:.2%}"
        assert upper == approx(
            row["upper_bound"], abs=error_bound
        ), f"for {confint_type}, upper bound should be {row['upper_bound']:.2%}, but got {upper:.2%}"

    print("test_confidence_interval passed")


if __name__ == "__main__":
    load_test_data()
    test_confidence_interval()
