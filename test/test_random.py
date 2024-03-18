"""
"""

import os

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

artifact_dir = os.environ.get("ARTIFACT_DIR", None)
random_data_amount = int(os.environ.get("RANDOM_DATA_AMOUNT", 10000))
random_data_range = int(os.environ.get("RANDOM_DATA_RANGE", 1000000))


def test_random():
    # test errors would not be raised
    # when `compute_confidence_interval` is
    # tested with a large amount of randomly generated data
    from diff_binom_confint import compute_confidence_interval
    from diff_binom_confint._binom_confint import _supported_methods

    rng = np.random.default_rng()
    err_quadruples = []
    with tqdm(range(random_data_amount), mininterval=1.0, total=random_data_amount) as pbar:
        for _ in pbar:
            n_total = rng.integers(1, random_data_range)
            n_positive = rng.integers(0, n_total + 1)
            for method in _supported_methods:
                try:
                    compute_confidence_interval(
                        n_positive,
                        n_total,
                        conf_level=0.975,
                        method=method,
                    )
                except Exception as e:
                    err_quadruples.append((n_positive, n_total, method, e.__class__.__name__))
                    pbar.set_postfix_str(f"err count: {len(err_quadruples)}")

    if artifact_dir is not None and len(err_quadruples) > 0:
        df_err = pd.DataFrame(err_quadruples, columns=["n_total", "n_positive", "method", "error"])
        df_err.to_csv(os.path.join(artifact_dir, "bci_random_data_errors.csv"), index=False)
    elif len(err_quadruples) > 0:
        raise ValueError(f"Errors occurred: {err_quadruples}")


def test_dbci_random():
    # test errors would not be raised
    # when `compute_difference_confidence_interval` is
    # tested with a large amount of randomly generated data
    from diff_binom_confint import compute_difference_confidence_interval
    from diff_binom_confint._diff_binom_confint import _supported_methods

    np.random.default_rng()
    errors = []
    with tqdm(range(random_data_amount), mininterval=1.0, total=random_data_amount) as pbar:
        for _ in pbar:
            n_total = np.random.randint(1, random_data_range)
            n_positive = np.random.randint(0, n_total + 1)
            ref_total = np.random.randint(1, random_data_range)
            ref_positive = np.random.randint(0, ref_total + 1)
            for method in _supported_methods:
                try:
                    compute_difference_confidence_interval(n_positive, n_total, ref_positive, ref_total, method=method)
                except Exception as e:
                    errors.append(
                        (
                            n_positive,
                            n_total,
                            ref_positive,
                            ref_total,
                            method,
                            e.__class__.__name__,
                        )
                    )
                    pbar.set_postfix_str(f"err count: {len(errors)}")

    if artifact_dir is not None and len(errors) > 0:
        df_errors = pd.DataFrame(
            errors,
            columns=[
                "n_positive",
                "n_total",
                "ref_positive",
                "ref_total",
                "method",
                "error",
            ],
        )
        df_errors.to_csv(os.path.join(artifact_dir, "dbci_random_data_errors.csv"), index=False)
    elif len(errors) > 0:
        raise ValueError(f"Errors occurred: {errors}")
