import itertools
import shutil
from pathlib import Path

import pandas as pd
import pytest

from diff_binom_confint import make_risk_report

_TEST_DATA_DIR = Path(__file__).parent / "test-data"
_TMP_DIR = Path(__file__).parent / "tmp-make-risk-report"


def test_make_risk_report():
    _TMP_DIR.mkdir(exist_ok=True)
    df_train = pd.read_csv(_TEST_DATA_DIR / "make-risk-report-train.csv", index_col=0)
    df_val = pd.read_csv(_TEST_DATA_DIR / "make-risk-report-val.csv", index_col=0)
    df_test = pd.concat((df_train, df_val), ignore_index=True)

    ref_classes = {
        "Sex": "Male",
        "ExtentOfResection": "全切",
        "WHOGrading": "WHOIV",
    }

    grid = itertools.product(
        [df_test, (df_train, df_val)],  # data_source
        [ref_classes, None],  # ref_classes
        ["Seizure", None],  # risk_name
        ["pd", "dict", "latex", "md", "markdown", "html"],  # return_type
        [str(_TMP_DIR / "risk-report"), None],  # save_path
    )

    for data_source, ref_classes, risk_name, return_type, save_path in grid:
        report = make_risk_report(
            data_source=data_source,
            ref_classes=ref_classes,
            risk_name=risk_name,
            return_type=return_type,
            save_path=save_path,
            target="HasSeizure",
            positive_class="Yes",
        )

        if return_type == "pd":
            assert isinstance(report, pd.DataFrame)
        elif return_type == "dict":
            assert isinstance(report, dict)
        elif return_type in ("latex", "md", "markdown", "html"):
            assert isinstance(report, str)

    with pytest.raises(ValueError, match=f"target {repr('xxx')} not in the columns"):
        make_risk_report(
            data_source=df_test,
            risk_name="Seizure",
            target="xxx",
            positive_class="Yes",
        )
    with pytest.raises(ValueError, match=f"target {repr('WHOGrading')} is not binary"):
        make_risk_report(
            data_source=df_test,
            risk_name="Seizure",
            target="WHOGrading",
            positive_class="WHOIV",
        )
    with pytest.raises(ValueError, match="Unable to automatically determine the positive class"):
        make_risk_report(
            data_source=df_test,
            risk_name="Seizure",
            target="HasSeizure",
        )
    with pytest.raises(ValueError, match=f"positive_class {repr('是')} not in the target column"):
        make_risk_report(
            data_source=df_test,
            risk_name="Seizure",
            target="HasSeizure",
            positive_class="是",
        )
    with pytest.raises(AssertionError, match="ref_classes should be a subset of the features"):
        make_risk_report(
            data_source=df_test,
            risk_name="Seizure",
            target="HasSeizure",
            positive_class="Yes",
            ref_classes={
                "Sex": "Male",
                "xxx": "全切",
            },
        )
    with pytest.raises(ValueError, match=f"ref class {repr('xxx')} not in the feature {repr('WHOGrading')}"):
        make_risk_report(
            data_source=df_test,
            risk_name="Seizure",
            target="HasSeizure",
            positive_class="Yes",
            ref_classes={
                "Sex": "Male",
                "ExtentOfResection": "全切",
                "WHOGrading": "xxx",
            },
        )

    df_test["HasSeizure"] = df_test["HasSeizure"].map({"Yes": 1, "No": 0})
    with pytest.warns(RuntimeWarning, match=f"positive_class is None, automatically set to {repr(1)}"):
        make_risk_report(
            data_source=df_test,
            risk_name="Seizure",
            target="HasSeizure",
        )

    shutil.rmtree(_TMP_DIR)
