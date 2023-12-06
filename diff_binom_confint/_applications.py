import warnings
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import pandas as pd

from ._binom_confint import compute_confidence_interval
from ._diff_binom_confint import compute_difference_confidence_interval

__all__ = [
    "make_risk_report",
]


def make_risk_report(
    data_source: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]],
    target: str,
    positive_class: Optional[Union[str, int, float]] = None,
    ref_classes: Optional[Sequence[Dict[str, str]]] = None,
    risk_name: Optional[str] = None,
    conf_level: float = 0.95,
    method: str = "wilson",
    diff_method: str = "wilson",
    dropna: bool = True,
    save_path: Optional[Union[Path, str]] = None,
    return_type: str = "pd",
    **kwargs,
) -> Union[pd.DataFrame, dict, str]:
    """Make risk report for binomial confidence intervals.

    Parameters
    ----------
    data_source : pandas.DataFrame or tuple of two pandas.DataFrame
        Data source table.
        Each column should be categorical (including binary).
        Numerical columns should be discretized by the users themselves
        before passing to this function.
        If is a tuple of two :class:`~pandas.DataFrame` s, the two tables are
        train/validation tables, respectively.
    target : str
        Target column name.
    positive_class : str, int or float
        Positive class. If is None, non-null value (making if statement True)
        will be considered as positive class.
    ref_classes : list, optional
        Reference classes (for difference computation). If is None,
        reference classes will be chosen as the largest classes for each column.
    risk_name : str, optional
        Risk name. If is None, the risk name will be given by the positive class name
        and the target column name.
    conf_level : float, default 0.95
        Confidence level, should be inside the interval ``(0, 1)``.
    method : str, default "wilson"
        Type (computation method) of the confidence interval.
        For a full list of the available methods, see
        :func:`diff_binom_confint.list_confidence_interval_methods`.
    diff_method : str, default "wilson"
        Type (computation method) of the confidence interval of the difference.
        For a full list of the available methods, see
        :func:`diff_binom_confint.list_difference_confidence_interval_methods`.
    dropna: bool, default True
        Whether to drop missing values (column-wise).
        A better way is that the users deal with missing values themselves.
    save_path : str or pathlib.Path, optional
        Path to save the report table.
        If is None, the report table will not be saved.
    return_type : {"pd", "dict", "latex", "md", "markdown", "html"}, default "pd"
        The type of the returned report table.
        - "pd": pandas.DataFrame
        - "dict": dict
        - "latex": LaTeX table
        - "md" or "markdown": Markdown table
        - "html": HTML table
    **kwargs: dict, optional
        Other parameters passed to
        :func:`diff_binom_confint.compute_confidence_interval` and
        :func:`diff_binom_confint.compute_difference_confidence_interval`.

    Returns
    -------
    Union[pandas.DataFrame, dict, str]
        Report table.

    """
    if isinstance(data_source, pd.DataFrame):
        df = data_source.copy()
        is_split = False
    else:
        df_train, df_val = data_source
        df = pd.concat(data_source, ignore_index=True)
        is_split = True
    # fillna with "NA"
    df = df.fillna("NA")

    # check target, it should be in the columns and be binary
    if target not in df.columns:
        raise ValueError(f"target {repr(target)} not in the columns")
    if len(df[target].unique()) != 2:
        raise ValueError(f"target {repr(target)} is not binary")

    # convert all columns other than target to str type
    for col in df.columns:
        if col != target:
            df[col] = df[col].astype(str)
    if is_split:
        for col in df_train.columns:
            if col != target:
                df_train[col] = df_train[col].astype(str)
        for col in df_val.columns:
            if col != target:
                df_val[col] = df_val[col].astype(str)

    # check positive_class, it should be in df[target].unique()
    if positive_class is None:
        positive_class = [item for item in df[target].unique() if bool(item)]
        if len(positive_class) != 1:
            raise ValueError("Unable to automatically determine the positive class, please specify it manually.")
        positive_class = positive_class[0]
        warnings.warn(
            f"positive_class is None, automatically set to {repr(positive_class)}",
            RuntimeWarning,
        )
    if positive_class not in df[target].unique():
        raise ValueError(f"positive_class {repr(positive_class)} not in the target column")

    features = df.columns.drop(target)

    # check ref_classes
    default_ref_classes = {}
    for feature in features:
        default_ref_classes[feature] = df[feature].value_counts().index[0]
    if ref_classes is None:
        ref_classes = default_ref_classes
    else:
        _ref_classes = default_ref_classes.copy()
        _ref_classes.update(ref_classes)
        ref_classes = _ref_classes.copy()
        del _ref_classes
    assert set(ref_classes) <= set(features), "ref_classes should be a subset of the features"
    for feature, ref_cls in ref_classes.items():
        if ref_cls not in df[feature].unique():
            raise ValueError(f"ref class {repr(ref_cls)} not in the feature {repr(feature)}")
    ref_indicator = " (Ref.)"

    risk_name = risk_name or f"{positive_class} {target}"

    rows = []
    ret_dict = {}

    # row 1 - 2
    rows.extend(
        [
            [
                "Feature",
                "",
                "Affected",
                "",
                f"{risk_name} Risk ({str(int(conf_level * 100))}% CI)",
                "",
                f"{risk_name} Risk Difference ({str(int(conf_level * 100))}% CI)",
            ],
            ["", "", "n", "%", "n", "%", ""],
        ]
    )
    if is_split:
        rows[0].insert(4, "")
        rows[1].insert(4, "t/v")

    # row 3: overall statitics
    n_positive = df[df[target] == positive_class].shape[0]
    rows.append(
        [
            "Total",
            "",
            f"{len(df)}",
            "100%",
            f"{n_positive}",
            f"{n_positive / len(df):.1%}",
            "-",
        ]
    )
    if is_split:
        rows[-1].insert(4, f"{len(df_train)}/{len(df_val)}")

    feature_classes = {col: sorted(df[col].unique().tolist()) for col in features}
    # put ref item at the beginning
    for col in features:
        ref_item = ref_classes[col]
        feature_classes[col].remove(ref_item)
        feature_classes[col].insert(0, ref_item)

    for col in features:
        n_affected = {item: df[df[col] == item].shape[0] for item in feature_classes[col]}
        n_positive = {item: df[(df[col] == item) & (df[target] == positive_class)].shape[0] for item in feature_classes[col]}
        positive_target_risk = {}
        ref_item = ref_classes[col]
        for item in feature_classes[col]:
            positive_target_risk[item] = {
                "risk": n_positive[item] / n_affected[item],
                "confidence_interval": compute_confidence_interval(
                    n_positive[item], n_affected[item], conf_level, method
                ).astuple(),
            }
        positive_target_risk_diff = {}
        for item in feature_classes[col]:
            if item == ref_item:
                positive_target_risk_diff[f"{item} (Ref.)"] = {
                    "risk_difference": 0,
                    "confidence_interval": (0, 0),
                }
                continue
            positive_target_risk_diff[item] = {
                "risk_difference": positive_target_risk[item]["risk"] - positive_target_risk[ref_item]["risk"],
                "confidence_interval": compute_difference_confidence_interval(
                    n_positive[item],
                    n_affected[item],
                    n_positive[ref_item],
                    n_affected[ref_item],
                    conf_level,
                    method,
                    **kwargs,
                ).astuple(),
            }

        rows.append([col, "", "", "", "", "", ""])
        if is_split:
            rows[-1].insert(4, "")
        ret_dict[col] = {}
        for item in feature_classes[col]:
            if dropna and item == "NA":
                continue
            rows.append(
                [
                    "",
                    item,
                    f"{n_affected[item]}",
                    f"{n_affected[item] / len(df):.1%}",
                    f"{n_positive[item]}",
                    f"{positive_target_risk[item]['risk']:.1%} (from {positive_target_risk[item]['confidence_interval'][0]:.1%} to {positive_target_risk[item]['confidence_interval'][1]:.1%})",
                    f"{positive_target_risk_diff[item]['risk_difference']:.1%} (from {positive_target_risk_diff[item]['confidence_interval'][0]:.1%} to {positive_target_risk_diff[item]['confidence_interval'][1]:.1%})"
                    if item != ref_item
                    else "REF",
                ]
            )
            key = str(item) + (ref_indicator if item == ref_item else "")
            ret_dict[col][key] = {
                "Affected": {
                    "n": n_affected[item],
                    "percent": n_affected[item] / len(df),
                },
                f"{risk_name} Risk": {
                    "n": n_positive[item],
                    "percent": positive_target_risk[item]["risk"],
                    "confidence_interval": positive_target_risk[item]["confidence_interval"],
                },
                f"{risk_name} Risk Difference": {
                    "risk_difference": positive_target_risk_diff[item]["risk_difference"] if item != ref_item else 0,
                    "confidence_interval": positive_target_risk_diff[item]["confidence_interval"]
                    if item != ref_item
                    else (0, 0),
                },
            }
            if is_split:
                train_affected = df_train[df_train[col] == item].shape[0]
                train_positive = df_train[(df_train[col] == item) & (df_train[target] == positive_class)].shape[0]
                rows[-1].insert(4, f"{train_affected}/{train_positive}")
                ret_dict[col][key]["Affected"]["t/v"] = f"{train_affected}/{train_positive}"

    df_risk_table = pd.DataFrame(rows)

    if save_path is not None:
        save_path = Path(save_path)

    if save_path is not None:
        df_risk_table.to_csv(save_path.with_suffix(".csv"), index=False, header=False)
        df_risk_table.to_excel(save_path.with_suffix(".xlsx"), index=False, header=False)

    if return_type.lower() == "pd":
        return df_risk_table
    elif return_type.lower() == "latex":
        rows = [line.replace("%", r"\%") for line in df_risk_table.to_latex(header=False, index=False).splitlines()]
        rows[0] = r"\begin{tabular}{@{\extracolsep{6pt}}lllllll@{}}"
        rows[
            2
        ] = r"\multicolumn{2}{l}{Feature} & \multicolumn{affected_cols}{l}{Affected} & \multicolumn{2}{l}{risk_name Risk ($95\%$ CI)} & risk_name Risk Difference  ($95\%$ CI) \\ \cline{1-2}\cline{3-4}\cline{5-6}\cline{7-7}"
        rows[2].replace("risk_name", risk_name).replace("95", str(int(conf_level * 100)))
        if is_split:
            rows[2].replace("affected_cols", "3")
        else:
            rows[2].replace("affected_cols", "2")
        ret_lines = "\n".join(rows)
        if save_path is not None:
            save_path.with_suffix(".tex").write_text(ret_lines)
        return ret_lines
    elif return_type.lower() in ["md", "markdown"]:
        return df_risk_table.to_markdown(index=False)
    elif return_type.lower() == "html":
        return df_risk_table.to_html(index=False)
    elif return_type.lower() == "dict":
        return ret_dict
