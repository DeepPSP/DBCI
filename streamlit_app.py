from pathlib import Path

import pandas as pd
import streamlit as st

from diff_binom_confint import (
    __version__,
    compute_confidence_interval,
    compute_difference_confidence_interval,
    make_risk_report,
)
from diff_binom_confint._binom_confint import _supported_methods
from diff_binom_confint._diff_binom_confint import _supported_methods as _supported_diff_methods

st.set_page_config(
    page_title="DBCI",
    page_icon=":books:",
    layout="centered",
)


@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode("utf-8")


st.title("Binomial Confidence Intervals")


# show the version number on the sidebar

st.sidebar.markdown(f"version: {__version__}")

# configurations on the sidebar

st.sidebar.title("Configurations")
# select the method to compute the confidence interval
method = st.sidebar.selectbox(
    label="Method",
    options=_supported_methods,
    index=0,
    key="method",
)
diff_method = st.sidebar.selectbox(
    label="Difference Method",
    options=_supported_diff_methods,
    index=0,
    key="diff_method",
)
# set confidence level
conf_level = st.sidebar.slider(
    label="Confidence Level",
    min_value=0.5,
    max_value=0.999,
    value=0.95,
    step=0.001,
    key="conf_level",
)
# select left- or right- or two-sided
sides = st.sidebar.selectbox(
    label="sides",
    options=["left", "right", "two"],
    index=2,
    key="sides",
)
# toggle to clip or not
clip = st.sidebar.toggle(
    label="Clip the confidence interval to [0, 1]",
    value=True,
    key="clip",
)

tab_compute, tab_report = st.tabs(["🖩 Compute", "📋 Report"])

with tab_compute:
    # input on the tab_compute page

    st.header("Input")

    # input the number of trials
    n_total = st.number_input(
        label="Number of trials",
        min_value=1,
        max_value=None,
        value=10,
        step=1,
        key="n_total",
    )
    # input the number of positives
    n_positive = st.number_input(
        label="Number of positives",
        min_value=0,
        max_value=n_total,
        value="min",
        step=1,
        key="n_positive",
    )
    # input the number of trials for the reference group
    n_total_ref = st.number_input(
        label="Number of trials for the reference group",
        min_value=1,
        max_value=None,
        value=10,
        step=1,
        key="n_total_ref",
    )
    # input the number of positives for the reference group
    n_positive_ref = st.number_input(
        label="Number of positives for the reference group",
        min_value=0,
        max_value=n_total_ref,
        value="min",
        step=1,
        key="n_positive_ref",
    )

    # compute the confidence interval
    button = st.button(label="Compute CI", key="button")
    diff_button = st.button(label="Compute Difference CI", key="diff_button")

    if button:
        st.header("Output")
        st.subheader("Confidence Interval")
        st.markdown(
            compute_confidence_interval(
                n_total=n_total,
                n_positive=n_positive,
                method=method,
                conf_level=conf_level,
                sides=sides,
                clip=clip,
            ).astable(to="markdown")
        )
    if diff_button:
        st.header("Output")
        st.subheader("Difference Confidence Interval")
        st.markdown(
            compute_difference_confidence_interval(
                n_total=n_total,
                n_positive=n_positive,
                ref_total=n_total_ref,
                ref_positive=n_positive_ref,
                method=diff_method,
                conf_level=conf_level,
                sides=sides,
                clip=clip,
            ).astable(to="markdown")
        )

with tab_report:
    st.header("Risk Report")
    st.markdown(
        "The risk report is a summary table of the confidence intervals. "
        "The user should convert each continuous column to a categorical column themselves in the uploaded data."
    )

    # upload the data
    uploaded_file = st.file_uploader(
        label="Upload the data",
        type=["csv", "xlsx", "xls"],
        key="uploaded_file",
        accept_multiple_files=False,
    )

    target = st.text_input(
        label="Target column name (required)",
        value="",
        max_chars=100,
        key="target",
    )
    positive_class = st.text_input(
        label="Positive class (optional)",
        value="",
        max_chars=100,
        key="positive_class",
    )
    ref_classes = st.text_input(
        # TODO: replace with tags input
        label="Reference classes, each of the form `feature:ref_class`, separated by comma (optional)",
        value="",
        max_chars=100,
        key="ref_classes",
    )
    risk_name = st.text_input(
        label="Risk name (optional)",
        value="",
        max_chars=100,
        key="risk_name",
    )

    make_report_button = st.button(label="Make Report", key="make_report_button")

    if make_report_button:
        if uploaded_file is None:
            st.error("Please upload a file.")
            st.stop()
        if target == "":
            st.error("Please input the target column name.")
            st.stop()
        if positive_class == "":
            positive_class = None
        if ref_classes == "":
            ref_classes = None
        else:
            # convert to dict
            ref_classes = dict([x.strip().split(":") for x in ref_classes.split(",")])
        if risk_name == "":
            risk_name = None
        # read the data
        if Path(uploaded_file.name).suffix == ".xlsx":
            df_data = pd.read_excel(uploaded_file)
        else:
            df_data = pd.read_csv(uploaded_file)
        # make the risk report
        try:
            report_table = make_risk_report(
                data_source=df_data,
                target=target,
                positive_class=positive_class,
                ref_classes=ref_classes,
                risk_name=risk_name,
                method=method,
                diff_method=diff_method,
                conf_level=conf_level,
                sides=sides,
                clip=clip,
            )
        except Exception as e:
            st.error(e)
            st.stop()
        # show the report
        st.table(report_table)

        # provide the download link
        csv = convert_df(report_table)
        st.download_button(
            label="Download the report as CSV",
            data=csv,
            file_name="risk_report.csv",
            mime="text/csv",
        )

# command to run:
# nohup streamlit run streamlit_app.py --server.port 8502 > .logs/streamlit_app.log 2>&1 & echo $! > .logs/streamlit_app.pid
