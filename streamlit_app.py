import streamlit as st

from diff_binom_confint import __version__, compute_confidence_interval, compute_difference_confidence_interval
from diff_binom_confint._binom_confint import _supported_methods
from diff_binom_confint._diff_binom_confint import _supported_methods as _supported_diff_methods

st.set_page_config(
    page_title="DBCI",
    page_icon=":books:",
    layout="centered",
)


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


# input on the main page

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

# command to run:
# nohup streamlit run streamlit_app.py --server.port 8502 > .logs/streamlit_app.log 2>&1 & echo $! > .logs/streamlit_app.pid
