# Home.py
"""Landing page for the **Improving‑Your‑Statistical‑Inferences** Streamlit app.

Use the sidebar’s page selector or the quick‑links below to jump directly to a tool:

1. **P‑Value Simulation** – sample one‑sample *t*‑tests and watch the p‑value distribution & empirical power.
2. **Power & Errors** – visualise Type I & II error regions, power, and Cohen’s *d* for a one‑sample *Z*‑test.
3. **Likelihood Curve** – see the full binomial likelihood across θ ∈ [0,1] for any data (*n*, *x*).
4. **Likelihood Ratio** – compare two point hypotheses with a live likelihood‑ratio plot.
"""

import streamlit as st

st.set_page_config(
    page_title="Improving Statistical Inferences",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Improving Your Statistical Inferences")

st.markdown(
    """
Explore a collection of interactive statistical visualisations designed to build
intuition about hypothesis testing, power, and likelihoods.

### 🔍 Quick‑links
    """,
    unsafe_allow_html=True,
)

# Quick links (Streamlit ≥ 1.29)
st.page_link("pages/1_P_Value_Simulation.py", label="P‑Value Simulation", icon="📊")
st.page_link("pages/2_Power_Curve.py", label="Power & Errors", icon="⚡")
st.page_link("pages/3_Likelihood_Curve.py", label="Likelihood Curve", icon="📈")
st.page_link("pages/4_Likelihood_Ratio.py", label="Likelihood Ratio", icon="🔀")
st.page_link("pages/5_Bayes_Factor.py", label="Binomial Bayes‑Factor Calculator", icon="📈")

st.divider()

st.markdown(
    """
**Tip:** you can also use the **sidebar** to switch between pages while keeping
your current parameter settings.
"""
)
