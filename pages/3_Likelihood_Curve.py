# pages/3_Likelihood_Curve.py
"""Interactive binomial likelihood curve.

* **Sliders** let you pick number of trials *n* and observed successes *x*.
* The likelihood function  L(θ) = Binom(x | n, θ) is plotted for θ∈[0,1].
* The maximum‑likelihood estimate  θ̂ = x / n is marked with a red dot and label.
* The y‑axis is padded so the label never collides with the title.
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Likelihood Curve", page_icon="📈", layout="wide")

st.title("📈 Binomial Likelihood Curve")

# ----------------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Parameters")
    n = st.slider("Number of trials (n)", 1, 200, 10, 1)
    x = st.slider("Number of successes (x)", 0, n, 8, 1)

# ----------------------------------------------------------------------------
# Likelihood computation
# ----------------------------------------------------------------------------
θ = np.linspace(0, 1, 400)
likelihood = stats.binom.pmf(x, n, θ)

θ_hat = x / n  # MLE
L_hat = stats.binom.pmf(x, n, θ_hat)

# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(θ, likelihood, lw=2, label="Likelihood")

# Pad y‑axis 20 % above peak so annotation never touches title
ax.set_ylim(0, L_hat * 1.20)

# Red dot and label for MLE
ax.plot(θ_hat, L_hat, marker="o", color="#d62728")
ax.annotate(fr"$\hat{{\theta}}$ = {θ_hat:.2f}",
            xy=(θ_hat, L_hat),
            xytext=(θ_hat, L_hat * 1.10),
            ha="center", color="#d62728",
            arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=1))

ax.set_xlabel(r"$\theta$ (probability of success)")
ax.set_ylabel("Likelihood")
ax.set_title(f"Likelihood Curve   x = {x},   n = {n}")
ax.grid(alpha=0.2, ls=":")

# ----------------------------------------------------------------------------
# Display in Streamlit
# ----------------------------------------------------------------------------
col_left, col_fig, col_right = st.columns([1, 3, 1])
with col_fig:
    st.pyplot(fig, use_container_width=True)

st.markdown(f"Maximum‑likelihood estimate: **θ̂ = {θ_hat:.2f}**")
