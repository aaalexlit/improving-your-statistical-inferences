# pages/4_Likelihood_Ratio.py
"""
Interactive likelihood-ratio visualisation for a binomial model.

* Choose the data: total trials **n** and observed successes **x**.
* Specify two point-hypotheses **H₀** and **H₁** (θ values in [0,1]).
* The page shows
  - the likelihood curve **L(θ | x,n)**,
  - the two likelihood values L(H₀) and L(H₁),
  - the maximum-likelihood estimate θ̂ = x / n,
  - dashed guide lines identical to the original R figure,
  - the likelihood-ratio Λ = L(H₀) / L(H₁) (and its reciprocal) in the title.
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Likelihood Ratio", page_icon="📈", layout="wide")
st.title("📈 Binomial Likelihood-Ratio Plot")

with st.sidebar:
    st.header("Data")
    n = st.slider("Total trials n", 1, 100, 13, 1)
    x = st.slider("Successes x", 0, n, 8, 1)

    st.header("Point hypotheses")
    H0 = st.number_input("H₀ (θ₀)", 0.00, 1.00, 0.50, 0.01)
    H1 = st.number_input("H₁ (θ₁)", 0.00, 1.00, 0.05, 0.01)

# guard against undefined likelihood when θ=0 or 1 with incompatible x
eps = 1e-12
H0 = np.clip(H0, eps, 1 - eps)
H1 = np.clip(H1, eps, 1 - eps)

# ──────────────────────────────────────────────────────────────────────────────
# Likelihood curve and LR
# ──────────────────────────────────────────────────────────────────────────────
θ = np.linspace(0, 1, 401)
L = stats.binom.pmf(x, n, θ)

L_H0 = stats.binom.pmf(x, n, H0)
L_H1 = stats.binom.pmf(x, n, H1)

LR_H0_H1 = L_H0 / L_H1
LR_H1_H0 = 1.0 / LR_H0_H1

θ_hat = x / n
L_hat = stats.binom.pmf(x, n, θ_hat)

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(θ, L, lw=2, label="Likelihood")
ax.set_xlabel(r"$\theta$ (probability of success)")
ax.set_ylabel("Likelihood")
ax.set_ylim(0, L_hat * 1.25)  # leave room for annotation

# points for H0, H1, and vertical/horizontal guides
ax.plot([H0], [L_H0], "o", color="#d62728")
ax.plot([H1], [L_H1], "o", color="#2ca02c")
ax.plot([θ_hat], [L_hat], "o", color="black", markersize=5)

# horizontal dashed to θ̂
ax.hlines(L_H0, min(H0, θ_hat), max(H0, θ_hat), linestyles="dashed", lw=1)
ax.hlines(L_H1, min(H1, θ_hat), max(H1, θ_hat), linestyles="dashed", lw=1)

# vertical dashed between likelihoods at θ̂
ax.vlines(θ_hat, min(L_H0, L_H1), max(L_H0, L_H1), linestyles="dashed", lw=1)

# labels
ax.annotate(r"$\hat{\theta}$", xy=(θ_hat, L_hat), xytext=(θ_hat, L_hat * 1.10),
            ha="center", fontsize=9, color="black")
ax.annotate("H₀", xy=(H0, L_H0), xytext=(H0, L_H0 * 1.08),
            ha="center", fontsize=9, color="#d62728")
ax.annotate("H₁", xy=(H1, L_H1), xytext=(H1, L_H1 * 1.08),
            ha="center", fontsize=9, color="#2ca02c")

# title with LR values
ax.set_title(
    f"Likelihood Ratio  H₀/H₁: {LR_H0_H1:.2f}     "
    f"H₁/H₀: {LR_H1_H0:.2f}"
)

ax.grid(alpha=0.15, ls=":")
fig.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit layout
# ──────────────────────────────────────────────────────────────────────────────
_, col_fig, _ = st.columns([1, 3, 1])
with col_fig:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    f"For n = **{n}** and x = **{x}**, "
    f"θ̂ = x / n = **{θ_hat:.2f}**, "
    f"L(H₀)/L(H₁) = **{LR_H0_H1:.2f}** and "
    f"L(H₁)/L(H₀) = **{LR_H1_H0:.2f}**."
)