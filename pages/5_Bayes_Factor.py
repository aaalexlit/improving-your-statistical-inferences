# pages/5_Bayes_Factor.py
"""Binomial Bayes‑Factor demonstration (BF₁₀).

* Point null hypothesis **H₀** at θ₀.
* Conjugate Beta(α, β) prior.
* Data: *n* Bernoulli trials with *x* successes.
* Shows prior, likelihood, and posterior Beta curves plus the Bayes Factor:

  BF₁₀ = posterior density at θ₀ ÷ prior density at θ₀.
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bayes Factor", page_icon="📈", layout="wide")

st.title("📈 Binomial Bayes‑Factor Calculator")

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    n = st.slider("Total trials n", min_value=1, max_value=200, value=20, step=1)
    x = st.slider("Successes x", min_value=0, max_value=n, value=10, step=1)

    st.header("Prior Beta(α, β)")
    α_prior = st.number_input("α (alpha)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)
    β_prior = st.number_input("β (beta)", min_value=0.01, max_value=100.0, value=1.0, step=0.1)

    st.header("Point null θ₀")
    H0 = st.number_input("θ₀", 0.0, 1.0, 0.50, 0.005, format="%0.3f")

# guard for boundaries in Beta density
eps = 1e-9
θ0 = np.clip(H0, eps, 1 - eps)

# ──────────────────────────────────────────────────────────────────────────────
# Conjugate updating
# ──────────────────────────────────────────────────────────────────────────────
α_lik = x + 1
β_lik = (n - x) + 1
α_post = α_prior + x
β_post = β_prior + (n - x)

θ = np.linspace(0, 1, 500)
prior_dens = stats.beta.pdf(θ, α_prior, β_prior)
likelihood_dens = stats.beta.pdf(θ, α_lik, β_lik)
posterior_dens = stats.beta.pdf(θ, α_post, β_post)

# Bayes Factor (posterior / prior at θ0)
BF10 = stats.beta.pdf(θ0, α_post, β_post) / stats.beta.pdf(θ0, α_prior, β_prior)

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(θ, posterior_dens, lw=3, label="Posterior", color="black")
ax.plot(θ, prior_dens, lw=3, label="Prior", color="grey")
ax.plot(θ, likelihood_dens, lw=3, label="Likelihood", color="dodgerblue", linestyle="--")

# vertical line for θ0 and dots
ax.axvline(θ0, color="black", lw=1, ls=":")
ax.plot([θ0], [stats.beta.pdf(θ0, α_post, β_post)], "o", color="black")
ax.plot([θ0], [stats.beta.pdf(θ0, α_prior, β_prior)], "o", color="grey")
ax.vlines(θ0,
          stats.beta.pdf(θ0, α_prior, β_prior),
          stats.beta.pdf(θ0, α_post, β_post),
          lw=1, ls="dashed")

# y‑axis padding so labels stay clear of title
ymax = posterior_dens.max() * 1.25
ax.set_ylim(0, ymax)

ax.set_xlabel(r"$\theta$ (probability of success)")
ax.set_ylabel("Density")
ax.set_title(f"Bayes Factor  BF₁₀ = {BF10:.2f}")
ax.legend(frameon=False)
ax.grid(alpha=0.2, ls=":")
fig.tight_layout()

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit layout
# ──────────────────────────────────────────────────────────────────────────────
_, col_fig, _ = st.columns([1, 3, 1])
with col_fig:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    f"Posterior Beta(**α** = {α_post:.2f}, **β** = {β_post:.2f})  |  "
    f"Prior Beta(**α** = {α_prior:.2f}, **β** = {β_prior:.2f})  |  "
    f"BF₁₀ ≈ **{BF10:.2f}**"
)
