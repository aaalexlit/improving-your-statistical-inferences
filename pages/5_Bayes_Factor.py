# pages/5_Bayes_Factor.py
"""
Binomial Bayes-Factor & Posterior Summary

* Point null **H₀** at θ₀  →  BF₁₀ = p(θ₀ | data) / p(θ₀ | prior)
* Conjugate Beta(α, β) prior
* Interactive controls for *n*, *x*, prior (α, β), and θ₀
* Shows prior, likelihood, posterior, posterior mean, 95 % central credible interval,
  and 95 % highest-density interval (HDI)
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bayes Factor", page_icon="📈", layout="wide")
st.title("📈 Binomial Bayes-Factor & Posterior Summary")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data")
    n = st.slider("Total trials n", 1, 200, 20, 1)
    x = st.slider("Successes x", 0, n, 10, 1)

    st.header("Prior Beta(α, β)")
    α_prior = st.number_input("α (alpha)", 0.01, 100.0, 1.0, 0.1)
    β_prior = st.number_input("β (beta)", 0.01, 100.0, 1.0, 0.1)

    st.header("Point null θ₀")
    θ0 = st.number_input("θ₀", 0.0, 1.0, 0.50, 0.005, format="%0.3f")

eps = 1e-9
θ0 = np.clip(θ0, eps, 1 - eps)

# ─────────────────────────────────────────────────────────────────────────────
# Conjugate updating
# ─────────────────────────────────────────────────────────────────────────────
α_lik, β_lik = x + 1, n - x + 1
α_post, β_post = α_prior + x, β_prior + (n - x)

θ = np.linspace(0, 1, 600)
prior_dens      = stats.beta.pdf(θ, α_prior, β_prior)
likelihood_dens = stats.beta.pdf(θ, α_lik,  β_lik)
posterior_dens  = stats.beta.pdf(θ, α_post, β_post)

# Bayes factor
BF10 = stats.beta.pdf(θ0, α_post, β_post) / stats.beta.pdf(θ0, α_prior, β_prior)

# Posterior summaries
post_mean = α_post / (α_post + β_post)
LL, UL = stats.beta.ppf([0.025, 0.975], α_post, β_post)  # central 95 % CI

# HDI (95 %) by brute-force shortest interval on fine grid
cred_mass = 0.95
cdf = stats.beta.cdf(θ, α_post, β_post)
# for each index i find j s.t. cdf[j]-cdf[i]≈cred_mass
idx = np.searchsorted(cdf, cdf + cred_mass, side="right") - 1
valid = idx < len(θ)
interval_widths = θ[valid] - θ[idx[valid]]
min_idx = interval_widths.argmin()
HDI_low, HDI_high = θ[min_idx], θ[idx[valid][min_idx]]

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))

ax.plot(θ, posterior_dens, lw=3, label="Posterior", color="black")
ax.plot(θ, prior_dens,     lw=3, label="Prior",     color="grey")
ax.plot(θ, likelihood_dens,lw=3, label="Likelihood", color="dodgerblue", ls="--")

# Shade tails outside central CI
ax.fill_between(θ, 0, posterior_dens, where=(θ < LL) | (θ > UL),
                color="lightgrey", alpha=0.4, label="Central 95 % CI tails")

# Shade HDI
ax.fill_between(θ, 0, posterior_dens, where=(θ >= HDI_low) & (θ <= HDI_high),
                color="#d4f4dd", alpha=0.5, label="HDI 95 %")

# Vertical lines
ax.axvline(post_mean, color="black", lw=1)
ax.axvline(LL, color="grey", lw=1, ls=":")
ax.axvline(UL, color="grey", lw=1, ls=":")
ax.axvline(HDI_low,  color="#2ca02c", lw=1, ls="--")
ax.axvline(HDI_high, color="#2ca02c", lw=1, ls="--")

# θ₀ line and dots
ax.axvline(θ0, color="black", lw=1, ls=":")
ax.plot([θ0], [stats.beta.pdf(θ0, α_post, β_post)], "o", color="black")
ax.plot([θ0], [stats.beta.pdf(θ0, α_prior, β_prior)], "o", color="grey")
ax.vlines(θ0,
          stats.beta.pdf(θ0, α_prior, β_prior),
          stats.beta.pdf(θ0, α_post, β_post),
          lw=1, ls="dashed")

ymax = posterior_dens.max() * 1.30
ax.set_ylim(0, ymax)
ax.set_xlabel(r"$\theta$ (probability of success)")
ax.set_ylabel("Density")
ax.set_title(
    f"BF₁₀ = {BF10:.2f}   |   Posterior mean = {post_mean:.3f}\n"
    f"Central 95 % CI [{LL:.3f}, {UL:.3f}]   |   HDI [{HDI_low:.3f}, {HDI_high:.3f}]"
)
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.2, ls=":")
fig.tight_layout()

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit display
# ─────────────────────────────────────────────────────────────────────────────
_, col_fig, _ = st.columns([1, 3, 1])
with col_fig:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    f"Posterior Beta(**α** = {α_post:.2f}, **β** = {β_post:.2f}) • "
    f"Posterior mean **{post_mean:.3f}** • "
    f"Central 95 % CI **[{LL:.3f}, {UL:.3f}]** • "
    f"HDI **[{HDI_low:.3f}, {HDI_high:.3f}]** • "
    f"BF₁₀ ≈ **{BF10:.2f}**"
)