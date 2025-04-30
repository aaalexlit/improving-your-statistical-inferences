"""Binomial likelihood curve: interactively vary number of trials (*n*) and successes (*x*)."""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Likelihood Curve", page_icon="📈", layout="wide")

st.title("📈 Binomial Likelihood Curve")

with st.sidebar:
    st.header("Parameters")
    n = st.slider("Number of trials (n)", 1, 200, 10, 1)
    x = st.slider("Number of successes (x)", 0, n, 8, 1)

# Theta grid and likelihood ----------------------------------------------------
theta = np.linspace(0, 1, 400)
likelihood = stats.binom.pmf(x, n, theta)

# Plot ------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(theta, likelihood, lw=2)
ax.set_xlabel(r"$\theta$ (probability of success)")
ax.set_ylabel("Likelihood")
ax.set_title(f"Likelihood Curve:  x = {x},  n = {n}")
ax.set_ylim(0, likelihood.max() * 1.05)

# Maximum likelihood marker ----------------------------------------------------
theta_hat = x / n if n else 0
max_like = stats.binom.pmf(x, n, theta_hat)
ax.plot([theta_hat], [max_like], marker="o", color="#d62728")
ax.annotate(r"$\hat\theta$ = {:.2f}".format(theta_hat), xy=(theta_hat, max_like), xytext=(theta_hat, max_like * 1.1),
            ha="center", fontsize=10, color="#d62728")

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.pyplot(fig, use_container_width=True)

st.markdown(
    f"For n = **{n}** trials and x = **{x}** successes, the maximum-likelihood estimate is **θ̂ = {theta_hat:.2f}**.")
