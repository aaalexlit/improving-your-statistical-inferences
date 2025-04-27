# 2_Power_Curve.py

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Power and Errors Visualization", page_icon="📈", layout="wide")
st.title("📈 Type I and II Errors, Power, and Effect Size Visualization")

# Sidebar Settings
with st.sidebar:
    st.header("Settings 🎛️")
    solve_for = st.radio("Solve for:", ("Power", "Alpha", "n", "d"))
    power = st.slider("Power (1-β)", 0.5, 0.99, 0.8, 0.01)
    alpha = st.slider("Significance level (α)", 0.001, 0.2, 0.05, 0.001)
    sample_size = st.slider("Sample size (n)", 5, 500, 20, 1)
    effect_size = st.slider("Effect size (Cohen's d)", 0.01, 2.0, 0.63, 0.01)
    tail_type = st.radio("Test Type:", ("One-tailed", "Two-tailed"))

# Adjust calculations based on solve_for (basic implementation)
if solve_for == "Power":
    df = sample_size - 1
    se = 1 / np.sqrt(sample_size)
    non_central_param = effect_size / se
    if tail_type == "One-tailed":
        z_alpha = stats.t.ppf(1 - alpha, df)
        z_beta = z_alpha - non_central_param
        beta = stats.t.cdf(z_beta, df)
    else:
        z_alpha = stats.t.ppf(1 - alpha/2, df)
        z_beta = z_alpha - non_central_param
        beta = stats.t.cdf(z_beta, df)
    calculated_power = 1 - beta
else:
    calculated_power = power

# Define distributions
x = np.linspace(-4, 4, 1000)
h0 = stats.norm.pdf(x, 0, 1)
ha = stats.norm.pdf(x, effect_size, 1)

# Critical values
if tail_type == "One-tailed":
    z_crit = stats.norm.ppf(1 - alpha)
else:
    z_crit = stats.norm.ppf(1 - alpha / 2)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, h0, label=r"$H_0$", color='black', linestyle='--')
ax.plot(x, ha, label=r"$H_a$", color='skyblue')

# Fill regions
if tail_type == "One-tailed":
    ax.fill_between(x, 0, h0, where=(x > z_crit), color='red', alpha=0.3, label="Type I error (α)")
    ax.fill_between(x, 0, ha, where=(x < z_crit), color='navy', alpha=0.3, label="Type II error (β)")
else:
    ax.fill_between(x, 0, h0, where=(x > z_crit), color='red', alpha=0.3)
    ax.fill_between(x, 0, h0, where=(x < -z_crit), color='red', alpha=0.3)
    ax.fill_between(x, 0, ha, where=(x < -z_crit) | (x > z_crit), color='skyblue', alpha=0.3)
    ax.fill_between(x, 0, ha, where=(x > -z_crit) & (x < z_crit), color='navy', alpha=0.3, label="Type II error (β)")

# Annotations
ax.axvline(x=z_crit, color='black', linestyle='solid', label=r"$Z_{crit}$")
if tail_type == "Two-tailed":
    ax.axvline(x=-z_crit, color='black', linestyle='solid')

ax.annotate(f"Cohen's d: {effect_size:.2f}", xy=(effect_size/2, 0.4), xytext=(effect_size/2, 0.45),
            arrowprops=dict(facecolor='black', shrink=0.05), ha='center')

ax.set_xlabel("Sample space")
ax.set_ylabel("Probability Density")
ax.set_ylim(0, 0.8)
ax.legend()
plt.tight_layout()

# Layout control
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.pyplot(fig)

# Display stats
st.subheader("Summary")
st.write(f"Calculated Power: **{calculated_power:.2%}**")
st.write(f"Sample Size (n): **{sample_size}**")
st.write(f"Effect Size (Cohen's d): **{effect_size:.2f}**")
st.write(f"Significance Level (α): **{alpha:.3f}**")

st.caption("""
Visualization based on a one-sample Z-test. You can vary the sample size, power, significance level, and effect size to see how the sampling distributions change.
""")
