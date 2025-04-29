# 2_Power_Curve.py – interactive visualisation of Type I & II errors, power and effect size

"""
This page lets the user explore how α, power (1-β), sample size *n* and effect size *(Cohen’s d)* trade-off in a **one–sample Z-test**.
Four “solve-for” modes are offered – Power, α, *n* or *d*.  Whichever quantity is selected is **calculated automatically** from the other three and shown inside the plot, while the sliders for the remaining parameters stay active.

The plot places every annotation **inside** the figure (no separate legend) and shades:
  • Type I error (α) – red    • Type II error (β) – navy    • Power – sky-blue.
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt

st.set_page_config(page_title="Power & Errors", page_icon="📈", layout="wide")

# ----------------------------------------------------------------------------
# Sidebar: user controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings 🎛️")
    solve_for = st.radio("Solve for…", ["Power", "Alpha", "n", "d"], horizontal=True)

    # We still show all sliders; the one being solved-for is ignored and greyed-out using st.slider(disabled=True)
    # but Streamlit (<1.30) doesn’t have disabled sliders, so we just visually hint by the label.

    power_in  = st.slider("Power (1 – β)", 0.50, 0.99, 0.80, 0.01)
    alpha_in  = st.slider("Significance level (α)", 0.001, 0.20, 0.05, 0.001)
    n_in      = st.slider("Sample size (n)", 5, 500, 20, 1)
    d_in      = st.slider("Effect size (Cohen’s d)", 0.01, 2.0, 0.44, 0.01)
    tails     = st.radio("Tail", ["One-tailed", "Two-tailed"], horizontal=True)

# Helper ---------------------------------------------------------------------
NORMAL = stats.norm()

def crit_value(alpha: float, two_tailed: bool) -> float:
    """Return positive critical Z for given α and tail selection."""
    return NORMAL.ppf(1 - alpha/2) if two_tailed else NORMAL.ppf(1 - alpha)

def solve(power, alpha, n, d, mode, two_tailed):
    """Solve for requested parameter using standard z-test formulae.
    Returns (power, alpha, n, d)."""
    two = two_tailed
    z_alpha = crit_value(alpha, two)

    if mode == "Power":  # compute power from alpha, n, d
        z_beta = z_alpha - d * sqrt(n)
        if two:
            beta  = NORMAL.cdf( z_beta) - NORMAL.cdf(-z_alpha - d*sqrt(n))
        else:
            beta  = NORMAL.cdf( z_beta)
        power = 1 - beta

    elif mode == "Alpha":  # compute alpha from power, n, d
        z_beta = NORMAL.ppf(power)
        z_alpha = d*sqrt(n) - z_beta
        alpha = 2*(1-NORMAL.cdf(z_alpha)) if two else (1 - NORMAL.cdf(z_alpha))
        alpha = max(min(alpha, 0.5), 1e-6)  # clamp to sensible range

    elif mode == "n":  # compute sample size n from power, alpha, d
        z_beta  = NORMAL.ppf(power)
        z_alpha = crit_value(alpha, two)
        n = ((z_alpha + z_beta) / d)**2
        n = max(n, 2)

    elif mode == "d":  # compute effect size from power, alpha, n
        z_beta  = NORMAL.ppf(power)
        z_alpha = crit_value(alpha, two)
        d = (z_alpha + z_beta) / sqrt(n)

    # Re-calculate power with final values to ensure consistency when mode ≠ power
    z_alpha = crit_value(alpha, two)
    z_beta  = z_alpha - d*sqrt(n)
    if two:
        beta = NORMAL.cdf( z_beta) - NORMAL.cdf(-z_alpha - d*sqrt(n))
    else:
        beta = NORMAL.cdf(z_beta)
    power = 1 - beta

    return power, alpha, n, d

# Solve ----------------------------------------------------------------------
TWOTAILED = tails == "Two-tailed"
POWER, ALPHA, N, D = solve(power_in, alpha_in, n_in, d_in, solve_for, TWOTAILED)

# ----------------------------------------------------------------------------
# Build the plot
# ----------------------------------------------------------------------------
# X-axis range – cover ±4 SD around both means
x_min = -4
x_max = max(4, D + 4)
xx = np.linspace(x_min, x_max, 2000)

h0_pdf = NORMAL.pdf(xx)                        # H0 ~ N(0,1)
ha_pdf = NORMAL.pdf(xx, loc=D, scale=1)        # Ha ~ N(d,1)

zcrit = crit_value(ALPHA, TWOTAILED)

fig, ax = plt.subplots(figsize=(8, 4.5))

# Plot PDFs
ax.plot(xx, h0_pdf, color="black", linestyle=":", linewidth=1)
ax.plot(xx, ha_pdf, color="#4aa6ff", linewidth=2)

# --- Shading ----------------------------------------------------------------
if TWOTAILED:
    # Type I error (α) – red tails under H0
    ax.fill_between(xx, 0, h0_pdf, where=(xx <= -zcrit) | (xx >= zcrit), color="#c23b22", alpha=0.4)
    # Type II error (β) – dark area under Ha between −zcrit ↔ zcrit
    ax.fill_between(xx, 0, ha_pdf, where=(xx > -zcrit) & (xx < zcrit), color="#16233a", alpha=0.5)
    # Power – light blue tails under Ha beyond ±zcrit
    ax.fill_between(xx, 0, ha_pdf, where=(xx <= -zcrit) | (xx >= zcrit), color="#4aa6ff", alpha=0.4)
else:
    # Type I error (α) – right tail under H0
    ax.fill_between(xx, 0, h0_pdf, where=(xx >= zcrit), color="#c23b22", alpha=0.4)
    # Type II error (β) – left area under Ha before zcrit
    ax.fill_between(xx, 0, ha_pdf, where=(xx < zcrit), color="#16233a", alpha=0.5)
    # Power – right tail under Ha beyond zcrit
    ax.fill_between(xx, 0, ha_pdf, where=(xx >= zcrit), color="#4aa6ff", alpha=0.4)

# Critical lines
ax.axvline(zcrit, color="black", linewidth=1)
if TWOTAILED:
    ax.axvline(-zcrit, color="black", linewidth=1)

# Arrow & label for Cohen's d
ax.annotate(r"Cohen's $d$: {:.2f}".format(D), xy=(0.5*D, 0.37), ha="center", fontsize=12, weight='bold')
ax.annotate("", xy=(0, 0.33), xytext=(D, 0.33), arrowprops=dict(arrowstyle="<->", color="black"))

# Axis & ticks
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 0.45)
ax.set_xlabel("Standardised effect (z)")
ax.set_ylabel("Probability density")
ax.set_yticks([])

# Inline text annotations under the axis (mimicking the original design)
ax.text(0, -0.04, r"$\beta$", ha="center", va="top", fontsize=11)
ax.text(zcrit + 0.02, -0.04, r"$\alpha$" if not TWOTAILED else r"$\alpha/2$", ha="left", va="top", fontsize=11)

# ---------------------------------------------------------------------------
# Show figure centred with margins via columns
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns([1, 2.5, 1])
with col2:
    st.pyplot(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary numbers (large, beneath plot)
# ---------------------------------------------------------------------------
col_a, col_b, col_c, col_d = st.columns(4)
col_a.markdown(f"<h3 style='text-align:center;color:#c23b22;'>{ALPHA*100:.0f}%</h3><p style='text-align:center;'>Type I error</p>", unsafe_allow_html=True)
col_b.markdown(f"<h3 style='text-align:center;color:#16233a;'>{(1-POWER)*100:.0f}%</h3><p style='text-align:center;'>Type II error</p>", unsafe_allow_html=True)
col_c.markdown(f"<h3 style='text-align:center;color:#4aa6ff;'>{POWER*100:.0f}%</h3><p style='text-align:center;'>Power</p>", unsafe_allow_html=True)
col_d.markdown(f"<h3 style='text-align:center;color:#1c7c54;'>{int(round(N,0))}</h3><p style='text-align:center;'>Sample size</p>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Provide numeric outputs in an expander for detail
# ---------------------------------------------------------------------------
with st.expander("Show numeric details"):
    st.write(f"Power (1 – β): **{POWER:.3f}**")
    st.write(f"Type I error α: **{ALPHA:.3f}**")
    st.write(f"Type II error β: **{1-POWER:.3f}**")
    st.write(f"Sample size n: **{N:.2f}**")
    st.write(f"Effect size d: **{D:.3f}**")
    st.write(f"Tail: **{'Two-tailed' if TWOTAILED else 'One-tailed'}**")