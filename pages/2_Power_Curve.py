# 2_Power_Curve.py – corrected power calculations & shading
"""
Interactive visualisation for Type I & II errors, power and effect size for a **one‑sample Z‑test**.
Now uses exact formulas for power / β for both one‑ and two‑tailed tests so the shaded areas match
textbook diagrams.
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from math import sqrt

st.set_page_config(page_title="Power & Errors", page_icon="📈", layout="wide")

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings 🎛️")
    mode = st.radio("Solve for …", ["Power", "Alpha", "n", "d"], horizontal=True)
    power_in = st.slider("Power (1 – β)", 0.50, 0.99, 0.80, 0.01)
    alpha_in = st.slider("Significance level α", 0.001, 0.20, 0.05, 0.001)
    n_in     = st.slider("Sample size n", 5, 500, 50, 1)
    d_in     = st.slider("Effect size (Cohen’s d)", 0.01, 2.0, 0.20, 0.01)
    tails    = st.radio("Tail", ["One‑tailed", "Two‑tailed"], horizontal=True)

# convenience ------------------------------------------------------------------
NORM = stats.norm()
TWOTAILED = tails == "Two‑tailed"

def zcrit(alpha: float, two_tailed: bool) -> float:
    """Return +Z_crit for the chosen α and tail setting."""
    return NORM.ppf(1 - alpha/2) if two_tailed else NORM.ppf(1 - alpha)

# -----------------------------------------------------------------------------
# Power / alpha / n / d relationships (one‑sample z‑test)
# -----------------------------------------------------------------------------

def calc_power(alpha: float, n: float, d: float, two_tailed: bool) -> float:
    z_a = zcrit(alpha, two_tailed)
    shift = d * sqrt(n)
    if two_tailed:
        # Reject when |Z| > Z_α/2
        return (1 - NORM.cdf(z_a - shift)) + NORM.cdf(-z_a - shift)
    else:
        # Right‑tail test (H1 > H0)
        return 1 - NORM.cdf(z_a - shift)

def solve_unknown(power, alpha, n, d, mode, two_tailed):
    # Returning consistent (power, alpha, n, d)
    if mode == "Power":
        power = calc_power(alpha, n, d, two_tailed)

    elif mode == "Alpha":
        # numeric root‑find on α ∈ (1e‑6,0.5)
        from scipy.optimize import brentq
        f = lambda a: calc_power(a, n, d, two_tailed) - power
        alpha = brentq(f, 1e-6, 0.5)

    elif mode == "n":
        from scipy.optimize import brentq
        f = lambda nn: calc_power(alpha, nn, d, two_tailed) - power
        n = brentq(f, 2, 10_000)

    elif mode == "d":
        from scipy.optimize import brentq
        f = lambda dd: calc_power(alpha, n, dd, two_tailed) - power
        d = brentq(f, 1e-3, 5)

    # Ensure final power consistent
    power = calc_power(alpha, n, d, two_tailed)
    return power, alpha, n, d

POWER, ALPHA, N, D = solve_unknown(power_in, alpha_in, n_in, d_in, mode, TWOTAILED)
BETA = 1 - POWER
Z_ALPHA = zcrit(ALPHA, TWOTAILED)
SHIFT   = D * sqrt(N)

# -----------------------------------------------------------------------------
# Build distributions & shading
# -----------------------------------------------------------------------------
# X‑axis domain covering ±4 SD around both means
x_min = -4
x_max = max(4, SHIFT + 4)
xx = np.linspace(x_min, x_max, 2000)

h0_pdf = NORM.pdf(xx)                         # N(0,1)
ha_pdf = stats.norm.pdf(xx, loc=SHIFT, scale=1)  # N(d√n,1)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(xx, h0_pdf, color="0.3", ls=":", lw=1)
ax.plot(xx, ha_pdf, color="#5aa9ff", lw=2)

# --- Critical region lines ----------------------------------------------------
ax.axvline( Z_ALPHA, color="black", lw=1)
if TWOTAILED:
    ax.axvline(-Z_ALPHA, color="black", lw=1)

# --- Shading ------------------------------------------------------------------
if TWOTAILED:
    # α region under H0 tails
    ax.fill_between(xx, 0, h0_pdf, where=(xx >= Z_ALPHA) | (xx <= -Z_ALPHA), color="#c23b22", alpha=0.4)
    # β region under Ha inside acceptance band
    ax.fill_between(xx, 0, ha_pdf, where=(xx > -Z_ALPHA) & (xx < Z_ALPHA), color="#14233a", alpha=0.5)
    # Power region under Ha tails
    ax.fill_between(xx, 0, ha_pdf, where=(xx >= Z_ALPHA) | (xx <= -Z_ALPHA), color="#5aa9ff", alpha=0.4)
else:
    ax.fill_between(xx, 0, h0_pdf, where=(xx >= Z_ALPHA), color="#c23b22", alpha=0.4)
    ax.fill_between(xx, 0, ha_pdf, where=(xx < Z_ALPHA), color="#14233a", alpha=0.5)
    ax.fill_between(xx, 0, ha_pdf, where=(xx >= Z_ALPHA), color="#5aa9ff", alpha=0.4)

# --- Cohen's d arrow ----------------------------------------------------------
ax.annotate(r"Cohen's $d$: {:.2f}".format(D), xy=(SHIFT/2, 0.35), ha="center", fontsize=12, weight='bold')
ax.annotate("", xy=(0, 0.33), xytext=(SHIFT, 0.33), arrowprops=dict(arrowstyle="<->", color="black"))

# Axis formatting --------------------------------------------------------------
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, 0.45)
ax.set_xlabel("Standardised effect (Z)")
ax.set_ylabel("Probability density")
ax.set_yticks([])

# Inline α & β markers on baseline --------------------------------------------
ax.text(0, -0.035, r"$\beta$", ha="center", fontsize=11)
alpha_label = r"$\alpha$" if not TWOTAILED else r"$\alpha/2$"
ax.text(Z_ALPHA + 0.02, -0.035, alpha_label, ha="left", fontsize=11)

# -----------------------------------------------------------------------------
# Show in Streamlit layout
# -----------------------------------------------------------------------------
col_left, col_plot, col_right = st.columns([1, 3, 1])
with col_plot:
    st.pyplot(fig, use_container_width=True)

# Summary boxes ----------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.markdown(f"<h3 style='text-align:center;color:#c23b22;'>{ALPHA*100:.0f}%</h3><p style='text-align:center;'>Type I error</p>", unsafe_allow_html=True)
col2.markdown(f"<h3 style='text-align:center;color:#14233a;'>{BETA*100:.0f}%</h3><p style='text-align:center;'>Type II error</p>", unsafe_allow_html=True)
col3.markdown(f"<h3 style='text-align:center;color:#3b8aff;'>{POWER*100:.0f}%</h3><p style='text-align:center;'>Power</p>", unsafe_allow_html=True)
col4.markdown(f"<h3 style='text-align:center;color:#1c7c54;'>{int(round(N))}</h3><p style='text-align:center;'>Sample size</p>", unsafe_allow_html=True)

# Numeric details --------------------------------------------------------------
with st.expander("Show numeric details"):
    st.write(f"Power (1 – β): **{POWER:.3f}**")
    st.write(f"Type I error α: **{ALPHA:.3f}**")
    st.write(f"Type II error β: **{BETA:.3f}**")
    st.write(f"Sample size n: **{N:.2f}**")
    st.write(f"Effect size d: **{D:.3f}**")
    st.write(f"Tail: **{'Two‑tailed' if TWOTAILED else 'One‑tailed'}**")
