# pages/6_P_Value_Over_Time.py
"""
P-Value Over Time

* Simulates p-values as sample size increases
* Shows how p-values fluctuate during data collection
* Interactive controls for total sample size and effect size
* Identifies the lowest p-value and when significance is first reached
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="P-Value Over Time", page_icon="📈", layout="wide")
st.title("📈 P-Value Over Time")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Settings")
    n = st.slider("Total number of datapoints (per condition)", 20, 2000, 200, 10)
    D = st.slider("True effect size (Cohen's d)", -1.0, 1.0, 0.0, 0.1)
    SD = st.slider("Standard deviation", 0.1, 2.0, 1.0, 0.1)

    # Add a button to re-run the simulation
    rerun_button = st.button("🔄 Re-run Simulation", help="Generate new random data with the current parameters")

    st.markdown("""
    This simulation shows how p-values fluctuate as sample size increases.

    The simulation:
    1. Starts with 10 participants per group
    2. Adds participants one by one
    3. Calculates p-value at each step
    4. Plots p-value against sample size
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
# Set a random seed that changes when the button is clicked
if 'last_run' not in st.session_state or rerun_button:
    st.session_state.last_run = np.random.randint(0, 1000000)

np.random.seed(st.session_state.last_run)

# Initialize arrays
total_n = n + 10  # Start after initial 10 participants
p_values = np.zeros(total_n)
x_values = np.zeros(total_n)
y_values = np.zeros(total_n)

# Generate random data for first 10 participants
for i in range(10):
    x_values[i] = np.random.normal(0, SD)
    y_values[i] = np.random.normal(D, SD)

# Simulate adding participants one by one
for i in range(10, total_n):
    x_values[i] = np.random.normal(0, SD)
    y_values[i] = np.random.normal(D, SD)

    # Perform t-test with current data
    t_test = stats.ttest_ind(x_values[:i+1], y_values[:i+1], equal_var=True)
    p_values[i] = t_test.pvalue

# Remove first 10 empty p-values
p_values = p_values[10:total_n]
sample_sizes = np.arange(10, total_n)

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(sample_sizes, p_values, lw=2)
ax.axhline(y=0.05, color="darkgrey", linestyle="--", lw=2)

ax.set_ylim(0, 1)
ax.set_xlim(10, total_n)
ax.set_xlabel('Sample size')
ax.set_ylabel('p-value')
ax.grid(alpha=0.2)

# Set x-axis ticks
tick_step = (total_n - 10) // 4
ax.set_xticks(np.arange(10, total_n + 1, tick_step))

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
min_p = np.min(p_values)
min_p_sample_size = sample_sizes[np.argmin(p_values)]

# Find when p-value first drops below 0.05 (if it does)
sig_indices = np.where(p_values < 0.05)[0]
first_sig_index = sig_indices[0] if len(sig_indices) > 0 else None
first_sig_sample_size = sample_sizes[first_sig_index] if first_sig_index is not None else None

# ─────────────────────────────────────────────────────────────────────────────
# Streamlit display
# ─────────────────────────────────────────────────────────────────────────────
_, col_fig, _ = st.columns([1, 3, 1])
with col_fig:
    st.pyplot(fig, use_container_width=True)

st.markdown("### Results")
st.markdown(f"The lowest p-value was **{min_p:.4f}**, observed at sample size **{min_p_sample_size}**.")

if first_sig_sample_size:
    st.markdown(f"The p-value first dropped below 0.05 at sample size **{first_sig_sample_size}**.")
else:
    st.markdown("The p-value never dropped below 0.05 in this simulation.")

st.markdown("""
### Notes
- P-values can fluctuate dramatically as sample size increases
- The lowest p-value might occur before reaching the maximum sample size
- Statistical significance (p < 0.05) can be reached and then lost again
- This demonstrates why it's problematic to stop data collection when reaching significance
""")
