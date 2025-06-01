# pages/7_Optional_Stopping.py
"""
Optional Stopping Simulation

* Simulates p-values across multiple looks at the data
* Shows how optional stopping affects Type 1 error rates
* Interactive controls for simulation parameters
* Visualizes the distribution of p-values under optional stopping
"""

import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Optional Stopping Simulation", page_icon="🔍", layout="wide")
st.title("🔍 Optional Stopping Simulation")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulation Settings")
    N = st.slider("Total number of datapoints (per condition)", 20, 500, 100, 10)
    Looks = st.slider("Number of looks at the data", 2, 10, 5, 1)
    nSim = st.slider("Number of simulated studies", 1000, 100000, 50000, 1000)
    alpha = st.slider("Alpha level", 0.01, 0.10, 0.05, 0.01)
    D = st.slider("True effect size (Cohen's d)", -1.0, 1.0, 0.0, 0.1)
    
    # Add a button to re-run the simulation
    rerun_button = st.button("🔄 Re-run Simulation", help="Generate new random data with the current parameters")
    
    st.markdown("""
    This simulation demonstrates how optional stopping affects Type 1 error rates.
    
    The simulation:
    1. Generates data for multiple studies
    2. Looks at the data at predefined sample sizes
    3. Performs t-tests at each look
    4. Calculates Type 1 error rates with and without optional stopping
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────
# Set a random seed that changes when the button is clicked
if 'last_run' not in st.session_state or rerun_button:
    st.session_state.last_run = np.random.randint(0, 1000000)

np.random.seed(st.session_state.last_run)

# Progress bar
progress_bar = st.progress(0)
status_text = st.empty()

# Determine at which N's to look
LookN = np.ceil(np.linspace(0, N, Looks + 1)).astype(int)
LookN = LookN[1:]  # remove look at 0
LookN = LookN[LookN > 2]  # Remove looks at N of 1 or 2 (not possible with t-test)
Looks = len(LookN)  # if looks are removed, change number of looks

# Matrix for p-values at sequential tests
matp = np.zeros((nSim, Looks))
# Variable to store positions of optional stopping
OptStop = np.zeros(nSim, dtype=int)
# Variable to save optional stopping p-values
p = np.zeros(nSim)

# Loop data generation for each study, then loop to perform a test for each N
for i in range(nSim):
    x = np.random.normal(0, 1, N)
    y = np.random.normal(D, 1, N)
    
    for j in range(Looks):
        # Perform the t-test, store p-value
        t_test = stats.ttest_ind(x[:LookN[j]], y[:LookN[j]], equal_var=True)
        matp[i, j] = t_test.pvalue
    
    # Update progress
    if i % (nSim // 100) == 0 or i == nSim - 1:
        progress_bar.progress((i + 1) / nSim)
        status_text.text(f"Simulating study {i+1} of {nSim}")

# Save Type 1 error rate for each look
SigSeq = np.sum(matp < alpha, axis=0)

# Get the positions at which studies are stopped, and then these p-values
for i in range(nSim):
    sig_indices = np.where(matp[i, :] < alpha)[0]
    if len(sig_indices) > 0:
        OptStop[i] = sig_indices[0]
    else:
        OptStop[i] = Looks - 1  # If nothing significant, take last p-value

for i in range(nSim):
    p[i] = matp[i, OptStop[i]]

# Clear progress indicators
status_text.empty()
progress_bar.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### Results")

# Display Type 1 error rates for each look
look_error_rates = SigSeq / nSim
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Type 1 error rates for each look:")
    for j in range(Looks):
        st.markdown(f"Look {j+1} (N = {LookN[j]}): **{look_error_rates[j]:.4f}**")

with col2:
    st.markdown("#### Optional stopping error rate:")
    opt_stop_error = np.sum(p < alpha) / nSim
    st.markdown(f"Type 1 error rate with optional stopping: **{opt_stop_error:.4f}**")
    
    # Calculate inflation factor
    inflation = opt_stop_error / alpha
    st.markdown(f"Inflation factor: **{inflation:.2f}x**")

# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Histogram of p-values with optional stopping
breaks = 50
ax1.hist(p, bins=breaks, color="grey", edgecolor="black", alpha=0.7)
ax1.axhline(y=nSim/breaks, color="red", linestyle="--", label="Uniform distribution")
ax1.set_xlabel("p-value")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of p-values with optional stopping")
ax1.legend()

# Plot 2: Type 1 error rates across looks
ax2.plot(range(1, Looks + 1), look_error_rates, 'o-', color='blue', label='Error rate at each look')
ax2.axhline(y=alpha, color="red", linestyle="--", label=f"Alpha = {alpha}")
ax2.axhline(y=opt_stop_error, color="green", linestyle="-.", label=f"Optional stopping error rate = {opt_stop_error:.4f}")
ax2.set_xlabel("Look number")
ax2.set_ylabel("Type 1 error rate")
ax2.set_title("Type 1 error rates across looks")
ax2.set_xticks(range(1, Looks + 1))
ax2.legend()

plt.tight_layout()
st.pyplot(fig)

# ─────────────────────────────────────────────────────────────────────────────
# Explanation
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
### Notes
- When researchers look at their data multiple times and stop when they find significance, the Type 1 error rate is inflated
- The nominal alpha level (e.g., 0.05) is only valid for a single, pre-planned analysis
- Optional stopping can substantially increase false positive rates
- This simulation demonstrates why pre-registration and stopping rules are important in research
""")