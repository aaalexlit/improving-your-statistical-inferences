# pages/7_Optional_Stopping.py
"""
Optional Stopping Simulation

* Simulates p-values across multiple looks at the data
* Shows how optional stopping affects Type 1 error rates
* Interactive controls for simulation parameters
* Visualizes the distribution of p-values under optional stopping

Performance optimizations:
* Generates all random data at once instead of per simulation
* Processes simulations in batches to improve memory efficiency
* Uses vectorized operations for t-test calculations
* Implements vectorized post-processing of results
* Reduces frequency of progress updates
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
    Looks = st.slider("Number of looks at the data", 2, 200, 5, 1)
    nSim = st.slider("Number of simulated studies", 1000, 100000, 50000, 1000)
    alpha = st.slider("Alpha level", 0.01, 0.10, 0.05, 0.01)
    D = 0.0

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

# Generate all random data at once
x_all = np.random.normal(0, 1, (nSim, N))
y_all = np.random.normal(D, 1, (nSim, N))

# Update progress less frequently to improve performance
update_freq = max(1, nSim // 20)  # Update at most 20 times

# Pre-allocate arrays for t-test statistics and p-values
# This is more efficient than calculating one simulation at a time
with st.spinner("Running simulations... This may take a moment."):
    # Process simulations in batches to avoid memory issues with very large nSim
    batch_size = min(1000, nSim)  # Process at most 1000 simulations at once
    num_batches = (nSim + batch_size - 1) // batch_size  # Ceiling division

    for batch in range(num_batches):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, nSim)
        current_batch_size = end_idx - start_idx

        for j in range(Looks):
            # Get the current sample size for this look
            current_n = LookN[j]

            # Extract the data for the current batch up to the current sample size
            x_batch = x_all[start_idx:end_idx, :current_n]
            y_batch = y_all[start_idx:end_idx, :current_n]

            # Perform t-tests for all simulations in this batch at once
            # We'll implement the t-test calculation manually for better performance
            # This is equivalent to scipy.stats.ttest_ind with equal_var=True

            # Calculate means for each sample
            mean_x = np.mean(x_batch, axis=1)
            mean_y = np.mean(y_batch, axis=1)

            # Calculate variances for each sample
            var_x = np.var(x_batch, axis=1, ddof=1)  # ddof=1 for sample variance
            var_y = np.var(y_batch, axis=1, ddof=1)

            # Calculate the pooled standard deviation
            n_x = x_batch.shape[1]
            n_y = y_batch.shape[1]
            pooled_var = ((n_x - 1) * var_x + (n_y - 1) * var_y) / (n_x + n_y - 2)
            pooled_std = np.sqrt(pooled_var)

            # Calculate the t-statistic
            t_stat = (mean_x - mean_y) / (pooled_std * np.sqrt(1/n_x + 1/n_y))

            # Calculate the degrees of freedom
            df = n_x + n_y - 2

            # Calculate the p-value (two-tailed test)
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

            # Store the p-values
            matp[start_idx:end_idx, j] = p_values

        # Update progress after each batch
        progress_bar.progress((end_idx) / nSim)
        status_text.text(f"Processed {end_idx} of {nSim} simulations")

# Save Type 1 error rate for each look
SigSeq = np.sum(matp < alpha, axis=0)

# Get the positions at which studies are stopped, and then these p-values - vectorized approach
# Create a mask of significant p-values
sig_mask = matp < alpha

# For each simulation, find the first significant look (or use the last look if none are significant)
# First, get the indices of significant looks for each simulation
first_sig_indices = np.argmax(sig_mask, axis=1)

# For simulations with no significant results, argmax returns 0, so we need to fix those
# Create a mask for simulations with no significant results
no_sig_mask = ~np.any(sig_mask, axis=1)

# Set those to use the last look
first_sig_indices[no_sig_mask] = Looks - 1

# Store the stopping positions
OptStop = first_sig_indices

# Get the p-values at the stopping positions
p = matp[np.arange(nSim), OptStop]

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
breaks = 100
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
