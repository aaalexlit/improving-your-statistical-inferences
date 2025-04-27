import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestPower

NUM_BINS = 20

st.set_page_config(page_title="P-Value Simulation", page_icon="📊", layout="wide")
st.title("📊 P-Value Simulation for One-Sample t-Test")

# Sidebar parameters
with st.sidebar:
    st.header("Simulation Settings 🎛️")
    n_sims = st.slider('Number of Simulations', 10000, 100000, 50000, step=10000)
    mean_sample = st.slider('Mean Sample (IQ)', 80, 120, 106)
    sample_size = st.slider('Sample Size', 10, 100, 26)
    std_dev = st.slider('Standard Deviation', 5, 30, 15)

np.random.seed(42)

# Simulate all experiments at once
samples = np.random.normal(loc=mean_sample, scale=std_dev, size=(n_sims, sample_size))

# Perform t-tests manually
sample_means = samples.mean(axis=1)
sample_stds = samples.std(axis=1, ddof=1)
standard_errors = sample_stds / np.sqrt(sample_size)
t_statistics = (sample_means - 100) / standard_errors
p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), df=sample_size - 1))

# Calculate power
empirical_power = np.mean(p_values < 0.05)
effect_size = (mean_sample - 100) / std_dev
formal_power = TTestPower().power(effect_size=effect_size, nobs=sample_size, alpha=0.05, alternative='two-sided')

# Results
st.subheader("Results")
st.write(f"Empirical Power: **{empirical_power:.4f}**")
st.write(f"Formal Power: **{formal_power:.4f}**")
st.write(f"Effect size: **{effect_size:.4f}**")

# Plot
# Plot
fig, ax = plt.subplots(figsize=(6, 4))  # Good internal resolution
ax.hist(p_values, bins=NUM_BINS, color='grey', edgecolor='black')
ax.set_xlabel("P-values")
ax.set_ylabel("Number of p-values")
ax.set_title(f"P-value Distribution with {formal_power * 100:.1f}% Power")
ax.set_xlim(0, 1)
ax.set_ylim(0, n_sims)
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.set_yticks(np.linspace(0, n_sims, 5))
ax.axhline(y=n_sims / NUM_BINS, color='red', linestyle='dotted')
plt.tight_layout()

# Use columns to control plot width
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.pyplot(fig)
