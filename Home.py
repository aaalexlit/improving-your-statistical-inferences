import streamlit as st

st.set_page_config(page_title="Improving Statistical Inferences", page_icon="📈", layout="wide")

st.title("📈 Improving Your Statistical Inferences")

with st.sidebar:
    st.header("Welcome! 👋")
    st.write("Use the sidebar to navigate between simulations and analyses.")

# Main body content
st.write("""
Welcome to the interactive app for exploring statistical concepts!

- Go to **P-Value Simulation** to simulate experiments
- (More features will be added soon!)
""")