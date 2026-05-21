"""
Voltown Decision Lab — Main Entry Point
=========================================
Sequential Decision Making Under Uncertainty

Run with:  streamlit run Inicio.py
"""

import streamlit as st

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Voltown Decision Lab",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔋 Voltown Decision Lab")
    st.caption("Material de apoyo. Optimización bajo Incertidumbre. Universidad de los Andes.")

# ─── Main ────────────────────────────────────────────────────
st.title("🔋 Voltown Decision Lab")
st.markdown(
    "An interactive laboratory for exploring **sequential decision-making under uncertainty**. "
    "Design, tune, and battle-test energy policies — from simple rules to stochastic optimization."
)

st.image("assets/energy_flow.png")

st.divider()

# ─── Page descriptions ───────────────────────────────────────
st.header("Pages")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📐 Formulation")
    st.markdown(
        "Powell's five elements, decision variables, constraints, "
        "and the complete taxonomy of policy classes — "
        "PFA, CFA, VFA, DLA — with LaTeX formulations."
    )

with col2:
    st.subheader("🔬 Laboratory")
    st.markdown(
        "Step through episodes period-by-period. Watch how each policy "
        "reacts to real-time price, demand, and solar signals. "
        "Run multi-policy tournaments."
    )

with col3:
    st.subheader("🏗️ Architecture")
    st.markdown(
        "Code structure, interfaces, and design decisions. "
        "Everything you need to replicate this framework "
        "in your own project."
    )
