"""
Page 1 — Mathematical Formulation
===================================
Complete mathematical reference for the Voltown energy storage problem.
Powell's five elements, decision variables, constraints, policy taxonomy,
and risk formulations — all rendered with LaTeX.

No engine imports — this page is purely static educational content.
"""

import streamlit as st

st.set_page_config(
    page_title="Formulation — Voltown",
    page_icon="📐",
    layout="wide",
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Accent boxes ── */
.accent-box {
    border-left: 4px solid #3498db;
    background: rgba(52,152,219,0.05);
    padding: 0.9rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
    color: #333;
    font-size: 0.95rem;
    line-height: 1.6;
}

.accent-pfa { border-left-color: #e74c3c; background: rgba(231,76,60,0.05); }
.accent-cfa { border-left-color: #3498db; background: rgba(52,152,219,0.05); }
.accent-vfa { border-left-color: #2ecc71; background: rgba(46,204,113,0.05); }
.accent-dla { border-left-color: #9b59b6; background: rgba(155,89,182,0.05); }

/* ── Taxonomy table ── */
.taxonomy-table {
    width: 100%;
    border-collapse: collapse;
    margin: 1rem 0;
}
.taxonomy-table th, .taxonomy-table td {
    border: 1px solid #ddd;
    padding: 0.85rem 1rem;
    text-align: center;
    font-size: 0.92rem;
}
.taxonomy-table th {
    background: #f5f7fa;
    color: #333;
    font-weight: 700;
}
.taxonomy-table td {
    color: #333;
}
.taxonomy-table .pfa-cell { background: rgba(231,76,60,0.05); }
.taxonomy-table .cfa-cell { background: rgba(52,152,219,0.05); }
.taxonomy-table .vfa-cell { background: rgba(46,204,113,0.05); }
.taxonomy-table .dla-cell { background: rgba(155,89,182,0.05); }

/* ── Energy flow diagram ── */
.flow-diagram {
    font-family: 'Courier New', monospace;
    background: #f8f9fa;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    font-size: 0.88rem;
    line-height: 1.65;
    color: #333;
    overflow-x: auto;
    white-space: pre;
}
.flow-diagram .highlight {
    color: #3498db;
    font-weight: bold;
}

/* ── Connection card ── */
.connection-card {
    background: #f5f7fa;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #333;
    font-size: 0.93rem;
    line-height: 1.65;
}
.connection-card strong { color: #222; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📐 Formulation")
    st.caption("Complete mathematical model.")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown("# 📐 Mathematical Formulation")

st.divider()


# ═══════════════════════════════════════════════════════════════
# 1. PROBLEM DESCRIPTION
# ═══════════════════════════════════════════════════════════════

st.markdown("## 1 · The Voltown Narrative")

col_story, col_diagram = st.columns([3, 2], gap="large")

with col_story:
    st.markdown("""
    **Voltown** is a small community powered by a mix of grid electricity and rooftop solar panels,
    equipped with a shared **battery storage system** of capacity $K$ kWh.

    Every period (hour), the energy manager must decide how to route energy between
    three sources (grid, solar, battery) and two sinks (demand, battery) to **minimize total
    grid electricity cost** over the planning horizon.

    The challenge: **prices, demand, and solar** are all **uncertain**.
    The manager must decide *now* with only current information, while future conditions
    remain unknown. This is the essence of a **sequential decision problem under uncertainty**.
    """)

with col_diagram:
    st.markdown("#### 🔋 Energy Flow Diagram")
    st.image("assets/energy_flow.png", caption="5 decision variables · 4 constraints · ∞ uncertainty")

st.divider()


# ═══════════════════════════════════════════════════════════════
# 2. POWELL'S FIVE ELEMENTS
# ═══════════════════════════════════════════════════════════════

st.markdown("## 2 · Powell's Five Elements of Sequential Decision Problems")

st.markdown("""
<div class="accent-box">
Every sequential decision problem can be modeled as a Markov Decision Process (MDP).
Powell decomposes the MDP into exactly <strong>five</strong> universal elements.
</div>
""", unsafe_allow_html=True)

elements = [
    ("**State** $S_t$",
     r"S_t = (R_t, \, W_t) = \bigl(R_t, \; (p_t, d_t, s_t)\bigr)",
     "Battery level $R_t$ (physical state) and the exogenous observations $W_t$ = (price, demand, solar). "
     "The state contains *everything* needed to make a decision and compute the transition."),

    ("**Decision** $x_t$",
     r"x_t = (x_{gb},\; x_{gd},\; x_{sd},\; x_{sb},\; x_{bd})",
     "The five energy flows: grid→battery, grid→demand, solar→demand, solar→battery, battery→demand. "
     "Collectively these determine all costs and the next battery state."),

    ("**Exogenous Information** $W_{t+1}$",
     r"W_{t+1} = (p_{t+1},\; d_{t+1},\; s_{t+1})",
     "New price, demand, and solar availability revealed at the *start* of period $t+1$. "
     "This is information we learn — not control — and it's the source of all uncertainty."),

    ("**Transition** $S^M$",
     r"R_{t+1} = \text{clip}\bigl(R_t + x_{gb} + x_{sb} - x_{bd},\; 0,\; K\bigr)",
     "The battery evolves deterministically given the decision. The *only* randomness in the state "
     "comes from $W_{t+1}$, which enters directly. This is the \"physics\" of the system."),

    ("**Objective / Cost** $C$",
     r"\min_{\pi} \;\; \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t \; C_t(S_t, x_t^\pi)\right] "
     r"\quad \text{where} \quad C_t = p_t \cdot (x_{gb} + x_{gd})",
     "We pay the grid price for every kWh drawn from the grid. Solar is free. "
     "The goal: find the policy $\\pi$ that minimizes expected discounted cost over the full horizon."),
]

for label, latex, desc in elements:
    with st.container():
        st.markdown(f"#### {label}")
        st.latex(latex)
        st.markdown(desc)
        st.markdown("")

st.divider()


# ═══════════════════════════════════════════════════════════════
# 3. DECISION VARIABLES
# ═══════════════════════════════════════════════════════════════

st.markdown("## 3 · Decision Variables")

vars_data = [
    ("$x_{gb}$", "Grid → Battery",    "Energy purchased from grid and stored in battery",       "kWh", "Charging cost"),
    ("$x_{gd}$", "Grid → Demand",     "Energy purchased from grid and sent directly to demand", "kWh", "Direct grid usage cost"),
    ("$x_{sd}$", "Solar → Demand",    "Free solar energy used to meet demand",                  "kWh", "Always zero (free)"),
    ("$x_{sb}$", "Solar → Battery",   "Free solar energy stored in the battery for later",      "kWh", "Always zero (free)"),
    ("$x_{bd}$", "Battery → Demand",  "Previously stored energy discharged to meet demand",     "kWh", "Zero (already paid at charging)"),
]

col_hdr = st.columns([1, 2, 4, 1, 2])
headers = ["Variable", "Flow", "Description", "Units", "Cost Impact"]
for c, h in zip(col_hdr, headers):
    c.markdown(f"**{h}**")

st.markdown("---")
for var, flow, desc, units, cost in vars_data:
    cols = st.columns([1, 2, 4, 1, 2])
    cols[0].markdown(var)
    cols[1].markdown(f"`{flow}`")
    cols[2].markdown(desc)
    cols[3].markdown(units)
    cols[4].markdown(cost)

st.divider()


# ═══════════════════════════════════════════════════════════════
# 4. CONSTRAINTS
# ═══════════════════════════════════════════════════════════════

st.markdown("## 4 · Constraints")

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown("#### ☀️ Solar Capacity")
    st.latex(r"x_{sb} + x_{sd} \;\leq\; s_t")
    st.markdown("Total solar allocation cannot exceed available solar generation.")

    st.markdown("#### 🔋 Battery Upper Bound")
    st.latex(r"R_t + x_{gb} + x_{sb} - x_{bd} \;\leq\; K")
    st.markdown("Battery charge after all flows must not exceed physical capacity $K$.")

with c2:
    st.markdown("#### 🏠 Demand Satisfaction")
    st.latex(r"x_{bd} + x_{gd} + x_{sd} \;\geq\; d_t")
    st.markdown("All energy delivered must meet or exceed current period demand.")

    st.markdown("#### 🔋 Battery Lower Bound")
    st.latex(r"R_t + x_{gb} + x_{sb} - x_{bd} \;\geq\; 0")
    st.markdown("Battery cannot be discharged below zero.")

st.markdown("")
st.markdown("#### Non-negativity")
st.latex(r"x_{gb},\; x_{gd},\; x_{sd},\; x_{sb},\; x_{bd} \;\geq\; 0")

st.divider()


# ═══════════════════════════════════════════════════════════════
# 5. POLICY FORMULATIONS
# ═══════════════════════════════════════════════════════════════

st.markdown("## 5 · Policy Classes — The Four Universal Approaches")

st.markdown("""
<div class="accent-box">
A <strong>policy</strong> is a function that maps the current state to a decision:
&nbsp; <em>X<sub>t</sub><sup>π</sup> = X<sup>π</sup>(S<sub>t</sub>)</em>.
<br>Powell organizes policies into four classes based on how they use information and optimization.
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ─── PFA ─────────────────────────────────────────────────────
with st.expander("🔴  PFA — Policy Function Approximation", expanded=True):
    st.markdown("""
    <div class="accent-box accent-pfa">
    <strong>Core idea:</strong> Write down a parametric function that directly maps state → decision.
    No optimization at decision time — just evaluate the function. Think: if-then rules,
    lookup tables, or neural networks.
    </div>
    """, unsafe_allow_html=True)

    pfa1, pfa2 = st.columns(2, gap="large")

    with pfa1:
        st.markdown("##### PFA — Threshold Rule")
        st.latex(r"""
        x_{gb} =
        \begin{cases}
        \min(K - R_t,\; 5) & \text{if } p_t < \theta_1 \;\text{ and }\; R_t < \theta_2 \\
        0 & \text{otherwise}
        \end{cases}
        """)
        st.latex(r"""
        x_{bd} =
        \begin{cases}
        \min(R_t,\; d_t - x_{sd}) & \text{if } R_t > \theta_3 \\
        0 & \text{otherwise}
        \end{cases}
        """)
        st.markdown("""
        **Intuition:** "Charge when electricity is cheap, discharge when the battery is full enough."
        Three tunable parameters $\\theta_1, \\theta_2, \\theta_3$ define the boundaries.
        This is what an experienced plant operator might do intuitively.
        """)

    with pfa2:
        st.markdown("##### PFA — Linear Signal Rule")
        st.latex(r"""
        \text{charge\_signal} = \max\bigl(0,\;\; \theta_1 - \theta_2 \cdot p_t + \theta_3 \cdot (K - R_t)\bigr)
        """)
        st.latex(r"""
        \text{discharge\_signal} = \max\bigl(0,\;\; \theta_4 \cdot R_t - \theta_5 / p_t\bigr)
        """)
        st.markdown("""
        **Intuition:** Continuous signals that blend price and battery level. The `max(0, ·)` makes this
        a **ReLU activation** — connecting PFA to single-layer neural networks.
        The agent charges or discharges depending on which signal dominates.
        """)

    st.markdown("""
    **Advantages:** Lightning-fast evaluation; explainable; trivial to implement.

    **Limitations:** The functional form must be designed *a priori* — if the true optimal policy has a shape
    your parameterization can't capture, no amount of tuning will help.
    """)

# ─── CFA ─────────────────────────────────────────────────────
with st.expander("🔵  CFA — Cost Function Approximation"):
    st.markdown("""
    <div class="accent-box accent-cfa">
    <strong>Core idea:</strong> Solve a <em>single-period</em> optimization problem, but with a
    <strong>modified cost function</strong> that implicitly values the future. The magic is in the objective,
    not the forecast.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### CFA — Storage Value Buffer")
    st.latex(r"""
    \min_{x} \quad p_t \cdot (x_{gb} + x_{gd}) \;-\; \theta \cdot R_{t+1}
    """)
    st.latex(r"\text{subject to all constraints at period } t")

    st.markdown("""
    The term $-\\theta \\cdot R_{t+1}$ is a **proxy for future value**: every kWh in the battery at the end of
    this period is "worth" $\\theta$ \\$/kWh. This creates an incentive to store energy even when there's no
    immediate need — the buffer against future price spikes.

    The parameter $\\theta$ is *not* a forecast — it's an engineering knob that encodes the operator's
    intuition about how valuable stored energy is. Powell calls this the "art" of CFA.
    """)

    st.markdown("##### Gurobi LP at each period")
    st.code("""
m = gp.Model('CFA')
m.setObjective(
    price * (x_gb + x_gd) - theta * R_next,
    GRB.MINIMIZE
)
# → Solve, implement, advance to next period
    """, language="python")

    st.markdown("""
    **Advantages:** Handles constraints naturally (LP feasibility). More adaptive than PFA
    because the LP responds to the current state. Very common in practice (electricity markets, inventory).

    **Limitations:** The proxy $\\theta$ is hand-tuned. If the future doesn't "look like" $\\theta$ suggests,
    the policy can store too much or too little.
    """)

# ─── VFA ─────────────────────────────────────────────────────
with st.expander("🟢  VFA — Value Function Approximation"):
    st.markdown("""
    <div class="accent-box accent-vfa">
    <strong>Core idea:</strong> Learn a function $\\bar V(S)$ that estimates the value of being in each state.
    Then at each period, pick the action that minimizes immediate cost + discounted future value.
    This is the Bellman equation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### Bellman Optimality Equation")
    st.latex(r"""
    V^*(S_t) \;=\; \min_{x_t \in \mathcal{X}_t} \left[
        C_t(S_t, x_t) \;+\; \gamma \;\mathbb{E}\bigl[V^*(S_{t+1}) \;\big|\; S_t, x_t\bigr]
    \right]
    """)

    vfa1, vfa2 = st.columns(2, gap="large")

    with vfa1:
        st.markdown("##### Tabular Value Iteration (VI)")
        st.latex(r"""
        V_{k+1}(s) = \min_{a} \left[
            C(s, a) + \gamma \sum_{s'} P(s' \mid s, a)\, V_k(s')
        \right]
        """)
        st.markdown("""
        **Discretize** the state space into $(R, \\text{price\\_regime}, \\text{demand\\_regime})$
        tuples and sweep until convergence. Exact solution — but only feasible for small state spaces.
        """)

    with vfa2:
        st.markdown("##### Q-Learning (Model-Free)")
        st.latex(r"""
        Q(s, a) \;\leftarrow\; Q(s, a) + \alpha \bigl[
            C + \gamma \min_{a'} Q(s', a') - Q(s, a)
        \bigr]
        """)
        st.markdown("""
        No transition model needed — learn directly from sample episodes.
        Each $(s, a, c, s')$ transition improves the Q-table. The classic RL approach.
        """)

    st.markdown("")
    st.markdown("##### ADP — Linear Value Approximation")
    st.latex(r"""
    \bar{V}(S) \;=\; \boldsymbol{\phi}(S)^\top \mathbf{w}
    \qquad \text{with feature vector } \boldsymbol{\phi}(S) = \bigl[\, 1,\; R,\; R^2,\; R \cdot \mathbb{1}_{p>p_\text{med}}\,\bigr]
    """)
    st.markdown("""
    When the state space is too large for a table, approximate $V$ with a linear combination of features.
    The weights $\\mathbf{w}$ are updated online via temporal difference (TD) learning.
    This scales to continuous state spaces while retaining interpretability.
    """)

    st.markdown("""
    **Advantages:** Provably optimal (in the tabular case). Handles complex dynamics naturally.

    **Limitations:** Curse of dimensionality for tabular methods. Approximation quality depends heavily
    on feature engineering. Training requires many episodes.
    """)

# ─── DLA ─────────────────────────────────────────────────────
with st.expander("🟣  DLA — Direct Lookahead Approximation"):
    st.markdown("""
    <div class="accent-box accent-dla">
    <strong>Core idea:</strong> Build and solve an <em>approximate model of the future</em> at each decision point.
    Use a rolling-horizon optimization — solve a multi-period LP using forecasts,
    implement only the first period, then re-solve with updated information.
    This is <strong>Model Predictive Control (MPC)</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("##### DLA — Deterministic Lookahead (MPC)")
    st.latex(r"""
    \min_{x_0, \ldots, x_{H-1}} \quad \sum_{t=0}^{H-1} \hat{p}_t \cdot (x_{gb,t} + x_{gd,t})
    """)
    st.latex(r"""
    \text{s.t.} \quad B_{t+1} = B_t + x_{gb,t} + x_{sb,t} - x_{bd,t}, \quad
    0 \leq B_t \leq K, \quad
    \text{demand \& solar constraints } \forall\, t
    """)

    st.markdown("""
    Uses a **point forecast** $\\hat{W}$ for future periods. Simple but ignores forecast uncertainty —
    the plan is only as good as the forecast.
    """)

    st.markdown("##### DLA — Stochastic Lookahead (Two-Stage SP)")
    st.latex(r"""
    \min \quad \frac{1}{|\Omega|} \sum_{\omega \in \Omega}
        \sum_{t=0}^{H-1} p_t^\omega \cdot (x_{gb,t}^\omega + x_{gd,t}^\omega)
    """)
    st.latex(r"""
    \text{Non-anticipativity:} \quad x_{0}^{\omega} = x_{0}^{\omega'} \quad \forall\, \omega, \omega'
    """)

    st.markdown("""
    Generates a **scenario fan** of possible futures and optimizes the *expected* cost
    across all scenarios. Period-0 decisions are **forced to coincide** (non-anticipativity) —
    this is exactly the two-stage stochastic program from **Module 2**, embedded in a rolling-horizon loop.

    **Advantages:** Directly leverages forecast information. Handles constraints over multiple periods.
    The rolling-horizon approach means you automatically correct for forecast errors.

    **Limitations:** Computationally expensive (solve an LP every period). Quality depends on
    forecast accuracy and scenario coverage. The "horizon $H$" is itself a tunable parameter.
    """)

st.divider()


# ═══════════════════════════════════════════════════════════════
# 6. POWELL'S 2×2 TAXONOMY
# ═══════════════════════════════════════════════════════════════

st.markdown("## 6 · Powell's 2×2 Policy Taxonomy")

st.markdown("""
Powell's fundamental insight: all policies are classified along **two** dimensions —
whether they use an **explicit model of the future** (lookahead) and
whether they **optimize at decision time**.
""")

st.markdown("""
<table class="taxonomy-table">
  <tr>
    <th></th>
    <th></th>
    <th colspan="2">Optimization at Decision Time</th>
  </tr>
  <tr>
    <th></th>
    <th></th>
    <th>No Optimization</th>
    <th>With Optimization</th>
  </tr>
  <tr>
    <td rowspan="2" style="font-weight:700; color:#333; writing-mode: vertical-lr;
        text-orientation: mixed; transform: rotate(180deg); padding: 0.5rem;">
        Model of Future
    </td>
    <td style="font-weight:600; color:#333;">Implicit<br><span style="font-size:0.8rem; color:#666;">(no explicit lookahead)</span></td>
    <td class="pfa-cell">
      <strong style="color:#e74c3c;">PFA</strong><br>
      <span style="font-size:0.82rem;">Policy Function Approx.</span><br>
      <span style="font-size:0.78rem; color:#666;">Threshold rules, lookup tables,<br>neural network policies</span>
    </td>
    <td class="cfa-cell">
      <strong style="color:#3498db;">CFA</strong><br>
      <span style="font-size:0.82rem;">Cost Function Approx.</span><br>
      <span style="font-size:0.78rem; color:#666;">Modified objective LP,<br>buffer/proxy value terms</span>
    </td>
  </tr>
  <tr>
    <td style="font-weight:600; color:#333;">Explicit<br><span style="font-size:0.8rem; color:#666;">(lookahead model)</span></td>
    <td class="vfa-cell">
      <strong style="color:#2ecc71;">VFA</strong><br>
      <span style="font-size:0.82rem;">Value Function Approx.</span><br>
      <span style="font-size:0.78rem; color:#666;">Bellman equation, Q-learning,<br>ADP with features</span>
    </td>
    <td class="dla-cell">
      <strong style="color:#9b59b6;">DLA</strong><br>
      <span style="font-size:0.82rem;">Direct Lookahead Approx.</span><br>
      <span style="font-size:0.78rem; color:#666;">MPC / rolling horizon,<br>stochastic programming</span>
    </td>
  </tr>
</table>
""", unsafe_allow_html=True)

st.markdown("""
<div class="accent-box">
<strong>Key insight:</strong> No class is universally "best." PFA is fastest but least adaptive.
DLA is most computationally expensive but leverages the most information. VFA converges to optimality
with enough data but suffers the curse of dimensionality. CFA is the pragmatic middle ground
favored by practitioners. The <strong>Tournament page</strong> lets you see this empirically.
</div>
""", unsafe_allow_html=True)

st.divider()


# ═══════════════════════════════════════════════════════════════
# 7. RISK AS MODIFIER
# ═══════════════════════════════════════════════════════════════

st.markdown("## 7 · Risk as a Modifier — From Module 2 to Module 3")

r1, r2 = st.columns(2, gap="large")

with r1:
    st.markdown("#### 🔵 CFA + Risk: Volatility-Inflated Buffer")
    st.latex(r"""
    \theta_{\text{risk}} \;=\; \theta + \kappa \cdot \sigma_{\text{price}}
    """)
    st.markdown("""
    When price volatility $\\sigma_{\\text{price}}$ is high, the effective storage value increases.
    This makes the policy **store more energy** during volatile periods — a rational response
    that creates a buffer against tail-risk price spikes.

    The parameter $\\kappa$ is the **risk aversion coefficient**: how much extra buffer
    per unit of volatility.
    """)

with r2:
    st.markdown("#### 🟣 DLA + CVaR: Rockafellar-Uryasev")
    st.latex(r"""
    \min \quad \lambda \cdot \mathbb{E}[\text{cost}] \;+\;
    (1-\lambda) \cdot \text{CVaR}_\alpha[\text{cost}]
    """)
    st.latex(r"""
    \text{CVaR}_\alpha = \eta + \frac{1}{\alpha} \sum_\omega \frac{1}{|\Omega|}
    \max(0,\; \text{cost}^\omega - \eta)
    """)
    st.markdown("""
    The **Rockafellar-Uryasev linearization** from Module 2, now embedded in the
    rolling-horizon DLA. The auxiliary variable $\\eta$ represents the VaR threshold;
    the $z_\\omega = \\max(0, \\text{cost}^\\omega - \\eta)$ terms capture tail losses.

    $\\lambda \\in [0, 1]$ blends between pure expected-cost minimization ($\\lambda = 1$)
    and pure CVaR minimization ($\\lambda = 0$).
    """)

st.divider()


# ═══════════════════════════════════════════════════════════════
# 8. CONNECTIONS
# ═══════════════════════════════════════════════════════════════

st.markdown("## 8 · Connections to Modules 1 and 2")

m1, m2, m3 = st.columns(3, gap="large")

with m1:
    st.markdown("""
    <div class="connection-card">
    <strong>Module 1 — Probability</strong><br><br>
    The exogenous process $W_t = (p_t, d_t, s_t)$ is the stochastic model from Module 1.
    Price follows a sinusoidal base + Gaussian noise; demand and solar have similar structures.
    <br><br>
    <strong>Extremistan</strong> mode adds heavy-tailed shocks — connecting to the fat-tail
    discussions and why Gaussian assumptions fail in practice.
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="connection-card">
    <strong>Module 2 — Optimization Under Uncertainty</strong><br><br>
    DLA Stochastic is literally a <strong>two-stage stochastic program</strong> from Module 2,
    but solved <em>repeatedly</em> in a rolling fashion.
    <br><br>
    CFA uses a <strong>modified objective</strong> — the same idea as recourse costs
    but in a simpler, single-period form. CVaR enters identically via the
    Rockafellar-Uryasev reformulation.
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="connection-card">
    <strong>Module 3 — Sequential Decisions</strong><br><br>
    Module 3 asks: how do you make decisions <em>over time</em>, not just once?
    <br><br>
    The <strong>policy</strong> concept unifies everything — PFA (parametric rules),
    CFA (modified costs), VFA (Bellman-based learning), and DLA (rolling optimization).
    Each is a different answer to the same question: <em>what should I do now?</em>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.markdown("")
st.caption("📐 Formulation — Voltown Decision Lab")
