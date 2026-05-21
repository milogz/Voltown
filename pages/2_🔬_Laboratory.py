"""
🔬 Laboratory — Interactive Sequential Decision Playground
============================================================
THE core interactive page of the Voltown Dashboard.

Three tabs let students:
  1. Explore the exogenous world (W) generation
  2. Evaluate individual policies with rich diagnostics
  3. Run multi-policy tournaments with statistical comparison

Every chart is Plotly with the 'plotly_white' template for consistency.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine.world import WorldParams, generate_W, generate_W_extremistan
from engine.policies import POLICY_REGISTRY, CLASS_COLORS
from engine.simulator import run_episode, run_tournament, compute_tournament_stats
from engine.vfa_training import (
    value_iteration, q_learning, adp_train,
    estimate_transition_matrices,
)
from engine.mdp import transition, period_cost

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Voltown Laboratory",
    page_icon="🔬",
    layout="wide",
)

PLOTLY_TEMPLATE = "plotly_white"

# ── Sidebar: shared world parameters ────────────────────────────
st.sidebar.header("🌍 World Parameters")

T = st.sidebar.slider("T (horizon)", 12, 48, 24)
seed = st.sidebar.number_input("Seed", value=42, step=1)
demand_noise = st.sidebar.slider("Demand noise σ", 0.1, 1.0, 0.4, 0.05)
price_noise = st.sidebar.slider("Price noise σ", 0.002, 0.02, 0.008, 0.001,
                                 format="%.3f")
solar_noise = st.sidebar.slider("Solar noise σ", 0.1, 1.0, 0.4, 0.05)
cycles = st.sidebar.slider("Cycles per episode", 1, 4, 2)
world_mode = st.sidebar.radio("World mode", ["Mediocristan", "Extremistan"])

# Build world params from sidebar
wp = WorldParams(
    T=T,
    cycles=cycles,
    demand_noise=demand_noise,
    price_noise=price_noise,
    solar_noise=solar_noise,
)

gen_fn = generate_W if world_mode == "Mediocristan" else generate_W_extremistan
W = gen_fn(wp, seed=int(seed))

# ═════════════════════════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════════════════════════
tab_world, tab_policy, tab_tourney = st.tabs(
    ["🌍 World", "🤖 Policy Evaluation", "🏆 Tournament"]
)


# ─────────────────────────────────────────────────────────────────
# TAB 1 — WORLD VISUALIZATION
# ─────────────────────────────────────────────────────────────────
with tab_world:
    st.header("Exogenous Information — W")
    st.markdown(
        "The *world* generates demand, price, and solar data each period. "
        "Adjust sidebar parameters to see how the stochastic process changes."
    )

    # LaTeX formula for the exogenous info vector
    st.latex(r"W_{t+1} = \bigl(D_{t+1},\; p_{t+1},\; s_{t+1}\bigr)")

    periods = np.arange(T)
    colors = {"demand": "#e74c3c", "price": "#2ecc71", "solar": "#f39c12"}
    labels = {"demand": "Demand (kWh)", "price": "Price ($/kWh)",
              "solar": "Solar (kWh)"}

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=list(labels.values()),
        vertical_spacing=0.07,
    )

    # Generate a small ensemble for the fill-between "confidence" band
    n_band = 20
    bands = {k: [] for k in ["demand", "price", "solar"]}
    for s in range(n_band):
        W_sample = gen_fn(wp, seed=int(seed) + s + 1)
        for k in bands:
            bands[k].append(W_sample[k])

    for row_idx, var in enumerate(["demand", "price", "solar"], start=1):
        arr = np.array(bands[var])
        lo = np.percentile(arr, 10, axis=0)
        hi = np.percentile(arr, 90, axis=0)
        c = colors[var]

        # Band (upper edge)
        fig.add_trace(go.Scatter(
            x=periods, y=hi,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ), row=row_idx, col=1)

        # Band (lower edge, filled to upper)
        fig.add_trace(go.Scatter(
            x=periods, y=lo,
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=c.replace(")", ", 0.15)").replace("rgb", "rgba")
                      if "rgb" in c else f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.15)",
            showlegend=False, hoverinfo="skip",
        ), row=row_idx, col=1)

        # Main realized trace
        fig.add_trace(go.Scatter(
            x=periods, y=W[var],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color=c, width=2.5),
            name=labels[var],
            hovertemplate=f"t=%{{x}}<br>{var}=%{{y:.3f}}<extra></extra>",
        ), row=row_idx, col=1)

    fig.update_layout(
        height=700, template=PLOTLY_TEMPLATE,
        margin=dict(l=60, r=30, t=50, b=40),
        legend=dict(orientation="h", y=-0.05),
    )
    fig.update_xaxes(title_text="Period t", row=3, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Quick stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Demand", f"{W['demand'].mean():.2f} kWh")
    c2.metric("Mean Price", f"${W['price'].mean():.4f}/kWh")
    c3.metric("Mean Solar", f"{W['solar'].mean():.2f} kWh")


# ─────────────────────────────────────────────────────────────────
# TAB 2 — POLICY EVALUATION
# ─────────────────────────────────────────────────────────────────
with tab_policy:
    st.header("Single-Policy Deep Dive")

    # Policy selector in sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Policy Settings")
    policy_name = st.sidebar.selectbox("Policy", list(POLICY_REGISTRY.keys()))
    reg = POLICY_REGISTRY[policy_name]

    # Dynamic parameter sliders
    policy_params: dict = {}
    for pkey, pspec in reg["params"].items():
        val = st.sidebar.slider(
            pspec["label"],
            min_value=float(pspec["min"]),
            max_value=float(pspec["max"]),
            value=float(pspec["default"]),
            step=float(pspec["step"]),
            key=f"pol_{pkey}",
        )
        policy_params[pkey] = val

    run_btn = st.sidebar.button("▶ Run Episode", type="primary",
                                 use_container_width=True)

    if run_btn:
        # ── VFA training (if needed) ────────────────────────────
        training_kind = reg.get("needs_training")

        if training_kind == "vi":
            with st.spinner("Training — Value Iteration (estimating transitions + solving)…"):
                price_trans, demand_trans = estimate_transition_matrices(
                    gen_fn, wp, n_samples=80
                )
                V_star, pi_star, vi_hist = value_iteration(
                    price_trans, demand_trans, K=wp.K
                )
            policy_params["policy"] = pi_star
            st.success(f"VI converged in {len(vi_hist)} iterations.")

        elif training_kind == "ql":
            with st.spinner("Training — Q-Learning (500 episodes)…"):
                Q_table, ql_costs, ql_visits = q_learning(
                    gen_fn, wp, n_episodes=500
                )
            policy_params["Q"] = Q_table
            st.success("Q-Learning complete.")

        # ── Run episode ─────────────────────────────────────────
        ep_params = {**policy_params, "_seed": int(seed)}
        if reg.get("needs_forecast"):
            # Determine variant for simulator routing
            variant = "det" if "det" in policy_name.lower() else "stoch"
            ep_params["_variant"] = variant

        with st.spinner("Simulating episode…"):
            result = run_episode(
                reg["fn"], W, ep_params,
                K=wp.K, R_init=wp.R_init,
                needs_forecast=reg.get("needs_forecast", False),
            )

        st.session_state["ep_result"] = result
        st.session_state["ep_policy_name"] = policy_name

    # ── Display results (persisted in session_state) ─────────
    if "ep_result" in st.session_state:
        result = st.session_state["ep_result"]
        pname = st.session_state["ep_policy_name"]
        pcls = POLICY_REGISTRY[pname]["class"]
        pcolor = POLICY_REGISTRY[pname]["color"]

        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Cost", f"${result['total_cost']:.2f}")
        k2.metric("Avg Battery", f"{result['battery'].mean():.1f} kWh")
        k3.metric("Max Battery", f"{result['battery'].max():.1f} kWh")
        k4.metric("Min Battery", f"{result['battery'].min():.1f} kWh")

        st.markdown(f"**Policy:** `{pname}` · **Class:** `{pcls}`")

        # ── 4-panel visualization ───────────────────────────────
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Battery Level R_t",
                "Decision Anatomy",
                "Cost Timeline",
                "Energy Flow Snapshot",
            ],
            vertical_spacing=0.13,
            horizontal_spacing=0.10,
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"secondary_y": True}, {"type": "xy"}],
            ],
        )

        periods = np.arange(len(result["costs"]))
        batt = result["battery"]

        # Panel 1: Battery level (area)
        fig2.add_trace(go.Scatter(
            x=np.arange(len(batt)), y=batt,
            fill="tozeroy",
            fillcolor=pcolor.replace(")", ",0.2)") if "rgb" in pcolor
                       else f"rgba({int(pcolor[1:3],16)},{int(pcolor[3:5],16)},{int(pcolor[5:7],16)},0.2)",
            line=dict(color=pcolor, width=2),
            name="Battery",
            hovertemplate="t=%{x}<br>R=%{y:.2f} kWh<extra></extra>",
        ), row=1, col=1)

        # Capacity line
        fig2.add_trace(go.Scatter(
            x=[0, len(batt) - 1], y=[wp.K, wp.K],
            mode="lines",
            line=dict(color="#999", width=1, dash="dot"),
            name="Capacity",
            showlegend=False,
        ), row=1, col=1)

        # Panel 2: Decision anatomy (grouped bars)
        dec_keys = ["x_gb", "x_sd", "x_sb", "x_bd", "x_gd"]
        dec_colors = {
            "x_gb": "#3498db",  # grid→battery
            "x_sd": "#f39c12",  # solar→demand
            "x_sb": "#2ecc71",  # solar→battery
            "x_bd": "#e74c3c",  # battery→demand
            "x_gd": "#95a5a6",  # grid→demand
        }
        dec_labels = {
            "x_gb": "Grid→Bat",
            "x_sd": "Solar→Dem",
            "x_sb": "Solar→Bat",
            "x_bd": "Bat→Dem",
            "x_gd": "Grid→Dem",
        }
        for dk in dec_keys:
            vals = [d[dk] for d in result["decisions"]]
            fig2.add_trace(go.Bar(
                x=periods, y=vals,
                name=dec_labels[dk],
                marker_color=dec_colors[dk],
                hovertemplate=f"{dec_labels[dk]}=%{{y:.2f}}<extra></extra>",
            ), row=1, col=2)

        # Panel 3: Cost timeline (bars + cumulative line on secondary y)
        cum_cost = np.cumsum(result["costs"])
        fig2.add_trace(go.Bar(
            x=periods, y=result["costs"],
            name="Period Cost",
            marker_color="#e67e22",
            opacity=0.7,
            hovertemplate="t=%{x}<br>Cost=$%{y:.3f}<extra></extra>",
        ), row=2, col=1, secondary_y=False)

        fig2.add_trace(go.Scatter(
            x=periods, y=cum_cost,
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(color="#e74c3c", width=2.5),
            name="Cumulative",
            hovertemplate="t=%{x}<br>Cum=$%{y:.3f}<extra></extra>",
        ), row=2, col=1, secondary_y=True)

        fig2.update_yaxes(title_text="Period ($)", row=2, col=1,
                          secondary_y=False)
        fig2.update_yaxes(title_text="Cumulative ($)", row=2, col=1,
                          secondary_y=True)

        # Panel 4: Energy flow network snapshot
        # We'll build it outside the main figure — use a separate Plotly chart
        # For now, put a placeholder trace
        fig2.add_trace(go.Scatter(
            x=[0], y=[0], mode="text",
            text=["See flow diagram below ↓"],
            textfont=dict(size=14, color="#333"),
            showlegend=False,
        ), row=2, col=2)

        fig2.update_layout(
            height=750,
            template=PLOTLY_TEMPLATE,
            barmode="group",
            margin=dict(l=50, r=30, t=50, b=40),
            legend=dict(orientation="h", y=-0.08, font=dict(size=10)),
        )

        st.plotly_chart(fig2, use_container_width=True)

        # ── Energy Flow Network Snapshot ────────────────────────
        st.subheader("⚡ Energy Flow Snapshot")
        snap_t = st.slider(
            "Select period for flow diagram", 0, len(result["costs"]) - 1, 0,
            key="snap_period",
        )

        dec = result["decisions"][snap_t]
        W_snap = {k: v[snap_t] for k, v in W.items()}

        # Node positions for the energy flow graph
        node_labels = ["Grid", "Solar", "Battery", "Demand"]
        node_x = [0, 0, 1, 2]
        node_y = [1, 0, 0.5, 0.5]
        node_colors = ["#3498db", "#f39c12", "#2ecc71", "#e74c3c"]
        node_sizes = [35, 35, 40, 35]

        # Edges: (from_idx, to_idx, value, label)
        edges = [
            (0, 2, dec["x_gb"], "x_gb"),  # grid → battery
            (0, 3, dec["x_gd"], "x_gd"),  # grid → demand
            (1, 3, dec["x_sd"], "x_sd"),  # solar → demand
            (1, 2, dec["x_sb"], "x_sb"),  # solar → battery
            (2, 3, dec["x_bd"], "x_bd"),  # battery → demand
        ]

        fig_flow = go.Figure()

        # Draw edges
        max_flow = max(e[2] for e in edges) if any(e[2] > 0 for e in edges) else 1.0
        edge_color_map = {
            "x_gb": "#3498db", "x_gd": "#95a5a6",
            "x_sd": "#f39c12", "x_sb": "#2ecc71",
            "x_bd": "#e74c3c",
        }

        for fr, to, val, label in edges:
            if val < 0.001:
                continue
            width = max(1.5, 12 * val / max(max_flow, 0.01))
            mid_x = (node_x[fr] + node_x[to]) / 2
            mid_y = (node_y[fr] + node_y[to]) / 2

            fig_flow.add_trace(go.Scatter(
                x=[node_x[fr], mid_x, node_x[to]],
                y=[node_y[fr], mid_y, node_y[to]],
                mode="lines",
                line=dict(width=width, color=edge_color_map.get(label, "#888")),
                hoverinfo="skip",
                showlegend=False,
            ))
            # Edge label
            fig_flow.add_annotation(
                x=mid_x, y=mid_y,
                text=f"<b>{val:.2f}</b>",
                showarrow=False,
                font=dict(size=11, color="#333"),
                bgcolor="rgba(255,255,255,0.85)",
                borderpad=2,
            )

        # Draw nodes
        fig_flow.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            marker=dict(size=node_sizes, color=node_colors,
                        line=dict(width=2, color="#333")),
            text=node_labels,
            textposition="top center",
            textfont=dict(size=14, color="#333"),
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        ))

        # Info annotations
        fig_flow.add_annotation(
            x=0, y=1.25, text=f"p={W_snap['price']:.4f}$/kWh",
            showarrow=False, font=dict(size=10, color="#2ecc71"),
        )
        fig_flow.add_annotation(
            x=0, y=-0.25, text=f"s={W_snap['solar']:.2f} kWh",
            showarrow=False, font=dict(size=10, color="#f39c12"),
        )
        fig_flow.add_annotation(
            x=1, y=0.2, text=f"R={batt[snap_t]:.1f} kWh",
            showarrow=False, font=dict(size=10, color="#2ecc71"),
        )
        fig_flow.add_annotation(
            x=2, y=0.25, text=f"d={W_snap['demand']:.2f} kWh",
            showarrow=False, font=dict(size=10, color="#e74c3c"),
        )

        fig_flow.update_layout(
            height=350,
            template=PLOTLY_TEMPLATE,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-0.5, 2.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       range=[-0.5, 1.5]),
            margin=dict(l=20, r=20, t=20, b=20),
        )

        st.plotly_chart(fig_flow, use_container_width=True)

    else:
        st.info("👈 Configure a policy in the sidebar and click **▶ Run Episode**.")


# ─────────────────────────────────────────────────────────────────
# TAB 3 — TOURNAMENT
# ─────────────────────────────────────────────────────────────────
with tab_tourney:
    st.header("Multi-Policy Tournament")

    st.sidebar.markdown("---")
    st.sidebar.header("🏆 Tournament Settings")

    # Policy checkboxes (exclude risk variants by default)
    default_exclude = {"CFA Risk-Averse", "DLA CVaR"}
    selected_policies = {}
    for pname_t in POLICY_REGISTRY:
        default_on = pname_t not in default_exclude
        if st.sidebar.checkbox(pname_t, value=default_on, key=f"tourn_{pname_t}"):
            selected_policies[pname_t] = POLICY_REGISTRY[pname_t]

    n_episodes = st.sidebar.slider("Episodes per policy", 10, 100, 30,
                                    key="tourn_n_ep")

    run_tourn = st.sidebar.button("🏁 Run Tournament", type="primary",
                                   use_container_width=True)

    if run_tourn and selected_policies:
        # ── Build configs ───────────────────────────────────────
        policy_configs = {}
        for pname_t, preg in selected_policies.items():
            cfg = {
                "fn": preg["fn"],
                "params": {k: v["default"] for k, v in preg["params"].items()},
                "needs_forecast": preg.get("needs_forecast", False),
                "stochastic": "Stoch" in pname_t or "CVaR" in pname_t,
            }

            # VFA training
            training_kind = preg.get("needs_training")
            if training_kind == "vi":
                with st.spinner(f"Training {pname_t}…"):
                    pt, dt = estimate_transition_matrices(gen_fn, wp, n_samples=80)
                    _, pi_star, _ = value_iteration(pt, dt, K=wp.K)
                cfg["params"]["policy"] = pi_star
            elif training_kind == "ql":
                with st.spinner(f"Training {pname_t}…"):
                    Q_table, _, _ = q_learning(gen_fn, wp, n_episodes=500)
                cfg["params"]["Q"] = Q_table

            policy_configs[pname_t] = cfg

        with st.spinner("Running tournament…"):
            df_results = run_tournament(
                policy_configs, gen_fn, wp,
                n_episodes=n_episodes, seed_start=int(seed),
            )
            stats = compute_tournament_stats(df_results)

        st.session_state["tourn_df"] = df_results
        st.session_state["tourn_stats"] = stats
        st.session_state["tourn_selected"] = list(selected_policies.keys())

    # ── Display tournament results ──────────────────────────────
    if "tourn_df" in st.session_state:
        df_results = st.session_state["tourn_df"]
        stats = st.session_state["tourn_stats"]
        tourn_names = st.session_state["tourn_selected"]

        st.subheader("📊 Summary Statistics")
        st.dataframe(
            stats.style.format("{:.4f}").background_gradient(
                cmap="RdYlGn_r", subset=["Mean"]
            ),
            use_container_width=True,
        )

        # ── Side-by-side: box plot + bar chart ──────────────────
        col_box, col_bar = st.columns(2)

        with col_box:
            fig_box = go.Figure()
            for pname_t in tourn_names:
                preg = POLICY_REGISTRY.get(pname_t, {})
                pcolor = preg.get("color", "#888")
                fig_box.add_trace(go.Box(
                    y=df_results[pname_t],
                    name=pname_t,
                    marker_color=pcolor,
                    boxmean=True,
                ))
            fig_box.update_layout(
                title="Cost Distribution by Policy",
                yaxis_title="Total Episode Cost ($)",
                template=PLOTLY_TEMPLATE,
                height=450,
                showlegend=False,
                margin=dict(l=50, r=20, t=50, b=80),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_box, use_container_width=True)

        with col_bar:
            fig_bar = go.Figure()
            means = stats["Mean"].values
            stds = stats["Std"].values
            bar_colors = [
                POLICY_REGISTRY.get(n, {}).get("color", "#888")
                for n in stats.index
            ]
            fig_bar.add_trace(go.Bar(
                x=list(stats.index),
                y=means,
                error_y=dict(type="data", array=stds, visible=True,
                             color="#333", thickness=1.5),
                marker_color=bar_colors,
                hovertemplate="%{x}<br>Mean=$%{y:.4f} ± %{error_y.array:.4f}<extra></extra>",
            ))
            fig_bar.update_layout(
                title="Mean Cost ± Std",
                yaxis_title="Mean Total Cost ($)",
                template=PLOTLY_TEMPLATE,
                height=450,
                margin=dict(l=50, r=20, t=50, b=80),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # ── Trajectory comparisons for episode #0 ───────────────
        st.subheader("📈 Episode #0 — Trajectory Comparison")

        W_ep0 = gen_fn(wp, seed=int(seed))
        fig_batt = go.Figure()
        fig_cum = go.Figure()

        for pname_t in tourn_names:
            preg = POLICY_REGISTRY.get(pname_t, {})
            pcolor = preg.get("color", "#888")

            cfg = {
                k: v["default"]
                for k, v in preg.get("params", {}).items()
            }
            cfg["_seed"] = int(seed)

            training_kind = preg.get("needs_training")
            if training_kind == "vi":
                pt, dt = estimate_transition_matrices(gen_fn, wp, n_samples=80)
                _, pi_star, _ = value_iteration(pt, dt, K=wp.K)
                cfg["policy"] = pi_star
            elif training_kind == "ql":
                Q_table, _, _ = q_learning(gen_fn, wp, n_episodes=500)
                cfg["Q"] = Q_table

            if preg.get("needs_forecast"):
                variant = "det" if "det" in pname_t.lower() else "stoch"
                cfg["_variant"] = variant

            r = run_episode(
                preg["fn"], W_ep0, cfg,
                K=wp.K, R_init=wp.R_init,
                needs_forecast=preg.get("needs_forecast", False),
            )

            fig_batt.add_trace(go.Scatter(
                x=np.arange(len(r["battery"])),
                y=r["battery"],
                mode="lines",
                name=pname_t,
                line=dict(color=pcolor, width=2),
                hovertemplate=f"{pname_t}<br>t=%{{x}}<br>R=%{{y:.2f}}<extra></extra>",
            ))

            fig_cum.add_trace(go.Scatter(
                x=np.arange(len(r["costs"])),
                y=np.cumsum(r["costs"]),
                mode="lines",
                name=pname_t,
                line=dict(color=pcolor, width=2),
                hovertemplate=f"{pname_t}<br>t=%{{x}}<br>Cum=$%{{y:.3f}}<extra></extra>",
            ))

        # Battery capacity reference
        fig_batt.add_hline(y=wp.K, line_dash="dot", line_color="#999",
                           annotation_text="Capacity", opacity=0.5)

        fig_batt.update_layout(
            title="Battery Level Trajectories (Episode #0)",
            xaxis_title="Period t",
            yaxis_title="Battery R (kWh)",
            template=PLOTLY_TEMPLATE,
            height=400,
            legend=dict(orientation="h", y=-0.15),
        )
        fig_cum.update_layout(
            title="Cumulative Cost Trajectories (Episode #0)",
            xaxis_title="Period t",
            yaxis_title="Cumulative Cost ($)",
            template=PLOTLY_TEMPLATE,
            height=400,
            legend=dict(orientation="h", y=-0.15),
        )

        tc1, tc2 = st.columns(2)
        with tc1:
            st.plotly_chart(fig_batt, use_container_width=True)
        with tc2:
            st.plotly_chart(fig_cum, use_container_width=True)

    elif not run_tourn:
        st.info("👈 Select policies and click **🏁 Run Tournament**.")
