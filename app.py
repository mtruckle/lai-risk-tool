"""
LAI Risk Management Tool — Streamlit App (v2)

Run locally:  streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
import plotly.graph_objects as go
import plotly.express as px

from db import init_db, insert_position, update_position, delete_position, get_positions
from bbg_parser import parse_bbg_option
from market_data import get_spot, realized_vol, compute_beta
from analytics import (
    enrich_position, portfolio_summary, risk_curve,
    expected_return_table, stress_scenario, steady_state_gross,
    sensitivity_vol_sweep, sensitivity_spot_sweep,
    STRESS_SCENARIOS,
)
from vol_decay import LAI_ETF_MAP, is_lai_etf, get_lai_info


# -------------------------------------------------------------------------
st.set_page_config(
    page_title="LAI Risk Tool",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# -------------------------------------------------------------------------
BUILD_VERSION = "2026-04-20-r8"

st.markdown(f"""
<style>
    .main {{ padding-top: 1rem; }}
    h1, h2, h3 {{ color: #0A1931; font-family: Georgia, serif; }}
    [data-testid="stMetricValue"] {{ font-size: 24px; }}
    .build-badge {{
        position: fixed;
        top: 0.5rem;
        right: 1rem;
        z-index: 999999;
        background: #0A1931;
        color: #D4A017;
        padding: 0.3rem 0.8rem;
        border-radius: 6px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 12px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }}
</style>
<div class="build-badge">Build: {BUILD_VERSION}</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# HELPERS
def fmt_money(x, show_cents=False):
    """Format $ amount with comma separators. Minus sign before $ for negatives.
    Examples: $1,310,000  |  -$1,310,000  |  $5.25  |  -$0.75"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        x = float(x)
    except (ValueError, TypeError):
        return "—"
    if show_cents:
        if x < 0:
            return f"-${abs(x):,.2f}"
        return f"${x:,.2f}"
    if x < 0:
        return f"-${abs(x):,.0f}"
    return f"${x:,.0f}"


def fmt_int(x):
    """Integer with comma separators; preserves negative sign."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x):,.0f}"
    except (ValueError, TypeError):
        return str(x)


def fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    try:
        return f"{float(x)*100:.{digits}f}%"
    except (ValueError, TypeError):
        return "—"


# -------------------------------------------------------------------------
# SIDEBAR
st.sidebar.title("LAI Risk Tool")
page = st.sidebar.radio("Page", [
    "📋 Positions",
    "📊 Summary",
    "📈 Risk Curve",
    "💰 Expected Return",
    "⚠️ Stress Scenario",
    "⚙️ Settings / Data Refresh",
])

st.sidebar.markdown("---")
gross_limit = st.sidebar.number_input(
    "Gross Capital Limit ($M)",
    value=150.0, min_value=1.0, step=10.0
) * 1_000_000

# Underlying 1Y return assumption for vol decay is hardcoded to 0%
# This isolates the pure vol-decay alpha from any assumed directional drift
r_1y_assumption = 0.0

st.sidebar.caption(f"""
Data source: Yahoo Finance (15-min delayed)  
Risk-free: 4.0% | Div yield: 0.5%  
MC uses historical bootstrap (inherits historical drift)  
**Build:** {BUILD_VERSION}
""")


# -------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_enriched_positions(_trigger_refresh=0):
    raw = get_positions()
    enriched = [enrich_position(p) for p in raw]
    return enriched


def refresh_data():
    get_enriched_positions.clear()


# =========================================================================
# PAGE: POSITIONS
# =========================================================================
if page == "📋 Positions":
    st.title("Positions")
    
    tab1, tab2, tab3 = st.tabs(["Add Position", "Open Positions", "Closed Positions"])
    
    with tab1:
        st.subheader("Add a new position")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            bbg_ticker = st.text_input(
                "BBG Ticker",
                placeholder="e.g. 'SOXS US 01/21/28 P4' or 'SOXS1 01/15/27 P2' or 'SQQQ' (stock)",
                help="Bloomberg format for options. For stock shorts, just the ticker.",
            )
        
        with col2:
            parsed = None
            if bbg_ticker:
                try:
                    parsed = parse_bbg_option(bbg_ticker)
                    st.success(f"✓ Parsed: {parsed['instrument_type']}")
                except ValueError as e:
                    st.error(str(e))
        
        if parsed:
            with st.expander("Parsed details (edit if needed)", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.write(f"**Underlying:** {parsed['underlying']}")
                    if parsed["instrument_type"] == "OPTION":
                        st.write(f"**Strike:** {fmt_money(parsed['strike'], show_cents=True)}")
                with c2:
                    if parsed["instrument_type"] == "OPTION":
                        st.write(f"**Expiry:** {parsed['expiry']}")
                        st.write(f"**Type:** {'Call' if parsed['option_type']=='C' else 'Put'}")
                with c3:
                    st.write(f"**Root:** {parsed.get('option_root') or '—'}")
                    mult = st.number_input(
                        "Multiplier (shares per contract)",
                        value=int(parsed["multiplier"]), min_value=1, step=1,
                        help="100 normally; 5 for post-reverse-split options like SOXS1",
                    )
                    parsed["multiplier"] = mult
            
            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                quantity = st.number_input(
                    "Quantity (+ long / − short)",
                    value=10, step=1,
                    help="Contracts for options, shares for stock. Negative = short."
                )
            with c2:
                entry_price = st.number_input(
                    "Entry Price",
                    value=1.00, min_value=0.00, step=0.01, format="%.2f"
                )
            with c3:
                entry_date = st.date_input("Entry Date", value=date.today())
            with c4:
                notes = st.text_input("Notes (optional)")
            
            if st.button("Add Position", type="primary"):
                pos = {
                    **parsed,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "entry_date": entry_date.isoformat(),
                    "status": "OPEN",
                    "notes": notes or "",
                    "exit_price": None,
                    "exit_date": None,
                }
                new_id = insert_position(pos)
                refresh_data()
                st.success(f"Position #{new_id} added ✓")
                st.rerun()
    
    with tab2:
        st.subheader("Open positions")
        enriched = get_enriched_positions()
        open_pos = [p for p in enriched if p["status"] == "OPEN"]
        
        if not open_pos:
            st.info("No open positions. Add one in the 'Add Position' tab.")
        else:
            df_rows = []
            for p in open_pos:
                df_rows.append({
                    "ID": p["id"],
                    "Ticker": p["bbg_ticker"],
                    "Type": p["instrument_type"],
                    "Qty": fmt_int(p["quantity"]),
                    "Mult": fmt_int(p["multiplier"]),
                    "Entry": fmt_money(p["entry_price"], show_cents=True),
                    "Mid/Spot": fmt_money(p.get("mid_price"), show_cents=True),
                    "IV": fmt_pct(p.get("iv"), 1) if p.get("iv") else "—",
                    "Delta": f"{p.get('delta', 0):.3f}",
                    "Market Value": fmt_money(p.get("market_value", 0)),
                    "Unrealized P&L": fmt_money(p.get("unrealized_pnl", 0)),
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            st.markdown("---")
            st.subheader("Close or delete a position")
            col1, col2 = st.columns([1, 2])
            with col1:
                ids = [p["id"] for p in open_pos]
                pos_id = st.selectbox("Position ID", ids, key="close_id")
            selected = next((p for p in open_pos if p["id"] == pos_id), None)
            if selected:
                with col2:
                    st.write(f"**{selected['bbg_ticker']}** — Qty: {fmt_int(selected['quantity'])}, Entry: {fmt_money(selected['entry_price'], show_cents=True)}")
                
                cc1, cc2, cc3, cc4 = st.columns(4)
                with cc1:
                    exit_price = st.number_input(
                        "Exit Price", value=float(selected.get("mid_price") or selected["entry_price"]),
                        min_value=0.0, step=0.01, format="%.2f"
                    )
                with cc2:
                    exit_date = st.date_input("Exit Date", value=date.today(), key="exit_date")
                with cc3:
                    if st.button("✅ Close Position", type="primary"):
                        update_position(pos_id, {
                            "status": "CLOSED",
                            "exit_price": exit_price,
                            "exit_date": exit_date.isoformat(),
                        })
                        refresh_data()
                        st.success(f"Position #{pos_id} closed.")
                        st.rerun()
                with cc4:
                    if st.button("🗑️ Delete", help="Permanently delete"):
                        delete_position(pos_id)
                        refresh_data()
                        st.warning(f"Position #{pos_id} deleted.")
                        st.rerun()
    
    with tab3:
        st.subheader("Closed positions (realized P&L)")
        enriched = get_enriched_positions()
        closed_pos = [p for p in enriched if p["status"] == "CLOSED"]
        
        if not closed_pos:
            st.info("No closed positions yet.")
        else:
            df_rows = []
            total_realized = 0
            for p in closed_pos:
                rp = p.get("realized_pnl", 0)
                total_realized += rp
                df_rows.append({
                    "ID": p["id"],
                    "Ticker": p["bbg_ticker"],
                    "Qty": fmt_int(p["quantity"]),
                    "Entry": fmt_money(p["entry_price"], show_cents=True),
                    "Entry Date": p["entry_date"],
                    "Exit": fmt_money(p.get("exit_price"), show_cents=True),
                    "Exit Date": p.get("exit_date"),
                    "Realized P&L": fmt_money(rp),
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            st.metric("Total Realized P&L", fmt_money(total_realized))

# =========================================================================
# PAGE: SUMMARY
# =========================================================================
elif page == "📊 Summary":
    st.title("Portfolio Summary")
    
    enriched = get_enriched_positions()
    summary = portfolio_summary(enriched)
    
    if summary["n_open_positions"] == 0:
        st.info("No open positions. Add positions on the Positions page.")
    else:
        st.subheader("Portfolio metrics")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Option Premium", fmt_money(summary["total_premium"]))
        with c2:
            st.metric("Gross Exposure", fmt_money(summary["gross_exposure"]))
            st.caption(f"Limit: {fmt_money(gross_limit)}")
        with c3:
            st.metric("Delta Exposure", fmt_money(summary["total_delta_exp"]))
        with c4:
            st.metric("Beta-Adj Exposure", fmt_money(summary["total_beta_exp"]))
        
        st.subheader("Portfolio Greeks")
        g1, g2, g3, g4 = st.columns(4)
        with g1:
            st.metric("Total Delta (shares)", fmt_int(summary['total_delta']))
            st.caption("Net share-equivalent exposure")
        with g2:
            st.metric("$ Gamma", fmt_int(summary['total_gamma']))
            st.caption("Change in delta-shares per $1 underlying move")
        with g3:
            theta_day = summary["total_theta"] / 365
            st.metric("Theta (per day)", fmt_money(theta_day))
            st.caption("Daily time decay in $")
        with g4:
            vega_vol_pt = summary["total_vega"] / 100
            st.metric("Vega (per vol pt)", fmt_money(vega_vol_pt))
            st.caption("P&L per 1 vol point change")
        
        st.markdown("---")
        
        st.subheader("P&L")
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric("Unrealized P&L", fmt_money(summary["unrealized_pnl"]))
        with p2:
            st.metric("Realized P&L", fmt_money(summary["realized_pnl"]))
        with p3:
            st.metric("Total P&L", fmt_money(summary["total_pnl"]))
        
        st.markdown("---")
        
        st.subheader("By underlying")
        by_und = summary["by_underlying"]
        if by_und:
            rows = []
            for u, data in sorted(by_und.items()):
                rows.append({
                    "Underlying": u,
                    "# Positions": data["n_positions"],
                    "Option Premium": fmt_money(data["option_premium"]),
                    "Gross Notional": fmt_money(data["gross_notional"]),
                    "Delta Exposure": fmt_money(data["delta_exposure"]),
                    "Beta-Adj Exposure": fmt_money(data["beta_adj_exposure"]),
                    "Unrealized P&L": fmt_money(data["unrealized_pnl"]),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True)
            
            if len(by_und) > 1:
                pie_df = pd.DataFrame([
                    {"Underlying": u, "Premium": abs(d["option_premium"])}
                    for u, d in by_und.items() if abs(d["option_premium"]) > 0
                ])
                if len(pie_df) > 0:
                    fig = px.pie(pie_df, values="Premium", names="Underlying",
                                 title="Option Premium by Underlying")
                    st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# PAGE: RISK CURVE
# =========================================================================
elif page == "📈 Risk Curve":
    st.title("Risk Curve — SPX Scenario Analysis")
    st.caption(
        "Portfolio sensitivity to SPX moves (-15% to +15% in 2.5% increments). "
        "Options repriced via Black-Scholes assuming IV unchanged. "
        "LAI ETFs move at leverage × underlying beta × SPX move."
    )
    
    enriched = get_enriched_positions()
    if len([p for p in enriched if p["status"] == "OPEN"]) == 0:
        st.info("No open positions.")
    else:
        with st.spinner("Computing risk curve..."):
            rc = risk_curve(enriched)
        
        summary_df = rc["summary_df"]
        spx_x = summary_df["SPX Move (decimal)"] * 100
        
        # ========== SECTION 1: Exposure curve ==========
        st.subheader("Exposure under SPX shock")
        st.caption("Gross, net, and net beta-adjusted delta exposures as SPX moves.")
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=spx_x, y=summary_df["Gross Delta Exposure"]/1e6,
            mode="lines+markers", name="Gross Delta Exposure",
            line=dict(color="#0A1931", width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x:.1f}% SPX</b><br>Gross: $%{y:,.1f}M<extra></extra>",
        ))
        fig1.add_trace(go.Scatter(
            x=spx_x, y=summary_df["Net Delta Exposure"]/1e6,
            mode="lines+markers", name="Net Delta Exposure",
            line=dict(color="#D4A017", width=3),
            marker=dict(size=7),
            hovertemplate="<b>%{x:.1f}% SPX</b><br>Net Δ: $%{y:,.1f}M<extra></extra>",
        ))
        fig1.add_trace(go.Scatter(
            x=spx_x, y=summary_df["Net Beta-Adj Exposure"]/1e6,
            mode="lines+markers", name="Net Beta-Adj Exposure",
            line=dict(color="#8B0000", width=3, dash="dash"),
            marker=dict(size=7),
            hovertemplate="<b>%{x:.1f}% SPX</b><br>Net βAdj: $%{y:,.1f}M<extra></extra>",
        ))
        fig1.add_hline(y=0, line_dash="dot", line_color="#AAA")
        fig1.add_vline(x=0, line_dash="dot", line_color="#AAA")
        fig1.update_layout(
            xaxis_title="SPX Move (%)",
            yaxis_title="Exposure ($M)",
            height=500,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.3,
                xanchor="center", x=0.5,
            ),
            margin=dict(b=100),
        )
        fig1.update_xaxes(ticksuffix="%", gridcolor="#EEE")
        fig1.update_yaxes(tickprefix="$", ticksuffix="M", gridcolor="#EEE")
        st.plotly_chart(fig1, use_container_width=True)
        
        st.markdown("---")
        
        # ========== SECTION 2: Delta-Adj Exposure with granularity ==========
        st.subheader("Delta-Adjusted Exposure under SPX shock")
        
        granularity_a = st.selectbox(
            "Granularity",
            ["Total Portfolio", "By Beta Direction (Long / Short Beta)", "By Instrument"],
            key="granularity_delta",
        )
        
        fig2 = go.Figure()
        
        if granularity_a == "Total Portfolio":
            fig2.add_trace(go.Scatter(
                x=spx_x, y=rc["delta_exp_by_direction"]["Total"]/1e6,
                mode="lines+markers", name="Total Portfolio",
                line=dict(color="#0A1931", width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Total: $%{y:,.1f}M<extra></extra>",
            ))
        elif granularity_a == "By Beta Direction (Long / Short Beta)":
            fig2.add_trace(go.Scatter(
                x=spx_x, y=rc["delta_exp_by_direction"]["Long Beta"]/1e6,
                mode="lines+markers", name="Long Beta (short SQQQ/SOXS)",
                line=dict(color="#2E8B57", width=3),
                marker=dict(size=7),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Long β: $%{y:,.1f}M<extra></extra>",
            ))
            fig2.add_trace(go.Scatter(
                x=spx_x, y=rc["delta_exp_by_direction"]["Short Beta"]/1e6,
                mode="lines+markers", name="Short Beta (QQQ/SOXX puts)",
                line=dict(color="#B22222", width=3),
                marker=dict(size=7),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Short β: $%{y:,.1f}M<extra></extra>",
            ))
            fig2.add_trace(go.Scatter(
                x=spx_x, y=rc["delta_exp_by_direction"]["Total"]/1e6,
                mode="lines+markers", name="Net",
                line=dict(color="#0A1931", width=2, dash="dash"),
                marker=dict(size=6),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Net: $%{y:,.1f}M<extra></extra>",
            ))
        else:  # By Instrument
            palette = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
            for i, col in enumerate(rc["delta_exp_by_position"].columns):
                if col == "Total":
                    continue
                fig2.add_trace(go.Scatter(
                    x=spx_x, y=rc["delta_exp_by_position"][col]/1e6,
                    mode="lines+markers", name=col,
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=5),
                    hovertemplate=f"<b>%{{x:.1f}}% SPX</b><br>{col}: $%{{y:,.1f}}M<extra></extra>",
                ))
            fig2.add_trace(go.Scatter(
                x=spx_x, y=rc["delta_exp_by_position"]["Total"]/1e6,
                mode="lines+markers", name="Total",
                line=dict(color="#0A1931", width=3, dash="dash"),
                marker=dict(size=6),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Total: $%{y:,.1f}M<extra></extra>",
            ))
        
        fig2.add_hline(y=0, line_dash="dot", line_color="#AAA")
        fig2.add_vline(x=0, line_dash="dot", line_color="#AAA")
        fig2.update_layout(
            xaxis_title="SPX Move (%)",
            yaxis_title="Delta-Adj Exposure ($M)",
            height=500,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.3,
                xanchor="center", x=0.5,
            ),
            margin=dict(b=100),
        )
        fig2.update_xaxes(ticksuffix="%", gridcolor="#EEE")
        fig2.update_yaxes(tickprefix="$", ticksuffix="M", gridcolor="#EEE")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("---")
        
        # ========== SECTION 3: P&L payoff chart with granularity ==========
        st.subheader("P&L Payoff under SPX shock")
        
        granularity_b = st.selectbox(
            "Granularity",
            ["Total Portfolio", "By Beta Direction (Long / Short Beta)", "By Instrument"],
            key="granularity_pnl",
        )
        
        fig3 = go.Figure()
        
        if granularity_b == "Total Portfolio":
            fig3.add_trace(go.Scatter(
                x=spx_x, y=rc["pnl_by_direction"]["Total"]/1e6,
                mode="lines+markers", name="Portfolio P&L",
                line=dict(color="#0A1931", width=3),
                marker=dict(size=8, color="#D4A017"),
                fill="tozeroy", fillcolor="rgba(10,25,49,0.1)",
                hovertemplate="<b>%{x:.1f}% SPX</b><br>P&L: $%{y:,.1f}M<extra></extra>",
            ))
        elif granularity_b == "By Beta Direction (Long / Short Beta)":
            fig3.add_trace(go.Scatter(
                x=spx_x, y=rc["pnl_by_direction"]["Long Beta"]/1e6,
                mode="lines+markers", name="Long Beta (short SQQQ/SOXS)",
                line=dict(color="#2E8B57", width=3),
                marker=dict(size=7),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Long β P&L: $%{y:,.1f}M<extra></extra>",
            ))
            fig3.add_trace(go.Scatter(
                x=spx_x, y=rc["pnl_by_direction"]["Short Beta"]/1e6,
                mode="lines+markers", name="Short Beta (QQQ/SOXX puts)",
                line=dict(color="#B22222", width=3),
                marker=dict(size=7),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Short β P&L: $%{y:,.1f}M<extra></extra>",
            ))
            fig3.add_trace(go.Scatter(
                x=spx_x, y=rc["pnl_by_direction"]["Total"]/1e6,
                mode="lines+markers", name="Net Portfolio",
                line=dict(color="#0A1931", width=2, dash="dash"),
                marker=dict(size=6),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Net P&L: $%{y:,.1f}M<extra></extra>",
            ))
        else:
            palette = px.colors.qualitative.Set1 + px.colors.qualitative.Set2
            for i, col in enumerate(rc["pnl_by_position"].columns):
                if col == "Total":
                    continue
                fig3.add_trace(go.Scatter(
                    x=spx_x, y=rc["pnl_by_position"][col]/1e6,
                    mode="lines+markers", name=col,
                    line=dict(color=palette[i % len(palette)], width=2),
                    marker=dict(size=5),
                    hovertemplate=f"<b>%{{x:.1f}}% SPX</b><br>{col}: $%{{y:,.1f}}M<extra></extra>",
                ))
            fig3.add_trace(go.Scatter(
                x=spx_x, y=rc["pnl_by_position"]["Total"]/1e6,
                mode="lines+markers", name="Total",
                line=dict(color="#0A1931", width=3, dash="dash"),
                marker=dict(size=6),
                hovertemplate="<b>%{x:.1f}% SPX</b><br>Total: $%{y:,.1f}M<extra></extra>",
            ))
        
        fig3.add_hline(y=0, line_dash="dot", line_color="#AAA")
        fig3.add_vline(x=0, line_dash="dot", line_color="#AAA")
        fig3.update_layout(
            xaxis_title="SPX Move (%)",
            yaxis_title="P&L ($M)",
            height=500,
            hovermode="x unified",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h",
                yanchor="bottom", y=-0.3,
                xanchor="center", x=0.5,
            ),
            margin=dict(b=100),
        )
        fig3.update_xaxes(ticksuffix="%", gridcolor="#EEE")
        fig3.update_yaxes(tickprefix="$", ticksuffix="M", gridcolor="#EEE")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.markdown("---")
        
        # ========== SECTION 4: Pivoted detail tables ==========
        st.subheader("Scenario Detail — Per-Instrument × SPX Shock")
        st.caption("Each metric shown as a separate table with instruments as rows and SPX scenarios as columns.")
        
        # Format helper: takes DataFrame (rows=scenarios, cols=instruments + Total)
        # transposes to (rows=instruments, cols=scenarios) and formats as $M
        def pivot_for_display(df_scenarios_rows):
            t = df_scenarios_rows.T  # rows=instruments, cols=scenarios
            # Use .map on DataFrame (pandas 2.x replaces .applymap)
            def fmt_cell(x):
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return "—"
                return f"${x/1e6:,.2f}M" if x >= 0 else f"-${abs(x)/1e6:,.2f}M"
            fmt_t = t.copy()
            for col in fmt_t.columns:
                fmt_t[col] = fmt_t[col].apply(fmt_cell)
            fmt_t.index.name = "Instrument"
            return fmt_t.reset_index()
        
        st.markdown("**P&L ($M)** — Portfolio P&L by instrument across SPX shock scenarios")
        pnl_pivot = pivot_for_display(rc["pnl_by_position"])
        st.dataframe(pnl_pivot, hide_index=True, use_container_width=True)
        
        st.markdown("**Gross Delta Exposure ($M)** — absolute delta-adjusted exposure")
        gross_pivot = pivot_for_display(rc["gross_delta_by_position"])
        st.dataframe(gross_pivot, hide_index=True, use_container_width=True)
        
        st.markdown("**Net Delta Exposure ($M)** — signed delta-adjusted exposure")
        net_pivot = pivot_for_display(rc["delta_exp_by_position"])
        st.dataframe(net_pivot, hide_index=True, use_container_width=True)
        
        st.markdown("**Net Beta-Adjusted Exposure ($M)** — signed delta × beta")
        beta_pivot = pivot_for_display(rc["beta_adj_exp_by_position"])
        st.dataframe(beta_pivot, hide_index=True, use_container_width=True)

# =========================================================================
# PAGE: EXPECTED RETURN
# =========================================================================
elif page == "💰 Expected Return":
    st.title("Strategy 3 — Compounded Short + Vanilla Put Hedge")
    st.caption(
        "Compounded LAI short with daily top-up, hedged with monthly-rolled 3M vanilla puts on the underlying. "
        "Historical backtest replays actual 2020-2026 history. Forward MC uses 20-day joint block bootstrap "
        "of (underlying returns, VIX, VXN) to preserve spot-vol correlation and crash dynamics."
    )

    from monte_carlo import historical_backtest, forward_mc, load_joint_history
    
    # Cache joint history load
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cached_history():
        return load_joint_history()
    
    try:
        hist = _cached_history()
    except Exception as e:
        st.error(f"Could not load historical data: {e}")
        st.stop()
    
    st.info(
        f"Historical data loaded: {hist['n_days']} trading days "
        f"({pd.to_datetime(hist['dates'][0]).strftime('%Y-%m-%d')} to "
        f"{pd.to_datetime(hist['dates'][-1]).strftime('%Y-%m-%d')}). "
        "QQQ puts priced at VXN. SOXX puts priced at VIX × 1.33."
    )
    
    # --- Controls ---
    ctl1, ctl2, ctl3, ctl4 = st.columns(4)
    with ctl1:
        target_notional_m = st.number_input(
            "Short notional per leg ($M)", 
            value=15.0, step=1.0, min_value=0.1, max_value=1000.0,
        )
        target_notional = target_notional_m * 1e6
    with ctl2:
        tracking_leverage = st.selectbox(
            "LAI Tracking Leverage",
            options=[-3.0, -4.0, -5.0, -6.0], index=0,
            help="-3x is normal. Stress -4/-5/-6× for extreme regimes.",
        )
    with ctl3:
        horizon_years = st.selectbox("Forward MC horizon (years)", options=[1, 3, 5, 10], index=3)
    with ctl4:
        n_paths = st.selectbox("Forward MC paths", options=[1000, 3000, 5000, 10000], index=2)
    
    # Build config: both strike/ratio combos for side-by-side comparison
    configs = [
        {"label": "95% @ 3.45×", "strike_pct": 0.95, "ratio": 3.45},
        {"label": "90% @ 3.9×",  "strike_pct": 0.90, "ratio": 3.9},
    ]
    
    # ========================================================================
    # SECTION 1: HISTORICAL BACKTEST — reproduces spreadsheet
    # ========================================================================
    st.markdown("---")
    st.header("Historical Backtest — 2020-2026")
    st.caption(
        "Deterministic replay of actual historical returns. "
        "Unhedged version should match the hardcoded Bloomberg spreadsheet. "
        "Hedged version shows both 95%@3.45× and 90%@3.9× side-by-side for Calmar comparison."
    )
    
    with st.spinner("Running historical backtests..."):
        # Run unhedged and both hedged configs for both LAI ETFs
        hist_results = {}
        for lai, underlying in [('SQQQ', 'QQQ'), ('SOXS', 'SOXX')]:
            hist_results[f"{lai}_unhedged"] = historical_backtest(
                underlying=underlying, target_notional=target_notional,
                tracking_leverage=tracking_leverage, include_puts=False, history=hist,
            )
            for cfg in configs:
                hist_results[f"{lai}_{cfg['label']}"] = historical_backtest(
                    underlying=underlying, target_notional=target_notional,
                    tracking_leverage=tracking_leverage, include_puts=True,
                    put_strike_pct=cfg['strike_pct'], put_notional_ratio=cfg['ratio'],
                    history=hist,
                )
    
    # Summary table across all 6 configs
    st.subheader("Summary — Historical Backtest")
    summary_rows = []
    for lai in ['SQQQ', 'SOXS']:
        for label in ['unhedged', '95% @ 3.45×', '90% @ 3.9×']:
            r = hist_results[f"{lai}_{label}"]
            summary_rows.append({
                "Leg": lai,
                "Hedge Config": label,
                "Avg Annual Return": fmt_pct(r['avg_annual'], 1),
                "CAGR": fmt_pct(r['cagr'], 1),
                "MaxDD": fmt_pct(r['max_dd'], 1),
                "Ann Vol": fmt_pct(r['ann_vol'], 1),
                "Sharpe": f"{r['sharpe']:.2f}",
                "Calmar": f"{r['calmar']:.2f}",
                "Total P&L": fmt_money(r['total_pnl']),
                "Avg Gross": fmt_money(r['path_avg_gross']),
            })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)
    
    # Identify best Calmar per leg
    sqqq_calmars = [hist_results[f"SQQQ_{c['label']}"]['calmar'] for c in configs]
    soxs_calmars = [hist_results[f"SOXS_{c['label']}"]['calmar'] for c in configs]
    sqqq_best = configs[int(np.argmax(sqqq_calmars))]['label']
    soxs_best = configs[int(np.argmax(soxs_calmars))]['label']
    
    st.success(
        f"**Highest Calmar (historical, hedged):** "
        f"SQQQ → {sqqq_best} (Calmar {max(sqqq_calmars):.2f}) · "
        f"SOXS → {soxs_best} (Calmar {max(soxs_calmars):.2f})"
    )
    
    # Yearly breakdown per leg
    for lai in ['SQQQ', 'SOXS']:
        st.subheader(f"{lai} — Yearly Detail")
        # Combine all three configs into a single side-by-side yearly table
        yearly_combined = None
        for label in ['unhedged', '95% @ 3.45×', '90% @ 3.9×']:
            r = hist_results[f"{lai}_{label}"]
            df = r['yearly_df'].copy()
            df = df.rename(columns={
                'P&L ($)': f"P&L — {label}",
                'Return %': f"Return — {label}",
                'Avg Gross ($)': f"Avg Gross — {label}",
            })
            # Drop 'Days' column for all but first
            if yearly_combined is None:
                yearly_combined = df
            else:
                yearly_combined = yearly_combined.merge(
                    df.drop(columns=['Days']), on='Year', how='outer',
                )
        
        # Format for display
        disp = yearly_combined.copy()
        for col in disp.columns:
            if col == 'Year' or col == 'Days':
                continue
            if col.startswith('P&L') or col.startswith('Avg Gross'):
                disp[col] = disp[col].apply(lambda x: fmt_money(x) if pd.notna(x) else "—")
            elif col.startswith('Return'):
                disp[col] = disp[col].apply(lambda x: fmt_pct(x, 1) if pd.notna(x) else "—")
        st.dataframe(disp, hide_index=True, use_container_width=True)
        
        # Equity curve chart
        fig_eq = go.Figure()
        for label in ['unhedged', '95% @ 3.45×', '90% @ 3.9×']:
            r = hist_results[f"{lai}_{label}"]
            equity_curve = r['target_notional'] + r['cum_pnl']
            fig_eq.add_trace(go.Scatter(
                x=r['dates'], y=equity_curve / 1e6,
                mode='lines', name=label, line=dict(width=2),
                hovertemplate=f"{label}<br>%{{x|%b %Y}}<br>Equity: $%{{y:.2f}}M<extra></extra>",
            ))
        fig_eq.update_layout(
            title=f"{lai} — Equity Curve (starting ${target_notional/1e6:.0f}M)",
            xaxis_title="Date", yaxis_title="Equity ($M)",
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=40),
        )
        fig_eq.update_xaxes(gridcolor="#EEE")
        fig_eq.update_yaxes(tickprefix="$", ticksuffix="M", gridcolor="#EEE")
        st.plotly_chart(fig_eq, use_container_width=True)
    
    # ========================================================================
    # SECTION 2: FORWARD MONTE CARLO
    # ========================================================================
    st.markdown("---")
    st.header("Forward Monte Carlo — Distribution of Outcomes")
    st.caption(
        f"{n_paths:,} simulated {horizon_years}-year paths. 20-day joint block bootstrap of "
        f"(underlying return, VIX, VXN). Preserves spot-vol correlation and regime structure."
    )
    
    with st.spinner(f"Running {n_paths:,} Monte Carlo paths per config (6 configs total)..."):
        mc_results = {}
        for lai, underlying in [('SQQQ', 'QQQ'), ('SOXS', 'SOXX')]:
            mc_results[f"{lai}_unhedged"] = forward_mc(
                underlying=underlying, target_notional=target_notional,
                tracking_leverage=tracking_leverage, include_puts=False,
                horizon_years=horizon_years, n_paths=n_paths, history=hist,
            )
            for cfg in configs:
                mc_results[f"{lai}_{cfg['label']}"] = forward_mc(
                    underlying=underlying, target_notional=target_notional,
                    tracking_leverage=tracking_leverage, include_puts=True,
                    put_strike_pct=cfg['strike_pct'], put_notional_ratio=cfg['ratio'],
                    horizon_years=horizon_years, n_paths=n_paths, history=hist,
                )
    
    # Summary MC table (medians)
    st.subheader("Summary — Forward MC Medians")
    mc_summary = []
    for lai in ['SQQQ', 'SOXS']:
        for label in ['unhedged', '95% @ 3.45×', '90% @ 3.9×']:
            r = mc_results[f"{lai}_{label}"]
            mc_summary.append({
                "Leg": lai,
                "Hedge Config": label,
                "Median Avg Annual": fmt_pct(r['median_avg_annual'], 1),
                "Median CAGR": fmt_pct(r['median_cagr'], 1),
                "Median MaxDD": fmt_pct(r['median_max_dd'], 1),
                "Median Calmar": f"{r['median_calmar']:.2f}",
                "Median Sharpe": f"{r['median_sharpe']:.2f}",
                "Win Rate": fmt_pct(r['win_rate'], 1),
            })
    st.dataframe(pd.DataFrame(mc_summary), hide_index=True, use_container_width=True)
    
    # Percentile distribution per leg
    for lai in ['SQQQ', 'SOXS']:
        st.subheader(f"{lai} — Avg Annual Return Distribution (MC)")
        
        # Combine 3 configs into one percentile table
        pcts_to_show = [5, 10, 25, 50, 75, 90, 95]
        pct_rows = []
        for label in ['unhedged', '95% @ 3.45×', '90% @ 3.9×']:
            r = mc_results[f"{lai}_{label}"]
            pct_rows.append({
                "Hedge Config": label,
                **{f"{p}th": fmt_pct(r['avg_ann_percentiles'][p], 1) for p in pcts_to_show},
            })
        st.dataframe(pd.DataFrame(pct_rows), hide_index=True, use_container_width=True)
        
        # Bar chart showing percentiles side-by-side
        fig_pct = go.Figure()
        colors = {'unhedged': '#8B0000', '95% @ 3.45×': '#0A1931', '90% @ 3.9×': '#D4A017'}
        pct_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        for label in ['unhedged', '95% @ 3.45×', '90% @ 3.9×']:
            r = mc_results[f"{lai}_{label}"]
            values = [r['avg_ann_percentiles'][p] if p in r['avg_ann_percentiles'] 
                      else np.percentile(r['avg_annual_per_path'], p) for p in pct_levels]
            fig_pct.add_trace(go.Bar(
                x=[f"{p}th" for p in pct_levels],
                y=[v*100 for v in values],
                name=label, marker_color=colors.get(label, '#888'),
                text=[f"{v*100:+.1f}%" for v in values],
                textposition="outside", textfont=dict(size=9),
            ))
        fig_pct.add_hline(y=0, line_dash="dash", line_color="#555")
        fig_pct.update_layout(
            title=f"{lai} — Avg Annual Return by Percentile",
            xaxis_title="Percentile", yaxis_title="Avg Annual Return (%)",
            height=400, plot_bgcolor="white", paper_bgcolor="white",
            barmode="group", margin=dict(t=50),
        )
        fig_pct.update_yaxes(ticksuffix="%", gridcolor="#EEE")
        st.plotly_chart(fig_pct, use_container_width=True)
    
    # Methodology notes
    st.markdown("---")
    st.info(
        "**Methodology notes:**\n\n"
        "1. **Historical Backtest** replays actual 2020-2026 daily returns deterministically. "
        "Strategy mechanics: daily top-up short to target when short MV < target; let MV ride when above. "
        "Puts bought as 3M vanilla at entry, marked-to-market daily using dynamic IV (VXN for QQQ puts, "
        "VIX × 1.33 for SOXX puts), rolled monthly. This should reproduce the hardcoded Bloomberg spreadsheet.\n\n"
        "2. **Forward Monte Carlo** uses 20-day joint block bootstrap — random contiguous 20-day windows "
        "from history, concatenated with wraparound, for each simulated path. Preserves vol clustering, "
        "spot-vol correlation, and within-month regime dynamics. 5,000 paths per config.\n\n"
        "3. **Denominator for returns**: gross capital = short notional + |put delta| × contracts × 100 × spot, "
        "averaged across days within the year. So yearly return % = yearly P&L / yearly avg gross.\n\n"
        "4. **Avg Annual Return** (primary KPI) = arithmetic mean of yearly returns. "
        "**CAGR** = (Π(1+yearly_return))^(1/n) − 1 (reinvestment interpretation).\n\n"
        "5. **Dynamic IV**: QQQ puts priced at bootstrapped VXN; SOXX puts priced at bootstrapped VIX × 1.33 "
        "(SOXX beta to SPX). This captures the fact that IV spikes during crashes, making puts respond realistically."
    )


# =========================================================================
# PAGE: STRESS SCENARIO
# =========================================================================
elif page == "⚠️ Stress Scenario":
    st.title("Stress Scenario Analysis")
    
    st.subheader("Stress scenario parameters")
    st.caption(
        "Default scenario: observed 'extreme' LAI rallies with stressed leverage multipliers "
        "(SQQQ +132.1% at -4x, Dec 2021-Jun 2022; SOXS +229% at -6x, Jan-Apr 2025)."
    )
    
    # Historical reference panels
    st.markdown("#### Historical reference periods (intraday trough-to-high)")
    rc1, rc2 = st.columns(2)
    with rc1:
        with st.container(border=True):
            st.markdown("**SOXS Stress (22 Jan 2025 → 7 Apr 2025)**")
            try:
                import yfinance as yf
                soxs_hist = yf.Ticker("SOXS").history(start="2025-01-22", end="2025-04-08", auto_adjust=False)
                soxx_hist = yf.Ticker("SOXX").history(start="2025-01-22", end="2025-04-08", auto_adjust=False)
                if len(soxs_hist) > 0 and len(soxx_hist) > 0:
                    soxs_low = soxs_hist["Low"].min()
                    soxs_high = soxs_hist["High"].max()
                    soxx_low = soxx_hist["Low"].min()
                    soxx_high = soxx_hist["High"].max()
                    soxs_intraday = soxs_high/soxs_low - 1
                    soxx_intraday = soxx_low/soxx_high - 1
                    st.write(f"SOXX intraday high→low: {fmt_pct(soxx_intraday, 1)}")
                    st.write(f"SOXS intraday low→high: {fmt_pct(soxs_intraday, 1)}")
                    if soxx_intraday != 0:
                        st.caption(f"Implied stressed leverage: {soxs_intraday/soxx_intraday:.2f}x")
            except Exception:
                st.write("SOXS: +~229% (reference)")
                st.write("SOXX: -~38% (reference)")
    
    with rc2:
        with st.container(border=True):
            st.markdown("**SQQQ Stress (27 Dec 2021 → 16 Jun 2022)**")
            try:
                import yfinance as yf
                sqqq_hist = yf.Ticker("SQQQ").history(start="2021-12-27", end="2022-06-17", auto_adjust=False)
                qqq_hist = yf.Ticker("QQQ").history(start="2021-12-27", end="2022-06-17", auto_adjust=False)
                if len(sqqq_hist) > 0 and len(qqq_hist) > 0:
                    sqqq_low = sqqq_hist["Low"].min()
                    sqqq_high = sqqq_hist["High"].max()
                    qqq_low = qqq_hist["Low"].min()
                    qqq_high = qqq_hist["High"].max()
                    sqqq_intraday = sqqq_high/sqqq_low - 1
                    qqq_intraday = qqq_low/qqq_high - 1
                    st.write(f"QQQ intraday high→low: {fmt_pct(qqq_intraday, 1)}")
                    st.write(f"SQQQ intraday low→high: {fmt_pct(sqqq_intraday, 1)}")
                    if qqq_intraday != 0:
                        st.caption(f"Implied stressed leverage: {sqqq_intraday/qqq_intraday:.2f}x")
            except Exception:
                st.write("SQQQ: +~132% (reference)")
                st.write("QQQ: -~33% (reference)")
    
    st.markdown("---")
    
    st.markdown("#### Stress parameters (editable)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sqqq_pct = st.number_input("SQQQ move (%)", value=132.1, step=5.0) / 100
    with c2:
        sqqq_lev = st.number_input("SQQQ stressed leverage", value=-4.0, step=0.5)
    with c3:
        soxs_pct = st.number_input("SOXS move (%)", value=229.0, step=5.0) / 100
    with c4:
        soxs_lev = st.number_input("SOXS stressed leverage", value=-6.0, step=0.5)
    
    enriched = get_enriched_positions()
    if len([p for p in enriched if p["status"] == "OPEN"]) == 0:
        st.info("No open positions.")
    else:
        with st.spinner("Computing stress scenario..."):
            stress = stress_scenario(
                enriched,
                custom_sqqq_move=sqqq_pct,
                custom_soxs_move=soxs_pct,
                sqqq_leverage=int(sqqq_lev),
                soxs_leverage=int(soxs_lev),
            )
        
        st.subheader("Stress impact")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Stress P&L", fmt_money(stress["total_pnl"]))
        with c2:
            st.metric("Current Gross", fmt_money(stress["current_gross_exposure"]))
        with c3:
            st.metric("Stressed Gross", fmt_money(stress["new_gross_exposure"]))
        with c4:
            ratio = (stress["new_gross_exposure"] / stress["current_gross_exposure"]
                     if stress["current_gross_exposure"] > 0 else 0)
            st.metric("Gross Multiplier", f"{ratio:.2f}x")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Stressed Gross Delta-Adj Exposure", fmt_money(stress["stressed_delta_adj_gross"]))
        with c2:
            st.metric("Stressed Net Delta-Adj Exposure", fmt_money(stress["stressed_delta_adj_net"]))
        
        st.caption(
            f"**Implied moves:** QQQ = {fmt_pct(stress['qqq_implied'], 1)}, "
            f"SOXX = {fmt_pct(stress['soxx_implied'], 1)} "
            f"(derived from SQQQ {fmt_pct(sqqq_pct, 1)} at {int(sqqq_lev)}x and "
            f"SOXS {fmt_pct(soxs_pct, 1)} at {int(soxs_lev)}x)"
        )
        
        st.markdown("---")
        
        st.subheader("Steady-state gross deployment")
        ss = steady_state_gross(enriched, stress, gross_limit=gross_limit)
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Gross Limit", fmt_money(ss["gross_limit"]))
        with sc2:
            st.metric(
                "Stress Gross / Current Ratio",
                f"{ss['stress_gross_ratio']:.2f}x"
            )
        with sc3:
            st.metric("Max Steady-State Gross", fmt_money(ss["steady_state_gross"]))
        
        st.info(
            f"If you scale your current positions by {ss['scaling_factor']:.2f}x, "
            f"your stressed gross exposure would equal the {fmt_money(ss['gross_limit'])} limit. "
            f"Steady-state gross should not exceed {fmt_money(ss['steady_state_gross'])} "
            f"to survive this stress scenario without breaching the gross limit."
        )
        
        st.markdown("---")
        
        st.subheader("By-position detail under stress")
        by_pos = stress["by_position"]
        if len(by_pos) > 0:
            disp = by_pos.copy()
            disp["Move %"] = disp["Move %"].apply(fmt_pct)
            disp["Current Spot"] = disp["Current Spot"].apply(lambda x: fmt_money(x, show_cents=True))
            disp["Stressed Spot"] = disp["Stressed Spot"].apply(lambda x: fmt_money(x, show_cents=True))
            disp["Stressed Price"] = disp["Stressed Price"].apply(lambda x: fmt_money(x, show_cents=True))
            disp["Stressed Delta"] = disp["Stressed Delta"].apply(lambda x: f"{x:.3f}")
            disp["Current MV"] = disp["Current MV"].apply(fmt_money)
            disp["Stressed MV"] = disp["Stressed MV"].apply(fmt_money)
            disp["Stressed Gross"] = disp["Stressed Gross"].apply(fmt_money)
            disp["Stressed Delta-Adj Exposure"] = disp["Stressed Delta-Adj Exposure"].apply(fmt_money)
            disp["Stress P&L"] = disp["Stress P&L"].apply(fmt_money)
            st.dataframe(disp, hide_index=True, use_container_width=True)

# =========================================================================
# PAGE: SETTINGS
# =========================================================================
elif page == "⚙️ Settings / Data Refresh":
    st.title("Settings & Data")
    
    st.subheader("Data refresh")
    st.caption("Market data is cached for 60 seconds. Click below to force a refresh.")
    if st.button("🔄 Refresh all market data", type="primary"):
        refresh_data()
        st.success("Cache cleared. Reload other pages to pull fresh data.")
    
    st.markdown("---")
    
    st.subheader("LAI ETF mapping")
    st.caption("Leveraged/inverse ETFs recognised by the system.")
    lai_df = pd.DataFrame([
        {"Ticker": k, "Underlying": v["underlying"], "Leverage": v["leverage"]}
        for k, v in LAI_ETF_MAP.items()
    ])
    st.dataframe(lai_df, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("Data diagnostics")
    ticker_test = st.text_input("Test a ticker (spot, beta, vol)", value="SQQQ")
    if ticker_test:
        col1, col2, col3 = st.columns(3)
        with col1:
            spot = get_spot(ticker_test)
            st.metric("Spot", fmt_money(spot, show_cents=True) if spot else "N/A")
        with col2:
            lai = get_lai_info(ticker_test)
            if lai:
                beta = compute_beta(lai["underlying"])
                st.metric(f"Beta ({lai['underlying']} vs SPY)",
                          f"{beta:.3f}" if not np.isnan(beta) else "N/A")
                st.caption(f"Effective beta of {ticker_test}: {beta * lai['leverage']:.2f}")
            else:
                beta = compute_beta(ticker_test)
                st.metric(f"Beta vs SPY", f"{beta:.3f}" if not np.isnan(beta) else "N/A")
        with col3:
            if lai:
                vol = realized_vol(lai["underlying"], 260)
                st.metric(f"Vol ({lai['underlying']}, 260d)",
                          fmt_pct(vol, 1) if not np.isnan(vol) else "N/A")
            else:
                vol = realized_vol(ticker_test, 260)
                st.metric(f"Vol (260d)", fmt_pct(vol, 1) if not np.isnan(vol) else "N/A")
    
    st.markdown("---")
    
    st.subheader("Database")
    db_path = __import__("db").DB_PATH
    st.code(str(db_path))
    all_pos = get_positions()
    st.write(f"Total positions in DB: **{len(all_pos)}** "
             f"({sum(1 for p in all_pos if p['status']=='OPEN')} open, "
             f"{sum(1 for p in all_pos if p['status']=='CLOSED')} closed)")
