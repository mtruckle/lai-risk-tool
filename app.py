"""
LAI Risk Management Tool — Streamlit App

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
# CUSTOM CSS for a cleaner institutional look
st.markdown("""
<style>
    .main { padding-top: 1rem; }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0A1931;
    }
    .loss { color: #B22222; font-weight: bold; }
    .gain { color: #2E8B57; font-weight: bold; }
    h1, h2, h3 { color: #0A1931; font-family: Georgia, serif; }
    [data-testid="stMetricValue"] { font-size: 24px; }
</style>
""", unsafe_allow_html=True)

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

r_1y_assumption = st.sidebar.number_input(
    "Underlying 1Y Expected Return (for vol decay)",
    value=5.0, min_value=-20.0, max_value=50.0, step=1.0, format="%.1f"
) / 100

st.sidebar.caption(f"""
Data source: Yahoo Finance (15-min delayed)  
Risk-free: 4.0% | Div yield: 0.5%
""")


# -------------------------------------------------------------------------
# HELPERS
@st.cache_data(ttl=60, show_spinner=False)
def get_enriched_positions(_trigger_refresh=0):
    """Enrich all positions. _trigger_refresh lets us invalidate cache."""
    raw = get_positions()
    enriched = [enrich_position(p) for p in raw]
    return enriched


def refresh_data():
    """Clear cache to force data refresh."""
    get_enriched_positions.clear()


def fmt_money(x, millions=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if millions:
        return f"${x/1e6:,.2f}M"
    if abs(x) >= 1e6:
        return f"${x/1e6:,.2f}M"
    if abs(x) >= 1e3:
        return f"${x/1e3:,.1f}k"
    return f"${x:,.2f}"


def fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x*100:.{digits}f}%"


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
                        st.write(f"**Strike:** ${parsed['strike']}")
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
            # Display table
            df_rows = []
            for p in open_pos:
                df_rows.append({
                    "ID": p["id"],
                    "Ticker": p["bbg_ticker"],
                    "Type": p["instrument_type"],
                    "Qty": p["quantity"],
                    "Mult": p["multiplier"],
                    "Entry": p["entry_price"],
                    "Mid/Spot": p.get("mid_price"),
                    "IV": f"{p.get('iv')*100:.1f}%" if p.get("iv") else "—",
                    "Δ": f"{p.get('delta', 0):.3f}",
                    "MV": p.get("market_value", 0),
                    "Unreal. P&L": p.get("unrealized_pnl", 0),
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(
                df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "MV": st.column_config.NumberColumn(format="$%.0f"),
                    "Unreal. P&L": st.column_config.NumberColumn(format="$%.0f"),
                    "Entry": st.column_config.NumberColumn(format="$%.2f"),
                    "Mid/Spot": st.column_config.NumberColumn(format="$%.2f"),
                },
            )
            
            st.markdown("---")
            st.subheader("Close or delete a position")
            col1, col2 = st.columns([1, 2])
            with col1:
                ids = [p["id"] for p in open_pos]
                pos_id = st.selectbox("Position ID", ids, key="close_id")
            selected = next((p for p in open_pos if p["id"] == pos_id), None)
            if selected:
                with col2:
                    st.write(f"**{selected['bbg_ticker']}** — Qty: {selected['quantity']}, Entry: ${selected['entry_price']}")
                
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
                    if st.button("🗑️ Delete", help="Permanently delete (no history kept)"):
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
                    "Qty": p["quantity"],
                    "Entry": p["entry_price"],
                    "Entry Date": p["entry_date"],
                    "Exit": p.get("exit_price"),
                    "Exit Date": p.get("exit_date"),
                    "Realized P&L": rp,
                })
            df = pd.DataFrame(df_rows)
            st.dataframe(df, hide_index=True, use_container_width=True,
                         column_config={
                             "Realized P&L": st.column_config.NumberColumn(format="$%.0f"),
                             "Entry": st.column_config.NumberColumn(format="$%.2f"),
                             "Exit": st.column_config.NumberColumn(format="$%.2f"),
                         })
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
        # Top-level metrics
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
            st.metric("Total Δ (shares)", f"{summary['total_delta']:,.0f}")
        with g2:
            st.metric("Total Γ", f"{summary['total_gamma']:,.2f}")
        with g3:
            theta_day = summary["total_theta"] / 365
            st.metric("Total Θ (per day)", fmt_money(theta_day))
        with g4:
            vega_vol_pt = summary["total_vega"] / 100
            st.metric("Total Vega (per vol pt)", fmt_money(vega_vol_pt))
        
        st.markdown("---")
        
        # P&L
        st.subheader("P&L")
        p1, p2, p3 = st.columns(3)
        with p1:
            delta = "↑" if summary["unrealized_pnl"] >= 0 else "↓"
            st.metric("Unrealized P&L", fmt_money(summary["unrealized_pnl"]))
        with p2:
            st.metric("Realized P&L", fmt_money(summary["realized_pnl"]))
        with p3:
            st.metric("Total P&L", fmt_money(summary["total_pnl"]))
        
        st.markdown("---")
        
        # By underlying
        st.subheader("By underlying")
        by_und = summary["by_underlying"]
        if by_und:
            rows = []
            for u, data in sorted(by_und.items()):
                rows.append({
                    "Underlying": u,
                    "# Positions": data["n_positions"],
                    "Option Premium": data["option_premium"],
                    "Gross Notional": data["gross_notional"],
                    "Delta Exposure": data["delta_exposure"],
                    "Beta-Adj Exposure": data["beta_adj_exposure"],
                    "Unrealized P&L": data["unrealized_pnl"],
                })
            df = pd.DataFrame(rows)
            st.dataframe(
                df, hide_index=True, use_container_width=True,
                column_config={
                    "Option Premium": st.column_config.NumberColumn(format="$%.0f"),
                    "Gross Notional": st.column_config.NumberColumn(format="$%.0f"),
                    "Delta Exposure": st.column_config.NumberColumn(format="$%.0f"),
                    "Beta-Adj Exposure": st.column_config.NumberColumn(format="$%.0f"),
                    "Unrealized P&L": st.column_config.NumberColumn(format="$%.0f"),
                },
            )
            
            # Premium pie by underlying
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
        "Portfolio P&L by SPX move (-15% to +15% in 2.5% increments). "
        "Each position repriced via Black-Scholes assuming implied vol is unchanged. "
        "Leveraged/inverse ETFs move at their actual multiplier × underlying beta × SPX move."
    )
    
    enriched = get_enriched_positions()
    if len([p for p in enriched if p["status"] == "OPEN"]) == 0:
        st.info("No open positions.")
    else:
        with st.spinner("Computing risk curve..."):
            rc = risk_curve(enriched)
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rc["SPX Move (decimal)"] * 100,
            y=rc["Portfolio P&L ($)"] / 1e6,
            mode="lines+markers",
            line=dict(color="#0A1931", width=3),
            marker=dict(size=8, color="#D4A017"),
            name="Portfolio P&L",
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="#888")
        fig.add_vline(x=0, line_dash="dash", line_color="#888")
        fig.update_layout(
            xaxis_title="SPX Move (%)",
            yaxis_title="Portfolio P&L ($M)",
            height=500,
            hovermode="x unified",
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(ticksuffix="%", gridcolor="#EEE")
        fig.update_yaxes(tickprefix="$", ticksuffix="M", gridcolor="#EEE")
        st.plotly_chart(fig, use_container_width=True)
        
        # Table
        st.subheader("Scenario table")
        rc_display = rc.copy()
        rc_display["Portfolio P&L ($M)"] = rc_display["Portfolio P&L ($)"] / 1e6
        rc_display = rc_display[["SPX Move", "Portfolio P&L ($)", "Portfolio P&L ($M)"]]
        st.dataframe(
            rc_display, hide_index=True, use_container_width=True,
            column_config={
                "Portfolio P&L ($)": st.column_config.NumberColumn(format="$%.0f"),
                "Portfolio P&L ($M)": st.column_config.NumberColumn(format="$%.2fM"),
            }
        )

# =========================================================================
# PAGE: EXPECTED RETURN
# =========================================================================
elif page == "💰 Expected Return":
    st.title("Expected Return — Vol Decay on LAI ETF Options")
    st.caption(
        f"Assumes: underlying 1Y return = {r_1y_assumption*100:.1f}%, "
        "underlying vol = 260-day realized, constant implied vol. "
        "Expected return = intrinsic value at expiry (from decayed spot) vs today's mid price."
    )
    
    enriched = get_enriched_positions()
    if len([p for p in enriched if p["status"] == "OPEN"]) == 0:
        st.info("No open positions.")
    else:
        with st.spinner("Computing expected returns..."):
            result = expected_return_table(enriched, r_1y_assumption)
        
        table = result["table"]
        
        if len(table) == 0:
            st.warning("No LAI ETF options in portfolio. Expected return is based on LAI options only.")
        else:
            # Portfolio metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Weighted LAI Return (ann.)",
                          fmt_pct(result["weighted_lai_return"]))
            with c2:
                st.metric("Total LAI Premium", fmt_money(result["total_lai_mv"]))
            with c3:
                st.metric("Protection Cost (ann.)",
                          fmt_money(result["total_protection_cost_annual"]))
                st.caption(f"{fmt_pct(result['protection_cost_pct_of_lai'])} of LAI premium")
            with c4:
                net = result["net_expected_return"]
                color = "gain" if (net is not None and not np.isnan(net) and net >= 0) else "loss"
                st.metric("Net Expected Return (ann.)", fmt_pct(net))
            
            st.markdown("---")
            st.subheader(f"LAI option detail ({result['n_lai_options']} positions)")
            
            # Format display
            disp = table.copy()
            disp["Underlying Vol"] = disp["Underlying Vol"].apply(lambda x: f"{x*100:.1f}%")
            disp["Expected Decay"] = disp["Expected Decay"].apply(lambda x: f"{x*100:.1f}%")
            disp["Exp. Total Return"] = disp["Exp. Total Return"].apply(lambda x: f"{x*100:.1f}%")
            disp["Exp. Annual Return"] = disp["Exp. Annual Return"].apply(lambda x: f"{x*100:.1f}%")
            disp["Years"] = disp["Years"].apply(lambda x: f"{x:.2f}")
            disp["Exp. Spot @ Expiry"] = disp["Exp. Spot @ Expiry"].apply(lambda x: f"${x:.2f}")
            disp["Intrinsic @ Expiry"] = disp["Intrinsic @ Expiry"].apply(lambda x: f"${x:.2f}")
            disp["Current Mid"] = disp["Current Mid"].apply(lambda x: f"${x:.2f}" if x else "—")
            
            st.dataframe(disp, hide_index=True, use_container_width=True)
            
            st.info(
                f"**Methodology:** Vol decay formula = `(1 + r)^L × exp((L − L²) × σ² × T / 2) − 1`. "
                f"For each LAI option, expected fair value at expiry = `max(K − expected_spot, 0)`. "
                f"Annualized return = `(FV / mid)^(1/T) − 1`. "
                f"Protection cost is long non-LAI puts (e.g. QQQ, SOXX puts), annualized as "
                f"`premium_paid / years_to_expiry`."
            )

# =========================================================================
# PAGE: STRESS SCENARIO
# =========================================================================
elif page == "⚠️ Stress Scenario":
    st.title("Stress Scenario Analysis")
    
    # Scenario controls
    st.subheader("Stress scenario parameters")
    st.caption(
        "Default scenario models observed 'extreme' LAI rallies with stressed leverage multipliers "
        "(-4x SQQQ observed 2022, -6x SOXS observed April 2025)."
    )
    
    # Historical reference panels
    st.markdown("#### Historical reference periods")
    rc1, rc2 = st.columns(2)
    with rc1:
        sc_soxs = STRESS_SCENARIOS["SOXS (22 Jan 2025 – 7 Apr 2025)"]
        with st.container(border=True):
            st.markdown("**SOXS Stress (22 Jan 2025 → 7 Apr 2025)**")
            # Fetch OHLC moves for reference
            try:
                import yfinance as yf
                soxs_hist = yf.Ticker("SOXS").history(start=sc_soxs["start"], end="2025-04-08", auto_adjust=False)
                soxx_hist = yf.Ticker("SOXX").history(start=sc_soxs["start"], end="2025-04-08", auto_adjust=False)
                if len(soxs_hist) > 0 and len(soxx_hist) > 0:
                    soxs_pct = soxs_hist["Close"].iloc[-1] / soxs_hist["Open"].iloc[0] - 1
                    soxx_pct = soxx_hist["Close"].iloc[-1] / soxx_hist["Open"].iloc[0] - 1
                    st.write(f"SOXX: {soxx_pct*100:+.1f}%")
                    st.write(f"SOXS: {soxs_pct*100:+.1f}%")
                    st.caption(f"Implied leverage: {soxs_pct/soxx_pct:.2f}x")
            except Exception:
                st.write("SOXS: +~200% (reference)")
                st.write("SOXX: -~33% (reference)")
    
    with rc2:
        sc_sqqq = STRESS_SCENARIOS["SQQQ (27 Dec 2021 – 16 Jun 2022)"]
        with st.container(border=True):
            st.markdown("**SQQQ Stress (27 Dec 2021 → 16 Jun 2022)**")
            try:
                import yfinance as yf
                sqqq_hist = yf.Ticker("SQQQ").history(start=sc_sqqq["start"], end="2022-06-17", auto_adjust=False)
                qqq_hist = yf.Ticker("QQQ").history(start=sc_sqqq["start"], end="2022-06-17", auto_adjust=False)
                if len(sqqq_hist) > 0 and len(qqq_hist) > 0:
                    sqqq_pct = sqqq_hist["Close"].iloc[-1] / sqqq_hist["Open"].iloc[0] - 1
                    qqq_pct = qqq_hist["Close"].iloc[-1] / qqq_hist["Open"].iloc[0] - 1
                    st.write(f"QQQ: {qqq_pct*100:+.1f}%")
                    st.write(f"SQQQ: {sqqq_pct*100:+.1f}%")
                    st.caption(f"Implied leverage: {sqqq_pct/qqq_pct:.2f}x")
            except Exception:
                st.write("SQQQ: +~130% (reference)")
                st.write("QQQ: -~32.5% (reference)")
    
    st.markdown("---")
    
    # User-adjustable inputs
    st.markdown("#### Stress parameters (editable)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sqqq_pct = st.number_input("SQQQ move (%)", value=130.0, step=5.0) / 100
    with c2:
        sqqq_lev = st.number_input("SQQQ stressed leverage", value=-4.0, step=0.5)
    with c3:
        soxs_pct = st.number_input("SOXS move (%)", value=200.0, step=5.0) / 100
    with c4:
        soxs_lev = st.number_input("SOXS stressed leverage", value=-6.0, step=0.5)
    
    # Run stress
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
        
        # Headline metrics
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
        
        # Implied moves
        st.caption(
            f"**Implied moves:** QQQ = {stress['qqq_implied']*100:+.1f}%, "
            f"SOXX = {stress['soxx_implied']*100:+.1f}% "
            f"(derived from SQQQ {sqqq_pct*100:+.0f}% at {int(sqqq_lev)}x and "
            f"SOXS {soxs_pct*100:+.0f}% at {int(soxs_lev)}x)"
        )
        
        st.markdown("---")
        
        # Steady-state solver
        st.subheader("Steady-state gross deployment")
        ss = steady_state_gross(enriched, stress, gross_limit=gross_limit)
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.metric("Gross Limit", fmt_money(ss["gross_limit"]))
        with sc2:
            st.metric(
                "Stress Gross / Current Gross Ratio",
                f"{ss['stress_gross_ratio']:.2f}x"
            )
        with sc3:
            st.metric("Max Steady-State Gross", fmt_money(ss["steady_state_gross"]))
        
        st.info(
            f"If you scale your current positions by {ss['scaling_factor']:.2f}x, "
            f"your stressed gross exposure would equal the {fmt_money(ss['gross_limit'])} limit. "
            f"In other words, steady-state gross should not exceed {fmt_money(ss['steady_state_gross'])} "
            f"to survive this stress scenario without breaching the gross limit."
        )
        
        st.markdown("---")
        
        # By-position detail
        st.subheader("By-position detail under stress")
        by_pos = stress["by_position"]
        if len(by_pos) > 0:
            disp = by_pos.copy()
            disp["Move %"] = (disp["Move %"] * 100).round(1).astype(str) + "%"
            for col in ["Current Spot", "Stressed Spot", "Stressed Price"]:
                disp[col] = disp[col].apply(lambda x: f"${x:.2f}")
            for col in ["Current MV", "Stressed MV", "Stressed Gross", "Stress P&L"]:
                disp[col] = disp[col].apply(fmt_money)
            st.dataframe(disp, hide_index=True, use_container_width=True)

# =========================================================================
# PAGE: SETTINGS / DATA REFRESH
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
            st.metric("Spot", f"${spot:.2f}" if spot else "N/A")
        with col2:
            # Beta of the underlying if LAI, else direct
            lai = get_lai_info(ticker_test)
            if lai:
                beta = compute_beta(lai["underlying"])
                st.metric(f"β ({lai['underlying']} vs SPY)",
                          f"{beta:.3f}" if not np.isnan(beta) else "N/A")
                st.caption(f"Effective β of {ticker_test}: {beta * lai['leverage']:.2f}")
            else:
                beta = compute_beta(ticker_test)
                st.metric(f"β vs SPY", f"{beta:.3f}" if not np.isnan(beta) else "N/A")
        with col3:
            # Vol of the underlying if LAI
            if lai:
                vol = realized_vol(lai["underlying"], 260)
                st.metric(f"σ ({lai['underlying']}, 260d)",
                          f"{vol*100:.1f}%" if not np.isnan(vol) else "N/A")
            else:
                vol = realized_vol(ticker_test, 260)
                st.metric(f"σ (260d)", f"{vol*100:.1f}%" if not np.isnan(vol) else "N/A")
    
    st.markdown("---")
    
    st.subheader("Database")
    db_path = __import__("db").DB_PATH
    st.code(str(db_path))
    all_pos = get_positions()
    st.write(f"Total positions in DB: **{len(all_pos)}** "
             f"({sum(1 for p in all_pos if p['status']=='OPEN')} open, "
             f"{sum(1 for p in all_pos if p['status']=='CLOSED')} closed)")
