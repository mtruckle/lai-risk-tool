"""
Analytics engine: enriches positions with live market data, computes Greeks,
risk curves, expected returns, and stress scenarios.
"""
import numpy as np
import pandas as pd
from datetime import datetime

from market_data import (
    get_spot, get_option_quote, compute_beta, realized_vol
)
from pricing import bs_price, bs_greeks, implied_vol
from bbg_parser import years_to_expiry
from vol_decay import (
    LAI_ETF_MAP, get_lai_info, is_lai_etf,
    compute_vol_decay, compute_expected_return_to_expiry
)


RISK_FREE_RATE = 0.04
DIV_YIELD = 0.005


# ========== SCENARIO HELPERS ==========

# Stress scenarios (user-specified historical periods)
STRESS_SCENARIOS = {
    "SOXS (22 Jan 2025 – 7 Apr 2025)": {
        "instrument": "SOXS",
        "underlying": "SOXX",
        "start": "2025-01-22",
        "end": "2025-04-07",
        "lai_move": 2.29,       # +229% for SOXS intraday trough-to-high
        "implied_leverage": -6,  # observed stressed leverage
    },
    "SQQQ (27 Dec 2021 – 16 Jun 2022)": {
        "instrument": "SQQQ",
        "underlying": "QQQ",
        "start": "2021-12-27",
        "end": "2022-06-16",
        "lai_move": 1.321,      # +132.1% for SQQQ intraday trough-to-high
        "implied_leverage": -4,  # observed stressed leverage
    },
}


# ========== POSITION ENRICHMENT ==========

def enrich_position(pos: dict, asof_date=None) -> dict:
    """
    Given a position dict from the DB, add:
      - spot: current price of underlying
      - option_root spot (for options, same)
      - mid_price: current mid of option (or entry if stock)
      - iv: implied vol
      - T: years to expiry
      - greeks: delta, gamma, theta, vega (per contract)
      - market_value (signed): mid * qty * multiplier  (for option: premium)
      - notional_exposure (absolute): |qty| * multiplier * spot  (for options: underlying equivalent)
      - beta: beta of underlying to SPY
      - unrealized_pnl: (current - entry) * qty * multiplier
    """
    enriched = dict(pos)
    
    if pos["instrument_type"] == "OPTION":
        # Fetch spot of underlying using the option_root (post-split tickers may have suffix)
        spot = get_spot(pos["option_root"])
        if spot is None:
            # Try underlying ticker as fallback
            spot = get_spot(pos["underlying"])
        enriched["spot"] = spot
        
        # Try to get option quote via option_root
        quote = get_option_quote(
            pos["option_root"], pos["expiry"], pos["strike"], pos["option_type"]
        )
        
        # Fallback: try the underlying ticker (for pre-split options)
        if quote is None and pos["option_root"] != pos["underlying"]:
            quote = get_option_quote(
                pos["underlying"], pos["expiry"], pos["strike"], pos["option_type"]
            )
        
        if quote:
            enriched["mid_price"] = quote["mid"]
            enriched["bid"] = quote["bid"]
            enriched["ask"] = quote["ask"]
            enriched["last"] = quote["last"]
            enriched["iv"] = quote["iv"]
            enriched["oi"] = quote["openInterest"]
            enriched["volume"] = quote["volume"]
        else:
            # Can't fetch - use entry price as fallback mark
            enriched["mid_price"] = pos["entry_price"]
            enriched["bid"] = None
            enriched["ask"] = None
            enriched["last"] = None
            enriched["iv"] = None
            enriched["oi"] = None
            enriched["volume"] = None
        
        T = years_to_expiry(pos["expiry"], asof_date)
        enriched["T"] = T
        
        # Compute Greeks from yfinance IV (but fall back to backing out IV from mid)
        iv = enriched["iv"]
        if (iv is None or (isinstance(iv, float) and np.isnan(iv)) or iv <= 0) and spot:
            # Back out IV from mid price
            if enriched["mid_price"] > 0:
                iv = implied_vol(
                    enriched["mid_price"], spot, pos["strike"], T,
                    RISK_FREE_RATE, DIV_YIELD, pos["option_type"]
                )
                enriched["iv"] = iv
        
        if spot and iv and not (isinstance(iv, float) and np.isnan(iv)) and iv > 0 and T > 0:
            greeks = bs_greeks(
                spot, pos["strike"], T, RISK_FREE_RATE, DIV_YIELD,
                iv, pos["option_type"]
            )
        else:
            greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
        
        enriched["delta"] = greeks["delta"]
        enriched["gamma"] = greeks["gamma"]
        enriched["theta"] = greeks["theta"]
        enriched["vega"] = greeks["vega"]
        
        # Market value (signed by qty direction)
        qty = pos["quantity"]
        mult = pos["multiplier"]
        spot_val = spot if spot else 0
        enriched["market_value"] = (enriched["mid_price"] or 0) * qty * mult
        enriched["notional_gross"] = abs(qty) * mult * spot_val
        enriched["delta_exposure"] = qty * mult * greeks["delta"] * spot_val
        enriched["unrealized_pnl"] = ((enriched["mid_price"] or pos["entry_price"]) - pos["entry_price"]) * qty * mult
    
    else:
        # Stock position
        spot = get_spot(pos["underlying"])
        enriched["spot"] = spot
        enriched["mid_price"] = spot if spot else pos["entry_price"]
        enriched["T"] = 0
        enriched["iv"] = None
        enriched["delta"] = 1.0  # Stock has delta = 1
        enriched["gamma"] = 0
        enriched["theta"] = 0
        enriched["vega"] = 0
        
        qty = pos["quantity"]
        mult = pos["multiplier"]  # Should be 1 for stock
        price = enriched["mid_price"]
        enriched["market_value"] = price * qty * mult
        enriched["notional_gross"] = abs(qty) * mult * (spot if spot else 0)
        enriched["delta_exposure"] = qty * mult * (spot if spot else 0)
        enriched["unrealized_pnl"] = (price - pos["entry_price"]) * qty * mult
    
    # Beta
    # For LAI ETFs, beta = underlying_beta × leverage
    lai_info = get_lai_info(pos["underlying"])
    if lai_info is not None:
        und_beta = compute_beta(lai_info["underlying"])
        enriched["beta"] = und_beta * lai_info["leverage"] if not np.isnan(und_beta) else np.nan
    else:
        enriched["beta"] = compute_beta(pos["underlying"])
    
    if not np.isnan(enriched["beta"]) and enriched.get("delta_exposure"):
        enriched["beta_adj_exposure"] = enriched["delta_exposure"] * enriched["beta"]
    else:
        enriched["beta_adj_exposure"] = np.nan
    
    # Realized P&L if closed
    if pos["status"] == "CLOSED" and pos.get("exit_price"):
        enriched["realized_pnl"] = (pos["exit_price"] - pos["entry_price"]) * pos["quantity"] * pos["multiplier"]
    else:
        enriched["realized_pnl"] = 0
    
    return enriched


# ========== PORTFOLIO SUMMARY ==========

def portfolio_summary(enriched_positions: list) -> dict:
    """Aggregate portfolio-level metrics."""
    open_pos = [p for p in enriched_positions if p["status"] == "OPEN"]
    closed_pos = [p for p in enriched_positions if p["status"] == "CLOSED"]
    
    # Total premium: sum of market values for options (ignore stock for "premium" concept)
    total_premium = sum(p["market_value"] for p in open_pos if p["instrument_type"] == "OPTION")
    
    # Exposures
    total_delta_exp = sum((p.get("delta_exposure") or 0) for p in open_pos)
    total_beta_exp = sum((p.get("beta_adj_exposure") or 0) for p in open_pos if not np.isnan(p.get("beta_adj_exposure") or np.nan))
    
    # Greeks (aggregated across all options, scaled by qty * multiplier)
    total_delta = sum(p["delta"] * p["quantity"] * p["multiplier"] for p in open_pos)
    total_gamma = sum(p["gamma"] * p["quantity"] * p["multiplier"] for p in open_pos)
    total_theta = sum(p["theta"] * p["quantity"] * p["multiplier"] for p in open_pos)  # per year
    total_vega = sum(p["vega"] * p["quantity"] * p["multiplier"] for p in open_pos)    # per 1.00 vol
    
    # Gross exposure
    gross_exposure = sum(p["notional_gross"] for p in open_pos)
    
    # P&L
    unrealized_pnl = sum(p["unrealized_pnl"] for p in open_pos)
    realized_pnl = sum(p["realized_pnl"] for p in closed_pos)
    
    # By underlying
    by_underlying = {}
    for p in open_pos:
        u = p["underlying"]
        if u not in by_underlying:
            by_underlying[u] = {
                "n_positions": 0,
                "total_premium": 0,
                "delta_exposure": 0,
                "beta_adj_exposure": 0,
                "gross_notional": 0,
                "unrealized_pnl": 0,
                "option_premium": 0,
            }
        bu = by_underlying[u]
        bu["n_positions"] += 1
        bu["gross_notional"] += p["notional_gross"]
        bu["delta_exposure"] += (p.get("delta_exposure") or 0)
        if not np.isnan(p.get("beta_adj_exposure") or np.nan):
            bu["beta_adj_exposure"] += p["beta_adj_exposure"]
        bu["unrealized_pnl"] += p["unrealized_pnl"]
        if p["instrument_type"] == "OPTION":
            bu["option_premium"] += p["market_value"]
            bu["total_premium"] += p["market_value"]
    
    return {
        "total_premium": total_premium,
        "total_delta_exp": total_delta_exp,
        "total_beta_exp": total_beta_exp,
        "total_delta": total_delta,
        "total_gamma": total_gamma,
        "total_theta": total_theta,
        "total_vega": total_vega,
        "gross_exposure": gross_exposure,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "total_pnl": unrealized_pnl + realized_pnl,
        "by_underlying": by_underlying,
        "n_open_positions": len(open_pos),
        "n_closed_positions": len(closed_pos),
    }


# ========== RISK CURVE ==========

def _classify_beta_direction(p: dict) -> str:
    """
    Classify a position by its exposure direction relative to SPX.
    - 'long_beta': position gains when SPX goes up
    - 'short_beta': position gains when SPX goes down
    
    Rules:
    - Short cash SQQQ/SOXS = long beta (you profit when SPX up → SQQQ down)
    - Long put on LAI (SQQQ/SOXS) = short beta (LAI goes up when SPX down → put gains)
    - Long put on QQQ/SOXX/SPY = short beta
    - Long call on anything LAI positively correlated to SPX = long beta
    """
    und = p["underlying"].upper()
    lai = get_lai_info(und)
    qty = p["quantity"]
    
    if p["instrument_type"] == "STOCK":
        # Cash position: long SQQQ/SOXS (short beta), short SQQQ/SOXS (long beta)
        # Long QQQ/SOXX (long beta), short QQQ/SOXX (short beta)
        if lai and lai["leverage"] < 0:
            # Inverse ETF: long = short beta, short = long beta
            return "short_beta" if qty > 0 else "long_beta"
        else:
            # Regular stock or leveraged long ETF (correlated to SPX)
            return "long_beta" if qty > 0 else "short_beta"
    else:
        # Option
        opt_type = p["option_type"].upper()
        # For an inverse ETF (SQQQ/SOXS):
        #   long put on SQQQ: gains when SQQQ falls, i.e. when QQQ rises → long beta
        #   long call on SQQQ: gains when SQQQ rises → short beta
        # For a regular ETF (QQQ/SOXX/SPY):
        #   long put on QQQ: gains when QQQ falls → short beta
        #   long call on QQQ: gains when QQQ rises → long beta
        is_inverse = lai and lai["leverage"] < 0
        
        if is_inverse:
            # Inverse underlying
            if opt_type == "P":
                direction = "long_beta"  # long put on inverse = long beta
            else:
                direction = "short_beta"  # long call on inverse = short beta
        else:
            # Regular underlying (or unknown — assume regular)
            if opt_type == "P":
                direction = "short_beta"
            else:
                direction = "long_beta"
        
        # Flip if short
        if qty < 0:
            direction = "short_beta" if direction == "long_beta" else "long_beta"
        
        return direction


def _compute_new_spot(p: dict, spx_move: float) -> float:
    """Given a position, compute the new spot price of its underlying given an SPX move."""
    spot = p.get("spot")
    if not spot:
        return None
    
    lai = get_lai_info(p["underlying"])
    if lai:
        # LAI ETF: new_spot = spot × (1 + leverage × underlying_beta × spx_move)
        und_beta = compute_beta(lai["underlying"])
        if np.isnan(und_beta):
            und_beta = 1.0
        underlying_move_pct = und_beta * spx_move
        lai_move_pct = lai["leverage"] * underlying_move_pct
        return spot * (1 + lai_move_pct)
    else:
        beta = p.get("beta") or 1.0
        if np.isnan(beta):
            beta = 1.0
        return spot * (1 + beta * spx_move)


def _reprice_position(p: dict, new_spot: float) -> tuple:
    """
    Given a position and a new spot price for its underlying,
    return (new_price, new_pnl, new_delta).
    """
    if p["instrument_type"] == "OPTION":
        T = p["T"]
        iv = p.get("iv") or 0.3
        new_price = bs_price(
            new_spot, p["strike"], T, RISK_FREE_RATE, DIV_YIELD,
            iv, p["option_type"]
        )
        new_greeks = bs_greeks(
            new_spot, p["strike"], T, RISK_FREE_RATE, DIV_YIELD,
            iv, p["option_type"]
        )
        new_delta = new_greeks["delta"]
        pnl = (new_price - (p.get("mid_price") or 0)) * p["quantity"] * p["multiplier"]
    else:
        new_price = new_spot
        new_delta = 1.0
        pnl = (new_spot - (p.get("mid_price") or 0)) * p["quantity"] * p["multiplier"]
    
    return new_price, pnl, new_delta


# ========== RISK CURVE ==========

def risk_curve(enriched_positions: list, asof_date=None) -> dict:
    """
    Compute portfolio P&L and delta-adjusted exposure for SPX moves from -15% to +15%.
    
    Returns dict with:
      - summary_df: aggregate P&L and exposures by scenario (gross, net, net-beta-adj)
      - pnl_by_position: DataFrame (rows=scenarios, cols=positions + Total)
      - pnl_by_direction: DataFrame (rows=scenarios, cols=long_beta, short_beta, Total)
      - delta_exp_by_position: DataFrame (rows=scenarios, cols=positions + Total)
      - delta_exp_by_direction: DataFrame (rows=scenarios, cols=long_beta, short_beta, Total)
    
    Delta-adjusted exposure convention: signed (long = positive $, short = negative $).
    Delta-adj gross exposure aggregates absolute values; net aggregates signed values.
    """
    open_pos = [p for p in enriched_positions if p["status"] == "OPEN"]
    spx_moves = np.arange(-0.15, 0.15 + 0.025/2, 0.025)
    
    # Label rows for readability
    labels = [f"{m*100:+.1f}%" for m in spx_moves]
    
    # Position keys (for column names)
    pos_keys = [p["bbg_ticker"] for p in open_pos]
    
    # Classify positions by beta direction
    direction_map = {p["bbg_ticker"]: _classify_beta_direction(p) for p in open_pos}
    
    # Storage
    pnl_pos = {k: [] for k in pos_keys}
    delta_exp_pos = {k: [] for k in pos_keys}
    beta_adj_exp_pos = {k: [] for k in pos_keys}  # NEW: per-position beta-adj
    
    summary_rows = []
    
    for spx_move in spx_moves:
        row_pnl_pos = {}
        row_delta_exp_pos = {}
        
        total_pnl = 0
        gross_delta_exp = 0
        net_delta_exp = 0
        net_beta_adj_exp = 0
        
        for p in open_pos:
            k = p["bbg_ticker"]
            spot = p.get("spot")
            if not spot:
                pnl_pos[k].append(0)
                delta_exp_pos[k].append(0)
                beta_adj_exp_pos[k].append(0)
                continue
            
            new_spot = _compute_new_spot(p, spx_move)
            new_price, pnl, new_delta = _reprice_position(p, new_spot)
            
            qty = p["quantity"]
            mult = p["multiplier"]
            
            delta_exp = new_delta * qty * mult * new_spot
            
            # Beta-adjusted per-position
            beta = p.get("beta")
            if beta is None or np.isnan(beta):
                beta = 1.0
            pos_beta_adj = delta_exp * beta
            
            pnl_pos[k].append(pnl)
            delta_exp_pos[k].append(delta_exp)
            beta_adj_exp_pos[k].append(pos_beta_adj)
            
            total_pnl += pnl
            gross_delta_exp += abs(delta_exp)
            net_delta_exp += delta_exp
            net_beta_adj_exp += pos_beta_adj
        
        summary_rows.append({
            "SPX Move": labels[len(summary_rows)],
            "SPX Move (decimal)": spx_move,
            "Portfolio P&L": total_pnl,
            "Gross Delta Exposure": gross_delta_exp,
            "Net Delta Exposure": net_delta_exp,
            "Net Beta-Adj Exposure": net_beta_adj_exp,
        })
    
    summary_df = pd.DataFrame(summary_rows)
    
    pnl_by_position = pd.DataFrame(pnl_pos, index=labels)
    pnl_by_position["Total"] = pnl_by_position.sum(axis=1)
    
    delta_exp_by_position = pd.DataFrame(delta_exp_pos, index=labels)
    delta_exp_by_position["Total"] = delta_exp_by_position.sum(axis=1)
    
    beta_adj_exp_by_position = pd.DataFrame(beta_adj_exp_pos, index=labels)
    beta_adj_exp_by_position["Total"] = beta_adj_exp_by_position.sum(axis=1)
    
    # Gross delta exp by position = abs of delta exp per scenario
    gross_delta_by_position = delta_exp_by_position.drop(columns="Total").abs()
    gross_delta_by_position["Total"] = gross_delta_by_position.sum(axis=1)
    
    # By direction
    pnl_long = [sum(pnl_pos[k][i] for k in pos_keys if direction_map[k] == "long_beta")
                for i in range(len(labels))]
    pnl_short = [sum(pnl_pos[k][i] for k in pos_keys if direction_map[k] == "short_beta")
                 for i in range(len(labels))]
    pnl_by_direction = pd.DataFrame({
        "Long Beta": pnl_long,
        "Short Beta": pnl_short,
    }, index=labels)
    pnl_by_direction["Total"] = pnl_by_direction.sum(axis=1)
    
    dexp_long = [sum(delta_exp_pos[k][i] for k in pos_keys if direction_map[k] == "long_beta")
                 for i in range(len(labels))]
    dexp_short = [sum(delta_exp_pos[k][i] for k in pos_keys if direction_map[k] == "short_beta")
                  for i in range(len(labels))]
    delta_exp_by_direction = pd.DataFrame({
        "Long Beta": dexp_long,
        "Short Beta": dexp_short,
    }, index=labels)
    delta_exp_by_direction["Total"] = delta_exp_by_direction.sum(axis=1)
    
    return {
        "summary_df": summary_df,
        "pnl_by_position": pnl_by_position,
        "pnl_by_direction": pnl_by_direction,
        "delta_exp_by_position": delta_exp_by_position,
        "delta_exp_by_direction": delta_exp_by_direction,
        "beta_adj_exp_by_position": beta_adj_exp_by_position,
        "gross_delta_by_position": gross_delta_by_position,
        "direction_map": direction_map,
        "labels": labels,
    }


# ========== EXPECTED RETURN TABLE ==========

# Fallback underlying vols when yfinance data unavailable (used rarely)
VOL_FALLBACKS = {
    "QQQ": 0.22,
    "SOXX": 0.35,
    "SPY": 0.18,
    "IWM": 0.23,
    "XLK": 0.24,
    "XBI": 0.30,
}

def _get_underlying_vol(underlying_ticker: str, lookback_days: int = 260) -> float:
    """Get underlying realized vol with sensible fallback if yfinance unavailable."""
    vol = realized_vol(underlying_ticker, lookback_days)
    if vol is None or (isinstance(vol, float) and np.isnan(vol)):
        vol = VOL_FALLBACKS.get(underlying_ticker.upper(), 0.25)
    return vol


def expected_return_table(enriched_positions: list, r_1y_assumption=0.0, asof_date=None,
                          use_mc: bool = True, mc_paths: int = 5000,
                          tracking_leverage: float = -3.0,
                          roll_frequency_months: int = 1,
                          option_tenor_months: int = 3) -> dict:
    """
    Compute expected $ P&L by leg. Three legs:
    
    1. LAI options (long puts / short calls): expected $ return to fair-value at expiry,
       annualized via CAGR using the vol decay formula.
    
    2. Cash shorts on LAI ETFs — TWO METHODS:
       a) Simple formula: -decay_1y × notional  (reference only, overstates $ P&L)
       b) Monte Carlo (DEFAULT): simulates GBM path of underlying, applies daily top-up
          to maintain short at target notional. Captures path-dependent compounding.
    
    3. Long protection puts (non-LAI, e.g. QQQ/SOXX) — MONTHLY ROLL MODEL:
       Buy X-month put, sell it N-months later. Annualized cost = 12 × roll_cost / roll_freq.
       Assumes no change in underlying.
    """
    from monte_carlo import simulate_compounded_short
    
    lai_option_rows = []
    cash_short_rows = []
    protection_rows = []
    
    total_lai_option_annual_usd = 0
    total_cash_short_annual_usd = 0     # using MC if enabled
    total_cash_short_simple_usd = 0     # simple formula for reference
    total_protection_annual_cost_usd = 0
    total_gross_exposure = 0
    
    for p in enriched_positions:
        if p["status"] != "OPEN":
            continue
        
        spot = p.get("spot")
        mid = p.get("mid_price")
        total_gross_exposure += p.get("notional_gross") or 0
        
        # ===== CASH SHORT / LONG IN LAI ETF =====
        if p["instrument_type"] == "STOCK" and is_lai_etf(p["underlying"]):
            lai_info = get_lai_info(p["underlying"])
            underlying_vol = _get_underlying_vol(lai_info["underlying"], 260)
            
            qty = p["quantity"]
            notional = abs(qty) * (spot or mid)
            
            # Method A: simple formula (for reference)
            decay_1y = compute_vol_decay(
                r_1y_assumption, lai_info["leverage"], underlying_vol, 1.0
            )
            if qty < 0:
                simple_pnl = -decay_1y * notional
            else:
                simple_pnl = decay_1y * notional
            total_cash_short_simple_usd += simple_pnl
            
            # Method B: Monte Carlo (compounded top-up)
            mc_mean = np.nan
            mc_median = np.nan
            mc_p5 = np.nan
            mc_p95 = np.nan
            if use_mc and qty < 0:  # only for shorts (strategy doesn't apply to longs)
                mc_result = simulate_compounded_short(
                    target_notional=notional,
                    underlying_vol=underlying_vol,
                    tracking_leverage=tracking_leverage,
                    horizon_days=252,
                    underlying_drift=r_1y_assumption,
                    n_paths=mc_paths,
                    seed=42,
                )
                mc_mean = mc_result["mean_pnl"]
                mc_median = mc_result["median_pnl"]
                mc_p5 = mc_result["p5"]
                mc_p95 = mc_result["p95"]
                total_cash_short_annual_usd += mc_mean
            else:
                # Fallback: use simple formula
                total_cash_short_annual_usd += simple_pnl
            
            cash_short_rows.append({
                "Ticker": p["bbg_ticker"],
                "Direction": "SHORT" if qty < 0 else "LONG",
                "Qty": qty,
                "Spot": spot,
                "Notional ($)": notional,
                "Underlying Vol": underlying_vol,
                "1Y Simple Decay": decay_1y,
                "Simple $ P&L": simple_pnl,
                "MC Mean $ P&L": mc_mean,
                "MC Median $ P&L": mc_median,
                "MC 5th pct": mc_p5,
                "MC 95th pct": mc_p95,
            })
        
        # ===== OPTION POSITIONS =====
        elif p["instrument_type"] == "OPTION":
            # LAI option → use vol decay expected return to FV at expiry
            if is_lai_etf(p["underlying"]):
                lai_info = get_lai_info(p["underlying"])
                underlying_vol = _get_underlying_vol(lai_info["underlying"], 260)
                
                result = compute_expected_return_to_expiry(
                    p, spot, mid, underlying_vol, r_1y_assumption, asof_date
                )
                
                if result and p.get("market_value") is not None:
                    mv = p["market_value"]
                    annual_pnl_usd = mv * result["expected_return_annualized"]
                    total_lai_option_annual_usd += annual_pnl_usd
                    
                    lai_option_rows.append({
                        "Ticker": p["bbg_ticker"],
                        "Underlying": p["underlying"],
                        "Strike": p["strike"],
                        "Expiry": p["expiry"],
                        "Type": p["option_type"],
                        "Qty": p["quantity"],
                        "Years": result["years"],
                        "Leverage": result["leverage"],
                        "Underlying Vol": result["underlying_vol"],
                        "Expected Decay": result["expected_decay"],
                        "Exp. Spot @ Expiry": result["expected_spot_at_expiry"],
                        "Intrinsic @ Expiry": result["intrinsic_at_expiry"],
                        "Current Mid": mid,
                        "Exp. Total Return": result["expected_return_total"],
                        "Exp. Annual Return": result["expected_return_annualized"],
                        "Market Value ($)": mv,
                        "Annual P&L ($)": annual_pnl_usd,
                    })
            else:
                # Non-LAI option = protection (e.g. QQQ/SOXX put)
                # NEW: monthly-roll cost model
                T = p["T"]
                if T > 0 and mid and mid > 0 and spot and p["quantity"] > 0:
                    iv = p.get("iv") or 0.20
                    K = p["strike"]
                    opt_type = p["option_type"]
                    
                    # Buy price = current mid (at tenor_months, e.g. 3M)
                    buy_price = mid
                    # Sell price = BS price at (tenor_months - roll_freq_months) = 2M for 3M buy / 1M roll
                    sell_T = max((option_tenor_months - roll_frequency_months) / 12.0, 1/365)
                    sell_price = bs_price(
                        spot, K, sell_T, RISK_FREE_RATE, DIV_YIELD, iv, opt_type
                    )
                    per_roll_cost = (buy_price - sell_price) * p["quantity"] * p["multiplier"]
                    # Annualized cost = 12 rolls per year × per-roll cost
                    rolls_per_year = 12 / roll_frequency_months
                    annual_cost = per_roll_cost * rolls_per_year
                    
                    total_protection_annual_cost_usd += annual_cost
                    protection_rows.append({
                        "Ticker": p["bbg_ticker"],
                        "Underlying": p["underlying"],
                        "Strike": K,
                        "Expiry": p["expiry"],
                        "Type": opt_type,
                        "Qty": p["quantity"],
                        "Tenor (mo)": option_tenor_months,
                        "Roll Freq (mo)": roll_frequency_months,
                        "Buy Price": buy_price,
                        "Sell Price (after roll)": sell_price,
                        "Per-Roll Cost ($)": per_roll_cost,
                        "Annual Cost ($)": annual_cost,
                    })
    
    # Net expected $ return (using MC for shorts if enabled)
    net_annual_pnl_usd = (
        total_lai_option_annual_usd
        + total_cash_short_annual_usd
        - total_protection_annual_cost_usd
    )
    
    if total_gross_exposure > 0:
        return_on_gross = net_annual_pnl_usd / total_gross_exposure
    else:
        return_on_gross = np.nan
    
    return {
        "lai_option_table": pd.DataFrame(lai_option_rows),
        "cash_short_table": pd.DataFrame(cash_short_rows),
        "protection_table": pd.DataFrame(protection_rows),
        "total_lai_option_annual_usd": total_lai_option_annual_usd,
        "total_cash_short_annual_usd": total_cash_short_annual_usd,
        "total_cash_short_simple_usd": total_cash_short_simple_usd,
        "total_protection_annual_cost_usd": total_protection_annual_cost_usd,
        "net_annual_pnl_usd": net_annual_pnl_usd,
        "total_gross_exposure": total_gross_exposure,
        "return_on_gross": return_on_gross,
        "n_lai_options": len(lai_option_rows),
        "n_cash_shorts": len(cash_short_rows),
        "n_protection_options": len(protection_rows),
        "use_mc": use_mc,
        "tracking_leverage": tracking_leverage,
    }


def sensitivity_vol_sweep(enriched_positions: list,
                           vol_grid: list = None,
                           tracking_leverage: float = -3.0,
                           mc_paths: int = 3000,
                           asof_date=None) -> pd.DataFrame:
    """
    Vary each LAI short's underlying vol from 10% to 70%, compute total strategy annual P&L.
    Spot assumed unchanged (0% drift). Protection puts: fixed annualized roll cost at current IV.
    
    Returns DataFrame with rows=scenarios (by vol), cols=['Vol', 'Total P&L ($)', 'Return on Gross (%)']
    """
    from monte_carlo import simulate_compounded_short
    
    if vol_grid is None:
        vol_grid = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
    
    # Get open positions and classify
    open_pos = [p for p in enriched_positions if p["status"] == "OPEN"]
    
    # For each vol point, compute total annualized P&L
    # Components:
    #   - LAI cash shorts: MC-simulated P&L at that vol, per-position
    #   - LAI options: re-evaluate expected return with new vol (held constant across)
    #   - Protection: fixed annual cost (unchanged by vol sweep since roll at current IV)
    
    gross_exposure = sum(p.get("notional_gross") or 0 for p in open_pos)
    
    # Precompute protection cost (vol-insensitive for this sweep — holding IV constant at current)
    _, protection_cost = _compute_protection_cost(open_pos)
    
    rows = []
    for vol in vol_grid:
        total_short_pnl = 0
        total_lai_opt_pnl = 0
        
        for p in open_pos:
            if p["instrument_type"] == "STOCK" and is_lai_etf(p["underlying"]):
                qty = p["quantity"]
                spot = p.get("spot") or p.get("mid_price") or 0
                notional = abs(qty) * spot
                if qty < 0 and notional > 0:
                    r = simulate_compounded_short(
                        target_notional=notional,
                        underlying_vol=vol,
                        tracking_leverage=tracking_leverage,
                        horizon_days=252,
                        underlying_drift=0.0,
                        n_paths=mc_paths,
                        seed=42,
                    )
                    total_short_pnl += r["mean_pnl"]
            
            elif p["instrument_type"] == "OPTION" and is_lai_etf(p["underlying"]):
                lai_info = get_lai_info(p["underlying"])
                result = compute_expected_return_to_expiry(
                    p, p.get("spot"), p.get("mid_price"), vol, 0.0, asof_date
                )
                if result and p.get("market_value") is not None:
                    total_lai_opt_pnl += p["market_value"] * result["expected_return_annualized"]
        
        total_pnl = total_short_pnl + total_lai_opt_pnl - protection_cost
        pnl_pct = total_pnl / gross_exposure if gross_exposure > 0 else np.nan
        
        rows.append({
            "Vol": vol,
            "LAI Shorts P&L": total_short_pnl,
            "LAI Options P&L": total_lai_opt_pnl,
            "Protection Cost": protection_cost,
            "Total P&L ($)": total_pnl,
            "Total P&L ($M)": total_pnl / 1e6,
            "Return on Gross (%)": pnl_pct,
        })
    
    return pd.DataFrame(rows)


def sensitivity_spot_sweep(enriched_positions: list,
                            spot_grid: list = None,
                            tracking_leverage: float = -3.0,
                            mc_paths: int = 3000,
                            asof_date=None) -> pd.DataFrame:
    """
    Vary total 1Y underlying move from -50% to +50%, vol held at current realized.
    For each move, compute total strategy annual P&L.
    Applies move to underlying — LAI short price path includes that directional drift.
    Protection puts: repriced at new spot (intrinsic + remaining time value) and roll cost recomputed.
    """
    from monte_carlo import simulate_compounded_short
    
    if spot_grid is None:
        spot_grid = [-0.50, -0.40, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    
    open_pos = [p for p in enriched_positions if p["status"] == "OPEN"]
    gross_exposure = sum(p.get("notional_gross") or 0 for p in open_pos)
    
    rows = []
    for move in spot_grid:
        total_short_pnl = 0
        total_lai_opt_pnl = 0
        total_protection_cost = 0
        
        # Annualized continuous drift to achieve total 1y move
        drift = np.log(1.0 + move) if move > -1 else -5  # log(1+m) as annualized drift
        
        for p in open_pos:
            if p["instrument_type"] == "STOCK" and is_lai_etf(p["underlying"]):
                lai_info = get_lai_info(p["underlying"])
                underlying_vol = _get_underlying_vol(lai_info["underlying"], 260)
                qty = p["quantity"]
                spot = p.get("spot") or p.get("mid_price") or 0
                notional = abs(qty) * spot
                if qty < 0 and notional > 0:
                    r = simulate_compounded_short(
                        target_notional=notional,
                        underlying_vol=underlying_vol,
                        tracking_leverage=tracking_leverage,
                        horizon_days=252,
                        underlying_drift=drift,
                        n_paths=mc_paths,
                        seed=42,
                    )
                    total_short_pnl += r["mean_pnl"]
            
            elif p["instrument_type"] == "OPTION" and is_lai_etf(p["underlying"]):
                # LAI options: recompute expected return with underlying moving `move` over 1y
                lai_info = get_lai_info(p["underlying"])
                underlying_vol = _get_underlying_vol(lai_info["underlying"], 260)
                # Pass the move as r_1y to the decay formula
                result = compute_expected_return_to_expiry(
                    p, p.get("spot"), p.get("mid_price"), underlying_vol, move, asof_date
                )
                if result and p.get("market_value") is not None:
                    total_lai_opt_pnl += p["market_value"] * result["expected_return_annualized"]
            
            elif p["instrument_type"] == "OPTION" and not is_lai_etf(p["underlying"]) and p["quantity"] > 0:
                # Protection put: roll cost recomputed with stressed spot
                spot = p.get("spot")
                if spot is None:
                    continue
                stressed_spot = spot * (1 + move)
                iv = p.get("iv") or 0.20
                K = p["strike"]
                opt_type = p["option_type"]
                
                # Buy a new 3M 95% put at stressed spot
                # But wait — protection puts are typically struck at a FIXED strike (95% of original spot at time of entry)
                # For the roll cost, though, we assume strikes reset monthly to 95% of current spot
                # So when underlying moves: new strike = stressed_spot × 0.95, new ATM-95% option
                new_K = stressed_spot * (K / spot)  # preserve relative moneyness
                buy_price = bs_price(stressed_spot, new_K, 0.25, RISK_FREE_RATE, DIV_YIELD, iv, opt_type)
                sell_price = bs_price(stressed_spot, new_K, 2/12, RISK_FREE_RATE, DIV_YIELD, iv, opt_type)
                per_roll = (buy_price - sell_price) * p["quantity"] * p["multiplier"]
                annual_cost = per_roll * 12
                total_protection_cost += annual_cost
        
        total_pnl = total_short_pnl + total_lai_opt_pnl - total_protection_cost
        pnl_pct = total_pnl / gross_exposure if gross_exposure > 0 else np.nan
        
        rows.append({
            "Spot Move": move,
            "LAI Shorts P&L": total_short_pnl,
            "LAI Options P&L": total_lai_opt_pnl,
            "Protection Cost": total_protection_cost,
            "Total P&L ($)": total_pnl,
            "Total P&L ($M)": total_pnl / 1e6,
            "Return on Gross (%)": pnl_pct,
        })
    
    return pd.DataFrame(rows)


def _compute_protection_cost(open_positions: list) -> tuple:
    """Helper to compute annualized protection cost assuming current spot/IV."""
    total_cost = 0
    rows = []
    for p in open_positions:
        if (p["instrument_type"] == "OPTION"
            and not is_lai_etf(p["underlying"])
            and p["quantity"] > 0):
            spot = p.get("spot")
            mid = p.get("mid_price")
            iv = p.get("iv") or 0.20
            if spot is None or mid is None or mid <= 0:
                continue
            K = p["strike"]
            opt_type = p["option_type"]
            # Buy 3M → sell 2M — 1M roll
            buy_price = mid
            sell_price = bs_price(spot, K, 2/12, RISK_FREE_RATE, DIV_YIELD, iv, opt_type)
            per_roll = (buy_price - sell_price) * p["quantity"] * p["multiplier"]
            total_cost += per_roll * 12
    return rows, total_cost


# ========== STRESS SCENARIO ==========

def stress_scenario(enriched_positions: list, scenario_key: str = None,
                     custom_sqqq_move: float = None, custom_soxs_move: float = None,
                     sqqq_leverage: int = -4, soxs_leverage: int = -6) -> dict:
    """
    Compute P&L and gross exposure under a stress scenario.
    
    By default:
      - SQQQ +132.1% (implying QQQ -33.0% at -4x)
      - SOXS +229% (implying SOXX -38.2% at -6x)
    
    Returns dict with:
      - total_pnl
      - new_gross_exposure
      - by_position details including stressed delta-adj exposure
    """
    sqqq_move = custom_sqqq_move if custom_sqqq_move is not None else 1.321
    soxs_move = custom_soxs_move if custom_soxs_move is not None else 2.29
    
    # Implied underlying moves
    qqq_move = sqqq_move / sqqq_leverage   # +132.1% at -4x → QQQ -33.0%
    soxx_move = soxs_move / soxs_leverage  # +229% at -6x → SOXX -38.2%
    
    open_pos = [p for p in enriched_positions if p["status"] == "OPEN"]
    rows = []
    total_pnl = 0
    new_gross = 0
    stressed_delta_adj_gross = 0
    stressed_delta_adj_net = 0
    
    for p in open_pos:
        spot = p.get("spot")
        if not spot:
            continue
        
        # Determine this position's underlying move
        und = p["underlying"].upper()
        if und == "SQQQ":
            u_move = sqqq_move
        elif und == "SOXS":
            u_move = soxs_move
        elif und == "QQQ":
            u_move = qqq_move
        elif und == "SOXX":
            u_move = soxx_move
        else:
            # Generic mapping: use beta for other underlyings
            lai = get_lai_info(und)
            if lai:
                mapped = lai["underlying"]
                if mapped == "QQQ":
                    u_move = lai["leverage"] * qqq_move
                elif mapped == "SOXX":
                    u_move = lai["leverage"] * soxx_move
                else:
                    u_move = 0
            else:
                # SPX-like underlying: average of QQQ/SOXX
                avg_spx_like = (qqq_move + soxx_move) / 2
                beta = p.get("beta") or 1.0
                if np.isnan(beta):
                    beta = 1.0
                u_move = beta * avg_spx_like
        
        new_spot = spot * (1 + u_move)
        
        # Reprice & Greeks
        if p["instrument_type"] == "OPTION":
            T = p["T"]
            iv = p.get("iv") or 0.3
            new_price = bs_price(
                new_spot, p["strike"], T, RISK_FREE_RATE, DIV_YIELD,
                iv, p["option_type"]
            )
            new_greeks = bs_greeks(
                new_spot, p["strike"], T, RISK_FREE_RATE, DIV_YIELD,
                iv, p["option_type"]
            )
            new_delta = new_greeks["delta"]
            pnl = (new_price - p["mid_price"]) * p["quantity"] * p["multiplier"]
            new_mv_gross = abs(p["quantity"]) * p["multiplier"] * new_spot
        else:
            new_price = new_spot
            new_delta = 1.0
            pnl = (new_spot - p["mid_price"]) * p["quantity"] * p["multiplier"]
            new_mv_gross = abs(p["quantity"]) * p["multiplier"] * new_spot
        
        # Stressed delta-adj exposure: delta × qty × multiplier × new_spot
        stressed_delta_adj = new_delta * p["quantity"] * p["multiplier"] * new_spot
        
        total_pnl += pnl
        new_gross += new_mv_gross
        stressed_delta_adj_gross += abs(stressed_delta_adj)
        stressed_delta_adj_net += stressed_delta_adj
        
        rows.append({
            "Ticker": p["bbg_ticker"],
            "Underlying": und,
            "Current Spot": spot,
            "Stressed Spot": new_spot,
            "Move %": u_move,
            "Current MV": p["market_value"],
            "Stressed Price": new_price,
            "Stressed MV": new_price * p["quantity"] * p["multiplier"],
            "Stressed Gross": new_mv_gross,
            "Stressed Delta": new_delta,
            "Stressed Delta-Adj Exposure": stressed_delta_adj,
            "Stress P&L": pnl,
        })
    
    return {
        "total_pnl": total_pnl,
        "new_gross_exposure": new_gross,
        "current_gross_exposure": sum(p["notional_gross"] for p in open_pos),
        "stressed_delta_adj_gross": stressed_delta_adj_gross,
        "stressed_delta_adj_net": stressed_delta_adj_net,
        "by_position": pd.DataFrame(rows),
        "sqqq_move": sqqq_move,
        "soxs_move": soxs_move,
        "qqq_implied": qqq_move,
        "soxx_implied": soxx_move,
        "sqqq_leverage": sqqq_leverage,
        "soxs_leverage": soxs_leverage,
    }


def steady_state_gross(enriched_positions: list, stress_result: dict, gross_limit: float = 150e6) -> dict:
    """
    Compute the steady-state gross deployment such that under stress, the gross stays at the limit.
    Assumes linear scaling: if we scale all positions by factor k, stressed gross also scales by k.
    """
    current_gross = stress_result["current_gross_exposure"]
    stressed_gross = stress_result["new_gross_exposure"]
    
    if current_gross == 0 or stressed_gross == 0:
        return {"steady_state_gross": 0, "scaling_factor": 0, "ratio": 0}
    
    # Stressed gross scales linearly with position size
    # Want: k * stressed_gross = gross_limit
    # So: k = gross_limit / stressed_gross
    # Steady state gross = k * current_gross
    k = gross_limit / stressed_gross
    steady_state = k * current_gross
    
    return {
        "steady_state_gross": steady_state,
        "scaling_factor": k,
        "stress_gross_ratio": stressed_gross / current_gross if current_gross > 0 else 0,
        "gross_limit": gross_limit,
    }
