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
        "lai_move": 2.00,       # +200% for SOXS in stress
        "implied_leverage": -6,  # observed stressed leverage
    },
    "SQQQ (27 Dec 2021 – 16 Jun 2022)": {
        "instrument": "SQQQ",
        "underlying": "QQQ",
        "start": "2021-12-27",
        "end": "2022-06-16",
        "lai_move": 1.30,       # +130% for SQQQ in stress
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

def risk_curve(enriched_positions: list, asof_date=None) -> pd.DataFrame:
    """
    Compute portfolio P&L for SPX moves from -15% to +15% in 2.5% increments.
    Uses beta to translate SPX move to each underlying move, then reprices options via BS
    holding implied vol constant.
    
    For leveraged/inverse ETFs, uses 3x or -3x multiplier on the UNDERLYING move
    (not on SPX directly) to reflect the actual instrument behavior.
    """
    open_pos = [p for p in enriched_positions if p["status"] == "OPEN"]
    
    spx_moves = np.arange(-0.15, 0.15 + 0.025/2, 0.025)
    rows = []
    
    for spx_move in spx_moves:
        total_pnl = 0
        
        for p in open_pos:
            spot = p.get("spot")
            if not spot:
                continue
            
            # Translate SPX move to this underlying
            lai = get_lai_info(p["underlying"])
            if lai:
                # LAI ETF: SPX move → underlying move via underlying beta, then × leverage
                und_beta = compute_beta(lai["underlying"])
                if np.isnan(und_beta):
                    und_beta = 1.0
                underlying_move_pct = und_beta * spx_move
                lai_move_pct = lai["leverage"] * underlying_move_pct
                new_spot = spot * (1 + lai_move_pct)
            else:
                # Non-LAI: SPX move × beta
                beta = p.get("beta") or 1.0
                if np.isnan(beta):
                    beta = 1.0
                new_spot = spot * (1 + beta * spx_move)
            
            # Reprice
            if p["instrument_type"] == "OPTION":
                T = p["T"]
                iv = p.get("iv") or 0.3
                new_price = bs_price(
                    new_spot, p["strike"], T, RISK_FREE_RATE, DIV_YIELD,
                    iv, p["option_type"]
                )
                pnl = (new_price - p["mid_price"]) * p["quantity"] * p["multiplier"]
            else:
                pnl = (new_spot - p["mid_price"]) * p["quantity"] * p["multiplier"]
            
            total_pnl += pnl
        
        rows.append({
            "SPX Move": f"{spx_move*100:+.1f}%",
            "SPX Move (decimal)": spx_move,
            "Portfolio P&L ($)": total_pnl,
        })
    
    return pd.DataFrame(rows)


# ========== EXPECTED RETURN TABLE ==========

def expected_return_table(enriched_positions: list, r_1y_assumption=0.05, asof_date=None) -> dict:
    """
    For each LAI option position, compute:
      - expected decay of the LAI over T
      - expected fair intrinsic value at expiry
      - expected return to FV (total and annualized)
    
    Also weighted-average the portfolio and compute:
      - total expected LAI return (weighted by position MV)
      - total put-protection cost (vanilla puts on QQQ/SOXX etc.), annualized as theta drag
      - net expected portfolio return
    """
    rows = []
    lai_positions = []
    protection_positions = []
    
    for p in enriched_positions:
        if p["status"] != "OPEN":
            continue
        if p["instrument_type"] != "OPTION":
            continue
        
        spot = p.get("spot")
        mid = p.get("mid_price")
        
        # LAI option?
        if is_lai_etf(p["underlying"]):
            # Realized vol of the UNDERLYING (not the LAI itself)
            lai_info = get_lai_info(p["underlying"])
            underlying_vol = realized_vol(lai_info["underlying"], 260)
            
            result = compute_expected_return_to_expiry(
                p, spot, mid, underlying_vol, r_1y_assumption, asof_date
            )
            if result:
                row = {
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
                    "Market Value": p["market_value"],
                }
                rows.append(row)
                lai_positions.append({
                    "mv": abs(p["market_value"]),
                    "annual_ret": result["expected_return_annualized"],
                })
        else:
            # Non-LAI option = protection position (QQQ/SOXX puts, SPX etc.)
            # Annualized cost = theta / mid_price (expressed as cost of carry)
            # More accurately: total premium paid / years to expiry = annualized cost if held to expiry
            T = p["T"]
            if T > 0 and mid and mid > 0:
                # Premium paid (long) = cost to carry; annualized = premium / T
                # This is conservative: assumes full premium decays to 0
                premium = mid * abs(p["quantity"]) * p["multiplier"]
                annual_cost = premium / T  # $ per year
                protection_positions.append({
                    "ticker": p["bbg_ticker"],
                    "premium": premium,
                    "annual_cost": annual_cost,
                    "is_long": p["quantity"] > 0,
                })
    
    # Weighted average LAI expected return
    total_mv_lai = sum(x["mv"] for x in lai_positions)
    if total_mv_lai > 0:
        weighted_lai_return = sum(
            x["mv"] * x["annual_ret"] for x in lai_positions
        ) / total_mv_lai
    else:
        weighted_lai_return = np.nan
    
    # Total annual cost of protection (long non-LAI puts)
    total_protection_cost = sum(
        x["annual_cost"] for x in protection_positions if x["is_long"]
    )
    
    # Portfolio expected annualized return = LAI return (on LAI $) - protection cost (in $)
    # Converting to a portfolio return requires a common base: use total LAI premium
    if total_mv_lai > 0:
        protection_cost_pct = total_protection_cost / total_mv_lai
        net_expected_return = weighted_lai_return - protection_cost_pct
    else:
        protection_cost_pct = np.nan
        net_expected_return = np.nan
    
    return {
        "table": pd.DataFrame(rows),
        "weighted_lai_return": weighted_lai_return,
        "total_lai_mv": total_mv_lai,
        "total_protection_cost_annual": total_protection_cost,
        "protection_cost_pct_of_lai": protection_cost_pct,
        "net_expected_return": net_expected_return,
        "n_lai_options": len(lai_positions),
        "n_protection_options": len(protection_positions),
    }


# ========== STRESS SCENARIO ==========

def stress_scenario(enriched_positions: list, scenario_key: str = None,
                     custom_sqqq_move: float = None, custom_soxs_move: float = None,
                     sqqq_leverage: int = -4, soxs_leverage: int = -6) -> dict:
    """
    Compute P&L and gross exposure under a stress scenario.
    
    By default:
      - SQQQ +130% (implying QQQ -32.5% at -4x)
      - SOXS +200% (implying SOXX -33.3% at -6x)
    
    Returns dict with:
      - total_pnl
      - new_gross_exposure
      - by_position details
    """
    sqqq_move = custom_sqqq_move if custom_sqqq_move is not None else 1.30
    soxs_move = custom_soxs_move if custom_soxs_move is not None else 2.00
    
    # Implied underlying moves
    qqq_move = sqqq_move / sqqq_leverage   # +130% at -4x → QQQ -32.5%
    soxx_move = soxs_move / soxs_leverage  # +200% at -6x → SOXX -33.3%
    
    open_pos = [p for p in enriched_positions if p["status"] == "OPEN"]
    rows = []
    total_pnl = 0
    new_gross = 0
    
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
            # Generic mapping: use beta for other underlyings; assume avg of QQQ/SOXX move for SPX-like
            lai = get_lai_info(und)
            if lai:
                # Look up the mapped underlying in our scenario set
                mapped = lai["underlying"]
                if mapped == "QQQ":
                    u_move = qqq_move * lai["leverage"] / lai["leverage"]  # already mapped
                    u_move = lai["leverage"] * qqq_move
                elif mapped == "SOXX":
                    u_move = lai["leverage"] * soxx_move
                else:
                    u_move = 0  # Unknown
            else:
                # Assume SPX-like underlying proxied by average of QQQ/SOXX = ~-33%
                avg_spx_like = (qqq_move + soxx_move) / 2
                beta = p.get("beta") or 1.0
                if np.isnan(beta):
                    beta = 1.0
                u_move = beta * avg_spx_like
        
        new_spot = spot * (1 + u_move)
        
        # Reprice
        if p["instrument_type"] == "OPTION":
            T = p["T"]
            iv = p.get("iv") or 0.3
            new_price = bs_price(
                new_spot, p["strike"], T, RISK_FREE_RATE, DIV_YIELD,
                iv, p["option_type"]
            )
            pnl = (new_price - p["mid_price"]) * p["quantity"] * p["multiplier"]
            new_mv_gross = abs(p["quantity"]) * p["multiplier"] * new_spot  # gross exposure
        else:
            new_price = new_spot
            pnl = (new_spot - p["mid_price"]) * p["quantity"] * p["multiplier"]
            new_mv_gross = abs(p["quantity"]) * p["multiplier"] * new_spot
        
        total_pnl += pnl
        new_gross += new_mv_gross
        
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
            "Stress P&L": pnl,
        })
    
    return {
        "total_pnl": total_pnl,
        "new_gross_exposure": new_gross,
        "current_gross_exposure": sum(p["notional_gross"] for p in open_pos),
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
