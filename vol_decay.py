"""
Volatility decay expected-return calculations for leveraged / inverse ETF options.

Formula (per user spec):
  decay = (1 + r_1y_underlying)^A2 * EXP((A2 - A2^2) * sigma^2 * T / 2) - 1

Where:
  r_1y_underlying = assumed 1-year underlying price return (default 5%)
  A2 = leverage multiplier (e.g. -3 for SQQQ/SOXS)
  sigma = underlying annualized realized vol (from 260-day lookback)
  T = years to expiry
"""
import numpy as np
from bbg_parser import years_to_expiry


# Known leveraged/inverse ETFs and their leverage vs underlying.
# Underlying determines the vol/beta; leverage is the multiplier to apply.
LAI_ETF_MAP = {
    # Nasdaq 100 (-3x inverse)
    "SQQQ": {"underlying": "QQQ",  "leverage": -3},
    # Philadelphia Semiconductor (-3x inverse)
    "SOXS": {"underlying": "SOXX", "leverage": -3},
    # Other common ones (add as needed)
    "SPXS": {"underlying": "SPY",  "leverage": -3},  # -3x S&P 500
    "TZA":  {"underlying": "IWM",  "leverage": -3},  # -3x Russell 2000
    "TECS": {"underlying": "XLK",  "leverage": -3},  # -3x Tech
    "LABD": {"underlying": "XBI",  "leverage": -3},  # -3x Biotech
    # Long leveraged (in case of calls)
    "TQQQ": {"underlying": "QQQ",  "leverage":  3},
    "SOXL": {"underlying": "SOXX", "leverage":  3},
    "SPXL": {"underlying": "SPY",  "leverage":  3},
}


def is_lai_etf(underlying: str) -> bool:
    return underlying.upper() in LAI_ETF_MAP


def get_lai_info(underlying: str) -> dict:
    return LAI_ETF_MAP.get(underlying.upper())


def compute_vol_decay(
    r_1y: float,
    leverage: int,
    underlying_vol: float,
    years: float
) -> float:
    """
    Expected total price decay of a leveraged/inverse ETF over the given years,
    conditional on the assumed 1y underlying return and vol.
    
    Returns a decimal (e.g. -0.80 = -80% expected decay).
    """
    if years <= 0 or leverage == 0:
        return 0.0
    try:
        return (1 + r_1y) ** leverage * np.exp((leverage - leverage**2) * underlying_vol**2 * years / 2) - 1
    except Exception:
        return np.nan


def compute_expected_return_to_expiry(
    position: dict,
    spot: float,
    mid_price: float,
    underlying_vol: float,
    r_1y: float = 0.05,
    asof_date=None,
) -> dict:
    """
    Compute the expected return to fair value at expiry for an LAI option.
    
    Returns dict with:
      - expected_decay: fractional decay of the LAI ETF over T
      - expected_spot_at_expiry: spot * (1 + expected_decay)
      - intrinsic_at_expiry: for a put, max(K - exp_spot, 0)
      - expected_return_total: intrinsic_at_expiry / mid_price - 1
      - expected_return_annualized: CAGR equivalent
      - years: time to expiry
    
    Returns None for non-options or non-LAI positions.
    """
    if position.get("instrument_type") != "OPTION":
        return None
    
    lai_info = get_lai_info(position["underlying"])
    if lai_info is None:
        return None
    
    leverage = lai_info["leverage"]
    T = years_to_expiry(position["expiry"], asof_date)
    if T <= 0:
        return None
    if mid_price is None or mid_price <= 0:
        return None
    if spot is None or spot <= 0:
        return None
    if underlying_vol is None or np.isnan(underlying_vol):
        return None
    
    decay = compute_vol_decay(r_1y, leverage, underlying_vol, T)
    exp_spot = spot * (1 + decay)
    
    if position["option_type"].upper() == "P":
        intrinsic_at_exp = max(position["strike"] - exp_spot, 0)
    else:
        intrinsic_at_exp = max(exp_spot - position["strike"], 0)
    
    total_return = intrinsic_at_exp / mid_price - 1
    # CAGR: handle negative returns (CAGR undefined for negative total-return on positive capital at T<1)
    if 1 + total_return > 0:
        annualized = (1 + total_return) ** (1 / T) - 1
    else:
        # Total loss or worse → annualized effectively -100%
        annualized = -1.0
    
    return {
        "years": T,
        "leverage": leverage,
        "underlying_vol": underlying_vol,
        "expected_decay": decay,
        "expected_spot_at_expiry": exp_spot,
        "intrinsic_at_expiry": intrinsic_at_exp,
        "expected_return_total": total_return,
        "expected_return_annualized": annualized,
    }
