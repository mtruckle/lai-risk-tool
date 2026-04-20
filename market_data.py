"""
Market data layer using yfinance.

Provides:
- Spot prices
- Option chain snapshots (with Greeks from yfinance)
- Historical returns (for beta calculations)
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import lru_cache
import time


# Cache to avoid hammering yfinance
_price_cache = {}
_cache_ttl = 60  # seconds


def get_spot(ticker: str, stale_ok: bool = False) -> float:
    """Latest spot price (15-min delayed on yfinance)."""
    now = time.time()
    if ticker in _price_cache:
        ts, price = _price_cache[ticker]
        if now - ts < _cache_ttl:
            return price
    
    try:
        tk = yf.Ticker(ticker)
        # fast_info for quick quote
        fi = tk.fast_info
        price = float(fi.last_price) if fi.last_price else None
        if price is None or price <= 0:
            # Fallback to history
            hist = tk.history(period="5d", auto_adjust=False)
            if len(hist) > 0:
                price = float(hist["Close"].iloc[-1])
            else:
                return None
        _price_cache[ticker] = (now, price)
        return price
    except Exception as e:
        print(f"get_spot({ticker}) failed: {e}")
        return None


def get_option_chain(ticker: str, expiry: str):
    """
    Get full option chain for ticker at given expiry (YYYY-MM-DD).
    Returns dict with 'calls' and 'puts' DataFrames, or None if unavailable.
    """
    try:
        tk = yf.Ticker(ticker)
        # yfinance uses YYYY-MM-DD
        available = tk.options
        if expiry not in available:
            return None
        chain = tk.option_chain(expiry)
        return {"calls": chain.calls, "puts": chain.puts}
    except Exception as e:
        print(f"get_option_chain({ticker}, {expiry}) failed: {e}")
        return None


def get_option_quote(root: str, expiry: str, strike: float, opt_type: str) -> dict:
    """
    Look up a specific option contract.
    Returns dict with bid, ask, last, mid, iv, delta, gamma, theta, vega, openInterest.
    yfinance provides IV but NOT Greeks directly — we need to compute Greeks ourselves
    from the IV yfinance gives us (user requested this approach).
    """
    chain = get_option_chain(root, expiry)
    if chain is None:
        return None
    
    df = chain["calls"] if opt_type.upper() == "C" else chain["puts"]
    # Match by strike (yfinance uses exact float, need tolerance)
    matches = df[np.isclose(df["strike"], strike, atol=0.005)]
    if len(matches) == 0:
        return None
    row = matches.iloc[0]
    
    bid = float(row.get("bid", 0) or 0)
    ask = float(row.get("ask", 0) or 0)
    last = float(row.get("lastPrice", 0) or 0)
    iv = float(row.get("impliedVolatility", 0) or 0)
    
    # Mid: prefer (bid+ask)/2 if both positive, else last
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    else:
        mid = last
    
    return {
        "bid": bid,
        "ask": ask,
        "last": last,
        "mid": mid,
        "iv": iv,
        "volume": int(row.get("volume", 0) or 0),
        "openInterest": int(row.get("openInterest", 0) or 0),
    }


def get_historical_returns(ticker: str, period: str = "2y") -> pd.Series:
    """Daily returns series for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period, auto_adjust=True)
        if len(hist) == 0:
            return pd.Series(dtype=float)
        return hist["Close"].pct_change().dropna()
    except Exception as e:
        print(f"get_historical_returns({ticker}) failed: {e}")
        return pd.Series(dtype=float)


@lru_cache(maxsize=64)
def compute_beta(ticker: str, benchmark: str = "SPY") -> float:
    """
    Compute beta vs SPY (which proxies SPX) using:
    - 1/3 weight on daily returns, 1Y lookback
    - 2/3 weight on weekly returns, 2Y lookback
    """
    try:
        # Daily returns, 1y
        daily = get_historical_returns(ticker, period="1y")
        spy_daily = get_historical_returns(benchmark, period="1y")
        # Align
        df_d = pd.concat([daily.rename("t"), spy_daily.rename("b")], axis=1).dropna()
        if len(df_d) < 50:
            beta_d = np.nan
        else:
            cov_d = df_d.cov().loc["t", "b"]
            var_d = df_d["b"].var()
            beta_d = cov_d / var_d if var_d > 0 else np.nan
        
        # Weekly returns, 2y
        weekly_t = get_historical_returns(ticker, period="2y").resample("W-FRI").apply(
            lambda x: (1 + x).prod() - 1
        )
        weekly_b = get_historical_returns(benchmark, period="2y").resample("W-FRI").apply(
            lambda x: (1 + x).prod() - 1
        )
        df_w = pd.concat([weekly_t.rename("t"), weekly_b.rename("b")], axis=1).dropna()
        if len(df_w) < 20:
            beta_w = np.nan
        else:
            cov_w = df_w.cov().loc["t", "b"]
            var_w = df_w["b"].var()
            beta_w = cov_w / var_w if var_w > 0 else np.nan
        
        # Combine
        if np.isnan(beta_d) and np.isnan(beta_w):
            return np.nan
        if np.isnan(beta_d):
            return beta_w
        if np.isnan(beta_w):
            return beta_d
        return (1/3) * beta_d + (2/3) * beta_w
    except Exception as e:
        print(f"compute_beta({ticker}) failed: {e}")
        return np.nan


def realized_vol(ticker: str, lookback_days: int = 260) -> float:
    """Annualized realized vol from daily log returns over lookback window."""
    try:
        rets = get_historical_returns(ticker, period=f"{max(lookback_days+30, 365)}d")
        if len(rets) < 20:
            return np.nan
        log_rets = np.log(1 + rets.iloc[-lookback_days:])
        return float(log_rets.std() * np.sqrt(252))
    except Exception as e:
        print(f"realized_vol({ticker}) failed: {e}")
        return np.nan


def get_available_expiries(ticker: str) -> list:
    """List of available option expiries for ticker."""
    try:
        tk = yf.Ticker(ticker)
        return list(tk.options)
    except Exception as e:
        print(f"get_available_expiries({ticker}) failed: {e}")
        return []
