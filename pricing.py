"""
Black-Scholes pricing and Greeks.

Used for:
- Risk curve scenarios (repricing options at hypothetical underlying prices)
- Backing out Greeks from IV provided by yfinance
- Stress scenarios
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def bs_price(S, K, T, r, q, sigma, opt_type):
    """Black-Scholes price for a European call or put."""
    if T <= 0:
        if opt_type.upper() == "C":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    if sigma <= 0:
        fwd = S * np.exp((r - q) * T)
        pv_fwd = fwd * np.exp(-r * T)
        pv_K = K * np.exp(-r * T)
        if opt_type.upper() == "C":
            return max(pv_fwd - pv_K, 0)
        else:
            return max(pv_K - pv_fwd, 0)
    
    sigma = max(sigma, 1e-8)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if opt_type.upper() == "C":
        return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)


def bs_greeks(S, K, T, r, q, sigma, opt_type):
    """
    Black-Scholes Greeks.
    Returns dict: delta, gamma, theta (per year), vega (per 1.00 vol), rho.
    """
    if T <= 0 or sigma <= 0:
        intrinsic_sign = 1 if opt_type.upper() == "C" else -1
        if opt_type.upper() == "C":
            delta = 1.0 if S > K else 0.0
        else:
            delta = -1.0 if S < K else 0.0
        return {"delta": delta, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}
    
    sigma = max(sigma, 1e-8)
    sqT = np.sqrt(T)
    d1 = (np.log(S/K) + (r - q + 0.5*sigma**2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT
    
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    Nmd1 = norm.cdf(-d1)
    Nmd2 = norm.cdf(-d2)
    phi_d1 = norm.pdf(d1)
    
    if opt_type.upper() == "C":
        delta = np.exp(-q*T) * Nd1
        theta = (-S * np.exp(-q*T) * phi_d1 * sigma / (2 * sqT)
                 + q * S * np.exp(-q*T) * Nd1
                 - r * K * np.exp(-r*T) * Nd2)
        rho = K * T * np.exp(-r*T) * Nd2
    else:
        delta = -np.exp(-q*T) * Nmd1
        theta = (-S * np.exp(-q*T) * phi_d1 * sigma / (2 * sqT)
                 - q * S * np.exp(-q*T) * Nmd1
                 + r * K * np.exp(-r*T) * Nmd2)
        rho = -K * T * np.exp(-r*T) * Nmd2
    
    gamma = np.exp(-q*T) * phi_d1 / (S * sigma * sqT)
    vega = S * np.exp(-q*T) * phi_d1 * sqT  # per 1.00 vol change (divide by 100 for per-vol-point)
    
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta": float(theta),   # per year; divide by 365 for per day
        "vega": float(vega),     # per 1.00 vol change; divide by 100 for per vol-point
        "rho": float(rho),
    }


def implied_vol(price, S, K, T, r, q, opt_type):
    """Solve for IV given observed option price."""
    if T <= 0 or price <= 0:
        return np.nan
    
    intrinsic = max(0, (S - K) if opt_type.upper() == "C" else (K - S))
    if price < intrinsic * 0.999:
        return np.nan  # Arbitrage violation or stale price
    
    def objective(sig):
        return bs_price(S, K, T, r, q, sig, opt_type) - price
    
    try:
        iv = brentq(objective, 1e-4, 5.0, xtol=1e-6, maxiter=100)
        return iv
    except Exception:
        return np.nan
