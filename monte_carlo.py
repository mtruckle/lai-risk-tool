"""
Monte Carlo engine for compounded LAI short strategy.

Strategy mechanic:
- Short $TARGET_NOTIONAL of LAI ETF (e.g. $15M SQQQ)
- Each trading day close:
    * If short MV < TARGET_NOTIONAL: buy more to top up to TARGET (books the gain)
    * If short MV >= TARGET_NOTIONAL: take no action (ride the loss)
- Daily LAI return approximated as: leverage × underlying_daily_return
  (using tracking_leverage, default -3x; can be stressed to -4/-5/-6)

Simulates GBM on the UNDERLYING (e.g. QQQ), then mechanically applies leverage
to get LAI daily returns. Applies top-up rule each day. Tracks $ P&L per path.
"""
import numpy as np


def simulate_compounded_short(
    target_notional: float,
    underlying_vol: float,
    tracking_leverage: float = -3.0,
    horizon_days: int = 252,
    underlying_drift: float = 0.0,
    n_paths: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Monte Carlo expected annual P&L from a compounded LAI short with daily top-up.
    
    Args:
        target_notional: $ target to maintain short at (e.g. 15_000_000)
        underlying_vol: annualized vol of the UNDERLYING index (e.g. 0.22 for QQQ)
        tracking_leverage: leverage of LAI vs underlying (default -3.0, stress: -4/-5/-6)
        horizon_days: trading days to simulate (default 252 = 1 year)
        underlying_drift: annualized drift of underlying (default 0% per user spec)
        n_paths: Monte Carlo paths
        seed: RNG seed for reproducibility
    
    Returns dict with:
        mean_pnl: mean annual P&L in $
        median_pnl, p5, p25, p75, p95: percentiles
        mean_pnl_pct: mean annual return as % of target_notional
        paths_sample: sample of 50 paths (for display)
    """
    dt = 1.0 / 252.0
    rng = np.random.default_rng(seed)
    
    # Simulate underlying log-returns: dS/S = μ dt + σ dW
    # log_ret ~ N((μ - σ²/2) dt, σ²dt)
    mean_log = (underlying_drift - 0.5 * underlying_vol ** 2) * dt
    sd_log = underlying_vol * np.sqrt(dt)
    
    # Shape: (n_paths, horizon_days)
    underlying_log_rets = rng.normal(mean_log, sd_log, size=(n_paths, horizon_days))
    underlying_arith_rets = np.exp(underlying_log_rets) - 1.0
    
    # LAI daily arithmetic return = tracking_leverage × underlying arith return
    lai_rets = tracking_leverage * underlying_arith_rets
    # Floor at -99.9% to avoid LAI going negative (not realistic, keeps math stable)
    lai_rets = np.maximum(lai_rets, -0.999)
    
    # Simulate the compounded short P&L for each path
    # State: short_mv (signed negative, we track as positive for simplicity as liability)
    # We track "short liability" = current MV we owe to buy back; starts at TARGET
    # Daily pnl = -short_liability × lai_ret  (if LAI up 1%, short loses 1%)
    # After pnl, new_liability = short_liability × (1 + lai_ret)
    # Top-up rule: if new_liability < TARGET, we add (TARGET - new_liability) of short
    #   → cumulative P&L gets banked (TARGET - new_liability), then liability reset to TARGET
    # If new_liability >= TARGET: do nothing, liability continues
    
    # Vectorized path iteration
    cum_pnl = np.zeros(n_paths)
    short_liability = np.full(n_paths, target_notional, dtype=float)
    
    for t in range(horizon_days):
        r = lai_rets[:, t]
        # Daily P&L (for a short: positive r = loss)
        daily_pnl = -short_liability * r
        cum_pnl += daily_pnl
        # Update liability
        short_liability = short_liability * (1.0 + r)
        # Top-up rule
        below_target = short_liability < target_notional
        # For paths below target: the "gain" is (target - current_liability); already captured in cum_pnl
        # We just reset liability to target — no additional P&L is booked (already accounted for)
        short_liability = np.where(below_target, target_notional, short_liability)
    
    # Final unrealized loss from any remaining liability above target
    # When we end above target, there's an unrealized loss sitting on the position
    # That's already captured in cum_pnl since we've been marking daily
    
    mean_pnl = float(np.mean(cum_pnl))
    median_pnl = float(np.median(cum_pnl))
    p5 = float(np.percentile(cum_pnl, 5))
    p25 = float(np.percentile(cum_pnl, 25))
    p75 = float(np.percentile(cum_pnl, 75))
    p95 = float(np.percentile(cum_pnl, 95))
    win_rate = float((cum_pnl > 0).mean())
    
    return {
        "mean_pnl": mean_pnl,
        "median_pnl": median_pnl,
        "p5": p5,
        "p25": p25,
        "p75": p75,
        "p95": p95,
        "mean_pnl_pct": mean_pnl / target_notional,
        "median_pnl_pct": median_pnl / target_notional,
        "win_rate": win_rate,
        "n_paths": n_paths,
        "horizon_days": horizon_days,
        "target_notional": target_notional,
        "underlying_vol": underlying_vol,
        "tracking_leverage": tracking_leverage,
        "all_pnl": cum_pnl,  # full distribution for plotting
    }


def simulate_with_vol_sweep(
    target_notional: float,
    vol_grid: list,
    tracking_leverage: float = -3.0,
    horizon_days: int = 252,
    n_paths: int = 5000,
) -> dict:
    """Run MC across multiple vol points. Returns dict mapping vol -> mean_pnl."""
    results = {}
    for v in vol_grid:
        r = simulate_compounded_short(
            target_notional=target_notional,
            underlying_vol=v,
            tracking_leverage=tracking_leverage,
            horizon_days=horizon_days,
            n_paths=n_paths,
            seed=42,
        )
        results[v] = r["mean_pnl"]
    return results


def simulate_with_spot_sweep(
    target_notional: float,
    underlying_vol: float,
    spot_move_grid: list,
    tracking_leverage: float = -3.0,
    horizon_days: int = 252,
    n_paths: int = 5000,
) -> dict:
    """Run MC across multiple assumed underlying drifts (expressed as total 1y moves).
    For spot move m% over 1y, drift = log(1+m) / 1y (geometric average)."""
    results = {}
    for move in spot_move_grid:
        drift = np.log(1.0 + move)  # annualized continuous drift
        r = simulate_compounded_short(
            target_notional=target_notional,
            underlying_vol=underlying_vol,
            tracking_leverage=tracking_leverage,
            horizon_days=horizon_days,
            underlying_drift=drift,
            n_paths=n_paths,
            seed=42,
        )
        results[move] = r["mean_pnl"]
    return results
