"""
Monte Carlo engine for Strategy 3 (compounded LAI short + monthly-rolled vanilla put).

Implements EXACT mechanics from the user's hardcoded backtest spreadsheet:
  - Daily: position_size = shares × current_price
  - If position_size < target: buy more shares to restore target (books realized P&L)
  - If position_size > target: let it ride (MTM loss only)
  - Daily P&L = -shares × (new_price - old_price) = -position_size_prev × pct_change

Metrics per year:
  - Yearly P&L = sum of daily P&L
  - Yearly Avg Gross = mean of (short position_size + |put_delta| × put_contracts × 100 × spot) across days
  - Yearly Return % = Yearly P&L / Yearly Avg Gross

Aggregation across years:
  - Avg Annual Return = arithmetic mean of yearly returns (PRIMARY KPI)
  - CAGR = (Π(1 + yearly_return))^(1/n) - 1 (reinvestment interpretation)

Two simulators:
  1. historical_backtest(...) — deterministic replay of actual 2020-2026 history.
     Should reproduce user's spreadsheet: SQQQ unhedged 71.1% avg / 51-56% CAGR,
                                          SOXS unhedged 120.6% avg / 105-125% CAGR.
  2. forward_mc(...) — joint 20-day block bootstrap of (underlying_ret, VIX, VXN),
     preserves spot-vol correlation, used for forward-looking risk analysis.

Dynamic put IV:
  - QQQ puts priced at bootstrapped VXN (directly)
  - SOXX puts priced at bootstrapped VIX × SOXX-beta-to-SPX (1.33 default)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

DATA_DIR = Path(__file__).parent / "data"

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04
DIV_YIELD = 0.005
SOXX_BETA_TO_SPX = 1.33  # for scaling VIX into SOXX IV


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_csv(ticker: str) -> pd.DataFrame:
    path = DATA_DIR / f"{ticker}.csv"
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = ['Date', 'Price']
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna().sort_values('Date').reset_index(drop=True)
    return df


def load_joint_history() -> dict:
    """Load QQQ, SOXX, VIX, VXN aligned on common dates."""
    qqq = _load_csv("QQQ").rename(columns={'Price': 'QQQ'})
    soxx = _load_csv("SOXX").rename(columns={'Price': 'SOXX'})
    vix = _load_csv("VIX").rename(columns={'Price': 'VIX'})
    vxn = _load_csv("VXN").rename(columns={'Price': 'VXN'})
    df = qqq.merge(soxx, on='Date').merge(vix, on='Date').merge(vxn, on='Date').sort_values('Date').reset_index(drop=True)
    df['QQQ_ret'] = df['QQQ'].pct_change()
    df['SOXX_ret'] = df['SOXX'].pct_change()
    df = df.dropna().reset_index(drop=True)
    return {
        'dates': df['Date'].values,
        'qqq': df['QQQ'].values,
        'soxx': df['SOXX'].values,
        'qqq_ret': df['QQQ_ret'].values,
        'soxx_ret': df['SOXX_ret'].values,
        'vix': df['VIX'].values / 100.0,
        'vxn': df['VXN'].values / 100.0,
        'n_days': len(df),
    }


# ---------------------------------------------------------------------------
# Black-Scholes (vectorized)
# ---------------------------------------------------------------------------

def bs_put_price_vec(S, K, T, r, q, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T_safe = np.maximum(T, 1.0/365.0)
    sigma_safe = np.maximum(sigma, 0.01)
    sqT = np.sqrt(T_safe)
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqT)
    d2 = d1 - sigma_safe * sqT
    price = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S * np.exp(-q * T_safe) * norm.cdf(-d1)
    price = np.where(T <= 0, np.maximum(K - S, 0), price)
    return price


def bs_put_delta_vec(S, K, T, r, q, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T_safe = np.maximum(T, 1.0/365.0)
    sigma_safe = np.maximum(sigma, 0.01)
    sqT = np.sqrt(T_safe)
    d1 = (np.log(S/K) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqT)
    delta = -np.exp(-q * T_safe) * norm.cdf(-d1)
    delta_expired = np.where(S < K, -1.0, 0.0)
    delta = np.where(T <= 0, delta_expired, delta)
    return delta


# ---------------------------------------------------------------------------
# Joint block bootstrap
# ---------------------------------------------------------------------------

def joint_block_bootstrap(history: dict, horizon_days: int, n_paths: int,
                          block_length: int = 20, seed: int = 42) -> dict:
    """Return dict of arrays (n_paths, horizon_days) — qqq_ret, soxx_ret, vix, vxn."""
    rng = np.random.default_rng(seed)
    n_source = history['n_days']
    blocks_per_path = int(np.ceil(horizon_days / block_length))
    starts = rng.integers(0, n_source, size=(n_paths, blocks_per_path))
    offsets = np.arange(block_length)[None, None, :]
    idx = (starts[:, :, None] + offsets) % n_source
    idx = idx.reshape(n_paths, -1)[:, :horizon_days]
    return {
        'qqq_ret': history['qqq_ret'][idx],
        'soxx_ret': history['soxx_ret'][idx],
        'vix': history['vix'][idx],
        'vxn': history['vxn'][idx],
    }


# ---------------------------------------------------------------------------
# Core strategy simulator: works for both historical replay and MC
# ---------------------------------------------------------------------------

def _simulate_paths(
    underlying_rets: np.ndarray,    # (n_paths, horizon_days)
    iv_series: np.ndarray,          # (n_paths, horizon_days) — matching IV per day
    tracking_leverage: float,
    target_notional: float,
    put_strike_pct: float,
    put_notional_ratio: float,
    put_tenor_months: int,
    put_roll_months: int,
    include_puts: bool,
    initial_spot: float = 100.0,
) -> dict:
    """
    Core loop. Returns per-path arrays of daily P&L and daily gross exposure.

    Mechanics (per day, per path):
      1. Update underlying spot: S_t = S_{t-1} × (1 + u_return)
      2. LAI return = tracking_leverage × u_return (floored at -99.9%)
      3. Short leg:
         - Daily P&L = -position_size_prev × LAI_return (= -shares × price_change in LAI terms)
         - Update position_size *= (1 + LAI_return)
         - If position_size < target: top up (realizes profit)
      4. Put leg (if enabled):
         - Mark existing put at new spot, new IV, new T_remaining
         - Daily MTM P&L = (new_price - old_price) × contracts × 100
         - Every roll_days: sell current, buy fresh 3M at current spot & IV
    """
    n_paths, horizon_days = underlying_rets.shape

    lai_rets = np.maximum(tracking_leverage * underlying_rets, -0.999)

    # Spot path (n_paths, horizon_days+1)
    spot = np.empty((n_paths, horizon_days + 1))
    spot[:, 0] = initial_spot
    spot[:, 1:] = initial_spot * np.cumprod(1 + underlying_rets, axis=1)

    # State arrays
    position_size = np.full(n_paths, target_notional, dtype=float)  # short dollar exposure
    daily_pnl = np.zeros((n_paths, horizon_days))                    # per-day P&L
    gross_exposure = np.zeros((n_paths, horizon_days + 1))           # per-day gross (for denominator)
    gross_exposure[:, 0] = target_notional

    # Put leg state
    put_strike = np.zeros(n_paths)
    put_contracts = np.zeros(n_paths)
    put_days_elapsed = np.zeros(n_paths, dtype=int)
    put_old_price = np.zeros(n_paths)

    put_tenor_days = int(put_tenor_months * 21)
    roll_days = int(put_roll_months * 21)

    if include_puts:
        iv0 = iv_series[:, 0]
        put_strike[:] = put_strike_pct * spot[:, 0]
        # put $ notional = put_notional_ratio × short_notional
        # contracts such that contracts × spot × 100 = put_notional_ratio × short_notional
        put_contracts[:] = (put_notional_ratio * target_notional) / (spot[:, 0] * 100)
        put_old_price[:] = bs_put_price_vec(
            spot[:, 0], put_strike, put_tenor_months / 12.0,
            RISK_FREE_RATE, DIV_YIELD, iv0,
        )
        # Initial gross includes put delta exposure
        initial_delta = bs_put_delta_vec(
            spot[:, 0], put_strike, put_tenor_months / 12.0,
            RISK_FREE_RATE, DIV_YIELD, iv0,
        )
        gross_exposure[:, 0] = target_notional + np.abs(initial_delta) * put_contracts * 100 * spot[:, 0]

    # --- Daily loop ---
    for t in range(horizon_days):
        lai_r = lai_rets[:, t]
        new_spot = spot[:, t + 1]
        iv_today = iv_series[:, t]

        # Short leg P&L: loss when LAI rallies (+), gain when LAI falls (-)
        short_pnl = -position_size * lai_r
        position_size = position_size * (1 + lai_r)
        # Top-up rule: book gain if below target
        below_target = position_size < target_notional
        position_size = np.where(below_target, target_notional, position_size)

        day_pnl = short_pnl.copy()

        # Put leg
        if include_puts:
            put_days_elapsed += 1
            T_rem = np.maximum((put_tenor_days - put_days_elapsed) / 252.0, 1.0/365.0)
            new_put_price = bs_put_price_vec(
                new_spot, put_strike, T_rem,
                RISK_FREE_RATE, DIV_YIELD, iv_today,
            )
            put_mtm_pnl = (new_put_price - put_old_price) * put_contracts * 100
            day_pnl += put_mtm_pnl
            put_old_price = new_put_price

            # Monthly roll
            roll_mask = put_days_elapsed >= roll_days
            if roll_mask.any():
                fresh_strike = np.where(roll_mask, put_strike_pct * new_spot, put_strike)
                # Size fresh puts to CURRENT short notional (so put stays 3.45× short)
                fresh_contracts = np.where(
                    roll_mask,
                    (put_notional_ratio * position_size) / (new_spot * 100),
                    put_contracts,
                )
                fresh_price = bs_put_price_vec(
                    new_spot, fresh_strike, put_tenor_months / 12.0,
                    RISK_FREE_RATE, DIV_YIELD, iv_today,
                )
                # Roll P&L: sell old at new_put_price, buy fresh at fresh_price
                # Cash flow = proceeds from sell - cost of buy
                roll_pnl = np.where(
                    roll_mask,
                    new_put_price * put_contracts * 100 - fresh_price * fresh_contracts * 100,
                    0.0,
                )
                day_pnl += roll_pnl
                put_strike = fresh_strike
                put_contracts = fresh_contracts
                put_old_price = np.where(roll_mask, fresh_price, put_old_price)
                put_days_elapsed = np.where(roll_mask, 0, put_days_elapsed)

            # Update gross exposure with new put delta
            T_rem_new = np.maximum((put_tenor_days - put_days_elapsed) / 252.0, 1.0/365.0)
            put_delta_new = bs_put_delta_vec(
                new_spot, put_strike, T_rem_new,
                RISK_FREE_RATE, DIV_YIELD, iv_today,
            )
            gross_exposure[:, t + 1] = position_size + np.abs(put_delta_new) * put_contracts * 100 * new_spot
        else:
            gross_exposure[:, t + 1] = position_size

        daily_pnl[:, t] = day_pnl

    return {
        'daily_pnl': daily_pnl,                # (n_paths, horizon_days)
        'gross_exposure': gross_exposure,      # (n_paths, horizon_days + 1)
        'spot_path': spot,                     # (n_paths, horizon_days + 1)
    }


# ---------------------------------------------------------------------------
# Yearly metrics: compute avg annual return per path
# ---------------------------------------------------------------------------

def _compute_yearly_metrics(daily_pnl: np.ndarray, gross_exposure: np.ndarray) -> dict:
    """
    daily_pnl: (n_paths, horizon_days)
    gross_exposure: (n_paths, horizon_days + 1)
    Returns per-path yearly return stats and multi-year aggregates.
    """
    n_paths, horizon_days = daily_pnl.shape
    n_years = horizon_days // 252

    yearly_pnl = np.zeros((n_paths, n_years))
    yearly_avg_gross = np.zeros((n_paths, n_years))
    for y in range(n_years):
        start = y * 252
        end = (y + 1) * 252
        yearly_pnl[:, y] = daily_pnl[:, start:end].sum(axis=1)
        yearly_avg_gross[:, y] = gross_exposure[:, start + 1:end + 1].mean(axis=1)

    yearly_ret = yearly_pnl / yearly_avg_gross

    avg_annual = yearly_ret.mean(axis=1)                 # arithmetic mean of yearly returns
    # CAGR: compound yearly returns as reinvestment index
    equity = np.prod(1 + yearly_ret, axis=1)
    equity = np.where(equity > 0, equity, 1e-6)
    cagr = equity ** (1.0 / n_years) - 1

    # Max DD within path (peak-to-trough of cumulative P&L vs path avg gross)
    cum_pnl = np.cumsum(daily_pnl, axis=1)
    running_max = np.maximum.accumulate(cum_pnl, axis=1)
    path_avg_gross = gross_exposure[:, 1:].mean(axis=1)
    drawdown_dollar = cum_pnl - running_max
    max_dd = drawdown_dollar.min(axis=1) / path_avg_gross

    # Vol (annualized std of daily P&L / avg gross)
    daily_ret = daily_pnl / path_avg_gross[:, None]
    ann_vol = daily_ret.std(axis=1) * np.sqrt(252)
    sharpe = np.where(ann_vol > 0, avg_annual / ann_vol, 0)
    calmar = np.where(max_dd < 0, avg_annual / np.abs(max_dd), np.inf)

    return {
        'yearly_pnl': yearly_pnl,
        'yearly_avg_gross': yearly_avg_gross,
        'yearly_ret': yearly_ret,
        'avg_annual': avg_annual,
        'cagr': cagr,
        'max_dd': max_dd,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'calmar': calmar,
        'path_avg_gross': path_avg_gross,
        'total_pnl': cum_pnl[:, -1],
    }


# ---------------------------------------------------------------------------
# Public API: Historical Backtest (deterministic replay)
# ---------------------------------------------------------------------------

def historical_backtest(
    underlying: str,                # "QQQ" or "SOXX"
    target_notional: float = 1_000_000,
    tracking_leverage: float = -3.0,
    include_puts: bool = False,
    put_strike_pct: float = 0.95,
    put_notional_ratio: float = 3.45,
    put_tenor_months: int = 3,
    put_roll_months: int = 1,
    history: dict = None,
) -> dict:
    """
    Deterministic replay of actual 2020-2026 history.
    Should reproduce user's spreadsheet numbers when include_puts=False.
    """
    if history is None:
        history = load_joint_history()

    if underlying.upper() == "QQQ":
        u_rets = history['qqq_ret'].reshape(1, -1)
        iv = history['vxn'].reshape(1, -1)
    elif underlying.upper() == "SOXX":
        u_rets = history['soxx_ret'].reshape(1, -1)
        iv = history['vix'].reshape(1, -1) * SOXX_BETA_TO_SPX
    else:
        raise ValueError(f"Unsupported underlying: {underlying}")

    initial_spot = history['qqq'][0] if underlying.upper() == "QQQ" else history['soxx'][0]

    sim = _simulate_paths(
        underlying_rets=u_rets,
        iv_series=iv,
        tracking_leverage=tracking_leverage,
        target_notional=target_notional,
        put_strike_pct=put_strike_pct,
        put_notional_ratio=put_notional_ratio,
        put_tenor_months=put_tenor_months,
        put_roll_months=put_roll_months,
        include_puts=include_puts,
        initial_spot=initial_spot,
    )

    # Build yearly summary by actual calendar year (using history dates)
    # history['dates'] is already aligned with returns (after dropna in load).
    # length matches u_rets.shape[1]
    dates = pd.to_datetime(history['dates'])
    years = dates.year
    unique_years = sorted(set(years))

    yearly_data = []
    for y in unique_years:
        mask = (years == y)
        year_pnl = sim['daily_pnl'][0, mask].sum()
        year_avg_gross = sim['gross_exposure'][0, 1:][mask].mean()
        year_ret = year_pnl / year_avg_gross if year_avg_gross > 0 else 0
        n_days = mask.sum()
        yearly_data.append({
            'Year': int(y),
            'Days': int(n_days),
            'P&L ($)': float(year_pnl),
            'Avg Gross ($)': float(year_avg_gross),
            'Return %': float(year_ret),
        })

    yearly_df = pd.DataFrame(yearly_data)

    # Aggregate metrics
    yearly_rets = np.array([r['Return %'] for r in yearly_data])
    avg_annual = yearly_rets.mean()
    # CAGR (reinvestment interpretation over n_periods)
    equity = np.prod(1 + yearly_rets)
    n_periods = len(yearly_rets)
    cagr = equity ** (1 / n_periods) - 1 if n_periods > 0 else 0

    # Max DD on cumulative P&L / path avg gross
    cum_pnl = np.cumsum(sim['daily_pnl'][0])
    path_avg_gross = sim['gross_exposure'][0, 1:].mean()
    running_max = np.maximum.accumulate(cum_pnl)
    max_dd = float((cum_pnl - running_max).min() / path_avg_gross)

    daily_ret_series = sim['daily_pnl'][0] / path_avg_gross
    ann_vol = float(daily_ret_series.std() * np.sqrt(252))
    sharpe = avg_annual / ann_vol if ann_vol > 0 else 0
    calmar = avg_annual / abs(max_dd) if max_dd < 0 else np.inf

    return {
        'yearly_df': yearly_df,
        'avg_annual': float(avg_annual),
        'cagr': float(cagr),
        'max_dd': float(max_dd),
        'ann_vol': float(ann_vol),
        'sharpe': float(sharpe),
        'calmar': float(calmar),
        'total_pnl': float(cum_pnl[-1]),
        'final_equity_multiple': float(1 + cum_pnl[-1] / target_notional),
        'path_avg_gross': float(path_avg_gross),
        'n_days': len(sim['daily_pnl'][0]),
        'n_years': n_periods,
        'daily_pnl': sim['daily_pnl'][0],
        'gross_exposure': sim['gross_exposure'][0],
        'cum_pnl': cum_pnl,
        'dates': dates,
        'underlying': underlying,
        'include_puts': include_puts,
        'put_strike_pct': put_strike_pct,
        'put_notional_ratio': put_notional_ratio,
        'target_notional': target_notional,
        'tracking_leverage': tracking_leverage,
    }


# ---------------------------------------------------------------------------
# Public API: Forward Monte Carlo (bootstrap distribution)
# ---------------------------------------------------------------------------

def forward_mc(
    underlying: str,
    target_notional: float = 15_000_000,
    tracking_leverage: float = -3.0,
    include_puts: bool = True,
    put_strike_pct: float = 0.95,
    put_notional_ratio: float = 3.45,
    put_tenor_months: int = 3,
    put_roll_months: int = 1,
    horizon_years: int = 10,
    n_paths: int = 5000,
    block_length: int = 20,
    seed: int = 42,
    history: dict = None,
) -> dict:
    """
    Forward-looking MC: joint block bootstrap across (qqq_ret, soxx_ret, vix, vxn).
    Returns distribution of CAGR, avg annual, max DD, Calmar across paths.
    """
    if history is None:
        history = load_joint_history()

    horizon_days = horizon_years * 252
    bootstrap = joint_block_bootstrap(history, horizon_days, n_paths, block_length, seed)

    if underlying.upper() == "QQQ":
        u_rets = bootstrap['qqq_ret']
        iv = bootstrap['vxn']
        initial_spot = history['qqq'][0]
    elif underlying.upper() == "SOXX":
        u_rets = bootstrap['soxx_ret']
        iv = bootstrap['vix'] * SOXX_BETA_TO_SPX
        initial_spot = history['soxx'][0]
    else:
        raise ValueError(f"Unsupported underlying: {underlying}")

    sim = _simulate_paths(
        underlying_rets=u_rets,
        iv_series=iv,
        tracking_leverage=tracking_leverage,
        target_notional=target_notional,
        put_strike_pct=put_strike_pct,
        put_notional_ratio=put_notional_ratio,
        put_tenor_months=put_tenor_months,
        put_roll_months=put_roll_months,
        include_puts=include_puts,
        initial_spot=initial_spot,
    )

    metrics = _compute_yearly_metrics(sim['daily_pnl'], sim['gross_exposure'])

    # Summary stats
    pcts = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
    avg_ann_percentiles = {p: float(np.percentile(metrics['avg_annual'], p)) for p in pcts}
    cagr_percentiles = {p: float(np.percentile(metrics['cagr'], p)) for p in pcts}
    maxdd_percentiles = {p: float(np.percentile(metrics['max_dd'], p)) for p in pcts}
    calmar_percentiles = {p: float(np.percentile(metrics['calmar'][np.isfinite(metrics['calmar'])], p))
                          if np.isfinite(metrics['calmar']).any() else 0 for p in pcts}

    return {
        'underlying': underlying,
        'include_puts': include_puts,
        'put_strike_pct': put_strike_pct,
        'put_notional_ratio': put_notional_ratio,
        'target_notional': target_notional,
        'tracking_leverage': tracking_leverage,
        'horizon_years': horizon_years,
        'n_paths': n_paths,
        'block_length': block_length,
        # Distribution arrays
        'avg_annual_per_path': metrics['avg_annual'],
        'cagr_per_path': metrics['cagr'],
        'max_dd_per_path': metrics['max_dd'],
        'calmar_per_path': metrics['calmar'],
        'sharpe_per_path': metrics['sharpe'],
        'ann_vol_per_path': metrics['ann_vol'],
        'total_pnl_per_path': metrics['total_pnl'],
        'path_avg_gross': metrics['path_avg_gross'],
        # Summary (medians)
        'median_avg_annual': float(np.median(metrics['avg_annual'])),
        'mean_avg_annual': float(np.mean(metrics['avg_annual'])),
        'median_cagr': float(np.median(metrics['cagr'])),
        'median_max_dd': float(np.median(metrics['max_dd'])),
        'median_calmar': float(np.median(metrics['calmar'][np.isfinite(metrics['calmar'])])) if np.isfinite(metrics['calmar']).any() else 0,
        'median_sharpe': float(np.median(metrics['sharpe'])),
        'win_rate': float((metrics['avg_annual'] > 0).mean()),
        'median_path_avg_gross': float(np.median(metrics['path_avg_gross'])),
        # Percentile tables
        'avg_ann_percentiles': avg_ann_percentiles,
        'cagr_percentiles': cagr_percentiles,
        'maxdd_percentiles': maxdd_percentiles,
        'calmar_percentiles': calmar_percentiles,
    }


# ---------------------------------------------------------------------------
# Backward-compat shims for legacy analytics.py callers (Risk Curve tab)
# ---------------------------------------------------------------------------

def load_historical_returns(underlying: str) -> np.ndarray:
    """Legacy: return just daily returns for the specified underlying."""
    hist = load_joint_history()
    if underlying.upper() == "QQQ":
        return hist['qqq_ret']
    elif underlying.upper() == "SOXX":
        return hist['soxx_ret']
    return None


def simulate_compounded_short(
    target_notional: float,
    underlying: str = "QQQ",
    tracking_leverage: float = -3.0,
    horizon_days: int = 252,
    n_paths: int = 3000,
    block_length: int = 20,
    seed: int = 42,
    source_returns: np.ndarray = None,
    **kwargs,
) -> dict:
    """
    Legacy shim: returns a simplified dict matching the old API.
    Uses forward_mc under the hood but without puts.
    """
    horizon_years = max(1, horizon_days // 252)
    # Use the forward_mc path but override returns if caller supplied custom (e.g. scaled)
    if source_returns is not None:
        # Build a minimal history for the override
        fake_history = {
            'qqq_ret': source_returns,
            'soxx_ret': source_returns,
            'vix': np.full_like(source_returns, 0.20),
            'vxn': np.full_like(source_returns, 0.25),
            'dates': np.arange(len(source_returns)),
            'qqq': np.cumprod(1 + source_returns) * 100,
            'soxx': np.cumprod(1 + source_returns) * 100,
            'n_days': len(source_returns),
        }
        r = forward_mc(
            underlying=underlying, target_notional=target_notional,
            tracking_leverage=tracking_leverage, include_puts=False,
            horizon_years=horizon_years, n_paths=n_paths,
            block_length=block_length, seed=seed, history=fake_history,
        )
    else:
        r = forward_mc(
            underlying=underlying, target_notional=target_notional,
            tracking_leverage=tracking_leverage, include_puts=False,
            horizon_years=horizon_years, n_paths=n_paths,
            block_length=block_length, seed=seed,
        )
    
    # Convert to legacy return shape
    total_pnl = r['total_pnl_per_path']
    return {
        'mean_pnl': float(np.mean(total_pnl)) / horizon_years,
        'median_pnl': float(np.median(total_pnl)) / horizon_years,
        'p5': float(np.percentile(total_pnl, 5)) / horizon_years,
        'p95': float(np.percentile(total_pnl, 95)) / horizon_years,
        'win_rate': r['win_rate'],
        'all_pnl': total_pnl / horizon_years,
    }
