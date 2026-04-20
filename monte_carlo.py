"""
Monte Carlo engine for compounded LAI short strategy — HISTORICAL BLOCK BOOTSTRAP.

Replaces earlier GBM simulator. We resample actual historical daily returns of the
UNDERLYING (QQQ for SQQQ, SOXX for SOXS) in 5-day blocks to preserve short-term
serial correlation (vol clustering, momentum, mean reversion within a week).

Block bootstrap specifics:
- 5-day blocks
- Circular: blocks can wrap around end of sample so edge days aren't under-sampled
- Random block starts (with replacement)
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def load_historical_returns(underlying: str) -> np.ndarray:
    """Load historical daily arithmetic returns. Tries local CSV, falls back to yfinance."""
    ticker = underlying.upper()
    csv_path = DATA_DIR / f"{ticker}.csv"
    
    if csv_path.exists():
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        df.columns = ['Date', 'Price']
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna().sort_values('Date').reset_index(drop=True)
        return df['Price'].pct_change().dropna().values
    
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period="15y", auto_adjust=True)
        if len(hist) < 100:
            return None
        return hist['Close'].pct_change().dropna().values
    except Exception as e:
        print(f"load_historical_returns({ticker}) failed: {e}")
        return None


def block_bootstrap_returns(
    source_returns: np.ndarray,
    horizon_days: int,
    n_paths: int,
    block_length: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Circular block bootstrap. Returns array of shape (n_paths, horizon_days)."""
    rng = np.random.default_rng(seed)
    n_source = len(source_returns)
    if n_source < block_length:
        raise ValueError(f"Need >= {block_length} historical days, got {n_source}")
    
    blocks_per_path = int(np.ceil(horizon_days / block_length))
    starts = rng.integers(0, n_source, size=(n_paths, blocks_per_path))
    offsets = np.arange(block_length)[None, None, :]
    idx = (starts[:, :, None] + offsets) % n_source
    idx = idx.reshape(n_paths, -1)[:, :horizon_days]
    return source_returns[idx]


def simulate_compounded_short(
    target_notional: float,
    underlying: str = "QQQ",
    tracking_leverage: float = -3.0,
    horizon_days: int = 252,
    n_paths: int = 5000,
    block_length: int = 5,
    seed: int = 42,
    source_returns: np.ndarray = None,
) -> dict:
    """
    MC compounded short P&L using historical block bootstrap of the underlying.
    Daily top-up rule: if short MV < target, top up to target (books the gain).
    """
    if source_returns is None:
        source_returns = load_historical_returns(underlying)
        if source_returns is None or len(source_returns) < 100:
            raise ValueError(f"Could not load historical returns for {underlying}")
    
    underlying_rets = block_bootstrap_returns(
        source_returns, horizon_days, n_paths, block_length, seed
    )
    lai_rets = np.maximum(tracking_leverage * underlying_rets, -0.999)
    
    # Compounded short with daily top-up (vectorized path iteration)
    cum_pnl = np.zeros(n_paths)
    liability = np.full(n_paths, target_notional, dtype=float)
    for t in range(horizon_days):
        r = lai_rets[:, t]
        cum_pnl += -liability * r
        liability = liability * (1.0 + r)
        below = liability < target_notional
        liability = np.where(below, target_notional, liability)
    
    # Static short (no rebalancing) for comparison
    static_liability = np.full(n_paths, target_notional, dtype=float)
    for t in range(horizon_days):
        static_liability = static_liability * (1.0 + lai_rets[:, t])
    static_pnl = target_notional - static_liability
    
    # LAI terminal ratio (for diagnostics)
    lai_terminal = np.prod(1.0 + lai_rets, axis=1)
    
    return {
        "mean_pnl": float(np.mean(cum_pnl)),
        "median_pnl": float(np.median(cum_pnl)),
        "p5": float(np.percentile(cum_pnl, 5)),
        "p25": float(np.percentile(cum_pnl, 25)),
        "p75": float(np.percentile(cum_pnl, 75)),
        "p95": float(np.percentile(cum_pnl, 95)),
        "mean_pnl_pct": float(np.mean(cum_pnl)) / target_notional,
        "median_pnl_pct": float(np.median(cum_pnl)) / target_notional,
        "win_rate": float((cum_pnl > 0).mean()),
        "n_paths": n_paths,
        "horizon_days": horizon_days,
        "target_notional": target_notional,
        "tracking_leverage": tracking_leverage,
        "block_length": block_length,
        "underlying": underlying,
        "all_pnl": cum_pnl,
        "static_all_pnl": static_pnl,
        "static_mean_pnl": float(np.mean(static_pnl)),
        "static_median_pnl": float(np.median(static_pnl)),
        "lai_terminal_p5": float(np.percentile(lai_terminal, 5)),
        "lai_terminal_median": float(np.median(lai_terminal)),
        "lai_terminal_p95": float(np.percentile(lai_terminal, 95)),
        "lai_terminal_max": float(lai_terminal.max()),
        "source_n_days": len(source_returns),
    }
