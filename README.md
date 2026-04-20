# LAI Risk Tool

Local-hosted risk management platform for a long-volatility hedge fund strategy that shorts leveraged/inverse ETFs (SQQQ, SOXS) and hedges with vanilla puts on the underlying (QQQ, SOXX).

## What it does

- **Positions** — Add/close/track options and stock positions. Supports Bloomberg-format tickers (`SOXS US 01/21/28 P4`) including post-reverse-split options (`SOXS1 01/15/27 P2` with multiplier 5).
- **Summary** — Portfolio value, delta/beta-adjusted exposure, all Greeks, breakdown by underlying, realized + unrealized P&L.
- **Risk Curve** — P&L scenarios for SPX moves from -15% to +15% (2.5% increments), repricing every option via Black-Scholes at constant IV.
- **Expected Return** — LAI vol-decay expected return per option (using the formula you specified), weighted portfolio average, net of put-protection cost.
- **Stress Scenario** — SQQQ +130% / SOXS +200% (customisable), with historical reference periods (Dec 2021–Jun 2022 for SQQQ; Jan 2025–Apr 2025 for SOXS). Computes stress P&L, stressed gross exposure, and the steady-state gross deployment consistent with a $150M stress-gross limit.

## Installation

```bash
# Requires Python 3.9+
pip install streamlit yfinance pandas numpy scipy plotly
```

## Running

```bash
cd lai_risk_tool
streamlit run app.py
```

This will launch the app at `http://localhost:8501` and open your browser automatically.

## File overview

| File | Purpose |
|------|---------|
| `app.py` | Streamlit UI — 5 pages (Positions, Summary, Risk Curve, Expected Return, Stress) |
| `db.py` | SQLite persistence for positions |
| `bbg_parser.py` | Parse Bloomberg-format option tickers; handles post-split tickers |
| `market_data.py` | Yahoo Finance wrappers for spot, option chains, beta, realized vol |
| `pricing.py` | Black-Scholes pricing + Greeks + IV solver |
| `vol_decay.py` | Leveraged-ETF decay formula + expected-return-to-expiry |
| `analytics.py` | Position enrichment, portfolio summary, risk curve, stress, steady-state solver |

## Key methodology

### Vol decay formula (user-specified)

```
decay(T) = (1 + r_1y_underlying)^L × exp((L − L²) × σ² × T / 2) − 1
```

Where:
- `r_1y_underlying` = assumed 1-year underlying return (default 5%)
- `L` = leverage multiplier (e.g. -3 for SQQQ/SOXS)
- `σ` = underlying annualized realized vol (260-day lookback)
- `T` = years to expiry

**Note on interpretation:** your typed formula (`D5^C1/2`) seems to imply vol-raised-to-years, which is dimensionally odd. I've interpreted it as `σ² × T / 2` (variance-time), which is the standard leveraged-ETF formula. Confirm if you want the other interpretation.

### Beta (vs SPY)

```
β = (1/3) × β_daily_1y + (2/3) × β_weekly_2y
```

For leveraged/inverse ETFs, the effective beta = underlying β × leverage. E.g. SQQQ effective β ≈ 1.12 × -3 = -3.36.

### Risk curve

For each SPX scenario:
- Non-LAI underlyings: move = β × SPX move
- LAI ETFs: move = leverage × underlying_β × SPX move
- Options repriced via BS with IV held constant

### Stress scenario

Given stressed LAI moves and stressed leverage:
- SQQQ +130% at -4x → QQQ -32.5%
- SOXS +200% at -6x → SOXX -33.3%

Gross exposure under stress grows because short ETF MVs grow with the rally (you need margin to stay short) and put MVs grow as they go deep ITM.

**Steady-state gross** = gross_limit / (stressed_gross / current_gross). If stress multiplies gross by 3x, steady-state = $150M / 3 = $50M.

## Gotchas

1. **yfinance is 15-min delayed.** Good for risk analysis, not for execution. Consider IBKR integration for production.
2. **Option chains** — yfinance only has current snapshot, not historical. For backtesting historical portfolios, you'd need a different data source.
3. **Greeks** — sourced from yfinance's IV, but Greeks themselves are computed by this code via Black-Scholes for consistency.
4. **Post-split options** — `SOXS1` root is hardcoded with multiplier 5. If new post-split tickers appear, add them to `POST_SPLIT_MULTIPLIERS` in `bbg_parser.py`.

## Database

Positions are stored in SQLite at `positions.db` in the project directory. It survives app restarts. Back it up as needed.
