"""
Parse Bloomberg-format option tickers.

Examples:
  "SOXS US 01/21/28 P4"  -> underlying=SOXS, root=SOXS, expiry=2028-01-21, strike=4, type=P, mult=100
  "SOXS1 01/15/27 P2"    -> underlying=SOXS, root=SOXS1, expiry=2027-01-15, strike=2, type=P, mult=5 (post 20-for-1 reverse split)
  "SQQQ US 06/18/27 P30" -> underlying=SQQQ, root=SQQQ, expiry=2027-06-18, strike=30, type=P, mult=100

Also supports plain stock shorts entered as "SOXS" / "SQQQ" (no expiry/strike/type).
"""
import re
from datetime import datetime


# Known reverse-split-derived option roots and their new multipliers.
# After a reverse split, the existing contracts get a numeric suffix (e.g. SOXS1)
# and the deliverable shares per contract change.
# Add entries here as the manager identifies them.
POST_SPLIT_MULTIPLIERS = {
    "SOXS1": 5,    # 20-for-1 reverse split
    # "SQQQ1": X,  # if/when applicable
}


def parse_bbg_option(ticker: str) -> dict:
    """
    Parse a Bloomberg option ticker. Returns dict with:
      underlying, option_root, expiry (YYYY-MM-DD), strike, option_type, multiplier,
      or raises ValueError on bad format.
    Also accepts plain stock tickers like 'SQQQ' or 'SOXS US Equity'.
    """
    t = ticker.strip().upper()
    
    # Strip BBG exchange markers ("US", "US Equity", "Equity") anywhere in ticker
    t = re.sub(r"\s+US\s+EQUITY\b", " ", t)
    t = re.sub(r"\bUS\s+EQUITY\b", " ", t)
    t = re.sub(r"\s+EQUITY\b", " ", t)
    t = re.sub(r"\s+US(?=\s|$)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    
    # Check for full option format:
    # ROOT [US] MM/DD/YY [C|P]STRIKE
    # Handle optional 'US' already removed. Now looking for ROOT MM/DD/YY [CP]STRIKE
    opt_match = re.match(
        r"^(\S+?)\s+(\d{1,2}/\d{1,2}/\d{2,4})\s+([CP])([\d.]+)$",
        t
    )
    if opt_match:
        root, date_str, opt_type, strike_str = opt_match.groups()
        # Parse date - handle both 2-digit and 4-digit year
        try:
            dt = datetime.strptime(date_str, "%m/%d/%y")
        except ValueError:
            dt = datetime.strptime(date_str, "%m/%d/%Y")
        expiry = dt.strftime("%Y-%m-%d")
        strike = float(strike_str)
        
        # Derive underlying (strip numeric suffix)
        underlying = re.sub(r"\d+$", "", root)
        
        # Determine multiplier
        multiplier = POST_SPLIT_MULTIPLIERS.get(root, 100)
        
        return {
            "bbg_ticker": ticker,
            "instrument_type": "OPTION",
            "underlying": underlying,
            "option_root": root,
            "expiry": expiry,
            "strike": strike,
            "option_type": opt_type,
            "multiplier": multiplier,
        }
    
    # Fallback: treat as plain stock/ETF (for cash short positions)
    if re.match(r"^[A-Z]+\d*$", t):
        underlying = re.sub(r"\d+$", "", t)
        return {
            "bbg_ticker": ticker,
            "instrument_type": "STOCK",
            "underlying": underlying,
            "option_root": None,
            "expiry": None,
            "strike": None,
            "option_type": None,
            "multiplier": 1,
        }
    
    raise ValueError(
        f"Could not parse ticker: '{ticker}'. "
        f"Expected BBG option format (e.g. 'SOXS US 01/21/28 P4') "
        f"or plain ticker (e.g. 'SOXS')."
    )


def years_to_expiry(expiry_str: str, asof_date=None) -> float:
    """Compute ACT/365 years between asof_date and expiry."""
    if asof_date is None:
        asof_date = datetime.now().date()
    elif isinstance(asof_date, str):
        asof_date = datetime.strptime(asof_date, "%Y-%m-%d").date()
    
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (exp - asof_date).days
    return max(days, 0) / 365.0
