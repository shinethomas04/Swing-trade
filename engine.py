"""
Swing Trade Engine — shared analysis & persistence layer
=========================================================
A desktop application that:
  1. Identifies the top 50 publicly traded U.S. stocks by dollar volume
     averaged over the last 4 trading days.
  2. Pulls 30 days of daily OHLCV history for each.
  3. Runs a multi-factor swing-trade algorithm (trend, momentum, mean-reversion,
     volatility breakout) and produces a ranked list of candidates with
     entry / stop / target levels and a A+/A/B/C/D conviction rating.
  4. Saves every recommendation to a local SQLite database, then later
     replays them against actual price action to show:
        - Per-trade win/loss, P&L %, days held
        - Strategy back-test equity curves (e.g. "Long A+ only", "All longs",
          "Top 5 by score", etc.) given a fixed capital allocation per trade.

Dependencies:
    pip install yfinance pandas numpy customtkinter
    (falls back to tkinter automatically if customtkinter is not installed)

Run:
    python swing_trader.py
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Data: yfinance is the simplest free source ----------------------------
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None  # GUI will surface a clear error when the user clicks Run.



# ===========================================================================
#                                UNIVERSE
# ===========================================================================
# Liquid U.S. equity universe to scan. ~400 of the most actively traded names
# across all major sectors, popular ETFs, and high-beta movers. We re-rank
# these by 4-day dollar volume on every run, then walk the ranked list until
# we've found N A-or-better setups (configurable in the UI).
UNIVERSE: List[str] = sorted(set([
    # Mega-caps & FAANG+
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "BRK-B", "JPM", "V", "MA", "UNH", "HD", "WMT", "PG", "XOM", "CVX",
    "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    "KO", "PEP", "MCD", "VZ", "T", "TMUS", "CSCO", "IBM", "ACN", "TXN",

    # Semis / AI
    "AMD", "INTC", "MU", "TSM", "ASML", "QCOM", "ARM", "MRVL", "SMCI",
    "AMAT", "LRCX", "KLAC", "ON", "MPWR", "ADI", "NXPI", "MCHP", "SWKS",
    "QRVO", "TER", "ENTG", "WOLF", "AEHR", "CRDO", "ALAB", "ASTS", "INDI",
    "PLTR", "SNOW", "CRWD", "PANW", "NET", "DDOG", "MDB", "ZS", "S",
    "FTNT", "OKTA", "TENB", "VRNS", "CYBR", "QLYS", "RPD", "PATH", "BRZE",
    "AI", "SOUN", "BBAI", "RXRX", "SERV", "GTLB", "FROG", "ESTC",

    # Software / Internet
    "ORCL", "CRM", "ADBE", "NOW", "INTU", "SHOP", "UBER", "ABNB", "DASH",
    "SQ", "PYPL", "COIN", "HOOD", "RBLX", "U", "ROKU", "SPOT", "PINS",
    "SNAP", "DKNG", "BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV",
    "WDAY", "TEAM", "ZM", "DOCU", "TWLO", "HUBS", "BILL", "ASAN", "MNDY",
    "DOCN", "FSLY", "AKAM", "VEEV", "ANET", "FFIV", "CIEN", "JNPR",

    # Financials
    "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK", "AXP", "COF", "USB",
    "PNC", "TFC", "BK", "STT", "AMP", "MET", "PRU", "AIG", "TRV", "ALL",
    "PGR", "CB", "SPGI", "MCO", "ICE", "CME", "NDAQ", "MKTX", "CBOE",
    "FI", "FIS", "GPN", "DFS", "SOFI", "AFRM", "UPST", "LC", "OPEN",

    # Industrials / Aerospace / Defense
    "BA", "CAT", "DE", "GE", "HON", "RTX", "LMT", "NOC", "GD", "TDG",
    "LHX", "TXT", "HII", "AXON", "PH", "EMR", "ETN", "ITW", "ROK",
    "PCAR", "WAB", "URI", "FAST", "GWW", "DOV", "XYL", "CMI",
    "CARR", "OTIS", "JCI", "FLR", "PWR", "MAS", "BLD", "WMS",

    # Auto / EV / Mobility
    "F", "GM", "TSLA", "RIVN", "LCID", "NKLA", "FFIE", "QS", "CHPT",
    "BLNK", "EVGO", "WKHS", "GOEV", "RIDE", "MULN", "FSR", "PSNY",
    "NIO", "LI", "XPEV", "ZK", "STLA", "TM", "HMC", "RACE",

    # Energy / Oil & Gas
    "XOM", "CVX", "COP", "OXY", "MRO", "DVN", "EOG", "SLB", "HAL",
    "MPC", "PSX", "VLO", "PXD", "FANG", "CTRA", "APA", "OVV", "AR",
    "SM", "CHK", "MTDR", "RRC", "SWN", "NOG", "CNX", "TPL", "TRGP",
    "WMB", "KMI", "OKE", "ET", "EPD", "MPLX", "PAGP", "ENB", "TRP",

    # Renewable / Clean energy
    "PLUG", "FCEL", "ENPH", "FSLR", "RUN", "SEDG", "NOVA", "SHLS",
    "ARRY", "BE", "CWEN", "STEM", "AMPS", "BLDP", "HASI", "NEE",

    # Consumer / Retail
    "COST", "MCD", "SBUX", "NKE", "LULU", "TGT", "DG", "DLTR", "BBY",
    "TJX", "ROST", "ULTA", "CMG", "DPZ", "QSR", "YUM", "DRI", "WEN",
    "PZZA", "CAKE", "TXRH", "BLMN", "EAT", "CAVA", "WING", "SHAK",
    "CROX", "DECK", "RL", "TPR", "CPRI", "PVH", "LVS", "WYNN", "MGM",
    "DIS", "NFLX", "CMCSA", "PARA", "WBD", "EA", "ATVI", "TTWO", "RBLX",
    "ETSY", "EBAY", "W", "CHWY", "PETS", "BARK", "FIGS", "REVG",

    # Biotech / Pharma
    "MRNA", "BNTX", "NVAX", "GILD", "REGN", "VRTX", "AMGN", "BIIB",
    "ILMN", "ALNY", "BMRN", "INCY", "CRSP", "EDIT", "NTLA", "BEAM",
    "VERV", "RXRX", "RARE", "SAGE", "BLUE", "FATE", "ARWR", "IONS",
    "HALO", "EXEL", "UTHR", "JAZZ", "ACAD", "NBIX", "NKTR", "SRPT",
    "TWST", "PACB", "DNA", "RGNX", "MRTX", "RNA", "RVMD", "INSM",

    # Metals / Mining / Materials
    "FCX", "NEM", "GOLD", "AEM", "WPM", "FNV", "RGLD", "AG", "CDE",
    "PAAS", "HL", "AU", "KGC", "BTG", "EXK", "FSM", "EQX", "OR",
    "NUE", "STLD", "MT", "X", "CLF", "CENX", "KALU", "RS", "CMC",
    "AA", "BHP", "RIO", "VALE", "SCCO", "TECK", "FCX", "SQM", "ALB",

    # ETFs (very high dollar volume, useful for context AND tradeable)
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV", "VEA", "EEM",
    "XLF", "XLE", "XLK", "XLV", "XLY", "XLP", "XLI", "XLU", "XLRE",
    "XLB", "XLC", "XBI", "XOP", "XHB", "XME", "ITB", "KRE", "KBE",
    "SOXL", "SOXS", "SOXX", "SMH", "TQQQ", "SQQQ", "UPRO", "SPXU",
    "TNA", "TZA", "FAS", "FAZ", "TLT", "TBT", "IEF", "SHY", "AGG",
    "HYG", "JNK", "LQD", "GLD", "SLV", "GDX", "GDXJ", "USO", "UNG",
    "BOIL", "KOLD", "UVXY", "VXX", "VIXY", "BITO", "ETHE", "GBTC",
    "WEED", "MJ", "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ", "ICLN", "TAN",

    # Crypto-adjacent
    "MARA", "RIOT", "CLSK", "MSTR", "HUT", "BITF", "WULF", "CIFR",
    "IREN", "BTBT", "GREE", "CAN", "EBON", "SOS", "BTCM",
    "COIN", "HOOD", "GLXY", "GLBE",

    # Travel / Airlines / Hotels / Cruise
    "AAL", "DAL", "UAL", "LUV", "ALK", "JBLU", "SAVE", "ALGT", "HA",
    "CCL", "NCLH", "RCL", "VIK", "MAR", "HLT", "H", "IHG", "CHH",
    "BKNG", "EXPE", "ABNB", "TRIP", "TCOM", "MMYT",

    # Real Estate
    "PLD", "AMT", "EQIX", "PSA", "CCI", "WELL", "DLR", "O", "SPG",
    "AVB", "EQR", "VICI", "EXR", "ARE", "VTR", "WPC", "STAG", "REXR",
    "IRM", "BXP", "HST", "MAA", "ESS", "UDR", "INVH", "AMH",

    # Telecom / Media
    "T", "VZ", "TMUS", "CMCSA", "CHTR", "DISH", "LBRDK", "LBRDA",
    "PARA", "WBD", "FOXA", "FOX", "NWSA", "NWS", "NYT", "GCI",
    "TTD", "MGNI", "PUBM", "DV", "APP", "RAMP",

    # Popular high-volume movers / meme
    "GME", "AMC", "BB", "DJT", "TRUMP", "BBBY", "ATER", "BBIG", "PROG",
    "MMTLP", "GREE", "HKD", "AMTD", "EZGO", "NEGG", "SPCE", "CFVI",
    "CIFR", "BFRG", "MGOL", "GFAI", "HOLO", "WW", "WEN", "NEGG",

    # Lithium / battery / commodities
    "LIT", "ALB", "SQM", "LAC", "PLL", "MP", "REE", "USAR",

    # Misc high-volume names not yet covered
    "PFE", "GILD", "WBA", "HSY", "K", "GIS", "SJM", "CAG", "CPB",
    "MO", "PM", "BTI", "STZ", "TAP", "BUD", "CL", "KMB", "CHD",
    "EL", "COTY", "SBH", "ELF", "ULTA", "OLPX", "BBWI",
    "DHR", "TMO", "WAT", "MTD", "BIO", "TECH", "RVTY", "PKI",
    "ZTS", "IDXX", "XRAY", "ALGN", "DXCM", "ISRG", "ILMN", "PODD",
]))


# ===========================================================================
#                          UNIVERSE-LIMIT CONSTANTS
# ===========================================================================
# Hard caps so we don't hammer Yahoo's API or wait forever on a quiet day.
DEFAULT_TARGET_HITS = 10        # default # of A-or-better setups to find
MAX_TICKERS_TO_ANALYZE = 200    # don't analyze more than this many in one scan
ANALYSIS_BATCH_SIZE = 25        # download bars in batches of this size


# ===========================================================================
#                              ANALYSIS RESULT
# ===========================================================================
@dataclass
class Candidate:
    ticker: str
    last_price: float
    dollar_volume: float          # 4-day average daily $ volume
    score: float                  # 0..100, higher = better swing setup
    rating: str                   # "A+", "A", "B", "C", "D" — qualitative grade
    stars: int                    # 1..5 — visual representation of rating
    win_probability: float        # 0..1 — heuristic likelihood of hitting target before stop
    direction: str                # "LONG" or "SHORT"
    setup: str                    # short label of the dominant pattern
    rsi: float
    atr: float
    atr_pct: float
    sma20: float
    sma50: float                  # uses 30 days max so this is shorter avg
    trend: str                    # "Up", "Down", "Sideways"
    entry: float
    stop: float
    target: float
    risk_reward: float
    reasons: List[str] = field(default_factory=list)
    # --- New v2 fields (filters, warnings, context) ---
    sector: str = "Unknown"
    next_earnings: Optional[str] = None      # ISO date string or None
    days_to_earnings: Optional[int] = None   # int days until earnings, or None
    earnings_warning: bool = False           # earnings within "danger" window
    market_regime: str = "Unknown"           # SPY-based regime label
    regime_aligned: bool = True              # is this trade aligned with regime?
    warnings: List[str] = field(default_factory=list)
    ml_probability: Optional[float] = None   # XGBoost-blended probability if model loaded


def score_to_rating(score: float, risk_reward: float, trend_aligned: bool) -> Tuple[str, int, float]:
    """Translate a 0–100 raw score into a letter grade, star count, and an
    estimated win-probability for the target-before-stop outcome.

    The probability model:
        - Raw score -> base p (0..1) via logistic-ish curve.
        - Adds a bump if the trade is aligned with the prevailing trend
          (trend-following has a higher historical hit rate than counter-trend).
        - Adds a small bump for unusually clean R:R.
        - Caps at 80% to avoid implying false certainty.
    """
    s = max(0.0, min(100.0, float(score)))

    # Logistic-style base probability centered around score=55
    # (a "good" setup; below that, p drops fast)
    base_p = 1.0 / (1.0 + math.exp(-(s - 55.0) / 10.0))

    # Adjustments
    if trend_aligned:
        base_p += 0.05
    if risk_reward >= 2.5:
        base_p += 0.03
    elif risk_reward < 1.5:
        base_p -= 0.05

    # Floor and cap
    p = max(0.10, min(0.80, base_p))

    # Letter grade — map score thresholds; require strong score AND good R:R for A+
    if s >= 80 and risk_reward >= 2.0:
        rating, stars = "A+", 5
    elif s >= 70:
        rating, stars = "A", 4
    elif s >= 55:
        rating, stars = "B", 3
    elif s >= 40:
        rating, stars = "C", 2
    else:
        rating, stars = "D", 1

    return rating, stars, round(p, 3)


# ===========================================================================
#                            TECHNICAL INDICATORS
# ===========================================================================
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # Wilder smoothing == EMA with alpha = 1/period
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range (Wilder)."""
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger(series: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    upper = mid + n_std * std
    lower = mid - n_std * std
    # %B: where price sits in the band; 0=lower, 1=upper
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return mid, upper, lower, pct_b


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX – trend strength."""
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat(
        [(high - low),
         (high - close.shift(1)).abs(),
         (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


# ===========================================================================
#                             SCORING ENGINE
# ===========================================================================
def analyze_one(ticker: str, df: pd.DataFrame, dollar_vol_4d: float) -> Optional[Candidate]:
    """Run the full algo on one symbol's daily bars (≈30 rows) and return a
    Candidate, or None if there isn't enough data."""

    if df is None or len(df) < 20:
        return None

    df = df.copy()
    df.dropna(subset=["Close"], inplace=True)
    if len(df) < 20:
        return None

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    vol = df["Volume"]

    last_price = float(close.iloc[-1])
    if not math.isfinite(last_price) or last_price <= 0:
        return None

    # --- Indicators -------------------------------------------------------
    sma10 = close.rolling(10).mean()
    sma20 = close.rolling(20).mean()
    # Use a shorter "long MA" because we only have ~30 bars
    sma_long = close.rolling(min(50, max(15, len(close) - 1))).mean()

    rsi_series = rsi(close, 14)
    atr_series = atr(df, 14)
    _, _, macd_hist = macd(close)
    _, bb_up, bb_lo, pct_b = bollinger(close, 20, 2.0)
    adx_series = adx(df, 14)

    cur_rsi = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else 50.0
    cur_atr = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    cur_atr_pct = (cur_atr / last_price) * 100 if last_price else 0.0
    cur_sma20 = float(sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else last_price
    cur_smaL = float(sma_long.iloc[-1]) if pd.notna(sma_long.iloc[-1]) else last_price
    cur_adx = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else 0.0
    cur_pctb = float(pct_b.iloc[-1]) if pd.notna(pct_b.iloc[-1]) else 0.5
    cur_macdh = float(macd_hist.iloc[-1]) if pd.notna(macd_hist.iloc[-1]) else 0.0
    prev_macdh = float(macd_hist.iloc[-2]) if len(macd_hist) > 1 and pd.notna(macd_hist.iloc[-2]) else 0.0

    # 20-day Donchian
    hh20 = float(high.rolling(20).max().iloc[-1])
    ll20 = float(low.rolling(20).min().iloc[-1])

    # Volume confirmation: today vs 20-day average
    avg_vol20 = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else float(vol.mean())
    vol_ratio = (float(vol.iloc[-1]) / avg_vol20) if avg_vol20 else 1.0

    # --- Trend classification -------------------------------------------
    if last_price > cur_sma20 > cur_smaL and cur_adx >= 18:
        trend = "Up"
    elif last_price < cur_sma20 < cur_smaL and cur_adx >= 18:
        trend = "Down"
    else:
        trend = "Sideways"

    # --- Build long & short scores from independent factors --------------
    long_score = 0.0
    short_score = 0.0
    long_reasons: List[str] = []
    short_reasons: List[str] = []

    # 1. Trend (max 25)
    if trend == "Up":
        long_score += 25
        long_reasons.append(f"Uptrend: price>SMA20>SMA{int(min(50, len(close)-1))}, ADX={cur_adx:.0f}")
    elif trend == "Down":
        short_score += 25
        short_reasons.append(f"Downtrend: price<SMA20<long MA, ADX={cur_adx:.0f}")
    else:
        # mild penalty handled by lack of points
        pass

    # 2. RSI – favor pullbacks in trend, oversold/overbought reversals (max 20)
    if 40 <= cur_rsi <= 55 and trend == "Up":
        long_score += 18
        long_reasons.append(f"Pullback RSI={cur_rsi:.0f} in uptrend")
    elif cur_rsi < 30:
        long_score += 20
        long_reasons.append(f"Oversold RSI={cur_rsi:.0f} (mean-reversion long)")
    elif 45 <= cur_rsi <= 60 and trend == "Down":
        short_score += 18
        short_reasons.append(f"Bounce RSI={cur_rsi:.0f} in downtrend")
    elif cur_rsi > 70:
        short_score += 20
        short_reasons.append(f"Overbought RSI={cur_rsi:.0f} (mean-reversion short)")

    # 3. MACD histogram momentum (max 15)
    if cur_macdh > 0 and cur_macdh > prev_macdh:
        long_score += 15
        long_reasons.append("MACD histogram rising > 0")
    elif cur_macdh < 0 and cur_macdh < prev_macdh:
        short_score += 15
        short_reasons.append("MACD histogram falling < 0")
    elif cur_macdh > 0 > prev_macdh:
        long_score += 10
        long_reasons.append("MACD bullish cross")
    elif cur_macdh < 0 < prev_macdh:
        short_score += 10
        short_reasons.append("MACD bearish cross")

    # 4. Bollinger %B – breakout or mean-reversion edge (max 15)
    if cur_pctb <= 0.05:
        long_score += 12
        long_reasons.append(f"At lower Bollinger ({cur_pctb:.2f})")
    elif cur_pctb >= 0.95:
        short_score += 12
        short_reasons.append(f"At upper Bollinger ({cur_pctb:.2f})")
    if last_price >= hh20 * 0.999 and trend != "Down":
        long_score += 10
        long_reasons.append("20-day breakout (Donchian high)")
    if last_price <= ll20 * 1.001 and trend != "Up":
        short_score += 10
        short_reasons.append("20-day breakdown (Donchian low)")

    # 5. Volume confirmation (max 10)
    if vol_ratio >= 1.5:
        bonus = min(10, (vol_ratio - 1) * 8)
        if cur_macdh >= 0 or trend == "Up":
            long_score += bonus
            long_reasons.append(f"Volume {vol_ratio:.1f}× 20-day avg")
        if cur_macdh <= 0 or trend == "Down":
            short_score += bonus
            short_reasons.append(f"Volume {vol_ratio:.1f}× 20-day avg")

    # 6. Volatility filter – penalize ultra-quiet or insanely volatile names (max ±5)
    if 1.5 <= cur_atr_pct <= 6.0:
        long_score += 5
        short_score += 5
    elif cur_atr_pct < 1.0:
        long_score -= 3
        short_score -= 3
    elif cur_atr_pct > 10:
        long_score -= 3
        short_score -= 3

    # 7. Liquidity bonus (max 10) – scaled log of dollar volume
    if dollar_vol_4d > 0:
        liq_bonus = min(10.0, math.log10(max(dollar_vol_4d, 1)) - 7)  # $10M=0, $100M=1...
        liq_bonus = max(0.0, liq_bonus)
        long_score += liq_bonus
        short_score += liq_bonus

    # --- Pick the dominant side -----------------------------------------
    long_score = max(0.0, min(100.0, long_score))
    short_score = max(0.0, min(100.0, short_score))

    if long_score >= short_score:
        direction = "LONG"
        score = long_score
        reasons = long_reasons
    else:
        direction = "SHORT"
        score = short_score
        reasons = short_reasons

    # --- Setup label (whichever factor scored highest) -------------------
    if direction == "LONG":
        if cur_rsi < 30:
            setup = "Oversold reversal"
        elif last_price >= hh20 * 0.999 and trend != "Down":
            setup = "20-day breakout"
        elif 40 <= cur_rsi <= 55 and trend == "Up":
            setup = "Trend pullback"
        elif cur_pctb <= 0.05:
            setup = "Lower-band tag"
        else:
            setup = "Momentum long"
    else:
        if cur_rsi > 70:
            setup = "Overbought reversal"
        elif last_price <= ll20 * 1.001 and trend != "Up":
            setup = "20-day breakdown"
        elif 45 <= cur_rsi <= 60 and trend == "Down":
            setup = "Trend bounce short"
        elif cur_pctb >= 0.95:
            setup = "Upper-band tag"
        else:
            setup = "Momentum short"

    # --- Entry / Stop / Target using ATR --------------------------------
    if cur_atr <= 0 or not math.isfinite(cur_atr):
        cur_atr = last_price * 0.02  # fallback 2%

    if direction == "LONG":
        entry = last_price
        stop = round(last_price - 1.5 * cur_atr, 2)
        target = round(last_price + 3.0 * cur_atr, 2)
    else:
        entry = last_price
        stop = round(last_price + 1.5 * cur_atr, 2)
        target = round(last_price - 3.0 * cur_atr, 2)

    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = round(reward / risk, 2) if risk > 0 else 0.0

    # --- Rating / win-probability ---------------------------------------
    trend_aligned = (
        (direction == "LONG" and trend == "Up") or
        (direction == "SHORT" and trend == "Down")
    )
    rating, stars, win_prob = score_to_rating(score, rr, trend_aligned)

    return Candidate(
        ticker=ticker,
        last_price=round(last_price, 2),
        dollar_volume=dollar_vol_4d,
        score=round(score, 1),
        rating=rating,
        stars=stars,
        win_probability=win_prob,
        direction=direction,
        setup=setup,
        rsi=round(cur_rsi, 1),
        atr=round(cur_atr, 2),
        atr_pct=round(cur_atr_pct, 2),
        sma20=round(cur_sma20, 2),
        sma50=round(cur_smaL, 2),
        trend=trend,
        entry=round(entry, 2),
        stop=stop,
        target=target,
        risk_reward=rr,
        reasons=reasons,
    )


# ===========================================================================
#                            DATA DOWNLOADS
# ===========================================================================
def download_bulk(tickers: List[str], period: str = "35d") -> dict:
    """Download daily bars for a list of tickers in one batch.
    Returns {ticker: DataFrame}."""
    if yf is None:
        raise RuntimeError("yfinance is not installed.")
    data = yf.download(
        tickers=" ".join(tickers),
        period=period,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    out = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            if t in data.columns.levels[0]:
                df = data[t].dropna(how="all")
                if not df.empty:
                    out[t] = df
    else:
        # Single ticker case
        if not data.empty:
            out[tickers[0]] = data
    return out


# ===========================================================================
#                    EARNINGS, SECTOR, REGIME (v2 enhancements)
# ===========================================================================
# Module-level caches so we don't re-fetch within a single scan
_earnings_cache: Dict[str, Optional[pd.Timestamp]] = {}
_sector_cache: Dict[str, str] = {}


def get_next_earnings_date(ticker: str) -> Optional[pd.Timestamp]:
    """Fetch the next upcoming earnings date for a ticker.

    Returns a normalized Timestamp (date only, no time), or None if unknown.
    Cached per process.
    """
    if ticker in _earnings_cache:
        return _earnings_cache[ticker]
    if yf is None:
        _earnings_cache[ticker] = None
        return None
    try:
        t = yf.Ticker(ticker)
        # Try the new 'calendar' attribute first
        cal = getattr(t, "calendar", None)
        next_date = None
        if cal is not None:
            # Recent yfinance: dict-like
            if isinstance(cal, dict):
                ed = cal.get("Earnings Date")
                if ed:
                    if isinstance(ed, list) and ed:
                        next_date = pd.Timestamp(ed[0]).normalize()
                    else:
                        next_date = pd.Timestamp(ed).normalize()
            # Older yfinance: DataFrame
            elif hasattr(cal, "loc") and "Earnings Date" in getattr(cal, "index", []):
                ed_val = cal.loc["Earnings Date"]
                if hasattr(ed_val, "iloc"):
                    next_date = pd.Timestamp(ed_val.iloc[0]).normalize()
                else:
                    next_date = pd.Timestamp(ed_val).normalize()
        # Fallback: get_earnings_dates() returns past + future
        if next_date is None:
            try:
                eds = t.get_earnings_dates(limit=8)
                if eds is not None and not eds.empty:
                    today = pd.Timestamp.now().normalize()
                    # Index might be tz-aware; strip tz for comparison
                    idx = pd.to_datetime(eds.index).tz_localize(None).normalize()
                    future = idx[idx >= today]
                    if len(future) > 0:
                        next_date = future.min()
            except Exception:
                pass
        _earnings_cache[ticker] = next_date
        return next_date
    except Exception:
        _earnings_cache[ticker] = None
        return None


def get_sector(ticker: str) -> str:
    """Return the GICS sector for a ticker, or 'Unknown'. Cached per process."""
    if ticker in _sector_cache:
        return _sector_cache[ticker]
    if yf is None:
        _sector_cache[ticker] = "Unknown"
        return "Unknown"
    try:
        t = yf.Ticker(ticker)
        info = t.info if hasattr(t, "info") else {}
        sector = info.get("sector") or "Unknown"
        # ETFs typically have no sector — try to classify from quoteType
        if sector == "Unknown":
            qt = info.get("quoteType", "").upper()
            if qt == "ETF":
                sector = "ETF"
        _sector_cache[ticker] = sector
        return sector
    except Exception:
        _sector_cache[ticker] = "Unknown"
        return "Unknown"


def fetch_earnings_dates_bulk(tickers: List[str]) -> Dict[str, Optional[pd.Timestamp]]:
    """Look up earnings dates for many tickers; returns {ticker: Timestamp_or_None}."""
    return {t: get_next_earnings_date(t) for t in tickers}


def fetch_sectors_bulk(tickers: List[str]) -> Dict[str, str]:
    """Look up sectors for many tickers."""
    return {t: get_sector(t) for t in tickers}


# ---------------------------------------------------------------- Market regime
@dataclass
class MarketRegime:
    label: str           # "Strong Uptrend" | "Uptrend" | "Sideways" | "Downtrend" | "Strong Downtrend"
    spy_price: float
    spy_sma20: float
    spy_sma50: float
    spy_above_sma20: bool
    spy_above_sma50: bool
    spy_trend_strength: float   # ADX value
    favors_long: bool
    favors_short: bool
    note: str


def detect_market_regime(spy_df: Optional[pd.DataFrame] = None) -> MarketRegime:
    """Classify the broader market regime from SPY's daily chart.

    Logic:
        - Strong Uptrend  : price > SMA20 > SMA50, ADX ≥ 25
        - Uptrend         : price > SMA20 > SMA50
        - Sideways        : neither alignment
        - Downtrend       : price < SMA20 < SMA50
        - Strong Downtrend: price < SMA20 < SMA50, ADX ≥ 25
    """
    if spy_df is None or spy_df.empty or len(spy_df) < 50:
        # Try to fetch
        if yf is not None:
            try:
                data = yf.download("SPY", period="90d", interval="1d",
                                    auto_adjust=False, progress=False, threads=False)
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                spy_df = data.dropna(how="all")
            except Exception:
                spy_df = None

    if spy_df is None or spy_df.empty or len(spy_df) < 50:
        return MarketRegime(
            label="Unknown", spy_price=0.0, spy_sma20=0.0, spy_sma50=0.0,
            spy_above_sma20=False, spy_above_sma50=False,
            spy_trend_strength=0.0, favors_long=True, favors_short=True,
            note="SPY data unavailable; no regime filter applied."
        )

    close = spy_df["Close"]
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    last = float(close.iloc[-1])
    spy_adx = float(adx(spy_df, 14).iloc[-1]) if len(spy_df) >= 28 else 0.0

    above20 = last > sma20
    above50 = last > sma50

    if above20 and above50 and sma20 > sma50:
        label = "Strong Uptrend" if spy_adx >= 25 else "Uptrend"
        favors_long, favors_short = True, False
        note = f"SPY in {label.lower()}; longs preferred, shorts risky."
    elif (not above20) and (not above50) and sma20 < sma50:
        label = "Strong Downtrend" if spy_adx >= 25 else "Downtrend"
        favors_long, favors_short = False, True
        note = f"SPY in {label.lower()}; shorts preferred, longs risky."
    else:
        label = "Sideways"
        favors_long, favors_short = True, True
        note = "SPY mixed; no strong directional bias. Use selectivity."

    return MarketRegime(
        label=label, spy_price=round(last, 2),
        spy_sma20=round(sma20, 2), spy_sma50=round(sma50, 2),
        spy_above_sma20=above20, spy_above_sma50=above50,
        spy_trend_strength=round(spy_adx, 1),
        favors_long=favors_long, favors_short=favors_short,
        note=note,
    )


def rank_by_dollar_volume(price_data: dict, lookback_days: int = 4) -> List[tuple]:
    """Return [(ticker, avg_dollar_volume_4d), ...] sorted desc."""
    rows = []
    for t, df in price_data.items():
        if df is None or df.empty or len(df) < lookback_days:
            continue
        last = df.tail(lookback_days)
        # Use typical price * volume as a clean proxy for dollar volume
        typical = (last["High"] + last["Low"] + last["Close"]) / 3.0
        dv = float((typical * last["Volume"]).mean())
        if math.isfinite(dv) and dv > 0:
            rows.append((t, dv))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


# ===========================================================================
#                  ITERATIVE SCANNER  (target N A-rated hits)
# ===========================================================================
def scan_until_target_hits(
    target_a_count: int = DEFAULT_TARGET_HITS,
    max_to_analyze: int = MAX_TICKERS_TO_ANALYZE,
    batch_size: int = ANALYSIS_BATCH_SIZE,
    progress_cb=None,
    # --- v2 enhancements (all default to "on" with sensible behavior) ---
    earnings_window_days: int = 5,    # warn on earnings within N days; 0 disables
    skip_on_earnings: bool = False,   # True = remove from results entirely
    apply_regime_filter: bool = True, # gate longs/shorts by SPY regime
    sector_cap: int = 2,              # max A+/A per sector; 0 disables
    use_ml_blend: bool = True,        # if a trained model is available, blend
) -> Tuple[List[Candidate], dict]:
    """Walk the dollar-volume-ranked universe, analyzing names until we've
    found `target_a_count` candidates rated A or A+, or until we've analyzed
    `max_to_analyze` names (safety cap), whichever happens first.

    v2 adds:
        - Earnings filter: tags or skips names with earnings within
          `earnings_window_days`.
        - Market regime overlay: classifies SPY trend and warns/downgrades
          counter-regime trades.
        - Sector concentration cap: max `sector_cap` A+/A per sector.
        - ML probability blend: if an XGBoost model is trained on the tracker
          DB, blends its probability into win_probability.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed.")

    def _progress(stage, current, total, msg):
        if progress_cb:
            progress_cb(stage, current, total, msg)

    # ---- Step 0: market regime ----
    regime = MarketRegime("Unknown", 0, 0, 0, False, False, 0,
                          True, True, "regime check skipped")
    if apply_regime_filter:
        _progress("regime", 0, 1, "Detecting market regime from SPY…")
        regime = detect_market_regime()
        _progress("regime", 1, 1, f"Market regime: {regime.label}")

    # ---- Step 0.5: load ML model if available ----
    ml_model = None
    if use_ml_blend:
        try:
            ml_model = load_ml_model()
        except Exception:
            ml_model = None

    # ---- Step 1: rank the entire universe by 4-day dollar volume ----
    _progress("ranking", 0, len(UNIVERSE),
              f"Downloading 5d quotes for {len(UNIVERSE)} tickers…")
    five_d = download_bulk(UNIVERSE, period="6d")
    ranked = rank_by_dollar_volume(five_d, lookback_days=4)
    if not ranked:
        return [], _empty_stats(target_a_count, max_to_analyze, regime,
                                 "No data returned from Yahoo.")

    _progress("ranking", len(UNIVERSE), len(UNIVERSE),
              f"Ranked {len(ranked)} symbols by dollar volume")

    # ---- Step 2: walk the ranked list, batch-downloading history ----
    walk_list = ranked[:max_to_analyze]

    all_candidates: List[Candidate] = []
    a_count = 0
    analyzed = 0
    skipped_earnings = 0
    sector_counts: Dict[str, int] = {}
    stop_reason = ""

    today = pd.Timestamp.now().normalize()

    for batch_start in range(0, len(walk_list), batch_size):
        batch = walk_list[batch_start: batch_start + batch_size]
        batch_tickers = [t for t, _ in batch]

        _progress("scanning", analyzed, target_a_count,
                  f"Downloading 30d bars for batch "
                  f"{batch_start // batch_size + 1}  "
                  f"({len(batch_tickers)} symbols)…")

        try:
            hist = download_bulk(batch_tickers, period="35d")
        except Exception as e:
            print(f"batch download failed: {e}")
            continue

        for ticker, dvol in batch:
            df = hist.get(ticker)
            if df is None:
                continue
            try:
                cand = analyze_one(ticker, df, dvol)
            except Exception as e:
                print(f"analyze_one failed for {ticker}: {e}")
                continue
            if cand is None:
                continue

            # ----- Apply v2 enhancements per-candidate -----
            cand = _apply_enhancements(
                cand, regime,
                earnings_window_days=earnings_window_days,
                skip_on_earnings=skip_on_earnings,
                apply_regime_filter=apply_regime_filter,
                ml_model=ml_model,
                today=today,
            )
            if cand is None:
                # Skipped due to earnings filter
                skipped_earnings += 1
                continue

            # Sector concentration cap (only for A+/A)
            if sector_cap > 0 and cand.rating in ("A+", "A"):
                sec = cand.sector
                if sector_counts.get(sec, 0) >= sector_cap:
                    # Demote to B so it doesn't count against the target
                    cand.rating = "B"
                    cand.stars = 3
                    cand.warnings.append(
                        f"Sector cap: already {sector_cap} A-rated in {sec}"
                    )
                else:
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1

            all_candidates.append(cand)
            analyzed += 1
            if cand.rating in ("A+", "A"):
                a_count += 1

            _progress("scanning", a_count, target_a_count,
                      f"Analyzed {analyzed}  ·  "
                      f"found {a_count}/{target_a_count} A-or-better  "
                      f"(latest: {ticker} = {cand.rating})")

            if a_count >= target_a_count:
                stop_reason = (
                    f"Target reached: {a_count} A-or-better setups "
                    f"found after analyzing {analyzed} symbols."
                )
                break

        if a_count >= target_a_count:
            break

    if not stop_reason:
        if analyzed >= max_to_analyze:
            stop_reason = (
                f"Analyzed the safety cap of {max_to_analyze} symbols. "
                f"Only {a_count} A-or-better setups found — market may be "
                f"quiet today, or try lowering your target."
            )
        else:
            stop_reason = (
                f"Reached the end of the universe after analyzing "
                f"{analyzed} symbols. {a_count} A-or-better setups found."
            )

    # Sort: A+ first, then A, then by score
    rating_order = {"A+": 0, "A": 1, "B": 2, "C": 3, "D": 4}
    all_candidates.sort(
        key=lambda c: (rating_order.get(c.rating, 5), -c.score)
    )

    stats = {
        "analyzed": analyzed,
        "a_plus": sum(1 for c in all_candidates if c.rating == "A+"),
        "a":      sum(1 for c in all_candidates if c.rating == "A"),
        "b":      sum(1 for c in all_candidates if c.rating == "B"),
        "c":      sum(1 for c in all_candidates if c.rating == "C"),
        "d":      sum(1 for c in all_candidates if c.rating == "D"),
        "long":   sum(1 for c in all_candidates if c.direction == "LONG"),
        "short":  sum(1 for c in all_candidates if c.direction == "SHORT"),
        "stopped_at": analyzed,
        "target": target_a_count,
        "max_cap": max_to_analyze,
        "reason": stop_reason,
        "universe_size": len(UNIVERSE),
        "ranked_size": len(ranked),
        "regime": regime.label,
        "regime_note": regime.note,
        "skipped_earnings": skipped_earnings,
        "ml_model_loaded": ml_model is not None,
    }
    return all_candidates, stats


def _empty_stats(target, max_cap, regime, reason):
    return {"analyzed": 0, "a_plus": 0, "a": 0, "b": 0, "c": 0, "d": 0,
            "long": 0, "short": 0, "stopped_at": 0,
            "target": target, "max_cap": max_cap, "reason": reason,
            "universe_size": len(UNIVERSE), "ranked_size": 0,
            "regime": regime.label, "regime_note": regime.note,
            "skipped_earnings": 0, "ml_model_loaded": False}


def _apply_enhancements(cand, regime, earnings_window_days,
                        skip_on_earnings, apply_regime_filter,
                        ml_model, today):
    """Apply earnings filter, regime overlay, sector lookup, and ML blend
    to a single candidate. Returns the enhanced candidate, or None if it
    should be filtered out entirely."""

    # ---- Earnings ----
    if earnings_window_days > 0:
        ed = get_next_earnings_date(cand.ticker)
        if ed is not None:
            days = int((ed - today).days)
            cand.next_earnings = ed.strftime("%Y-%m-%d")
            cand.days_to_earnings = days
            if 0 <= days <= earnings_window_days:
                cand.earnings_warning = True
                cand.warnings.append(f"⚠ Earnings in {days}d ({cand.next_earnings})")
                if skip_on_earnings:
                    return None
                # Soft-downgrade: A+ → A, A → B
                if cand.rating == "A+":
                    cand.rating = "A"
                    cand.stars = 4
                elif cand.rating == "A":
                    cand.rating = "B"
                    cand.stars = 3

    # ---- Sector ----
    cand.sector = get_sector(cand.ticker)

    # ---- Regime overlay ----
    cand.market_regime = regime.label
    if apply_regime_filter and regime.label != "Unknown":
        if cand.direction == "LONG":
            cand.regime_aligned = regime.favors_long
            if not regime.favors_long:
                cand.warnings.append(
                    f"⚠ {regime.label}: longs are counter-regime"
                )
                # Demote one rating step
                if cand.rating == "A+":
                    cand.rating, cand.stars = "A", 4
                elif cand.rating == "A":
                    cand.rating, cand.stars = "B", 3
        else:  # SHORT
            cand.regime_aligned = regime.favors_short
            if not regime.favors_short:
                cand.warnings.append(
                    f"⚠ {regime.label}: shorts are counter-regime"
                )
                if cand.rating == "A+":
                    cand.rating, cand.stars = "A", 4
                elif cand.rating == "A":
                    cand.rating, cand.stars = "B", 3

    # ---- ML probability blend ----
    if ml_model is not None:
        try:
            ml_p = predict_with_model(ml_model, cand)
            cand.ml_probability = ml_p
            # Blend 50/50 with heuristic
            cand.win_probability = round((cand.win_probability + ml_p) / 2, 3)
        except Exception:
            pass

    return cand


# ===========================================================================
#                       PERSISTENCE  (SQLite or hosted Turso)
# ===========================================================================
# Storage strategy:
#   - If TURSO_URL + TURSO_TOKEN are configured (env vars or Streamlit
#     Secrets), use the hosted Turso database. History persists across
#     deploys, container restarts, and sleep cycles.
#   - Otherwise, fall back to a local SQLite file at ~/.swing_trader/.
#
# The two backends speak the same SQL, so the rest of the codebase doesn't
# care which one is active.

DB_PATH = Path.home() / ".swing_trader" / "history.db"


def _get_turso_config() -> Tuple[str, str]:
    """Return (url, token) from env vars, Streamlit secrets, or ('','')."""
    url = os.environ.get("TURSO_URL", "").strip()
    token = os.environ.get("TURSO_TOKEN", "").strip()

    # Also check Streamlit secrets when running inside Streamlit
    if not url:
        try:
            import streamlit as _st  # local import; not required for engine
            secrets = getattr(_st, "secrets", None)
            if secrets is not None:
                # st.secrets behaves like a dict; .get works in modern versions
                url = (secrets.get("TURSO_URL", "")
                       if hasattr(secrets, "get") else "") or url
                token = (secrets.get("TURSO_TOKEN", "")
                         if hasattr(secrets, "get") else "") or token
        except Exception:
            pass
    return url, token


# Cache a single libsql client per process. Each call to db_connect() gets
# a fresh "wrapper" that exposes the standard sqlite3 API.
_libsql_client = None


def _try_libsql():
    """Lazy import; returns the libsql module or None if not installed."""
    try:
        import libsql_experimental as libsql  # type: ignore
        return libsql
    except ImportError:
        try:
            import libsql_client as libsql  # type: ignore
            return libsql
        except ImportError:
            return None


def using_turso() -> bool:
    url, token = _get_turso_config()
    return bool(url and token and _try_libsql() is not None)


_DDL = """
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_id TEXT NOT NULL,
        scan_time TEXT NOT NULL,
        ticker TEXT NOT NULL,
        direction TEXT NOT NULL,
        setup TEXT,
        score REAL,
        rating TEXT,
        stars INTEGER,
        win_probability REAL,
        entry REAL,
        stop REAL,
        target REAL,
        risk_reward REAL,
        atr REAL,
        atr_pct REAL,
        rsi REAL,
        trend TEXT,
        dollar_volume REAL,
        reasons TEXT,
        evaluated INTEGER DEFAULT 0,
        outcome TEXT,
        exit_price REAL,
        exit_date TEXT,
        days_held INTEGER,
        pnl_pct REAL,
        max_favorable_pct REAL,
        max_adverse_pct REAL
    )
"""
_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_scan_id ON recommendations(scan_id)",
    "CREATE INDEX IF NOT EXISTS idx_ticker ON recommendations(ticker)",
    "CREATE INDEX IF NOT EXISTS idx_evaluated ON recommendations(evaluated)",
]


def db_connect():
    """Return a connection — Turso if configured, else local SQLite.

    The returned object always exposes .execute(sql, params=()),
    .executemany(sql, rows), .commit(), and .close(). For pandas
    read_sql_query, it also satisfies pandas' duck-typed expectations.
    """
    url, token = _get_turso_config()
    libsql = _try_libsql()
    if url and token and libsql is not None:
        # Hosted Turso (or any libsql-compatible) database
        conn = libsql.connect(database=url, auth_token=token)
        # Idempotent schema setup
        try:
            conn.execute(_DDL)
            for idx in _INDEXES:
                conn.execute(idx)
            conn.commit()
        except Exception:
            pass  # tables/indexes may already exist
        return conn

    # Local SQLite fallback
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_DDL)
    for idx in _INDEXES:
        conn.execute(idx)
    conn.commit()
    return conn


def save_recommendations(candidates: List[Candidate]) -> str:
    """Persist a batch of candidates under one scan_id. Returns the scan_id."""
    if not candidates:
        return ""
    scan_time = datetime.now().isoformat(timespec="seconds")
    scan_id = scan_time.replace(":", "").replace("-", "")
    conn = db_connect()
    try:
        rows = [
            (
                scan_id, scan_time, c.ticker, c.direction, c.setup,
                c.score, c.rating, c.stars, c.win_probability,
                c.entry, c.stop, c.target, c.risk_reward,
                c.atr, c.atr_pct, c.rsi, c.trend, c.dollar_volume,
                json.dumps(c.reasons),
            )
            for c in candidates
        ]
        conn.executemany("""
            INSERT INTO recommendations
            (scan_id, scan_time, ticker, direction, setup, score, rating, stars,
             win_probability, entry, stop, target, risk_reward, atr, atr_pct,
             rsi, trend, dollar_volume, reasons)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
        conn.commit()
    finally:
        conn.close()
    return scan_id


def load_all_recommendations() -> pd.DataFrame:
    conn = db_connect()
    sql = "SELECT * FROM recommendations ORDER BY scan_time DESC, score DESC"
    try:
        # sqlite3.Connection works directly with pandas
        if isinstance(conn, sqlite3.Connection):
            df = pd.read_sql_query(sql, conn)
        else:
            # libsql / Turso: fetch manually and build a DataFrame
            cur = conn.execute(sql)
            rows = cur.fetchall() if hasattr(cur, "fetchall") else list(cur)
            cols = _column_names_from_cursor(cur)
            df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return df


def _column_names_from_cursor(cur) -> List[str]:
    """Extract column names from any DB-API-ish cursor result."""
    desc = getattr(cur, "description", None)
    if desc:
        return [d[0] for d in desc]
    # Fallback: full schema (must match the CREATE TABLE order)
    return ["id", "scan_id", "scan_time", "ticker", "direction", "setup",
            "score", "rating", "stars", "win_probability",
            "entry", "stop", "target", "risk_reward",
            "atr", "atr_pct", "rsi", "trend", "dollar_volume", "reasons",
            "evaluated", "outcome", "exit_price", "exit_date", "days_held",
            "pnl_pct", "max_favorable_pct", "max_adverse_pct"]


# ===========================================================================
#                       EVALUATION  (did the recommendation work?)
# ===========================================================================
DEFAULT_HOLD_DAYS = 10  # Max bars to give a swing trade to reach target/stop


def evaluate_pending(hold_days: int = DEFAULT_HOLD_DAYS,
                     progress_cb=None) -> Tuple[int, int]:
    """Walk every un-evaluated recommendation, replay subsequent daily bars,
    and mark it WIN / LOSS / EXPIRED / OPEN.

    Logic:
      - Trade is considered entered at the close of the scan day at `entry`.
      - For LONG: WIN if High >= target before Low <= stop (within hold_days).
      - For SHORT: WIN if Low <= target before High >= stop.
      - If neither hits within hold_days, mark EXPIRED and use the close on
        the last evaluated day for P&L.
      - If we don't yet have hold_days of bars after the scan, mark OPEN and
        leave evaluated=0 so it'll be retried later.

    Returns (evaluated_count, still_open_count).
    """
    conn = db_connect()
    try:
        cur = conn.execute(
            "SELECT id, scan_time, ticker, direction, entry, stop, target "
            "FROM recommendations WHERE evaluated = 0"
        )
        pending = cur.fetchall()
    finally:
        conn.close()

    if not pending:
        return 0, 0

    # Group by ticker so we can do one yfinance call per symbol
    by_ticker: Dict[str, list] = {}
    for row in pending:
        by_ticker.setdefault(row[2], []).append(row)

    if yf is None:
        return 0, len(pending)

    evaluated = 0
    still_open = 0
    total_tickers = len(by_ticker)

    for i, (ticker, rows) in enumerate(by_ticker.items(), start=1):
        if progress_cb:
            progress_cb(i, total_tickers, ticker)
        try:
            # Pull enough history to cover the oldest pending recommendation
            oldest = min(datetime.fromisoformat(r[1]) for r in rows)
            start = (oldest - timedelta(days=2)).strftime("%Y-%m-%d")
            df = yf.download(ticker, start=start, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
            if df is None or df.empty:
                still_open += len(rows)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        except Exception:
            still_open += len(rows)
            continue

        # Normalize index to date-only
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()

        for rec_id, scan_time, _t, direction, entry, stop, target in rows:
            scan_date = pd.Timestamp(scan_time).normalize()
            future = df[df.index > scan_date]
            if len(future) < 1:
                still_open += 1
                continue

            window = future.head(hold_days)
            outcome, exit_price, exit_date, days, mfe, mae = _replay(
                window, direction, entry, stop, target
            )
            if outcome == "OPEN":
                still_open += 1
                continue

            pnl_pct = ((exit_price - entry) / entry * 100.0
                       if direction == "LONG"
                       else (entry - exit_price) / entry * 100.0)

            conn = db_connect()
            try:
                conn.execute("""
                    UPDATE recommendations
                    SET evaluated=1, outcome=?, exit_price=?, exit_date=?,
                        days_held=?, pnl_pct=?, max_favorable_pct=?, max_adverse_pct=?
                    WHERE id=?
                """, (outcome, float(exit_price), exit_date, int(days),
                      float(pnl_pct), float(mfe), float(mae), rec_id))
                conn.commit()
            finally:
                conn.close()
            evaluated += 1

    return evaluated, still_open


def _replay(window: pd.DataFrame, direction: str,
            entry: float, stop: float, target: float):
    """Bar-by-bar replay. Returns (outcome, exit_price, exit_date, days, mfe, mae).

    Conservative tie-break: if both stop and target are inside the same day's
    range, assume the stop hit first (the safer assumption for a real trader).
    """
    mfe = 0.0  # max favorable excursion %
    mae = 0.0  # max adverse excursion %

    for day_idx, (ts, row) in enumerate(window.iterrows(), start=1):
        hi = float(row["High"])
        lo = float(row["Low"])

        if direction == "LONG":
            mfe = max(mfe, (hi - entry) / entry * 100)
            mae = min(mae, (lo - entry) / entry * 100)
            stop_hit = lo <= stop
            tgt_hit = hi >= target
            if stop_hit and tgt_hit:
                return ("LOSS", stop, ts.strftime("%Y-%m-%d"), day_idx, mfe, mae)
            if stop_hit:
                return ("LOSS", stop, ts.strftime("%Y-%m-%d"), day_idx, mfe, mae)
            if tgt_hit:
                return ("WIN", target, ts.strftime("%Y-%m-%d"), day_idx, mfe, mae)
        else:  # SHORT
            mfe = max(mfe, (entry - lo) / entry * 100)
            mae = min(mae, (entry - hi) / entry * 100)
            stop_hit = hi >= stop
            tgt_hit = lo <= target
            if stop_hit and tgt_hit:
                return ("LOSS", stop, ts.strftime("%Y-%m-%d"), day_idx, mfe, mae)
            if stop_hit:
                return ("LOSS", stop, ts.strftime("%Y-%m-%d"), day_idx, mfe, mae)
            if tgt_hit:
                return ("WIN", target, ts.strftime("%Y-%m-%d"), day_idx, mfe, mae)

    # Neither hit — close out at the final bar
    if len(window) >= DEFAULT_HOLD_DAYS:
        last = window.iloc[-1]
        last_date = window.index[-1].strftime("%Y-%m-%d")
        return ("EXPIRED", float(last["Close"]), last_date, len(window), mfe, mae)

    # Not enough bars elapsed yet — keep it open
    return ("OPEN", 0.0, "", 0, 0.0, 0.0)


# ===========================================================================
#                            STRATEGY BACK-TESTER
# ===========================================================================
@dataclass
class StrategyResult:
    name: str
    n_trades: int
    n_wins: int
    n_losses: int
    n_expired: int
    win_rate: float
    avg_pnl_pct: float
    total_pnl_pct: float          # sum of % returns (informational)
    final_equity: float           # $ value after compounding fixed allocation
    starting_capital: float
    capital_per_trade: float
    max_drawdown_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    expectancy_pct: float         # avg win × win_rate − avg loss × loss_rate
    equity_curve: List[Tuple[str, float]] = field(default_factory=list)


# Each strategy is a (name, predicate) — predicate takes a row Series and
# returns True if the recommendation belongs in this strategy.
STRATEGIES: Dict[str, callable] = {
    "All recommendations":
        lambda r: True,
    "All longs":
        lambda r: r["direction"] == "LONG",
    "All shorts":
        lambda r: r["direction"] == "SHORT",
    "A+ rated only":
        lambda r: r["rating"] == "A+",
    "A or A+ rated":
        lambda r: r["rating"] in ("A", "A+"),
    "A+ longs only":
        lambda r: r["rating"] == "A+" and r["direction"] == "LONG",
    "Score ≥ 70":
        lambda r: (r["score"] or 0) >= 70,
    "Score ≥ 60 + R:R ≥ 2":
        lambda r: (r["score"] or 0) >= 60 and (r["risk_reward"] or 0) >= 2.0,
    "Trend-aligned only":
        lambda r: ((r["direction"] == "LONG" and r["trend"] == "Up")
                    or (r["direction"] == "SHORT" and r["trend"] == "Down")),
    "Breakout setups only":
        lambda r: "breakout" in (r["setup"] or "").lower()
                   or "breakdown" in (r["setup"] or "").lower(),
    "Mean-reversion setups only":
        lambda r: any(k in (r["setup"] or "").lower()
                      for k in ("oversold", "overbought", "band")),
}


def backtest_strategy(name: str, df_evaluated: pd.DataFrame,
                      starting_capital: float = 10_000.0,
                      capital_per_trade: float = 1_000.0) -> StrategyResult:
    """Simulate fixed-dollar trades chronologically.

    Each qualifying trade gets `capital_per_trade` invested. The % return on
    that trade is applied to the position; cash is replenished from a shared
    pool. (Simple, doesn't model concurrent open positions or fees.)
    """
    pred = STRATEGIES.get(name, lambda r: True)
    df = df_evaluated.copy()
    df = df[df["evaluated"] == 1]
    df = df[df["outcome"].isin(["WIN", "LOSS", "EXPIRED"])]
    df = df[df.apply(pred, axis=1)] if not df.empty else df
    df = df.sort_values("scan_time")

    if df.empty:
        return StrategyResult(
            name=name, n_trades=0, n_wins=0, n_losses=0, n_expired=0,
            win_rate=0.0, avg_pnl_pct=0.0, total_pnl_pct=0.0,
            final_equity=starting_capital,
            starting_capital=starting_capital,
            capital_per_trade=capital_per_trade,
            max_drawdown_pct=0.0, best_trade_pct=0.0, worst_trade_pct=0.0,
            expectancy_pct=0.0,
            equity_curve=[(datetime.now().strftime("%Y-%m-%d"),
                            starting_capital)],
        )

    equity = starting_capital
    peak = starting_capital
    max_dd = 0.0
    curve = [(df.iloc[0]["scan_time"][:10], equity)]

    wins = losses = expired = 0
    pnl_list: List[float] = []

    for _, r in df.iterrows():
        pnl_pct = float(r["pnl_pct"] or 0.0)
        pnl_list.append(pnl_pct)
        # Apply $-allocation × pct return to overall equity
        equity += capital_per_trade * (pnl_pct / 100.0)
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        outcome = r["outcome"]
        if outcome == "WIN":
            wins += 1
        elif outcome == "LOSS":
            losses += 1
        else:
            expired += 1
        curve.append((r.get("exit_date") or r["scan_time"][:10], equity))

    n = len(pnl_list)
    win_rate = wins / n if n else 0.0
    avg = float(np.mean(pnl_list)) if pnl_list else 0.0
    avg_win = float(np.mean([p for p in pnl_list if p > 0])) if any(p > 0 for p in pnl_list) else 0.0
    avg_loss = float(np.mean([p for p in pnl_list if p <= 0])) if any(p <= 0 for p in pnl_list) else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return StrategyResult(
        name=name, n_trades=n, n_wins=wins, n_losses=losses, n_expired=expired,
        win_rate=round(win_rate * 100, 1),
        avg_pnl_pct=round(avg, 2),
        total_pnl_pct=round(sum(pnl_list), 2),
        final_equity=round(equity, 2),
        starting_capital=starting_capital,
        capital_per_trade=capital_per_trade,
        max_drawdown_pct=round(max_dd, 2),
        best_trade_pct=round(max(pnl_list), 2),
        worst_trade_pct=round(min(pnl_list), 2),
        expectancy_pct=round(expectancy, 2),
        equity_curve=curve,
    )


# ===========================================================================
#                  EMPIRICAL RATING CALIBRATION
# ===========================================================================
# Once enough closed trades have accumulated in the tracker DB, we can replace
# the hand-picked score thresholds (A+ ≥ 80, A ≥ 70, etc.) with thresholds
# derived from actual win rates. This avoids rating creep and makes the
# letter grades mean what they say.

CALIBRATION_PATH = DB_PATH.parent / "calibration.json"
MIN_TRADES_FOR_CALIBRATION = 30  # need at least this many closed trades


def calibrate_rating_thresholds(min_trades: int = MIN_TRADES_FOR_CALIBRATION
                                ) -> Optional[dict]:
    """Recompute score thresholds from actual outcomes in the tracker DB.

    Strategy:
        - Pull every closed trade (WIN/LOSS/EXPIRED) with its score
        - Compute win rate by score bucket (deciles)
        - Set rating thresholds at percentiles that produce target win rates:
            A+: top 10% of scores AND win rate ≥ 65%
            A : top 25% of scores AND win rate ≥ 55%
            B : top 50% of scores AND win rate ≥ 45%
            C : top 75%
            D : bottom 25%
        - Write to calibration.json for the engine to use on next scan

    Returns the new thresholds dict, or None if not enough data.
    """
    df = load_all_recommendations()
    closed = df[df["evaluated"] == 1]
    closed = closed[closed["outcome"].isin(["WIN", "LOSS", "EXPIRED"])]

    if len(closed) < min_trades:
        return None

    # Treat WIN as 1, EXPIRED with positive pnl as 0.5, LOSS as 0
    def outcome_value(row):
        if row["outcome"] == "WIN":
            return 1.0
        if row["outcome"] == "LOSS":
            return 0.0
        # EXPIRED — credit by sign of P&L
        return 0.5 if (row.get("pnl_pct") or 0) > 0 else 0.0

    closed = closed.copy()
    closed["win_value"] = closed.apply(outcome_value, axis=1)
    closed = closed.sort_values("score", ascending=False).reset_index(drop=True)

    n = len(closed)

    # Helper: compute min-score threshold that gives target win rate over the
    # top-K scores (descending). Returns (threshold, observed_win_rate).
    def threshold_for(top_k: int) -> Tuple[float, float]:
        top_k = min(top_k, n)
        if top_k <= 0:
            return 999.0, 0.0
        top = closed.head(top_k)
        return float(top["score"].min()), float(top["win_value"].mean())

    aplus_n = max(int(n * 0.10), 3)
    a_n     = max(int(n * 0.25), 5)
    b_n     = max(int(n * 0.50), 10)
    c_n     = max(int(n * 0.75), 15)

    aplus_thr, aplus_wr = threshold_for(aplus_n)
    a_thr, a_wr         = threshold_for(a_n)
    b_thr, b_wr         = threshold_for(b_n)
    c_thr, c_wr         = threshold_for(c_n)

    # Sanity: enforce monotonic thresholds
    a_thr     = min(a_thr, aplus_thr - 0.5)
    b_thr     = min(b_thr, a_thr - 0.5)
    c_thr     = min(c_thr, b_thr - 0.5)

    cal = {
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "n_trades": int(n),
        "thresholds": {
            "A+": round(aplus_thr, 2),
            "A":  round(a_thr, 2),
            "B":  round(b_thr, 2),
            "C":  round(c_thr, 2),
        },
        "observed_win_rates": {
            "A+": round(aplus_wr, 3),
            "A":  round(a_wr, 3),
            "B":  round(b_wr, 3),
            "C":  round(c_wr, 3),
        },
    }

    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(cal, f, indent=2)
    return cal


def load_calibration() -> Optional[dict]:
    """Load the latest calibration from disk, if any."""
    if not CALIBRATION_PATH.exists():
        return None
    try:
        with open(CALIBRATION_PATH) as f:
            return json.load(f)
    except Exception:
        return None


# Override score_to_rating to honor calibration when available
_ORIGINAL_score_to_rating = score_to_rating


def score_to_rating(score: float, risk_reward: float, trend_aligned: bool):
    """Calibrated version: uses learned thresholds if available, else falls
    back to the hand-picked defaults."""
    cal = load_calibration()
    if cal is None:
        return _ORIGINAL_score_to_rating(score, risk_reward, trend_aligned)

    th = cal["thresholds"]
    s = max(0.0, min(100.0, float(score)))

    # Same probability model as before
    base_p = 1.0 / (1.0 + math.exp(-(s - 55.0) / 10.0))
    if trend_aligned:
        base_p += 0.05
    if risk_reward >= 2.5:
        base_p += 0.03
    elif risk_reward < 1.5:
        base_p -= 0.05
    p = max(0.10, min(0.80, base_p))

    if s >= th["A+"] and risk_reward >= 2.0:
        return "A+", 5, round(p, 3)
    elif s >= th["A"]:
        return "A", 4, round(p, 3)
    elif s >= th["B"]:
        return "B", 3, round(p, 3)
    elif s >= th["C"]:
        return "C", 2, round(p, 3)
    else:
        return "D", 1, round(p, 3)


# ===========================================================================
#                   ML CLASSIFIER (XGBoost — optional)
# ===========================================================================
# When enough labeled trades exist, train a gradient-boosted classifier on the
# numerical features (RSI, ATR%, score, etc.) to predict win probability.
# Used to *augment*, not replace, the rule-based score: the scan blends
# 50/50 with the heuristic when a model is loaded.

ML_MODEL_PATH = DB_PATH.parent / "ml_model.json"
ML_FEATURES = [
    "score", "rsi", "atr_pct", "risk_reward", "win_probability",
    "is_long", "is_uptrend", "is_downtrend",
]
MIN_TRADES_FOR_ML = 50  # ML needs more data than calibration


def _features_for_candidate(cand) -> List[float]:
    return [
        float(cand.score),
        float(cand.rsi),
        float(cand.atr_pct),
        float(cand.risk_reward),
        float(cand.win_probability),
        1.0 if cand.direction == "LONG" else 0.0,
        1.0 if cand.trend == "Up" else 0.0,
        1.0 if cand.trend == "Down" else 0.0,
    ]


def _features_for_db_row(row) -> List[float]:
    return [
        float(row["score"] or 0),
        float(row["rsi"] or 50),
        float(row["atr_pct"] or 0),
        float(row["risk_reward"] or 0),
        float(row["win_probability"] or 0.5),
        1.0 if row["direction"] == "LONG" else 0.0,
        1.0 if row["trend"] == "Up" else 0.0,
        1.0 if row["trend"] == "Down" else 0.0,
    ]


def train_ml_model(min_trades: int = MIN_TRADES_FOR_ML) -> Optional[dict]:
    """Train an XGBoost classifier on closed trades. Falls back to a logistic
    regression if XGBoost isn't installed.

    Returns metadata dict on success, None if not enough data.
    """
    df = load_all_recommendations()
    closed = df[df["evaluated"] == 1]
    closed = closed[closed["outcome"].isin(["WIN", "LOSS", "EXPIRED"])]

    if len(closed) < min_trades:
        return None

    # Label: WIN=1, LOSS=0, EXPIRED with +pnl=1, EXPIRED with -pnl=0
    def label(row):
        if row["outcome"] == "WIN":
            return 1
        if row["outcome"] == "LOSS":
            return 0
        return 1 if (row.get("pnl_pct") or 0) > 0 else 0

    closed = closed.copy()
    closed["label"] = closed.apply(label, axis=1)
    X = closed.apply(_features_for_db_row, axis=1, result_type="expand").values
    y = closed["label"].values

    # Try XGBoost first
    model_kind = "xgboost"
    try:
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=y, feature_names=ML_FEATURES)
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "eta": 0.1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
        }
        booster = xgb.train(params, dtrain, num_boost_round=120)
        booster.save_model(str(ML_MODEL_PATH))
    except ImportError:
        # Fallback: simple logistic regression we can serialize as JSON
        model_kind = "logistic"
        weights = _train_logistic(X, y)
        with open(ML_MODEL_PATH, "w") as f:
            json.dump({"kind": "logistic", "weights": weights,
                        "features": ML_FEATURES}, f)

    # Write meta BEFORE evaluating, so load_ml_model() works
    meta = {
        "kind": model_kind,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "n_trades": int(len(closed)),
        "in_sample_accuracy": 0.0,  # placeholder, updated below
        "feature_names": ML_FEATURES,
        "model_path": str(ML_MODEL_PATH),
    }
    meta_path = ML_MODEL_PATH.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Now load the model properly and compute in-sample accuracy
    loaded = load_ml_model()
    if loaded is not None:
        pred_p = predict_with_model(loaded, None, X_raw=X)
        pred = (np.asarray(pred_p) >= 0.5).astype(int)
        acc = float((pred == y).mean()) if len(y) else 0.0
    else:
        acc = 0.0

    meta["in_sample_accuracy"] = round(acc, 3)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def _train_logistic(X, y, lr=0.05, epochs=300, l2=0.01):
    """Tiny logistic regression. Used only when XGBoost isn't installed."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # z-score
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-9
    Xn = (X - mu) / sd
    # Add bias column
    Xb = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(epochs):
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = Xb.T @ (p - y) / len(y) + l2 * w
        w -= lr * grad
    return {"w": w.tolist(), "mean": mu.tolist(), "std": sd.tolist()}


def load_ml_model():
    """Load the trained model. Returns a dict with `kind` + handle, or None."""
    meta_path = ML_MODEL_PATH.with_suffix(".meta.json")
    if not ML_MODEL_PATH.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        kind = meta.get("kind", "logistic")
        if kind == "xgboost":
            try:
                import xgboost as xgb
                booster = xgb.Booster()
                booster.load_model(str(ML_MODEL_PATH))
                return {"kind": "xgboost", "booster": booster, "meta": meta}
            except ImportError:
                return None
        else:
            with open(ML_MODEL_PATH) as f:
                payload = json.load(f)
            return {"kind": "logistic", "params": payload["weights"],
                    "meta": meta}
    except Exception:
        return None


def predict_with_model(model, cand, X_raw=None) -> float:
    """Predict win probability. If `cand` is given, builds features from it.
    If `X_raw` is given (numpy array), batch-predicts and returns array."""
    if model is None:
        return 0.5
    if X_raw is not None:
        X = np.asarray(X_raw, dtype=float)
        if model["kind"] == "xgboost":
            import xgboost as xgb
            d = xgb.DMatrix(X, feature_names=ML_FEATURES)
            return model["booster"].predict(d)
        else:
            p = model["params"]
            mu = np.asarray(p["mean"])
            sd = np.asarray(p["std"])
            Xn = (X - mu) / sd
            Xb = np.hstack([Xn, np.ones((Xn.shape[0], 1))])
            w = np.asarray(p["w"])
            z = Xb @ w
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
    # Single-candidate prediction
    feats = np.asarray([_features_for_candidate(cand)], dtype=float)
    out = predict_with_model(model, None, X_raw=feats)
    return float(out[0])


# Convenience: re-export sector list helpers
SECTOR_OPTIONS = sorted({
    "Technology", "Communication Services", "Consumer Cyclical",
    "Consumer Defensive", "Healthcare", "Financial Services",
    "Industrials", "Energy", "Basic Materials", "Real Estate",
    "Utilities", "ETF", "Unknown",
})
