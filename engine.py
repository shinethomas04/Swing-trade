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
# Liquid U.S. equity universe to scan. ~150 of the most actively traded names
# across mega-caps, semis, AI, fintech, biotech, energy, retail, ETFs, and
# popular high-beta movers. We re-rank these by dollar volume each run.
UNIVERSE: List[str] = sorted(set([
    # Mega-caps & FAANG+
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "BRK-B", "JPM", "V", "MA", "UNH", "HD", "WMT", "PG", "XOM", "CVX",
    "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT", "DHR", "BMY",
    # Semis / AI
    "AMD", "INTC", "MU", "TSM", "ASML", "QCOM", "ARM", "MRVL", "SMCI",
    "AMAT", "LRCX", "KLAC", "ON", "MPWR", "ADI", "TXN", "NXPI",
    "PLTR", "SNOW", "CRWD", "PANW", "NET", "DDOG", "MDB", "ZS", "S",
    # Software / Internet
    "ORCL", "CRM", "ADBE", "NOW", "INTU", "SHOP", "UBER", "ABNB", "DASH",
    "SQ", "PYPL", "COIN", "HOOD", "RBLX", "U", "ROKU", "SPOT", "PINS",
    "SNAP", "DKNG", "BABA", "JD", "PDD", "BIDU", "NIO", "LI", "XPEV",
    # Financials
    "BAC", "WFC", "C", "GS", "MS", "SCHW", "BLK", "AXP", "COF", "USB",
    # Industrials / Energy
    "BA", "CAT", "DE", "GE", "HON", "RTX", "LMT", "F", "GM", "RIVN",
    "LCID", "NKLA", "PLUG", "FCEL", "ENPH", "FSLR", "RUN", "OXY", "MRO",
    "DVN", "EOG", "SLB", "HAL", "MPC", "PSX", "VLO",
    # Consumer / Retail
    "COST", "MCD", "SBUX", "NKE", "LULU", "TGT", "DG", "DLTR", "BBY",
    "TJX", "ROST", "ULTA", "CMG", "DPZ", "QSR",
    # Biotech / Pharma movers
    "MRNA", "BNTX", "NVAX", "GILD", "REGN", "VRTX", "AMGN", "BIIB",
    "ILMN", "PFE",
    # ETFs (very high dollar volume, useful for market context)
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XBI",
    "SOXL", "SOXS", "TQQQ", "SQQQ", "TLT", "GLD", "SLV", "USO",
    # Popular high-volume movers
    "GME", "AMC", "BB", "MARA", "RIOT", "CLSK", "MSTR", "HUT", "BITF",
    "DJT", "TRUMP", "SAVE", "CCL", "NCLH", "RCL", "AAL", "DAL", "UAL",
    "T", "VZ", "TMUS", "DIS", "NFLX", "CMCSA", "PARA",
]))


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
#                       PERSISTENCE  (SQLite recommendation log)
# ===========================================================================
DB_PATH = Path.home() / ".swing_trader" / "history.db"


def db_connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
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
            -- evaluation results (filled in later)
            evaluated INTEGER DEFAULT 0,
            outcome TEXT,                  -- WIN / LOSS / OPEN / EXPIRED
            exit_price REAL,
            exit_date TEXT,
            days_held INTEGER,
            pnl_pct REAL,
            max_favorable_pct REAL,
            max_adverse_pct REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scan_id ON recommendations(scan_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker ON recommendations(ticker)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluated ON recommendations(evaluated)")
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
    try:
        df = pd.read_sql_query(
            "SELECT * FROM recommendations ORDER BY scan_time DESC, score DESC",
            conn,
        )
    finally:
        conn.close()
    return df


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
