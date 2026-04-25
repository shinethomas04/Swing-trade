"""
Swing Trade Scanner — Streamlit web/mobile app
================================================
Phone-friendly UI on top of the same analysis engine used by the desktop app.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Then on your phone:
    1. Make sure phone & PC are on the same Wi-Fi.
    2. Find your PC's local IP (e.g. 192.168.1.42).
    3. On the phone, open http://192.168.1.42:8501
    4. Chrome menu → "Add to Home Screen" to get an app-style icon.

For internet-wide access you can deploy free to Streamlit Community Cloud
(https://streamlit.io/cloud) — it auto-installs requirements.txt and serves
the app at a public URL.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

import engine as eng

# ---------------------------------------------------------------- page config
st.set_page_config(
    page_title="Swing Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Swing Trade Scanner — educational use only."},
)

# Mobile-friendly CSS: tighter padding, sticky tab bar, larger tap targets,
# zebra-striped tables, color-coded outcome chips.
st.markdown("""
<style>
    /* compact top padding so the title sits high on phone screens */
    .block-container { padding-top: 1rem; padding-bottom: 4rem; }
    /* bigger tap targets on phones */
    .stButton > button {
        height: 3rem; font-weight: 600; font-size: 1rem;
        border-radius: 10px;
    }
    /* color the primary action button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        border: none; color: white;
    }
    /* Tabs spacing */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        height: 3rem; font-size: 1rem; font-weight: 600;
        padding: 0 16px;
    }
    /* metric labels readable on small screens */
    [data-testid="stMetricLabel"] { font-size: 0.85rem; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    /* table sizing */
    [data-testid="stDataFrame"] { font-size: 0.9rem; }
    /* hide the "deploy" header on mobile to save space */
    header { visibility: hidden; height: 0; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- session state
if "candidates" not in st.session_state:
    st.session_state.candidates = []
if "scan_run_at" not in st.session_state:
    st.session_state.scan_run_at = None
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None


# ---------------------------------------------------------------- helpers
def fmt_money(v: float) -> str:
    if v >= 1e9: return f"${v/1e9:,.2f}B"
    if v >= 1e6: return f"${v/1e6:,.1f}M"
    if v >= 1e3: return f"${v/1e3:,.1f}K"
    return f"${v:,.0f}"


def stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def candidates_to_df(cands) -> pd.DataFrame:
    rows = []
    for i, c in enumerate(cands, start=1):
        # Build a "Flags" column with chips for warnings
        flags = []
        if c.earnings_warning and c.days_to_earnings is not None:
            flags.append(f"🔔 EARN {c.days_to_earnings}d")
        if not c.regime_aligned:
            flags.append("⚠ Counter-regime")
        # Sector cap warning (rating-demoted by sector)
        if any("Sector cap" in w for w in (c.warnings or [])):
            flags.append("📊 Sector-capped")
        flags_str = "  ".join(flags) if flags else ""

        rows.append({
            "#": i,
            "Ticker": c.ticker,
            "Grade": c.rating,
            "Rating": stars(c.stars),
            "Side": c.direction,
            "Score": round(c.score, 1),
            "Win %": int(round(c.win_probability * 100)),
            "Flags": flags_str,
            "Setup": c.setup,
            "Sector": (c.sector or "Unknown")[:14],  # truncate for table
            "Trend": c.trend,
            "Price": round(c.last_price, 2),
            "4d $Vol": fmt_money(c.dollar_volume),
            "RSI": round(c.rsi, 1),
            "ATR%": round(c.atr_pct, 2),
            "Entry": round(c.entry, 2),
            "Stop": round(c.stop, 2),
            "Target": round(c.target, 2),
            "R:R": round(c.risk_reward, 2),
        })
    return pd.DataFrame(rows)


def style_candidate_df(df: pd.DataFrame):
    """Color-code by rating and side."""
    def row_style(row):
        rating = row.get("Grade", "")
        side = row.get("Side", "")
        bg = {
            "A+": "background-color: #14532d; color: white;",
            "A":  "background-color: #166534; color: white;",
            "B":  "background-color: #1e3a8a; color: white;",
            "C":  "background-color: #78350f; color: white;",
            "D":  "background-color: #7f1d1d; color: white;",
        }.get(rating, "")
        styles = [bg] * len(row)
        # Color the Side cell
        if "Side" in row.index:
            idx = list(row.index).index("Side")
            color = "#22c55e" if side == "LONG" else "#ef4444"
            styles[idx] = bg + f" color: {color}; font-weight: 700;"
        return styles
    return df.style.apply(row_style, axis=1)


# ---------------------------------------------------------------- header
st.title("📈 Swing Trade Scanner")
st.caption(
    "Top 50 U.S. stocks by 4-day dollar volume → 30-day chart scoring → "
    "A+/A/B/C/D conviction rating → tracked over time."
)


# ---------------------------------------------------------------- TABS
tab_scan, tab_track, tab_about = st.tabs(
    ["📡  Scanner", "📊  Performance Tracker", "ℹ️  About"]
)


# =====================================================================
# SCANNER TAB
# =====================================================================
with tab_scan:
    # Top control row — the "find N A-rated" workflow
    cfg_col1, cfg_col2, cfg_col3 = st.columns([1, 1, 1])
    with cfg_col1:
        target_hits = st.number_input(
            "Target A/A+ setups to find",
            min_value=1, max_value=50,
            value=eng.DEFAULT_TARGET_HITS, step=1,
            help="The scan walks the dollar-volume-ranked universe and stops "
                 "as soon as it has found this many A-or-better candidates.",
        )
    with cfg_col2:
        max_cap = st.number_input(
            "Max symbols to analyze",
            min_value=25, max_value=400,
            value=eng.MAX_TICKERS_TO_ANALYZE, step=25,
            help="Safety cap. If the market is quiet and the target isn't "
                 "hit, the scan stops here so it doesn't run forever.",
        )
    with cfg_col3:
        min_score = st.slider(
            "Display: min score", 0, 100, 0, 5,
            help="Hide candidates with lower scores (does not affect the scan)",
        )

    # ---- Advanced filters (v2 enhancements) ----
    with st.expander("🛡️  Advanced filters & risk controls", expanded=False):
        f1, f2, f3 = st.columns(3)
        with f1:
            earnings_window = st.number_input(
                "Earnings warning window (days)",
                min_value=0, max_value=30, value=5, step=1,
                help="Warn (and downgrade) candidates with earnings releases "
                     "within this many trading days. Set to 0 to disable.",
            )
            skip_earnings = st.checkbox(
                "Skip earnings names entirely",
                value=False,
                help="If checked, candidates with imminent earnings are removed "
                     "instead of just downgraded.",
            )
        with f2:
            regime_filter = st.checkbox(
                "Apply SPY market-regime overlay",
                value=True,
                help="Detects whether SPY is trending up, down, or sideways "
                     "and downgrades counter-regime trades.",
            )
            sector_cap = st.number_input(
                "Max A-rated per sector",
                min_value=0, max_value=10, value=2, step=1,
                help="Concentration limit. After this many A+/A setups in the "
                     "same GICS sector, the rest are demoted to B. Set to 0 "
                     "to disable.",
            )
        with f3:
            use_ml = st.checkbox(
                "Blend ML model (if trained)",
                value=True,
                help="When a model has been trained from your tracker history, "
                     "blend its win-probability 50/50 with the heuristic.",
            )
            # Show ML model status
            ml_meta_path = eng.ML_MODEL_PATH.with_suffix(".meta.json")
            if ml_meta_path.exists():
                try:
                    import json as _json
                    with open(ml_meta_path) as f:
                        meta = _json.load(f)
                    st.caption(
                        f"📦 Model loaded: **{meta.get('kind','?')}** trained on "
                        f"{meta.get('n_trades','?')} trades, "
                        f"in-sample acc={meta.get('in_sample_accuracy','?')}"
                    )
                except Exception:
                    st.caption("📦 Model file present but unreadable")
            else:
                st.caption("📦 No ML model trained yet (use Tracker tab)")

    run_clicked = st.button("▶  Run Scan", type="primary",
                             use_container_width=True)

    if run_clicked:
        if eng.yf is None:
            st.error("yfinance is not installed. Run: pip install yfinance")
            st.stop()

        prog = st.progress(0, text="Starting scan…")
        status_box = st.empty()
        try:
            def cb(stage, current, total, msg):
                if stage == "regime":
                    pct = 2
                elif stage == "ranking":
                    pct = 5 + int(15 * current / max(total, 1))
                else:  # scanning
                    pct = 25 + int(70 * current / max(total, 1))
                    pct = min(95, pct)
                prog.progress(pct, text=msg)
                status_box.caption(msg)

            results, stats = eng.scan_until_target_hits(
                target_a_count=int(target_hits),
                max_to_analyze=int(max_cap),
                progress_cb=cb,
                earnings_window_days=int(earnings_window),
                skip_on_earnings=bool(skip_earnings),
                apply_regime_filter=bool(regime_filter),
                sector_cap=int(sector_cap),
                use_ml_blend=bool(use_ml),
            )

            if not results:
                prog.empty()
                status_box.empty()
                st.error("No candidates produced. " + stats.get("reason", ""))
                st.stop()

            prog.progress(98, text="Saving recommendations to history…")
            try:
                eng.save_recommendations(results)
            except Exception as e:
                st.warning(f"Could not save scan: {e}")

            prog.progress(100, text="Done.")
            st.session_state.candidates = results
            st.session_state.scan_run_at = datetime.now()
            st.session_state.scan_stats = stats
            prog.empty()
            status_box.empty()
        except Exception as e:
            prog.empty()
            status_box.empty()
            st.error(f"Scan failed: {e}")

    # --- summary metrics ---
    cands = st.session_state.candidates
    if cands:
        st.caption(f"Last scan: {st.session_state.scan_run_at:%Y-%m-%d %H:%M:%S}")

        # Show the scan-stop reason (target reached, or hit cap, etc.)
        stats = st.session_state.get("scan_stats", {})
        if stats:
            target_met = stats.get("a_plus", 0) + stats.get("a", 0) >= stats.get("target", 0)
            icon = "✅" if target_met else "⚠️"
            extra = ""
            if stats.get("skipped_earnings", 0) > 0:
                extra = f"  ·  Earnings warnings on {stats['skipped_earnings']} names"
            if stats.get("ml_model_loaded"):
                extra += "  ·  🤖 ML probability blended"
            st.info(
                f"{icon}  **{stats.get('reason', '')}**  \n"
                f"Analyzed {stats.get('analyzed', 0)} of {stats.get('ranked_size', 0)} "
                f"ranked symbols (universe of {stats.get('universe_size', 0)}).{extra}"
            )

            # Market regime banner — color-coded
            regime = stats.get("regime", "Unknown")
            regime_note = stats.get("regime_note", "")
            regime_colors = {
                "Strong Uptrend": ("#14532d", "🟢"),
                "Uptrend": ("#166534", "🟢"),
                "Sideways": ("#78350f", "🟡"),
                "Downtrend": ("#991b1b", "🔴"),
                "Strong Downtrend": ("#7f1d1d", "🔴"),
                "Unknown": ("#374151", "⚪"),
            }
            bg, em = regime_colors.get(regime, ("#374151", "⚪"))
            st.markdown(
                f"<div style='background:{bg}; padding:10px 14px; "
                f"border-radius:8px; margin:4px 0;'>"
                f"<b>{em} Market regime: {regime}</b><br>"
                f"<span style='color:#cfcfcf; font-size:0.9rem'>{regime_note}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

        n_aplus = sum(1 for c in cands if c.rating == "A+")
        n_a     = sum(1 for c in cands if c.rating == "A")
        n_b     = sum(1 for c in cands if c.rating == "B")
        n_long  = sum(1 for c in cands if c.direction == "LONG")
        n_short = sum(1 for c in cands if c.direction == "SHORT")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("A+ setups", n_aplus)
        m2.metric("A setups", n_a)
        m3.metric("B setups", n_b)
        m4.metric("Longs", n_long)
        m5.metric("Shorts", n_short)

        # --- table ---
        df = candidates_to_df(
            [c for c in cands if c.score >= min_score]
        )
        if df.empty:
            st.info(f"No candidates above min score = {min_score}.")
        else:
            st.dataframe(
                style_candidate_df(df),
                use_container_width=True,
                hide_index=True,
                height=520,
            )

        # --- per-ticker detail (mobile-friendly accordion) ---
        st.markdown("### 🔍 Detail view")
        ticker_options = [c.ticker for c in cands if c.score >= min_score]
        if ticker_options:
            choice = st.selectbox(
                "Pick a ticker for the full breakdown",
                options=ticker_options,
                index=0,
            )
            cand = next((c for c in cands if c.ticker == choice), None)
            if cand:
                emoji = "🟢" if cand.direction == "LONG" else "🔴"
                color = "#22c55e" if cand.direction == "LONG" else "#ef4444"
                st.markdown(
                    f"### {emoji} {cand.ticker} "
                    f"<span style='color:{color}'>{cand.direction}</span>"
                    f"  &nbsp; · &nbsp;  **{cand.rating}** {stars(cand.stars)}",
                    unsafe_allow_html=True,
                )

                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Score", f"{cand.score:.1f}/100")
                d2.metric("Win prob.", f"{cand.win_probability*100:.0f}%")
                d3.metric("R:R", f"1 : {cand.risk_reward:.2f}")
                d4.metric("Trend", cand.trend)

                p1, p2, p3 = st.columns(3)
                p1.metric("Entry", f"${cand.entry:,.2f}")
                p2.metric("Stop", f"${cand.stop:,.2f}",
                            f"{(cand.stop-cand.entry)/cand.entry*100:+.2f}%")
                p3.metric("Target", f"${cand.target:,.2f}",
                            f"{(cand.target-cand.entry)/cand.entry*100:+.2f}%")

                # ---- Risk warnings prominently shown at top ----
                if cand.warnings:
                    st.warning("⚠ " + "  ·  ".join(cand.warnings))

                # ---- Context row: sector, regime, earnings, ML ----
                ctx1, ctx2, ctx3, ctx4 = st.columns(4)
                ctx1.metric("Sector", cand.sector or "Unknown")
                aligned = "Aligned ✓" if cand.regime_aligned else "Counter ✗"
                ctx2.metric("vs Regime", aligned)
                if cand.next_earnings and cand.days_to_earnings is not None:
                    ctx3.metric(
                        "Next earnings",
                        f"{cand.days_to_earnings}d",
                        cand.next_earnings,
                        delta_color=("inverse" if cand.earnings_warning
                                     else "off"),
                    )
                else:
                    ctx3.metric("Next earnings", "—")
                if cand.ml_probability is not None:
                    ctx4.metric("🤖 ML prob",
                                f"{cand.ml_probability*100:.0f}%",
                                f"{(cand.ml_probability - 0.5)*100:+.0f}% vs 50/50")
                else:
                    ctx4.metric("🤖 ML prob", "n/a")

                st.markdown("**Indicators**")
                st.markdown(
                    f"- Setup: **{cand.setup}**\n"
                    f"- RSI(14): **{cand.rsi:.1f}**\n"
                    f"- ATR(14): **{cand.atr:.2f}** ({cand.atr_pct:.2f}% of price)\n"
                    f"- SMA-20: ${cand.sma20:,.2f} · long MA: ${cand.sma50:,.2f}\n"
                    f"- 4-day avg $-volume: **{fmt_money(cand.dollar_volume)}**"
                )

                if cand.reasons:
                    st.markdown("**Why it triggered**")
                    for r in cand.reasons:
                        st.markdown(f"- {r}")

                st.caption(
                    "Suggested risk: "
                    f"{abs(cand.entry-cand.stop)/cand.entry*100:.2f}% / "
                    f"reward: {abs(cand.target-cand.entry)/cand.entry*100:.2f}%"
                )
    else:
        st.info(
            "Set your **target A/A+ count** above, then tap **Run Scan**. "
            "The app walks the dollar-volume-ranked universe of "
            f"{len(eng.UNIVERSE)} stocks and stops as soon as it has found "
            "the requested number of A-rated setups (or hits the safety cap). "
            "Each scan is saved for the Performance Tracker tab."
        )


# =====================================================================
# TRACKER TAB
# =====================================================================
with tab_track:
    st.subheader("Performance tracker")
    st.caption("Replays past recommendations against actual price action.")

    tcol1, tcol2 = st.columns([1, 2])
    with tcol1:
        eval_clicked = st.button("🔄  Evaluate & Refresh",
                                   type="primary",
                                   use_container_width=True,
                                   key="eval_btn")
    with tcol2:
        capital = st.number_input("Capital per trade ($)",
                                    min_value=100, max_value=1_000_000,
                                    value=1000, step=100,
                                    help="Fixed allocation used in strategy back-tests")

    if eval_clicked:
        if eng.yf is None:
            st.error("yfinance is not installed.")
        else:
            prog2 = st.progress(0, text="Evaluating pending recommendations…")
            placeholder = st.empty()

            def cb(i, total, ticker):
                pct = int(95 * i / max(total, 1))
                prog2.progress(pct, text=f"Replaying {ticker} ({i}/{total})…")

            try:
                ev, op = eng.evaluate_pending(progress_cb=cb)
                prog2.progress(100, text="Done.")
                placeholder.success(
                    f"Evaluated {ev} new outcomes · {op} still open "
                    "(need more time before they can be scored)."
                )
                prog2.empty()
            except Exception as e:
                prog2.empty()
                st.error(f"Evaluation failed: {e}")

    # ---- Calibration & ML training ----
    with st.expander("🧠  Calibration & ML (advanced)", expanded=False):
        st.markdown(
            "Once enough trades have been **evaluated** (WIN/LOSS/EXPIRED), "
            "you can recalibrate the rating thresholds against your actual "
            "outcomes, and train a machine-learning model that predicts win "
            "probability."
        )
        cal_col, ml_col = st.columns(2)

        # ---- CALIBRATE ----
        with cal_col:
            st.markdown("**Empirical rating calibration**")
            cal_existing = eng.load_calibration()
            if cal_existing:
                st.caption(
                    f"Last calibrated: {cal_existing.get('computed_at','?')[:16]}  "
                    f"on {cal_existing.get('n_trades','?')} trades"
                )
                with st.container():
                    th = cal_existing.get("thresholds", {})
                    wr = cal_existing.get("observed_win_rates", {})
                    st.caption(
                        f"A+ ≥ {th.get('A+','?'):.0f} (won {wr.get('A+',0)*100:.0f}%)  ·  "
                        f"A ≥ {th.get('A','?'):.0f} ({wr.get('A',0)*100:.0f}%)  ·  "
                        f"B ≥ {th.get('B','?'):.0f} ({wr.get('B',0)*100:.0f}%)"
                    )
            else:
                st.caption("No calibration computed yet")

            if st.button(
                "📐 Recalibrate from history",
                use_container_width=True,
                key="cal_btn",
                help=f"Needs ≥{eng.MIN_TRADES_FOR_CALIBRATION} closed trades."
            ):
                cal = eng.calibrate_rating_thresholds()
                if cal is None:
                    st.warning(
                        f"Need at least {eng.MIN_TRADES_FOR_CALIBRATION} "
                        f"evaluated trades. You have fewer right now."
                    )
                else:
                    st.success(
                        f"Calibration updated from {cal['n_trades']} trades. "
                        f"New thresholds will be used on the next scan."
                    )
                    st.rerun()

        # ---- TRAIN ML ----
        with ml_col:
            st.markdown("**ML probability model**")
            ml_meta_path = eng.ML_MODEL_PATH.with_suffix(".meta.json")
            if ml_meta_path.exists():
                try:
                    import json as _json
                    with open(ml_meta_path) as f:
                        meta = _json.load(f)
                    st.caption(
                        f"📦 {meta.get('kind','?')} model trained "
                        f"{meta.get('trained_at','?')[:16]} on "
                        f"{meta.get('n_trades','?')} trades  ·  "
                        f"in-sample acc = {meta.get('in_sample_accuracy','?')}"
                    )
                except Exception:
                    st.caption("Model file present but unreadable")
            else:
                st.caption("No model trained yet")

            if st.button(
                "🤖 Train / retrain ML model",
                use_container_width=True,
                key="ml_btn",
                help=f"Needs ≥{eng.MIN_TRADES_FOR_ML} closed trades. "
                     "Uses XGBoost if installed, else logistic regression."
            ):
                with st.spinner("Training…"):
                    meta = eng.train_ml_model()
                if meta is None:
                    st.warning(
                        f"Need at least {eng.MIN_TRADES_FOR_ML} evaluated "
                        f"trades to train a model."
                    )
                else:
                    st.success(
                        f"Trained {meta['kind']} on {meta['n_trades']} trades. "
                        f"In-sample accuracy: {meta['in_sample_accuracy']}. "
                        f"It will be auto-blended into the next scan."
                    )
                    st.rerun()

    # --- load all data from DB ---
    df = eng.load_all_recommendations()

    if df.empty:
        st.info("No saved recommendations yet. Run a scan first.")
    else:
        # ---- summary metrics ----
        evaluated = df[df["evaluated"] == 1]
        wins = (evaluated["outcome"] == "WIN").sum()
        losses = (evaluated["outcome"] == "LOSS").sum()
        expired = (evaluated["outcome"] == "EXPIRED").sum()
        n_open = (df["evaluated"] == 0).sum()
        winrate = (wins / max(len(evaluated), 1)) * 100

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Total recs", len(df))
        s2.metric("Wins", int(wins))
        s3.metric("Losses", int(losses))
        s4.metric("Expired/Open", f"{int(expired)} / {int(n_open)}")
        s5.metric("Overall win rate", f"{winrate:.1f}%")

        # ---- strategy back-test table ----
        st.markdown("### 🎯 Strategy back-test")
        st.caption(
            f"Each strategy filters the same trade history and applies "
            f"${capital:,.0f}/trade. Final equity assumes a starting bankroll "
            f"of ${capital*10:,.0f}."
        )

        starting = capital * 10
        results = [
            eng.backtest_strategy(name, df,
                                    starting_capital=starting,
                                    capital_per_trade=capital)
            for name in eng.STRATEGIES.keys()
        ]
        results.sort(key=lambda r: r.final_equity, reverse=True)

        strat_df = pd.DataFrame([{
            "Strategy": r.name,
            "Trades": r.n_trades,
            "Wins": r.n_wins,
            "Losses": r.n_losses,
            "Win %": round(r.win_rate, 1),
            "Expectancy %": round(r.expectancy_pct, 2),
            "Avg P&L %": round(r.avg_pnl_pct, 2),
            "Best %": round(r.best_trade_pct, 2),
            "Worst %": round(r.worst_trade_pct, 2),
            "Max DD %": round(r.max_drawdown_pct, 2),
            "Final equity": round(r.final_equity, 2),
            "P&L $": round(r.final_equity - r.starting_capital, 2),
        } for r in results])

        # Color-code Final equity vs starting
        def color_pnl(v):
            if v > 0: return "color: #22c55e; font-weight: 700;"
            if v < 0: return "color: #ef4444; font-weight: 700;"
            return ""
        styled = strat_df.style.map(color_pnl, subset=["P&L $"]) \
                                .format({
                                    "Final equity": "${:,.2f}",
                                    "P&L $": "${:+,.2f}",
                                    "Win %": "{:.1f}%",
                                    "Expectancy %": "{:+.2f}%",
                                    "Avg P&L %": "{:+.2f}%",
                                    "Best %": "{:+.2f}%",
                                    "Worst %": "{:+.2f}%",
                                    "Max DD %": "{:.2f}%",
                                })
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # ---- equity curve chart for the user-selected strategy ----
        st.markdown("### 📈 Equity curve")
        strat_names = [r.name for r in results]
        chosen = st.selectbox("Strategy to chart", options=strat_names, index=0)
        chosen_res = next((r for r in results if r.name == chosen), None)
        if chosen_res and chosen_res.equity_curve:
            curve_df = pd.DataFrame(chosen_res.equity_curve,
                                      columns=["date", "equity"])
            curve_df["date"] = pd.to_datetime(curve_df["date"])
            curve_df = curve_df.set_index("date")
            st.line_chart(curve_df, height=280)
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Trades", chosen_res.n_trades)
            ec2.metric("Final equity",
                        f"${chosen_res.final_equity:,.2f}",
                        f"{(chosen_res.final_equity-chosen_res.starting_capital)/chosen_res.starting_capital*100:+.2f}%")
            ec3.metric("Max drawdown", f"{chosen_res.max_drawdown_pct:.2f}%")

        # ---- recommendation history ----
        st.markdown("### 📋 Recommendation history")
        hist_df = df[[
            "scan_time", "ticker", "direction", "rating", "score", "setup",
            "entry", "stop", "target", "outcome", "exit_price", "days_held",
            "pnl_pct"
        ]].copy()
        hist_df.columns = ["Scan time", "Ticker", "Side", "Grade", "Score",
                            "Setup", "Entry", "Stop", "Target", "Outcome",
                            "Exit", "Days", "P&L %"]
        hist_df["Scan time"] = pd.to_datetime(hist_df["Scan time"]).dt.strftime("%Y-%m-%d %H:%M")
        hist_df["Outcome"] = hist_df["Outcome"].fillna("OPEN")

        # outcome filter
        outcomes = ["All", "WIN", "LOSS", "EXPIRED", "OPEN"]
        oc_filter = st.selectbox("Filter by outcome", options=outcomes, index=0)
        if oc_filter != "All":
            hist_df = hist_df[hist_df["Outcome"] == oc_filter]

        def color_outcome(val):
            colors = {
                "WIN": "color: #22c55e; font-weight: 700;",
                "LOSS": "color: #ef4444; font-weight: 700;",
                "EXPIRED": "color: #cfcfcf;",
                "OPEN": "color: #fbbf24;",
            }
            return colors.get(val, "")

        styled_h = hist_df.style.map(color_outcome, subset=["Outcome"]) \
                                  .format({
                                      "Score": "{:.0f}",
                                      "Entry": "${:,.2f}",
                                      "Stop": "${:,.2f}",
                                      "Target": "${:,.2f}",
                                      "Exit": "${:,.2f}",
                                      "P&L %": "{:+.2f}%",
                                  }, na_rep="—")
        st.dataframe(styled_h, use_container_width=True, hide_index=True,
                      height=420)


# =====================================================================
# ABOUT TAB
# =====================================================================
with tab_about:
    st.subheader("How it works")
    st.markdown(f"""
**The scanner** pulls 6 days of daily bars for ~{len(eng.UNIVERSE)} of the most
liquid U.S. equities and ETFs and ranks them by 4-day average dollar volume.
Then it walks the ranked list from most-traded down, analyzing each name with
a multi-factor algorithm, until it has found **as many A-or-better setups as
you requested** (or hits the safety cap of {eng.MAX_TICKERS_TO_ANALYZE}
symbols, whichever comes first).

| Indicator | Purpose |
|---|---|
| SMA-20 + long MA | Trend direction |
| ADX(14) | Trend strength |
| RSI(14) | Pullback / oversold / overbought zones |
| MACD histogram | Momentum direction |
| Bollinger %B | Mean-reversion edge |
| 20-day Donchian high/low | Breakout level |
| ATR(14) | Stop placement |
| Volume vs 20d avg | Confirmation |

Trade plan: **1.5 × ATR stop** and **3.0 × ATR target** → built-in **1:2 R:R**.

**Conviction ratings (default thresholds — get recalibrated from your trade history):**

| Score | R:R | Grade |
|---|---|---|
| ≥ 80 | ≥ 2.0 | **A+** ★★★★★ |
| ≥ 70 | – | A ★★★★ |
| ≥ 55 | – | B ★★★ |
| ≥ 40 | – | C ★★ |
| < 40 | – | D ★ |

### Risk filters & enhancements (v2)

These are toggleable in the Scanner tab's "Advanced filters" expander:

- **🔔 Earnings filter.** Each candidate's next earnings date is checked.
  Names with earnings inside your window (default 5 days) are flagged with
  a warning chip and downgraded one rating step (A+ → A, A → B). You can
  also choose to skip them entirely.
- **📊 SPY market regime overlay.** SPY's daily chart is classified as
  Strong Uptrend / Uptrend / Sideways / Downtrend / Strong Downtrend.
  Counter-regime trades get demoted (e.g. a LONG setup in a Downtrend).
  This is the single biggest filter for avoiding "fighting the tape."
- **🏷️ Sector concentration cap.** Limits A+/A setups to N per sector
  (default 2). Stops you from accidentally betting your whole bankroll
  on semiconductors. Excess names in a sector are demoted to B.
- **📐 Empirical calibration.** Once you've evaluated ≥ {eng.MIN_TRADES_FOR_CALIBRATION}
  trades, the rating thresholds can be recomputed from your *actual* win
  rates instead of the hand-picked defaults. Click "Recalibrate" in the
  Tracker tab.
- **🤖 ML probability blend.** Once you've evaluated ≥ {eng.MIN_TRADES_FOR_ML}
  trades, an XGBoost classifier (or logistic-regression fallback) can be
  trained on (features → outcome) and its probability blended 50/50 with
  the heuristic win-probability. Train it from the Tracker tab.

### The tracker

Every recommendation is stored in a local SQLite database at
`{eng.DB_PATH}`. When you click *Evaluate & Refresh*, it pulls the actual
price action that followed each pending trade and marks it WIN / LOSS /
EXPIRED. Trades that haven't had ≥10 trading days yet stay OPEN.

The strategy back-test takes the same evaluated trades and replays them
through 11 different filters (A+ only, all longs, score ≥ 70, etc.), with
your chosen capital allocation per trade.

---

**Data:** Yahoo Finance via `yfinance` — free, no API key required.

**Educational use only — not investment advice.**
""")


# Footer
st.markdown("---")
if eng.using_turso():
    st.caption(
        "🟢 **Persistent storage active** (Turso) · "
        "Scan history survives redeploys and container restarts."
    )
else:
    st.caption(
        f"⚠️ Local DB: `{eng.DB_PATH}` · "
        "On Streamlit Cloud, history is wiped on redeploy. "
        "Configure `TURSO_URL` and `TURSO_TOKEN` in app secrets to persist."
    )
