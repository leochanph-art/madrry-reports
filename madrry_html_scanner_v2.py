"""
MADRRY Ultimate Scanner — v2 (hardened)
=======================================

A faster, more reliable rewrite of madrry_html_scanner.py. The *trading logic*
(M.E.T.A. momentum scoring, VCP/coil tiers, HVE episodic pivots, Post-HVE U&R,
trendline detection, entry/stop math) is preserved verbatim so signals are
identical to the original. What changed is everything *around* the logic:

  Reliability  - every network call retries with exponential backoff
               - each scan is isolated: one failure cannot kill the report
               - a visible DIAGNOSTICS panel surfaces errors instead of
                 swallowing them in stdout
               - outputs are written atomically (no half-written JSON)
  Speed        - yesterday's-watchlist tracking uses ONE batched yfinance
                 download instead of one network call per ticker
               - the 4 market-health probes run in parallel
               - sparklines reuse history already fetched for the coil scan
                 (zero extra network calls)
  Report / UX  - sticky table headers, click-to-sort columns, a live ticker
                 search box, inline SVG price sparklines, a run-summary bar,
                 and expandable M.E.T.A. detail that works on mobile
  Code quality - typed, dataclass config + diagnostics, logging, HTML escaping,
                 organised into clear sections

Run it the same way as the original:  python3 madrry_html_scanner_v2.py
"""

from __future__ import annotations

import concurrent.futures
import csv
import html as html_lib
import json
import hashlib
import logging
import math
import os
import re
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------------
# CONFIG  (paths unchanged — centralised so they live in one place)
# ----------------------------------------------------------------------------
WORKSPACE = "/Users/boundbythese/.openclaw/workspace"
LATEST_SETUPS_PATH = os.path.join(WORKSPACE, "latest_setups.json")
HVE_HISTORY_PATH = os.path.join(WORKSPACE, "hve_history.json")
BREADTH_HISTORY_PATH = os.path.join(WORKSPACE, "breadth_history.json")
EXT_PCTILE_PATH = os.path.join(WORKSPACE, "spy_qqq_extension_percentiles.csv")
FORWARD_BASERATE_PATH = os.path.join(WORKSPACE, "forward_baserates.json")
BREAKOUT_LOG_PATH = os.path.join(WORKSPACE, "breakout_log.json")
TIER_A_TRACKING_PATH = os.path.join(WORKSPACE, "tier_a_tracking.json")
META_WEIGHTS_PATH = os.path.join(WORKSPACE, "meta_weights.json")
EXCLUDED_TICKERS_PATH = os.path.join(WORKSPACE, "excluded_tickers.txt")

# ---- M.E.T.A. component weights (max points per component) -----------------
# The 11 scoring slots. "Candle" shares the "Flag" slot (one or the other fires),
# so it is NOT a separate slot and is not summed into the denominator.
# These defaults reproduce the original hardcoded 140-point architecture exactly;
# meta_weights.json (written by the weekend recalibration) overrides them.
META_WEIGHTS_DEFAULT = {
    "Trend": 15, "Proximity": 10, "10MA Quality": 15, "Vol Contraction": 15,
    "Vol Expansion": 10, "Flag": 10, "Base Quality": 15, "RS": 15,
    "Volatility": 10, "Supply Shock": 10, "Risk": 15,
}


def _load_meta_weights() -> Dict[str, float]:
    """Active weights = defaults overridden by meta_weights.json if present."""
    w = dict(META_WEIGHTS_DEFAULT)
    try:
        with open(META_WEIGHTS_PATH) as fh:
            blob = json.load(fh)
        for k, v in (blob.get("weights", blob) or {}).items():
            if k in w and isinstance(v, (int, float)):
                w[k] = v
    except Exception:  # noqa: BLE001 — missing/corrupt file => defaults
        pass
    return w


META_WEIGHTS = _load_meta_weights()
META_DENOM = sum(META_WEIGHTS.values()) or 1   # "Candle" excluded; `or 1` guards a 0-denom crash


def _load_excluded_tickers() -> set:
    """Tickers to keep OUT of the whole report — M&A targets / pending-delist names
    that pin near their deal price and masquerade as tight coils. One symbol per
    line in excluded_tickers.txt; blank lines and #-comments ignored. Edit that file
    (no code change needed) to add or drop names; the next scan picks it up."""
    out = set()
    try:
        with open(EXCLUDED_TICKERS_PATH) as fh:
            for line in fh:
                s = line.split("#", 1)[0].strip().upper()
                if s:
                    out.add(s)
    except OSError:
        pass
    return out


EXCLUDED_TICKERS = _load_excluded_tickers()

# ---- META v4 (enhanced signed-feature score) — ranking driver, with fallback ----
try:
    from madrry_meta_v4 import meta_v4_score as _meta_v4_score
except Exception:  # noqa: BLE001 — missing model/module => legacy score
    def _meta_v4_score(_df):
        return None

# ---- Fundamentals (past-2Q + next-2Q revenue/EPS) for the tap-to-expand narrative.
# Self-contained, disk-cached, never fatal. If the module is missing the helper is a
# no-op that returns the plain narrative unchanged.
try:
    import madrry_fundamentals as _fund
except Exception:  # noqa: BLE001
    _fund = None


def _narrative(ticker: str, inner_html: str) -> str:
    """Wrap a row's narrative (theme/sector/industry) so tapping it reveals the
    fundamentals panel. Falls back to the bare narrative if data/module unavailable."""
    if _fund is None:
        return inner_html
    try:
        return _fund.details_html(ticker, inner_html)
    except Exception:
        return inner_html


def _prefetch_fundamentals(tickers, budget_s: float = 90.0) -> None:
    """Warm the fundamentals cache for every ticker in the report. Called once for the
    main MADRRY batch (90s) and again, smaller, inside the Minervini/Trilogy generators
    — each call gets its OWN wall-clock budget, so keep the secondary ones short to bound
    the total fundamentals spend (most of their tickers are already cache-warm anyway)."""
    if _fund is None:
        return
    tk = [t for t in tickers if t]
    try:
        _fund.prefetch(tk, workers=8, budget_s=budget_s)
    except Exception:
        pass
    # Tier 2 (best-in-industry net-margin percentile): one batched scan per UNIQUE industry,
    # cached per day + idempotent, so the secondary Minervini/Trilogy calls are ~free. Own
    # short budget; failure just omits the "top X% in industry" clause.
    try:
        _fund.prefetch_industry_margins(tk, budget_s=min(30.0, budget_s / 3.0))
    except Exception:
        pass


REVISIONS_TOP_N = 12          # warm estimate-revision counts for this many top picks only


def _prefetch_revisions(tickers, budget_s: float = 25.0) -> None:
    """Tier 3 — warm analyst estimate-revision counts for a SMALL top-picks list. Per-ticker
    yfinance (NOT batched), so deliberately limited; never fatal."""
    if _fund is None:
        return
    try:
        _fund.prefetch_revisions([t for t in tickers if t], budget_s=budget_s)
    except Exception:
        pass


# --- sortable derived columns: price-to-MA distance + forward-quarter revenue YoY ---
def _ma_dist_data(closes) -> dict:
    """Price % extension above(+)/below(−) its trailing 10/20/50-day SMA, from a daily
    close series. Returns {10: pct|None, 20: pct|None, 50: pct|None} (None where fewer
    than `period` bars are available). Computed from the SAME close list the sparkline
    uses, so every table gets a uniform daily-SMA distance regardless of its own MA set."""
    vals = [float(c) for c in closes
            if c is not None and not (isinstance(c, float) and math.isnan(c))]
    out = {10: None, 20: None, 50: None}
    if not vals:
        return out
    px = vals[-1]
    for p in (10, 20, 50):
        if len(vals) >= p:
            ma = sum(vals[-p:]) / p
            out[p] = ((px - ma) / ma * 100.0) if ma else None
    return out


def _adr20(df, n: int = 20) -> float:
    """Average Daily Range over the last `n` sessions, as a percent — the canonical
    Qullamaggie ADR%:  100 × (mean(High / Low) − 1).

    This is THE single ADR definition for the whole report. It matches TradingView's
    native `ADRP` field (read directly on the tabs that scan TV — watchlist, new-highs)
    and the HTF engine's adr20, so "ADR" means the same slow 20-day range trait
    everywhere. It is used to COMPUTE ADR on the tabs fed by EXTERNAL history
    (Minervini / Trilogy), where no TradingView row exists. It is NOT TradingView's
    1-day `Volatility.D` — so the tightness gate (today's range ≤ ADR) compares today
    against the 20-day norm, not against itself.
    Returns 0.0 when history is too short/invalid (caller keeps its fallback)."""
    try:
        if df is None or len(df) < 2:
            return 0.0
        lo = df["Low"].astype(float).replace(0, np.nan)
        ratio = (df["High"].astype(float) / lo).iloc[-n:]
        v = float((ratio.mean() - 1.0) * 100.0)
        return v if v == v else 0.0          # NaN guard (== self is False for NaN)
    except Exception:  # noqa: BLE001
        return 0.0


def _ma_cells(d: Optional[dict]) -> str:
    """Three sortable <td>s — distance from the 10/20/50-day SMA. Display is the
    ABSOLUTE % with a ▲(above)/▼(below) arrow; data-sort is the absolute distance so
    clicking ascending puts the names TIGHTEST to the line first (the pullback/entry
    use-case). Missing → '—' with data-sort 9999 (parks last on ascending)."""
    d = d or {}
    cells = []
    for p in (10, 20, 50):
        v = d.get(p)
        if v is None:
            cells.append("<td class='num ma-cell' data-sort='9999'>—</td>")
        else:
            ab = abs(v)
            arrow = "▲" if v >= 0 else "▼"
            col = "var(--green)" if v >= 0 else "var(--red)"
            cells.append(
                f"<td class='num ma-cell' data-sort='{ab:.4f}'>{ab:.1f}"
                f"<span style='color:{col};font-size:var(--fs-micro);'>{arrow}</span></td>")
    return "".join(cells)


def _fwd_yoy_cell(ticker: str) -> str:
    """Sortable <td> — the FIRST forward (estimate) revenue quarter's YoY growth, read
    from the fundamentals cache (warmed by _prefetch_fundamentals). data-sort = the %
    so clicking descending puts the fastest-growing names first; missing → '—' with
    data-sort −999 (parks last on descending). Never raises."""
    y, lbl = None, ""
    if _fund is not None:
        try:
            rec = _fund.get(ticker)
            if rec:
                for r in rec.get("rev", []):
                    if r.get("est") and r.get("yoy") is not None:
                        y, lbl = r["yoy"], r.get("lbl", "")
                        break
        except Exception:
            y = None
    if y is None:
        return "<td class='num fy-cell' data-sort='-999'>—</td>"
    pct = y * 100.0
    col = "var(--green)" if pct > 0.5 else ("var(--red)" if pct < -0.5 else "var(--text-3)")
    sign = "+" if pct >= 0 else ""
    sub = (f"<br><span style='font-size:var(--fs-micro);color:var(--text-3);'>{esc(lbl)}</span>"
           if lbl else "")
    return (f"<td class='num fy-cell' data-sort='{pct:.2f}'>"
            f"<span style='color:{col};font-weight:600;'>{sign}{pct:.0f}%</span>{sub}</td>")


def _eps_accel_cell(ticker: str) -> str:
    """Sortable <td> — O'Neil earnings ACCELERATION: the TREND in quarterly EPS YoY growth
    (a rising growth RATE = accelerating, the CANSLIM 'C' refined). Headline = an
    acceleration arrow + the latest reported quarter's EPS YoY%; sub-line = the recent YoY
    path (+ a ✦TTM mark when trailing-12-month EPS is at a new high). data-sort = the change
    in YoY rate in pp, so clicking descending puts the fastest accelerators first; missing →
    '—' parked last (data-sort −9999). Reads the fundamentals cache; never raises."""
    a = None
    if _fund is not None:
        try:
            rec = _fund.get(ticker)
            a = rec.get("eps_accel") if rec else None
        except Exception:
            a = None
    score = a.get("accel_score") if a else None
    verdict = a.get("verdict") if a else None
    if score is None or verdict is None:
        return "<td class='num accel-cell' data-sort='-9999'>—</td>"
    if verdict == "accel":
        arrow, acol = ("▲▲" if score >= 20 else "▲"), "var(--green)"
    elif verdict == "decel":
        arrow, acol = ("▼▼" if score <= -20 else "▼"), "var(--red)"
    else:
        arrow, acol = "→", "var(--text-3)"
    qs = [q for q in (a.get("quarters") or []) if q.get("yoy") is not None]
    latest = qs[-1]["yoy"] * 100.0 if qs else None
    if latest is None:
        ynum = ""
    else:
        ycol = "var(--green)" if latest > 0.5 else ("var(--red)" if latest < -0.5 else "var(--text-3)")
        ynum = (f" <span style='color:{ycol};font-weight:600;'>"
                f"{'+' if latest >= 0 else ''}{latest:.0f}%</span>")
    path = "→".join(f"{q['yoy'] * 100:.0f}" for q in qs[-3:]) if qs else ""
    ttm = " <span style='color:var(--green);'>✦TTM</span>" if a.get("ttm_new_high") else ""
    if path:
        sub = (f"<br><span style='font-size:var(--fs-micro);color:var(--text-3);'>"
               f"{esc(path)}%{ttm}</span>")
    elif ttm:
        sub = f"<br><span style='font-size:var(--fs-micro);'>{ttm}</span>"
    else:
        sub = ""
    return (f"<td class='num accel-cell' data-sort='{score:.2f}'>"
            f"<span style='color:{acol};font-weight:700;'>{arrow}</span>{ynum}{sub}</td>")


# The FIVE sortable/reorderable column headers shared by every main stock table. The body
# cells are produced by _ma_cells(row['_ma_dist']) + _fwd_yoy_cell(ticker) + _eps_accel_cell(
# ticker), inserted at the SAME position so column N's header and cells move together on
# reorder. Adding/removing one here REQUIRES the matching body cell at all four sites.
_MA_YOY_HEADERS = (
    "<th class='num' data-col='d10' title='Distance from the 10-day SMA (|price−SMA|/SMA). Click ascending = tightest to the line first'>vs 10MA</th>"
    "<th class='num' data-col='d20' title='Distance from the 20-day SMA. Click ascending = tightest first'>vs 20MA</th>"
    "<th class='num' data-col='d50' title='Distance from the 50-day SMA. Click ascending = tightest first'>vs 50MA</th>"
    "<th class='num' data-col='fyoy' title='Next forward-quarter revenue YoY (consensus). Click descending = fastest-growing first'>Fwd YoY</th>"
    "<th class='num' data-col='accel' title='Earnings ACCELERATION (O&#39;Neil / CANSLIM C): the trend in quarterly EPS YoY growth &mdash; &#9650;&#9650;/&#9650; the growth RATE is rising, &#8594; steady, &#9660; slowing. Headline = latest reported-quarter EPS YoY%; &#10022;TTM = trailing-12-month EPS at a new high. Click descending = fastest accelerators first'>EPS Accel</th>"
)


def _ranking_meta_score(hist_df, legacy_score, market_modifier):
    """v4 enhanced score if available — raw 0-100 percentile of P(+2ADR win); it is a
    calibrated probability rank, so NOT regime-dampened. Falls back to the legacy
    M.E.T.A. score (×market_modifier, capped) when the v4 model is unavailable."""
    v4 = _meta_v4_score(hist_df)
    if v4 is not None:
        return v4
    return round(min(legacy_score * market_modifier, 100.0), 1)


def _meta_award(details: List[str], comp: str, frac: float, label: str,
                badges: Optional[List[str]] = None, badge: Optional[str] = None) -> int:
    """Award round(weight*frac) points for a component, append its detail line,
    and (optionally) a badge. Returns the points so the caller can accumulate."""
    mx = META_WEIGHTS.get(comp, 0)
    pts = round(mx * frac)
    details.append(f"{label} ({pts} penalty)" if frac < 0 else f"{label} ({pts}/{mx})")
    if badge and badges is not None:
        badges.append(badge)
    return pts
NH52_HISTORY_PATH = os.path.join(WORKSPACE, "nh52_history.json")
NH52_WATCH_DAYS = 20          # trading days a 52wk-high name stays on the daily monitor
STOCKBEE_SHEET_URL = ("https://docs.google.com/spreadsheet/pub?"
                      "key=0Am_cU8NLIU20dEhiQnVHN3Nnc3B1S3J6eGhKZFo0N3c&output=csv")
VIX_SYMBOL = "%5EVIX"
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "SMH"]
LEAD_SECTORS = {"XLK", "XLC", "SMH", "XLY"}   # the offense/leadership groups
# ---- HTF v2.4.1 (High Tight Flag) — owner-audited parameters (2026-06-12) ----
# universe gate (point-in-time, all must hold on the signal bar)
HTF_CAP_MIN = 3e9             # market cap > $3B
HTF_PRICE_MIN = 10.0          # close > $10
HTF_VOL_MIN = 5e5             # day volume AND 20/30/60/90-bar avg vol EACH > 500k
HTF_ADR_MIN = 4.0             # ADR(20) > 4%   [audited: 4.5 collapses — never raise]
HTF_OFF_HI_MAX = 25.0         # within 25% of the 52-week high
# entry (all seven gates)
HTF_THRUST_MIN = 0.60         # C/C[-40] − 1 > 60%
HTF_FLAG_BARS = 5             # flag = last 5 bars (signal bar included)
HTF_TIGHT_ADR_X = 3.0         # flag range < 3.0 × ADR20   (volatility-scaled)
HTF_FLAG_DEPTH_MAX = 0.30     # flag keeps < 30% of the pole  (load-bearing)
HTF_QAVG_ADR_X = 1.0          # avg flag-bar range ≤ 1.0 × ADR20   [never tighten]
HTF_QMAX_ADR_X = 1.25         # max flag-bar range ≤ 1.25 × ADR20  [never tighten]
HTF_RS_MIN = 85               # IBD-style RS percentile, full cross-section
HTF_COOLDOWN = 5              # trading bars between signals per name
# execution + exits
HTF_ADR_STOP_MULT = 1.5       # hard stop = close × (1 − 1.5 × ADR20); gap-through fills at open
HTF_EXT_X = 0.40              # blow-off sell: first close ≥ 40% above the 21-EMA
HTF_MAX_LEGS = 3              # ladder re-buy at the trade's peak after ANY exit
HTF_EQUITY = 100_000          # reference account size for the staged-ticket sizing
HTF_RISK_FRAC = 0.0075        # 0.75% account risk per HTF trade
HTF_MAX_POS_FRAC = 0.20       # cap position at 20% of equity
# New-52wk-high persistence: distinct weeks (of last 13) printing a fresh high.
NH_PERSIST_WEEKS = 5          # ⭐ Persistent Leader
NH_RELENTLESS_WEEKS = 9       # ⭐⭐ Relentless Leader
# ---- ANTS (David Ryan accumulation: up-days + volume + price + RS) — display + Top-Picks boost ----
# Per-stock institutional-accumulation read over a short window, graded 0-5 +
# a trailing consecutive-bar "chain". Classic David Ryan defaults. Decision-
# support only; does NOT affect the IBKR draft plan. ANTS rs_line (close/SPY vs
# its own MA) is DISTINCT from the Fred6725 RS percentile (resolve_rs).
ANTS_LOOKBACK = 15            # window for up-count / vol-gain / price-gain
ANTS_MIN_UP = 12             # >= this many up-days in the window => momentum_ok
ANTS_PRICE_PCT = 0.20        # close up >= 20% over the window
ANTS_VOL_PCT = 0.20          # avg volume up >= 20% vs the prior window
ANTS_USE_TREND = True        # require SMA10 > SMA20 for the price leg
ANTS_USE_RS = True           # enable the ELITE upgrade (rs_line rising vs SPY)
ANTS_COUNT_FULL_ONLY = False # chain counts any level>0 (True = FULL+ only)
ANTS_RS_FAST = 20            # is_rs_rising: rs_line > SMA(rs_line, 20)
ANTS_RS_SLOW = 50            # isStronger (info): rs_line > SMA(rs_line, 50)
ANTS_CHAIN_WINDOW = 60       # bars scanned for the trailing chain run
ANTS_BENCHMARK = "SPY"
ANTS_RS_HIGH_FRAC = 0.97     # RS line within 3% of its 1y high => relative-strength LEADER (standout)
ANTS_PX_LAG_FRAC = 0.95      # ...with price below 95% of its 1y high => stealth (RS leads price)
_ANTS_LABELS = {0: "NONE", 1: "MOM", 2: "MOM+VOL", 3: "MOM+PR", 4: "FULL", 5: "ELITE"}
_ANTS_EMPTY = {"level": 0, "chain": 0, "label": "NONE", "up_count": 0, "vol_gain": 0.0,
               "price_gain": 0.0, "rs_rising": False, "stronger": False, "ok": False,
               "ants_3m_peak": 0, "ants_3m_days": 0, "rs_line_rising": False,
               "rs_new_high": False, "rs_nh_before_price": False, "rs_nh_3m": False,
               "rs_spark_vals": []}
# ---- IBKR draft-order plan (TOP PICKS -> reviewable drafts; NEVER transmitted) ----
IBKR_ORDER_PLAN_PATH = os.path.join(WORKSPACE, "top_picks_orders.json")
IBKR_TOP_N = 3                # draft only the 3 strongest TOP PICKS
IBKR_RISK_FRAC = 0.005        # 0.5% of equity risked per draft (entry->stop distance)
IBKR_MAX_POS_FRAC = 0.10      # cap any ONE draft at 10% of equity (audit: 0.20 too high)
IBKR_MAX_SESSION_FRAC = 0.35  # cap TOTAL deployment across all drafts at 35% of equity
IBKR_MIN_PRICE = 1.00         # never draft sub-$1 names
IBKR_MIN_RPS_PCT = 0.005      # risk/share must be >= 0.5% of entry (else 1-cent-stop junk)
IBKR_ONE_PER_SECTOR = True    # at most one draft per sector
# GATE DROPPED 2026-06-15 (owner): the regime gate had been tuned on a breadth
# PROXY, not the live build_regime, and the true regime can't be replayed
# historically (T2108/Barchart/win-rate feeds don't exist as-of). So: ALWAYS FULL
# SIZE, draft in EVERY regime; the regime is RECORDED (not acted on) and top picks
# are forward-logged daily → reviewed weekly (Sat), then monthly → to re-tune a
# gate on the REAL regime once enough data accrues. Kept for reference only:
REGIME_SIZE_MULT = {"GREEN": 1.0, "YELLOW": 0.5}   # NOT used for gating anymore
# Persistent forward-tracking logs for the gate re-tune study (append-only, 1 row/session):
REGIME_HISTORY_PATH = os.path.join(WORKSPACE, "regime_history.jsonl")
TOP_PICKS_HISTORY_PATH = os.path.join(WORKSPACE, "top_picks_history.jsonl")
# Generate timestamped filenames to avoid overwriting previous reports on same day
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
HTML_REPORT_PATH = os.path.join(WORKSPACE, f"madrry_report_{TIMESTAMP}.html")
MD_REPORT_PATH = os.path.join(WORKSPACE, f"madrry_report_{TIMESTAMP}.md")
RS_CACHE_PATH = os.path.join(WORKSPACE, "memory", "rs_stocks_cache.csv")
RS_HISTORY_PATH = os.path.join(WORKSPACE, "memory", "rs_history.json")
RS_INDUSTRIES_URL = "https://raw.githubusercontent.com/Fred6725/rs-log/main/output/rs_industries.csv"
RS_INDUSTRIES_CACHE = os.path.join(WORKSPACE, "memory", "rs_industries_cache.csv")
IND_RS_STRONG = 90            # industry-group RS percentile that counts as "leadership"

TV_SCAN_URL = "https://scanner.tradingview.com/america/scan"
# Use the FULL ranked universe (rs_stocks.csv ≈ 5.9k names, percentile 0-99).
# The paginated rs_stocks_1.csv is only the TOP half (percentile 50-99), so every
# name in the bottom half wrongly fell through to the "< 50"/N/A default.
RS_CSV_URL = "https://raw.githubusercontent.com/Fred6725/rs-log/main/output/rs_stocks.csv"
HEADERS = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
HTTP_TIMEOUT = 30

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("madrry")


# ----------------------------------------------------------------------------
# DIAGNOSTICS + small infrastructure helpers
# ----------------------------------------------------------------------------
@dataclass
class Diagnostics:
    """Collects non-fatal problems + timings so they can be shown in the report."""
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    def error(self, msg: str) -> None:
        log.error(msg)
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.warnings.append(msg)


@contextmanager
def timed(diag: Diagnostics, name: str):
    start = time.time()
    try:
        yield
    finally:
        diag.timings[name] = time.time() - start
        log.info("%s took %.1fs", name, diag.timings[name])


def _request_json(
    url: str,
    *,
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = HTTP_TIMEOUT,
    retries: int = 3,
    backoff: float = 1.5,
    label: str = "request",
    diag: Optional[Diagnostics] = None,
) -> Any:
    """GET/POST JSON with exponential backoff. Raises on final failure."""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - network is intentionally broad
            last_err = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    if diag is not None:
        diag.error(f"{label}: failed after {retries} attempts ({last_err})")
    raise last_err if last_err else RuntimeError(label)


def tv_post(payload: dict, *, label: str = "tv_post", diag: Optional[Diagnostics] = None) -> Any:
    """POST a scanner payload to TradingView and return parsed JSON (with retries)."""
    return _request_json(
        TV_SCAN_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=HEADERS,
        label=label,
        diag=diag,
    )


def _atomic_write(path: str, text: str) -> None:
    """Write then rename, so a crash never leaves a truncated file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def esc(value: Any) -> str:
    """HTML-escape any dynamic text (themes/sectors can contain '&', '<', etc.)."""
    return html_lib.escape(str(value), quote=True)


_LEAD_EMOJI_RE = re.compile(r"^[^\w$+\-(]+\s*")


def _strip_lead_emoji(s: Any) -> str:
    """Drop a leading emoji/pictograph so flags read as plain tokens."""
    return _LEAD_EMOJI_RE.sub("", str(s))


def fetch_tv_last_bars(tickers: List[str]) -> Dict[str, dict]:
    """Latest completed/current session OHLCV per ticker straight from the
    TradingView scan API (the same vendor the scans use — it posts EOD data
    hours before Yahoo finishes consolidating). One POST per ~400 tickers."""
    out: Dict[str, dict] = {}
    for i in range(0, len(tickers), 400):
        chunk = tickers[i:i + 400]
        try:
            data = tv_post({
                "filter": [{"left": "name", "operation": "in_range", "right": chunk}],
                "columns": ["name", "open", "high", "low", "close", "volume"],
                "range": [0, len(chunk)],
            }, label="tv_last_bars")
            for r in data.get("data", []):
                d = r.get("d")
                if d and d[4] is not None:
                    # normalize class/preferred tickers (BRK.B vs BRK-B) so the
                    # patch lookup keyed elsewhere always hits (audit).
                    sym = str(d[0]).upper().replace(".", "-")
                    out[sym] = {"open": d[1], "high": d[2], "low": d[3],
                                "close": d[4], "volume": d[5]}
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch_tv_last_bars chunk failed: %s", exc)
    return out


def _patch_stale_histories(hmap: Dict[str, pd.DataFrame], label: str = "batch") -> int:
    """If Yahoo's daily bars end before the last completed US session (its bulk
    feed can lag/roll back during EOD consolidation), append the missing bar
    from TradingView so every consumer (HTF, persistence, grading, coil flags)
    works on CURRENT data instead of yesterday's. Returns #tickers patched."""
    exp = _expected_session_date()
    if not exp or not hmap:
        return 0
    stale = []
    for t, df in hmap.items():
        try:
            if df.index[-1].date().isoformat() < exp:
                stale.append(t)
        except Exception:  # noqa: BLE001
            continue
    if not stale:
        return 0
    bars = fetch_tv_last_bars(stale)
    patched = 0
    for t in stale:
        b = bars.get(t.upper().replace(".", "-"))   # match the normalized key (audit)
        df = hmap[t]
        if not b:
            continue
        last = df.iloc[-1]
        try:
            # identical close+volume => same bar (e.g. US holiday) — don't duplicate.
            # Use explicit None checks so a genuine 0-volume bar isn't masked (audit).
            _lv = last.get("Volume")
            _bv = b.get("volume")
            same_vol = (_lv is not None and _bv is not None and float(_lv) == float(_bv))
            if float(last["Close"]) == float(b["close"]) and same_vol:
                continue
            row = {}
            for col in df.columns:
                if col == "Open":
                    row[col] = b.get("open") if b.get("open") is not None else b["close"]
                elif col == "High":
                    row[col] = b.get("high") if b.get("high") is not None else b["close"]
                elif col == "Low":
                    row[col] = b.get("low") if b.get("low") is not None else b["close"]
                elif col == "Volume":
                    row[col] = b["volume"] if b.get("volume") is not None else 0
                else:                       # Close / Adj Close / anything else
                    row[col] = b["close"]
            # tz-MATCH the appended bar to the frame's index, else pd.concat
            # collapses the DatetimeIndex to object dtype and the downstream
            # freshness guards raise (tz-naive vs tz-aware) and silently null out
            # — which would BYPASS the anti-phantom-fire guards (audit C1).
            new_ts = pd.Timestamp(exp)
            if getattr(df.index, "tz", None) is not None and new_ts.tz is None:
                new_ts = new_ts.tz_localize(df.index.tz)
            hmap[t] = pd.concat([df, pd.DataFrame([row], index=[new_ts])])
            patched += 1
        except Exception:  # noqa: BLE001
            continue
    if patched:
        log.info("%s history lagged %s — patched %d/%d tickers with TradingView EOD bars",
                 label, exp, patched, len(stale))
    return patched


def _expected_session_date() -> Optional[str]:
    """Most recent US session whose 16:00 ET close has passed (weekends skipped;
    market holidays will look one day 'ahead' — treated as a soft warning only)."""
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:  # noqa: BLE001
        return None
    d = now_et.date()
    if now_et.hour < 16:
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.isoformat()


# Optional keyless price proxy (a Cloudflare Worker URL that relays Yahoo quotes
# with CORS). Leave "" to use the zero-setup public-proxy fallback. Set this once
# and every generated report's "🔄 Refresh Prices" button uses it.
LIVE_PRICE_PROXY = ""


def _lp(ticker: Any, close: Any, *, style: str = "",
        entry: Any = None, stop: Any = None, fmt: str = "{:.2f}") -> str:
    """A live-price <span>: tagged so the Refresh-Prices button can update it in
    place. data-snap holds the scan-time price (for up/down colouring)."""
    extra = ""
    if entry is not None:
        extra += f" data-entry='{entry}'"
    if stop is not None:
        extra += f" data-stop='{stop}'"
    return (f"<span class='lp' data-tkr='{esc(str(ticker))}' data-snap='{close}'{extra} "
            f"style='{style}'>${fmt.format(close)}</span>")


# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
def fetch_and_load_rs_scores(diag: Optional[Diagnostics] = None) -> Dict[str, Any]:
    """Download the latest RS strong stocks and return a ticker->percentile map."""
    log.info("Downloading latest RS scores...")
    try:
        content = None
        last_err = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(RS_CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read().decode("utf-8")
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                if attempt < 2:                      # don't sleep after the final attempt (audit)
                    time.sleep(1.5 * (attempt + 1))
        if content is not None:
            os.makedirs(os.path.dirname(RS_CACHE_PATH), exist_ok=True)
            _atomic_write(RS_CACHE_PATH, content)
            log.info("RS scores cached.")
        else:
            raise last_err if last_err else RuntimeError("no content")
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"RS download failed ({exc}); using local cache if present.")

    rs_map: Dict[str, Any] = {}
    if os.path.exists(RS_CACHE_PATH):
        try:
            with open(RS_CACHE_PATH, "r", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    ticker = row.get("Ticker", "").strip().upper()
                    percentile = row.get("Percentile", "").strip()
                    if ticker and percentile:
                        try:
                            rs_map[ticker] = int(percentile)
                        except ValueError:
                            rs_map[ticker] = percentile
        except Exception as exc:  # noqa: BLE001
            if diag:
                diag.warn(f"Error parsing RS cache: {exc}")
    _update_rs_history(rs_map)
    return rs_map


# Last-known RS percentile per ticker, so names that flicker out of the source
# (ADRs / OTC / recent spinoffs) keep their most recent score instead of N/A.
_RS_HISTORY: Dict[str, dict] = {}


def _update_rs_history(rs_map: Dict[str, Any]) -> Dict[str, dict]:
    """Persist today's RS percentiles; merge into the rolling history file."""
    global _RS_HISTORY
    hist: Dict[str, dict] = {}
    if os.path.exists(RS_HISTORY_PATH):
        try:
            with open(RS_HISTORY_PATH) as fh:
                hist = json.load(fh)
        except Exception:  # noqa: BLE001
            hist = {}
    today = _expected_session_date() or date.today().isoformat()   # ET session, not local (audit)
    for tk, p in rs_map.items():
        if isinstance(p, int):
            hist[tk] = {"rs": p, "asof": today}
    try:
        os.makedirs(os.path.dirname(RS_HISTORY_PATH), exist_ok=True)
        _atomic_write(RS_HISTORY_PATH, json.dumps(hist))
    except Exception:  # noqa: BLE001
        pass
    _RS_HISTORY = hist
    return hist


def resolve_rs(ticker: str, rs_map: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """Resolve a ticker's RS percentile. Returns (value, stale_asof):
    fresh today's value → (int, None); carried-forward → (int, 'YYYY-MM-DD');
    never seen → ('N/A', None)."""
    tk = (ticker or "").upper()
    v = rs_map.get(tk)
    if isinstance(v, int):
        return v, None
    h = _RS_HISTORY.get(tk)
    if h and isinstance(h.get("rs"), int):
        return h["rs"], h.get("asof")
    return "N/A", None


def fetch_and_load_industry_rs(diag: Optional[Diagnostics] = None) -> dict:
    """Download Fred6725 rs_industries.csv (144 industry groups ranked by RS) →
    {rows, by_ticker}. by_ticker maps each constituent ticker to its industry
    group's RS percentile — J Law's "leadership is an industry-group phenomenon"
    read, more granular than the 12 GICS sectors. Daily-fresh, cache fallback."""
    content = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(RS_INDUSTRIES_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                content = resp.read().decode("utf-8")
            break
        except Exception as exc:  # noqa: BLE001
            if diag:
                diag.warn(f"industry RS download attempt {attempt + 1}: {exc}")
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    if content is not None:
        try:
            os.makedirs(os.path.dirname(RS_INDUSTRIES_CACHE), exist_ok=True)
            _atomic_write(RS_INDUSTRIES_CACHE, content)
        except Exception:  # noqa: BLE001
            pass
    elif os.path.exists(RS_INDUSTRIES_CACHE):
        try:
            content = open(RS_INDUSTRIES_CACHE, encoding="utf-8").read()
        except Exception:  # noqa: BLE001
            content = None
    rows: List[dict] = []
    by_ticker: Dict[str, dict] = {}
    if content:
        try:
            for row in csv.DictReader(content.splitlines()):
                try:
                    pct = int(row["Percentile"])
                except (ValueError, KeyError, TypeError):
                    continue
                rec = {
                    "rank": int(row.get("Rank") or 0),
                    "industry": (row.get("Industry") or "").strip(),
                    "sector": (row.get("Sector") or "").strip(),
                    "pct": pct,
                    "p1m": int(row.get("1M_RS_Percentile") or 0),
                    "p3m": int(row.get("3M_RS_Percentile") or 0),
                }
                rows.append(rec)
                for t in (row.get("Tickers") or "").split(","):
                    t = t.strip().upper().replace(".", "-")
                    if t:
                        by_ticker[t] = rec
        except Exception as exc:  # noqa: BLE001
            if diag:
                diag.warn(f"industry RS parse failed: {exc}")
    rows.sort(key=lambda r: r["pct"], reverse=True)
    log.info("Industry RS: %d groups, %d tickers mapped", len(rows), len(by_ticker))
    return {"rows": rows, "by_ticker": by_ticker}


def attach_industry_rs(stocks: List[dict], by_ticker: Dict[str, dict]) -> None:
    """Tag each pick with its industry-group RS percentile + name (display-only;
    never touches the IBKR draft plan)."""
    for s in stocks:
        rec = by_ticker.get((s.get("ticker") or "").upper().replace(".", "-"))
        s["ind_rs"] = rec["pct"] if rec else None
        s["ind_name"] = rec["industry"] if rec else None


def fetch_stock_history(ticker: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Single-ticker history fallback (the coil scan uses the batch fetch)."""
    try:
        hist = yf.Ticker(ticker).history(period=period, interval=interval)
        if hist.empty or len(hist) < 200:
            return None
        return hist
    except Exception as exc:  # noqa: BLE001
        log.debug("history fetch failed for %s: %s", ticker, exc)
        return None


def fetch_histories_batch(tickers: List[str], period: str = "1y", min_rows: int = 200) -> Dict[str, pd.DataFrame]:
    """Batch-download daily history for many tickers in a SINGLE yfinance call."""
    if not tickers:
        return {}
    try:
        raw = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Batch history download failed: %s", exc)
        return {}

    if raw is None or len(raw) == 0:
        return {}

    out: Dict[str, pd.DataFrame] = {}
    multi = isinstance(raw.columns, pd.MultiIndex)
    for t in tickers:
        try:
            df = raw[t] if multi else raw
            df = df.dropna()
            if len(df) >= min_rows:
                out[t] = df
        except Exception:  # noqa: BLE001
            continue
    # Yahoo's bulk feed can lag (or roll back) a session vs the actual close —
    # fill any missing latest bar from TradingView so consumers stay current.
    _patch_stale_histories(out, label=f"batch({period})")
    return out


def _fetch_one_index(ticker: str) -> Optional[dict]:
    """Fetch + compute one market-health card (used in a thread pool)."""
    # 14mo (not 3mo) so we carry ≥200 bars for the SMA200 bull/bear regime that
    # the forward-base-rate lookup conditions on. The 10/20/50-SMA, dist-days and
    # asof/TV-patch logic all read only the tail, so the longer range is inert to them.
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=1d&range=14mo"
    data = _request_json(url, headers={"User-Agent": "Mozilla/5.0"}, label=f"market:{ticker}", retries=3)
    result = data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    vols = result["indicators"]["quote"][0]["volume"]

    valid = [(c, v) for c, v in zip(closes, vols) if c is not None and v is not None]
    if len(valid) < 50:
        return None

    # Trading date of the last daily bar (the scan's DATA vintage — distinct from
    # the wall-clock run time; a 6 AM pre-market run still carries yesterday's bar).
    asof = None
    try:
        ts = [t for t, c2 in zip(result.get("timestamp", []), closes) if c2 is not None]
        if ts:
            # Interpret the bar timestamp in ET (the market's tz), not the host's
            # local tz — otherwise the data-date is off by a day west of ET and
            # spuriously triggers the TV-append path (audit).
            try:
                from zoneinfo import ZoneInfo
                asof = datetime.fromtimestamp(ts[-1], ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            except Exception:  # noqa: BLE001
                asof = datetime.fromtimestamp(ts[-1]).strftime("%Y-%m-%d")
    except Exception:  # noqa: BLE001
        asof = None

    # If even the chart feed lags the last completed session, append the missing
    # bar from TradingView so the cards/data-date reflect the REAL latest close.
    exp = _expected_session_date()
    if exp and asof and asof < exp:
        b = fetch_tv_last_bars([ticker]).get(ticker.upper())
        if b and b.get("close") is not None and float(valid[-1][0]) != float(b["close"]):
            valid.append((float(b["close"]), float(b.get("volume") or 0)))
            asof = exp
            log.info("market:%s chart lagged — appended %s bar from TradingView", ticker, exp)

    c = valid[-1][0]
    prev_c = valid[-2][0] if len(valid) >= 2 else c
    change_pt = c - prev_c
    change_pct = (change_pt / prev_c * 100) if prev_c else 0.0
    sma10 = sum(x[0] for x in valid[-10:]) / 10
    sma20 = sum(x[0] for x in valid[-20:]) / 20
    sma21 = sum(x[0] for x in valid[-21:]) / 21
    sma50 = sum(x[0] for x in valid[-50:]) / 50
    # SMA200 + bull/bear regime for the forward-base-rate lookup (None if <200 bars).
    sma200 = (sum(x[0] for x in valid[-200:]) / 200) if len(valid) >= 200 else None
    above_200 = (c > sma200) if sma200 is not None else None
    ext_10 = ((c - sma10) / sma10) * 100
    ext_20 = ((c - sma20) / sma20) * 100
    ext_50 = ((c - sma50) / sma50) * 100
    spark = make_price_spark([x[0] for x in valid], 40)

    # Recent consolidation range (≈20 sessions, excluding the last 3) for the
    # topping-breakdown signal.
    win_r = [x[0] for x in valid[-23:-3]] if len(valid) >= 23 else [x[0] for x in valid[:-1]]
    rng_hi = max(win_r) if win_r else c
    rng_lo = min(win_r) if win_r else c
    rng_span = (rng_hi - rng_lo) or 1.0
    close_below_range = c < rng_lo
    range_pos = (c - rng_lo) / rng_span

    recent = valid[-20:]
    dist_days = 0
    for i in range(1, len(recent)):
        prev_c, prev_v = recent[i - 1]
        curr_c, curr_v = recent[i]
        if not prev_c or curr_v <= 0 or prev_v <= 0:   # a TV-patched bar can carry vol=0;
            continue                                   # skip rather than mis-judge volume (audit)
        pct_drop = (prev_c - curr_c) / prev_c * 100
        if pct_drop > 0.2 and curr_v > prev_v:
            dist_days += 1

    return {
        "ticker": ticker,
        "close": c,
        "asof": asof,
        "change_pt": change_pt,
        "change_pct": change_pct,
        "sma10": sma10,
        "sma21": sma21,
        "trend": "GREEN" if sma10 > sma21 else "RED",
        "ext_10": ext_10,
        "ext_20": ext_20,
        "ext_50": ext_50,
        "sma200": sma200,
        "above_200": above_200,
        "dist_days": dist_days,
        "spark": spark,
        "range_low": rng_lo,
        "close_below_range": close_below_range,
        "range_pos": range_pos,
    }


def fetch_sp_breadth(diag: Optional[Diagnostics] = None) -> dict:
    """S&P 500 breadth straight from Barchart (TradingView's EOD source for these
    indices): % of members above the 20/50/200-day MA = $S5TW / $S5FI / $S5TH,
    with the day-over-day percentage-point change. Returns {'ok': False} on failure."""
    import http.cookiejar
    import urllib.parse as _up
    try:
        cj = http.cookiejar.CookieJar()
        op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        op.addheaders = [("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"),
                         ("Accept", "text/html")]
        op.open("https://www.barchart.com/stocks/quotes/$S5FI/overview", timeout=15).read()
        xsrf = next((_up.unquote(c.value) for c in cj if c.name == "XSRF-TOKEN"), "")
        url = ("https://www.barchart.com/proxies/core-api/v1/quotes/get?"
               "symbols=$S5TW,$S5FI,$S5TH&fields=symbol,lastPrice,priceChange,tradeTime")
        req = urllib.request.Request(url, headers={
            "x-xsrf-token": xsrf, "Accept": "application/json",
            "User-Agent": "Mozilla/5.0", "Referer": "https://www.barchart.com/"})
        data = json.loads(op.open(req, timeout=15).read())
        m = {r["symbol"]: r for r in data.get("data", [])}

        def vc(sym):
            r = m[sym]
            return float(r["lastPrice"]), float(str(r["priceChange"]).replace("+", ""))

        a20, c20 = vc("$S5TW")
        a50, c50 = vc("$S5FI")
        a200, c200 = vc("$S5TH")
        return {"ok": True, "above20": a20, "above50": a50, "above200": a200,
                "chg20": c20, "chg50": c50, "chg200": c200,
                "asof": m.get("$S5FI", {}).get("tradeTime", "")}
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"S&P breadth (Barchart) fetch failed: {exc}")
        return {"ok": False, "above50": 50.0, "above200": 50.0}


def fetch_market_health(diag: Optional[Diagnostics] = None) -> Tuple[List[dict], dict]:
    """Index health (parallel Yahoo) + S&P-500 breadth (Barchart)."""
    tickers = ["QQQ", "SPY", "IWM"]
    market_data: List[dict] = []

    # The 4 Yahoo probes are independent -> fetch concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_fetch_one_index, t): t for t in tickers}
        for fut in concurrent.futures.as_completed(futures):
            t = futures[fut]
            try:
                card = fut.result()
                if card:
                    market_data.append(card)
            except Exception as exc:  # noqa: BLE001
                if diag:
                    diag.warn(f"Market health fetch error for {t}: {exc}")
    # Keep a stable display order regardless of completion order.
    order = {t: i for i, t in enumerate(tickers)}
    market_data.sort(key=lambda m: order.get(m["ticker"], 99))

    breadth = fetch_sp_breadth(diag)
    return market_data, breadth


def fetch_vix(diag: Optional[Diagnostics] = None) -> Optional[dict]:
    """VIX level + 1-day % change (info-only — VIX is coincident, not leading)."""
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{VIX_SYMBOL}?interval=1d&range=10d"
        data = _request_json(url, headers={"User-Agent": "Mozilla/5.0"}, label="vix", retries=2, timeout=12)
        cl = [c for c in data["chart"]["result"][0]["indicators"]["quote"][0]["close"] if c is not None]
        if len(cl) < 2:
            return None
        last, prev = cl[-1], cl[-2]
        return {"level": round(last, 2), "chg_pct": round((last - prev) / prev * 100, 1) if prev else 0.0}
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"VIX fetch failed: {exc}")
        return None


def fetch_t2108(diag: Optional[Diagnostics] = None) -> dict:
    """Real T2108 (% of stocks > 40-DMA) from the Stockbee Market Monitor's
    published Google Sheet CSV. T2108 is the 2nd-to-last column, S&P price the
    last; rows are most-recent-first → also yields the day change + divergence."""
    import csv as _csv
    import io as _io
    import re as _re
    try:
        req = urllib.request.Request(STOCKBEE_SHEET_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", "ignore")
        rows = list(_csv.reader(_io.StringIO(raw)))
        t_idx = spx_idx = None
        for r in rows:
            for i, cell in enumerate(r):
                cs = cell.strip()
                if cs == "T2108":
                    t_idx = i
                if cs in ("S&P", "S&P 500", "SP"):
                    spx_idx = i
            if t_idx is not None:
                break
        if t_idx is None:
            raise ValueError("T2108 column not found")
        if spx_idx is None:
            spx_idx = -1
        datere = _re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
        recs = []
        for r in rows:
            if r and datere.match(r[0].strip()):
                try:
                    t = float(r[t_idx])
                    s = float(r[spx_idx].replace(",", "")) if r[spx_idx] else None
                    recs.append((r[0].strip(), t, s))
                except (ValueError, IndexError):
                    continue
                if len(recs) >= 2:
                    break
        if not recs:
            raise ValueError("no T2108 data rows")
        cur = recs[0]
        prev = recs[1] if len(recs) > 1 else None
        out = {"ok": True, "t2108": round(cur[1], 1), "asof": cur[0], "spx": cur[2]}
        if prev:
            out["chg"] = round(cur[1] - prev[1], 1)
            out["divergence"] = bool(cur[2] and prev[2] and cur[2] > prev[2] and cur[1] < prev[1])
        else:
            out["chg"] = None
            out["divergence"] = False
        return out
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"T2108 (Stockbee sheet) fetch failed: {exc}")
        return {"ok": False}


def fetch_sector_rs(diag: Optional[Diagnostics] = None) -> Optional[dict]:
    """Per-sector RS vs SPY (1-month relative perf) + short-term momentum (price
    vs its own 50-DMA). Flags how many leadership sectors are rolling over."""
    try:
        raw = yf.download(tickers=SECTOR_ETFS + ["SPY"], period="3mo", interval="1d",
                          group_by="ticker", auto_adjust=False, threads=True, progress=False)
        if raw is None or len(raw) == 0:
            return None
        multi = isinstance(raw.columns, pd.MultiIndex)

        def closes(t):
            df = (raw[t] if multi else raw).dropna()
            return df["Close"] if len(df) >= 50 else None

        spy = closes("SPY")
        if spy is None:
            return None
        spy_1m = spy.iloc[-1] / spy.iloc[-21] - 1.0
        sectors = []
        for t in SECTOR_ETFS:
            c = closes(t)
            if c is None:
                continue
            rs = (c.iloc[-1] / c.iloc[-21] - 1.0) - spy_1m
            below50 = bool(c.iloc[-1] < c.iloc[-50:].mean())
            sectors.append({"etf": t, "rs": round(rs * 100, 1), "below50": below50,
                            "lead": t in LEAD_SECTORS})
        if not sectors:
            return None
        leaders = [s for s in sectors if s["lead"]]
        return {
            "sectors": sectors,
            "lead_weak": sum(1 for s in leaders if s["below50"] or s["rs"] < 0),
            "lead_n": len(leaders),
            "weak_total": sum(1 for s in sectors if s["below50"] or s["rs"] < 0),
            "n": len(sectors),
        }
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"Sector RS fetch failed: {exc}")
        return None


# ----------------------------------------------------------------------------
# ANALYTICS  (logic preserved verbatim from the original)
# ----------------------------------------------------------------------------
def is_tight_flag(hist_df, adr, days=3, max_range_pct=10.0):
    """Check whether the last N days form a tight flag / coil (VooDoo pattern)."""
    if hist_df is None or len(hist_df) < days or adr <= 0:
        return False

    recent = hist_df.tail(days).dropna(subset=["High", "Low", "Close", "Volume"])
    if len(recent) < days:
        return False

    daily_ranges = (recent["High"] - recent["Low"]) / recent["Close"] * 100
    if not daily_ranges.le(adr * 1.0).all():
        return False

    total_range = (recent["High"].max() - recent["Low"].min()) / recent["Close"].iloc[-1] * 100
    if total_range > max_range_pct:
        return False

    volumes = recent["Volume"].values
    if volumes[-1] > volumes[0] * 1.5:
        return False

    return True


DEAD_RANGE_DAYS = 5      # window for the auto dead-stock screen
DEAD_RANGE_PCT = 0.5     # mean daily range below this over the window => pinned/dead


def is_dead_pinned(hist_df, days: int = DEAD_RANGE_DAYS, thresh: float = DEAD_RANGE_PCT) -> bool:
    """Auto-screen 'dead' names — M&A targets pinned at a cash deal price, plus
    halted / pending-delist names. They flat-line: the recent daily range collapses
    to ~0 even while the 20-day ADR still looks alive from the pre-deal move. Drop
    any name whose MEAN daily (High/Low-1) over the last `days` sessions is below
    `thresh`%. Calibrated 2026-06-30: dead JHG 0.07% / NUVL 0.14% vs the lowest live
    momentum name ~1.5%+, so 0.5% sits in the gap with wide margin (no false drops)."""
    if hist_df is None or len(hist_df) < days:
        return False
    recent = hist_df.tail(days)
    lo = recent["Low"].values.astype(float)
    hi = recent["High"].values.astype(float)
    if (lo <= 0).any():
        return False
    rng = (hi / lo - 1.0) * 100.0
    return bool(np.mean(rng) < thresh)


def _vol_window_dryup(hist_df, n: int, thresh: float) -> bool:
    """True if the MEAN volume over the last `n` sessions is <= `thresh`% of EITHER
    the session immediately before that window OR the 50-day average volume.
    n=1 reduces to "today's volume vs yesterday / the 50-day avg" (the A- gate)."""
    if hist_df is None or len(hist_df) < n + 1:
        return False
    vols = hist_df["Volume"].values
    win = float(np.mean(vols[-n:]))
    prevd = float(vols[-(n + 1)])
    if prevd > 0 and win / prevd * 100 <= thresh:
        return True
    if len(vols) >= 50:
        v50 = float(np.mean(vols[-50:]))
        if v50 > 0 and win / v50 * 100 <= thresh:
            return True
    return False


def calculate_meta_momentum_score(stock_data, hist_df=None):
    """M.E.T.A. Momentum Score. Component MAX points come from META_WEIGHTS
    (meta_weights.json overrides the defaults); each tier awards
    round(weight * tier_fraction). With default weights this reproduces the
    original 140-point architecture exactly. Normalised to 100 by META_DENOM."""
    score = 0
    badges: List[str] = []
    details: List[str] = []

    adr = stock_data.get("adr", 0)
    is_flag = False
    if hist_df is not None and len(hist_df) >= 3 and adr > 0:
        is_flag = is_tight_flag(hist_df, adr, days=3, max_range_pct=10.0)

    # 1. Trend Strength
    perf_1m = stock_data.get("perf_1m", 0)
    perf_3m = stock_data.get("perf_3m", 0)
    if perf_1m >= 50 or perf_3m >= 100:
        score += _meta_award(details, "Trend", 1.0, "Trend: Explosive")
    elif perf_1m >= 25 or perf_3m >= 50:
        score += _meta_award(details, "Trend", 10 / 15, "Trend: Strong")
    else:
        score += _meta_award(details, "Trend", 5 / 15, "Trend: Established")

    # 2. Pivot Proximity
    dist_52w = stock_data.get("dist_52w", 0)
    if dist_52w <= 5.0:
        score += _meta_award(details, "Proximity", 1.0, "Proximity: Golden Zone 0-5%")
    elif dist_52w <= 10.0:
        score += _meta_award(details, "Proximity", 7 / 10, "Proximity: Near Pivot 5-10%")
    elif dist_52w <= 15.0:
        score += _meta_award(details, "Proximity", 3 / 10, "Proximity: Extended 10-15%")
    else:
        score += _meta_award(details, "Proximity", -1.0, "Proximity: Climax Run >15%",
                             badges, "⚠️ EXTENDED PIVOT")

    # 3. 10MA Quality
    close = stock_data.get("close", 0)
    sma10 = stock_data.get("sma10", 0)
    sma20 = stock_data.get("sma20", 0)
    if close > 0 and sma10 > 0 and sma20 > 0:
        dist10 = abs(close - sma10) / sma10 * 100
        dist20 = abs(close - sma20) / sma20 * 100
        if dist10 <= 3.0 and dist20 > 10.0 and close > sma20:
            score += _meta_award(details, "10MA Quality", 1.0, "10MA Quality: Dominance",
                                 badges, "👑 10MA DOMINANCE")
        elif dist10 <= 3.0:
            score += _meta_award(details, "10MA Quality", 12 / 15, "10MA Quality: Hugging 10MA",
                                 badges, "🎯 HUGGING 10MA")
        elif dist10 <= 5.0:
            score += _meta_award(details, "10MA Quality", 8 / 15, "10MA Quality: Near 10MA")
        else:
            score += _meta_award(details, "10MA Quality", 3 / 15, "10MA Quality: Resting")

    # 4. Volume Contraction (VooDoo)
    vol_pct = stock_data.get("vol_pct", 100)
    if vol_pct <= 55:
        score += _meta_award(details, "Vol Contraction", 1.0, "Vol Contraction: VooDoo <55%",
                             badges, "💧 VOODOO DAY")
    elif vol_pct <= 75:
        score += _meta_award(details, "Vol Contraction", 10 / 15, "Vol Contraction: Contracting")
    else:
        score += _meta_award(details, "Vol Contraction", 0.0, "Vol Contraction: Normal")

    # 5. Breakout Volume Expansion
    if vol_pct >= 250:
        score += _meta_award(details, "Vol Expansion", 1.0, "Vol Expansion: Massive >2.5x",
                             badges, "🚀 VOLUME SURGE")
    elif vol_pct >= 150:
        score += _meta_award(details, "Vol Expansion", 0.5, "Vol Expansion: Moderate >1.5x")
    else:
        score += _meta_award(details, "Vol Expansion", 0.0, "Vol Expansion: No Surge")

    # 6. Tight Flag Action (3-day) — Flag and Candle share the same slot
    if is_flag:
        score += _meta_award(details, "Flag", 1.0, "Flag: 3-Day Tight Coil",
                             badges, "🌀 TIGHT FLAG")
    else:
        day_range_pct = stock_data.get("day_range_pct", 0)
        if day_range_pct > 0 and adr > 0 and day_range_pct <= adr * 0.5:
            score += _meta_award(details, "Flag", 0.5, "Candle: Ultra Tight")
        else:
            score += _meta_award(details, "Flag", 0.0, "Candle: Loose")

    # 7. Base Quality (Depth)
    base_depth = 25.0
    if hist_df is not None and len(hist_df) >= 20:
        recent_high = hist_df["High"].iloc[-20:].max()
        recent_low = hist_df["Low"].iloc[-20:].min()
        if recent_high > 0:
            base_depth = (recent_high - recent_low) / recent_high * 100
    if base_depth < 35.0:
        score += _meta_award(details, "Base Quality", 1.0, f"Base Quality: Tight Base {base_depth:.1f}%")
    elif base_depth <= 50.0:
        score += _meta_award(details, "Base Quality", 8 / 15, f"Base Quality: Moderate Base {base_depth:.1f}%")
    else:
        score += _meta_award(details, "Base Quality", 0.0, f"Base Quality: Wide/Loose Base {base_depth:.1f}%")

    # 8. Relative Strength vs Market
    if dist_52w <= 5.0:
        score += _meta_award(details, "RS", 1.0, "RS: Leading - Near 52W High",
                             badges, "🔥 MARKET LEADER")
    elif perf_3m >= 60.0:
        score += _meta_award(details, "RS", 10 / 15, "RS: Strong Outperformance")
    else:
        score += _meta_award(details, "RS", 5 / 15, "RS: Market Performer")

    # 9. Controlled Volatility (EER)
    eer = 0.0
    if adr > 0:
        eer = perf_3m / adr
    if eer >= 5.0:
        score += _meta_award(details, "Volatility", 1.0, f"Volatility: Super Efficient EER {eer:.1f}")
    elif eer >= 3.0:
        score += _meta_award(details, "Volatility", 0.6, f"Volatility: Clean EER {eer:.1f}")
    else:
        score += _meta_award(details, "Volatility", 0.0, f"Volatility: Erratic EER {eer:.1f}")

    # 10. Supply Shock Potential
    float_shares = stock_data.get("float_shares", 0)
    mcap = stock_data.get("mcap", 10.0)
    if float_shares > 0:
        is_low_float = float_shares < 200e6
        float_desc = f"{float_shares / 1e6:.1f}M"
    else:
        is_low_float = mcap < 2.0
        float_desc = f"Cap {mcap:.1f}B"

    is_high_rvol = vol_pct >= 150
    if is_low_float and is_high_rvol:
        score += _meta_award(details, "Supply Shock", 1.0,
                             f"Supply Shock: Low Float ({float_desc}) + RVOL Surge",
                             badges, "💥 SUPPLY SHOCK")
    elif is_low_float or is_high_rvol:
        score += _meta_award(details, "Supply Shock", 0.5, "Supply Shock: Capable")
    else:
        score += _meta_award(details, "Supply Shock", 0.0, "Supply Shock: Large/Quiet")

    # 11. Risk Efficiency
    risk_pct = stock_data.get("risk_pct", 10.0)
    if risk_pct <= 3.5:
        score += _meta_award(details, "Risk", 1.0, f"Risk: Super Asymmetric {risk_pct}%",
                             badges, "🛡️ ASYMMETRIC RISK")
    elif risk_pct <= 5.0:
        score += _meta_award(details, "Risk", 10 / 15, f"Risk: Acceptable {risk_pct}%")
    else:
        score += _meta_award(details, "Risk", 0.0, f"Risk: Wide Stop {risk_pct}%")

    raw_score = max(0, score)
    final_score = (raw_score / META_DENOM) * 100.0

    if risk_pct <= 3.5:
        final_score *= 1.1
    elif risk_pct > 6.0:
        final_score *= 0.8

    final_score = round(min(final_score, 100.0), 1)

    if final_score >= 80.0:
        badges.insert(0, "🔥 SUPER MOMENTUM")
    elif final_score >= 60.0:
        badges.insert(0, "⚡ STRONG MOMENTUM")

    return {"score": final_score, "badges": badges, "details": details, "is_flag": is_flag}


def find_pivot_points(highs, lows, window=3):
    """Find swing highs and lows using pivot detection."""
    swing_highs = []
    swing_lows = []
    for i in range(window, len(highs) - window):
        if all(highs[i] >= highs[j] for j in range(i - window, i + window + 1) if j != i):
            if all(highs[i] > highs[j] for j in [i - window, i + window]):
                swing_highs.append((i, highs[i]))
        if all(lows[i] <= lows[j] for j in range(i - window, i + window + 1) if j != i):
            if all(lows[i] < lows[j] for j in [i - window, i + window]):
                swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def calculate_trendline_slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x2 == x1:
        return 0
    return (y2 - y1) / (x2 - x1)


def get_trendline_value_at_index(slope, point, target_idx):
    idx, price = point
    return price + slope * (target_idx - idx)


def _best_line(swings, want_rising: bool):
    """Shared helper: enumerate candidate trendlines and pick the best by touches+recency."""
    valid = []
    for i in range(len(swings) - 1):
        for j in range(i + 1, len(swings)):
            higher = swings[j][1] > swings[i][1]
            if higher != want_rising:
                continue
            slope = calculate_trendline_slope(swings[i], swings[j])
            if (want_rising and slope <= 0) or (not want_rising and slope >= 0):
                continue
            touches = 0
            for k in range(len(swings)):
                if k in (i, j):
                    continue
                expected = get_trendline_value_at_index(slope, swings[i], swings[k][0])
                if expected > 0 and abs(swings[k][1] - expected) / expected < 0.02:
                    touches += 1
            valid.append({"points": (swings[i], swings[j]), "slope": slope,
                          "touches": touches, "recency": swings[j][0]})
    if not valid:
        return None
    return max(valid, key=lambda x: (x["touches"], x["recency"]))


def calculate_trendline_analysis(ticker, hist_df=None):
    """M.E.T.A. Trendline Analysis (J Law). Detects UTL / TRL / DTL / TSL."""
    result = {
        "has_data": False,
        "utl": {"exists": False, "distance_pct": None, "slope": None, "score": 0},
        "trl": {"exists": False, "distance_pct": None, "slope": None, "score": 0},
        "dtl": {"exists": False, "distance_pct": None, "breakout": False, "score": 0},
        "tsl": {"exists": False, "distance_pct": None, "score": 0},
        "total_score": 0,
        "signals": [],
        "details": [],
    }

    if hist_df is None:
        hist_df = fetch_stock_history(ticker)
    if hist_df is None or len(hist_df) < 30:
        return result

    result["has_data"] = True
    highs = hist_df["High"].values
    lows = hist_df["Low"].values
    closes = hist_df["Close"].values
    current_price = closes[-1]
    current_idx = len(closes) - 1

    swing_highs, swing_lows = find_pivot_points(highs, lows, window=3)
    if len(swing_highs) < 2 and len(swing_lows) < 2:
        return result

    # === UTL (Up Trend Line) - Rising Support ===
    if len(swing_lows) >= 2:
        best_utl = _best_line(swing_lows, want_rising=True)
        if best_utl:
            slope = best_utl["slope"]
            point = best_utl["points"][0]
            utl_value_now = get_trendline_value_at_index(slope, point, current_idx)
            distance_pct = ((current_price - utl_value_now) / utl_value_now) * 100
            result["utl"].update({"exists": True, "slope": slope,
                                  "distance_pct": round(distance_pct, 2), "touches": best_utl["touches"]})
            if distance_pct <= 3:
                result["utl"]["score"] = 20
                result["signals"].append("🎯 UTL ENTRY ZONE (<3%)")
                result["details"].append(f"UTL: {distance_pct:.1f}% above support - PERFECT")
            elif distance_pct <= 8:
                result["utl"]["score"] = 10
                result["signals"].append("📐 Near UTL")
                result["details"].append(f"UTL: {distance_pct:.1f}% above support")
            elif distance_pct <= 15:
                result["utl"]["score"] = 5
                result["details"].append(f"UTL: {distance_pct:.1f}% above support - extended")
            else:
                result["utl"]["score"] = 0
                result["signals"].append("⚠️ FAR FROM UTL")
                result["details"].append(f"UTL: {distance_pct:.1f}% above support - OVEREXTENDED")

    # === TRL (Trend Resistance Line) - Profit Target ===
    if len(swing_highs) >= 2:
        best_trl = _best_line(swing_highs, want_rising=True)
        if best_trl:
            slope = best_trl["slope"]
            point = best_trl["points"][0]
            trl_value_now = get_trendline_value_at_index(slope, point, current_idx)
            distance_pct = ((trl_value_now - current_price) / current_price) * 100
            result["trl"].update({"exists": True, "slope": slope,
                                  "distance_pct": round(distance_pct, 2), "touches": best_trl["touches"]})
            if distance_pct > 0:
                result["trl"]["score"] = 0
                result["signals"].append(f"🎯 TRL Target: +{distance_pct:.1f}%")
                result["details"].append(f"TRL: +{distance_pct:.1f}% to resistance target")
            else:
                result["details"].append("TRL: Price above resistance (breakout)")

    # === DTL (Down Trend Line) - Breakout Candidate ===
    if len(swing_highs) >= 2:
        best_dtl = _best_line(swing_highs, want_rising=False)
        if best_dtl:
            slope = best_dtl["slope"]
            point = best_dtl["points"][0]
            dtl_value_now = get_trendline_value_at_index(slope, point, current_idx)
            distance_pct = ((dtl_value_now - current_price) / current_price) * 100
            result["dtl"].update({"exists": True, "slope": slope,
                                  "distance_pct": round(distance_pct, 2), "touches": best_dtl["touches"]})
            if distance_pct < 0:
                result["dtl"]["breakout"] = True
                result["dtl"]["score"] = 20
                result["signals"].append("🔥 DTL BREAKOUT!")
                result["details"].append(f"DTL: BREAKOUT! {abs(distance_pct):.1f}% above declining resistance")
            elif distance_pct <= 3:
                result["dtl"]["score"] = 10
                result["signals"].append("⚡ Near DTL Breakout")
                result["details"].append(f"DTL: {distance_pct:.1f}% from breakout")
            else:
                result["dtl"]["score"] = 0
                result["details"].append(f"DTL: {distance_pct:.1f}% from breakout")

    # === TSL (Trend Support Line) - In Downtrends ===
    if len(swing_lows) >= 2:
        best_tsl = _best_line(swing_lows, want_rising=False)
        if best_tsl:
            slope = best_tsl["slope"]
            point = best_tsl["points"][0]
            tsl_value_now = get_trendline_value_at_index(slope, point, current_idx)
            distance_pct = ((current_price - tsl_value_now) / tsl_value_now) * 100
            result["tsl"].update({"exists": True, "slope": slope,
                                  "distance_pct": round(distance_pct, 2), "touches": best_tsl["touches"]})
            if distance_pct > 0:
                result["tsl"]["score"] = 5
                result["details"].append(f"TSL: {distance_pct:.1f}% above declining support")

    result["total_score"] = (
        result["utl"]["score"] + result["trl"]["score"]
        + result["dtl"]["score"] + result["tsl"]["score"]
    )
    return result


def get_theme(ticker, industry):
    overrides = {
        "SNDK": "NAND Memory (AI Infra)", "MU": "DRAM / NAND Memory (AI Infra)",
        "INTC": "CPU / Foundry", "NVDA": "AI GPUs / Data Center", "AMD": "CPU / AI GPUs",
        "DELL": "AI Servers / Hardware", "SMCI": "AI Servers / Liquid Cooling",
        "CRWD": "Cybersecurity / Endpoint", "PLTR": "AI Software / Defense",
        "COIN": "Crypto Exchange", "MSTR": "Bitcoin Treasury", "HOOD": "Retail Trading / Crypto",
        "TEAM": "Enterprise Collaboration (SaaS)", "TWLO": "Cloud Communications (CPaaS)",
        "LITE": "Optical Components (AI Infra)", "JBL": "Electronics Mfg (AI Server Racks)",
        "LUNR": "Space Exploration / Lunar", "RKLB": "Aerospace / Rockets", "ASTS": "Space / Satellite Telecom",
        "SOUN": "Voice AI", "ALB": "Lithium / EV Battery", "AEHR": "Semi Test Equipment (SiC/EVs)",
        "CIFR": "Bitcoin Mining", "COHR": "Optical Components / Lasers (AI)",
        "SGML": "Lithium Mining (EVs)", "KOD": "Ophthalmology Biotech",
        "BAND": "Cloud Communications (CPaaS)", "ATOM": "Semiconductor Materials",
        "NVTS": "Silicon Carbide (EV/AI Power)", "FORM": "Semi Test Equipment",
    }
    if ticker in overrides:
        return overrides[ticker]
    ind = str(industry).strip()
    if ind in ("None", ""):
        return "N/A"
    ind_lower = ind.lower()
    if "software" in ind_lower:
        return "Software / SaaS"
    if "semiconductor" in ind_lower:
        return "Semiconductors"
    if "aerospace" in ind_lower:
        return "Aerospace & Defense"
    if "biotechnology" in ind_lower:
        return "Biotech"
    return ind


# MA overlay colours (distinct, GitHub-dark friendly) + the periods we draw.
# 50-MA deliberately the darkest/most-muted (slowest line, should recede).
_MA_SPEC = [(10, "#58a6ff"), (20, "#e3b341"), (50, "#8957e5")]   # blue / amber / dark-purple
# Darker, thinner variants for the small external-engine sparklines (Minervini /
# Trilogy) so the MA lines sit clearly BEHIND the bright green/red price line.
_MA_SPEC_DARK = [(10, "#1158c7"), (20, "#9e6a00"), (50, "#5a2da0")]  # dark blue / amber / purple
_MA_SPEC_10W = [(10, "#1158c7")]                                     # single 10-week MA (Trilogy)


def _trailing_sma(vals: List[float], period: int) -> List[Optional[float]]:
    """Simple trailing MA aligned to `vals`; None where < period bars precede it.
    O(n) running sum."""
    out: List[Optional[float]] = []
    s = 0.0
    for i, v in enumerate(vals):
        s += v
        if i >= period:
            s -= vals[i - period]
        out.append(s / period if i >= period - 1 else None)
    return out


def _polyline_segments(xy: List[Tuple[float, Optional[float]]], color: str, sw: float) -> str:
    """Draw a polyline that breaks across None gaps (so undefined MA bars don't
    connect). Returns concatenated <polyline> elements."""
    segs, cur = [], []
    for x, y in xy:
        if y is None:
            if len(cur) >= 2:
                segs.append(cur)
            cur = []
        else:
            cur.append(f"{x:.1f},{y:.1f}")
    if len(cur) >= 2:
        segs.append(cur)
    return "".join(
        f'<polyline points="{" ".join(s)}" fill="none" stroke="{color}" '
        f'stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round" opacity="0.75"/>'
        for s in segs)


def make_sparkline(closes: Iterable[float], width: int = 88, height: int = 26, pad: int = 2,
                   window: Optional[int] = None, show_ma: bool = False,
                   price_sw: float = 1.5, ma_sw: float = 1.1, label_fs: int = 7,
                   ma_spec: Optional[List[Tuple[int, str]]] = None, ma_labels: bool = True) -> str:
    """Inline SVG sparkline from a price series (reuses already-fetched history).

    show_ma=True overlays 10/20/50-period MAs (computed on the FULL series passed,
    so the 50-MA is valid across the whole window) and displays only the last
    `window` bars. Each MA gets a colour + a small end-label so it reads clearly.
    Backward-compatible: with show_ma=False / window=None it draws the single price
    line exactly as before (used by the external-engine weekly/low sparklines)."""
    vals = [float(c) for c in closes if c is not None and not (isinstance(c, float) and math.isnan(c))]
    if len(vals) < 2:
        return ""
    spec = ma_spec if ma_spec is not None else _MA_SPEC

    # MA series over the FULL history, THEN slice to the display window (so the
    # 50-MA exists at the left edge of the window instead of being blank).
    ma_full = {p: _trailing_sma(vals, p) for p, _ in spec} if show_ma else {}
    if window and len(vals) > window:
        disp = vals[-window:]
        ma_disp = {p: ma_full[p][-window:] for p in ma_full}
    else:
        disp = vals
        ma_disp = {p: ma_full[p][-len(disp):] for p in ma_full}
    n = len(disp)

    label_w = (label_fs * 2 + 4) if (show_ma and ma_labels) else 0
    inner_w = width - 2 * pad - label_w
    inner_h = height - 2 * pad

    # scale over the displayed price PLUS every defined MA point, so all lines fit
    ys = list(disp)
    for p in ma_disp:
        ys += [v for v in ma_disp[p] if v is not None]
    lo, hi = min(ys), max(ys)
    rng = (hi - lo) or 1.0

    def X(i):
        return pad + (i / (n - 1)) * inner_w

    def Y(v):
        return pad + (1 - (v - lo) / rng) * inner_h

    parts = []
    labels = []   # (y, period, colour) collected, then de-collided so none overlap
    # MA lines first (under the price), longest period at the back
    for p, col in spec:
        if p not in ma_disp:
            continue
        xy = [(X(i), (Y(v) if v is not None else None)) for i, v in enumerate(ma_disp[p])]
        parts.append(_polyline_segments(xy, col, ma_sw))
        # Label only an MA that actually drew a line (>=2 defined in-window points).
        # A 1-point MA renders no polyline (_polyline_segments needs >=2), so without
        # this gate its end-label would float orphaned with no visible line.
        drawn = [t for t in xy if t[1] is not None]
        if len(drawn) >= 2 and ma_labels:
            labels.append([min(max(drawn[-1][1], pad + label_fs), height - pad), p, col])
    # spread labels vertically so converging MAs stay legible
    gap = label_fs + 1
    labels.sort()
    for k in range(1, len(labels)):
        if labels[k][0] - labels[k - 1][0] < gap:
            labels[k][0] = labels[k - 1][0] + gap
    overflow = labels[-1][0] - (height - 1) if labels else 0
    if overflow > 0:                       # shift the whole stack up if it ran past the bottom
        for lab in labels:
            lab[0] -= overflow
    for y, p, col in labels:
        parts.append(
            f'<text x="{width - label_w + 2:.1f}" y="{y:.1f}" font-size="{label_fs}" '
            f'font-family="ui-monospace,monospace" fill="{col}">{p}</text>')

    color = "#3fb950" if disp[-1] >= disp[0] else "#ff7b72"
    ppts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(disp))
    parts.append(
        f'<polyline points="{ppts}" fill="none" stroke="{color}" '
        f'stroke-width="{price_sw}" stroke-linejoin="round" stroke-linecap="round"/>')
    parts.append(f'<circle cx="{X(n - 1):.1f}" cy="{Y(disp[-1]):.1f}" r="{price_sw * 1.25:.1f}" fill="{color}"/>')

    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'style="vertical-align:middle;">' + "".join(parts) + "</svg>")


def make_price_spark(closes: Iterable[float], window: int = 40) -> str:
    """MADRRY price-column sparkline: last `window` daily closes with 10/20/50-MA
    overlays. Larger canvas (uses the wide Price column), THICK price line, THIN
    muted MA lines so the price stands out and the MAs don't crowd it."""
    return make_sparkline(closes, width=190, height=58, pad=4, window=window,
                          show_ma=True, price_sw=2.4, ma_sw=0.8, label_fs=9)


def make_volume_bars(volumes: Iterable[float], ups: Optional[List[bool]] = None,
                     width: int = 88, height: int = 12, pad: int = 1) -> str:
    """Inline SVG volume histogram sized to sit directly under a make_sparkline
    price line (same width). Each bar is green on an up-close day, muted on a
    down/flat day, scaled to the window's max volume."""
    vals = [float(v) if (v is not None and not (isinstance(v, float) and math.isnan(v))) else 0.0
            for v in volumes]
    if len(vals) < 2:
        return ""
    hi = max(vals) or 1.0
    n = len(vals)
    inner_w = width - 2 * pad
    inner_h = height - 2 * pad
    bw = inner_w / n
    bars = []
    for i, v in enumerate(vals):
        h = (v / hi) * inner_h
        x = pad + i * bw
        y = height - pad - h
        col = "#3fb950" if (ups and i < len(ups) and ups[i]) else "#6e7681"
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bw - 0.4, 0.6):.1f}" '
                    f'height="{max(h, 0.4):.1f}" fill="{col}" opacity="0.65"/>')
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'style="display:block;margin-top:1px;">' + "".join(bars) + '</svg>')


# ----------------------------------------------------------------------------
# PLAYBOOK / FOOTPRINT ANALYSIS  (Martin Momentum method — additive, soft)
#
# Studies the ticker's recent "footprint": the base it built, how many
# higher-lows it has printed, whether the 9/21/50 EMAs are coiled (his favorite
# first-pullback condition) or spread/extended (his "out of the universe" zone),
# plus an Anchored VWAP confirmation. Everything here is informational — it adds
# badges, it never drops a stock or alters the M.E.T.A. score / tiers.
# ----------------------------------------------------------------------------
def _ema_last(close: pd.Series, span: int) -> float:
    return float(close.ewm(span=span, adjust=False).mean().iloc[-1])


def analyze_footprint(hist_df: Optional[pd.DataFrame]) -> dict:
    """Return base/EMA-coil/extension/AVWAP features + human-readable badges."""
    out = {
        "badges": [], "avwap_dist": None, "avwap_holding": False,
        "higher_lows": 0, "base_weeks": 0.0, "base_depth": None,
        "ema_spread": None, "ext9": None, "coiled": False, "extended": False,
    }
    if hist_df is None or len(hist_df) < 40:
        return out

    close, high, low = hist_df["Close"], hist_df["High"], hist_df["Low"]
    px = float(close.iloc[-1])
    if px <= 0:
        return out

    # --- EMA coil vs extended (Martin's core "footprint" read) ---
    e9, e21, e50 = _ema_last(close, 9), _ema_last(close, 21), _ema_last(close, 50)
    spread = (max(e9, e21, e50) - min(e9, e21, e50)) / px * 100.0
    ext9 = (px - e9) / e9 * 100.0 if e9 else 0.0
    out["ema_spread"] = round(spread, 1)
    out["ext9"] = round(ext9, 1)
    # Coiled = 9/21/50 clustered and price near them (Martin's first-pullback
    # condition). Extended = price far above the 9 EMA — his explicit "sell into
    # strength / out of the universe" rule. (Spread alone over-flags healthy
    # trends where the 50 EMA naturally trails, so it is not used as the guard.)
    out["coiled"] = spread <= 4.0 and ext9 < 8.0
    out["extended"] = ext9 >= 15.0

    # --- base: pivot high over the last ~90 sessions (excl. today) ---
    base_win = hist_df.iloc[-90:-1] if len(hist_df) > 90 else hist_df.iloc[:-1]
    piv_rel = int(base_win["High"].values.argmax())
    piv_pos = len(hist_df) - len(base_win) - 1 + piv_rel
    base_bars = max(len(hist_df) - 1 - piv_pos, 0)
    out["base_weeks"] = round(base_bars / 5.0, 1)
    base_high = float(base_win["High"].iloc[piv_rel])
    after = hist_df.iloc[piv_pos:]
    base_low = float(after["Low"].min())
    if base_high > 0:
        out["base_depth"] = round((base_high - base_low) / base_high * 100.0, 1)

    # --- higher-lows from swing lows inside the base (1st/2nd = premium) ---
    _, swing_lows = find_pivot_points(high.values, low.values, window=3)
    base_swing_lows = [p for (i, p) in swing_lows if i >= piv_pos]
    higher_lows = sum(1 for k in range(1, len(base_swing_lows))
                      if base_swing_lows[k] > base_swing_lows[k - 1])
    out["higher_lows"] = higher_lows

    # --- Anchored VWAP from the highest-volume candle in the last ~90 days ---
    # Exclude TODAY from the anchor search (iloc[-90:-1]); otherwise a breakout
    # day's own huge-volume bar self-anchors → single-row VWAP == today's price →
    # avwap_holding trivially True on any up-close (audit). Need >=2 bars.
    avwap_win = hist_df.iloc[-90:-1]
    sub = hist_df.iloc[0:0]
    if len(avwap_win) >= 1:
        anchor_rel = int(avwap_win["Volume"].values.argmax())
        anchor_pos = len(hist_df) - 1 - len(avwap_win) + anchor_rel
        sub = hist_df.iloc[anchor_pos:]
    cum_v = float(sub["Volume"].sum())
    if cum_v > 0 and len(sub) >= 2:
        tp = (sub["High"] + sub["Low"] + sub["Close"]) / 3.0
        avwap = float((tp * sub["Volume"]).sum() / cum_v)
        if avwap > 0:
            d = (px - avwap) / avwap * 100.0
            out["avwap_dist"] = round(d, 1)
            out["avwap_holding"] = (px >= avwap) and (0.0 <= d <= 4.0)

    # --- badges ---
    if higher_lows >= 1:
        out["badges"].append(f"🏗️ {higher_lows} Higher-Low" + ("s" if higher_lows > 1 else ""))
    if out["coiled"]:
        out["badges"].append("🪙 EMAs Coiled")
    if out["base_weeks"] >= 2:
        out["badges"].append(f"🧱 Base {out['base_weeks']:.0f}w")
    elif base_bars < 10:
        out["badges"].append("⚠️ Young Base (<2w)")
    if out["avwap_holding"]:
        out["badges"].append("📍 At AVWAP (holding)")
    elif out["avwap_dist"] is not None:
        out["badges"].append(f"〽️ AVWAP {out['avwap_dist']:+.0f}%")
    if out["extended"]:
        out["badges"].append(f"⚠️ Extended +{out['ext9']:.0f}% vs 9EMA")
    return out


# ----------------------------------------------------------------------------
# HVE HISTORY PERSISTENCE
# ----------------------------------------------------------------------------
def load_hve_history() -> dict:
    if os.path.exists(HVE_HISTORY_PATH):
        try:
            with open(HVE_HISTORY_PATH, "r") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_hve_history(history: dict) -> None:
    _atomic_write(HVE_HISTORY_PATH, json.dumps(history))


def cleanup_old_hve(history: dict, days: int = 5) -> dict:
    cutoff = datetime.now() - timedelta(days=days)
    cleaned = {}
    for ticker, data in history.items():
        try:
            if datetime.fromisoformat(data["date"]) >= cutoff:
                cleaned[ticker] = data
        except Exception:  # noqa: BLE001
            continue
    return cleaned


# ----------------------------------------------------------------------------
# YESTERDAY'S WATCHLIST TRACKING  (now batched — one network call, not N)
# ----------------------------------------------------------------------------
# Setup files are keyed by the scan's DATA DATE (the trading date of the last
# daily bar), NOT the wall-clock run time. This fixes the "yesterday was really
# my last run" bug: a mid-day re-run overwrites the SAME day's file instead of
# masquerading as a new day, and a 6 AM pre-market run keys to the prior close.
_SETUPS_DATE_RE = re.compile(r"^latest_setups_(\d{4}-\d{2}-\d{2})\.json$")


def _save_dated_setups(payload: str, data_date: str) -> None:
    """Write the run's picks under its data date (same-day re-runs overwrite, so
    each trading day ends up holding that day's FINAL run) and prune old files."""
    _atomic_write(os.path.join(WORKSPACE, f"latest_setups_{data_date}.json"), payload)
    try:
        dated = sorted(f for f in os.listdir(WORKSPACE) if _SETUPS_DATE_RE.match(f))
        for f in dated[:-14]:                       # keep ~14 trading days
            os.remove(os.path.join(WORKSPACE, f))
    except Exception:  # noqa: BLE001
        pass


def _previous_setups_path(data_date: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (path, date) of the newest dated setups file STRICTLY BEFORE the
    current scan's data date — i.e. the previous trading session's final picks,
    no matter how many times the scanner ran since. Falls back to the legacy
    un-dated latest_setups.json if no dated file exists yet."""
    try:
        dated = sorted(
            (m.group(1), f) for f in os.listdir(WORKSPACE)
            for m in [_SETUPS_DATE_RE.match(f)] if m
        )
    except Exception:  # noqa: BLE001
        dated = []
    prior = [(d, f) for d, f in dated if d < data_date]
    if prior:
        d, f = prior[-1]
        return os.path.join(WORKSPACE, f), d
    if os.path.exists(LATEST_SETUPS_PATH):
        return LATEST_SETUPS_PATH, None
    return None, None


def _append_breakout_log(records: list, data_date: Optional[str] = None) -> None:
    """Persist the session's A+/A/A- breakout outcomes keyed by DATA date
    (overwrites that day's entry on re-run, so the final run of a day wins)."""
    today = data_date or date.today().isoformat()
    log = {}
    if os.path.exists(BREAKOUT_LOG_PATH):
        try:
            with open(BREAKOUT_LOG_PATH) as fh:
                log = json.load(fh)
        except Exception:  # noqa: BLE001
            log = {}
    # MERGE per ticker (don't overwrite the whole day): a re-run updates each
    # re-graded ticker's outcome but never erases outcomes it didn't re-grade.
    day_recs = {r.get("ticker"): r for r in log.get(today, []) if r.get("ticker")}
    for r in records:
        if r.get("ticker"):
            day_recs[r["ticker"]] = r
    log[today] = list(day_recs.values())
    log = {k: log[k] for k in sorted(log)[-90:]}
    try:
        _atomic_write(BREAKOUT_LOG_PATH, json.dumps(log))
    except Exception:  # noqa: BLE001
        pass


def breakout_winrate(window: int = 50) -> Optional[dict]:
    """Rolling win-rate over the last `window` logged A+/A/A- breakout outcomes.
    Window widened to 50 so the score spans ~1-2 weeks now that the full A- pool
    is graded (otherwise a single busy day could fill the whole window)."""
    if not os.path.exists(BREAKOUT_LOG_PATH):
        return None
    try:
        with open(BREAKOUT_LOG_PATH) as fh:
            log = json.load(fh)
    except Exception:  # noqa: BLE001
        return None
    flat = []
    for d in sorted(log, reverse=True):
        flat.extend(r.get("outcome") for r in log[d])
        if len(flat) >= window:
            break
    flat = flat[:window]
    if not flat:
        return None
    wins = sum(1 for o in flat if o == "win")
    return {"winrate": round(100 * wins / len(flat)), "n": len(flat)}


def breakout_cumulative() -> Optional[dict]:
    """Accumulated win-rate over EVERY logged outcome in breakout_log.json
    (~90 days retained) — the report's true running win-rate, not just today's."""
    if not os.path.exists(BREAKOUT_LOG_PATH):
        return None
    try:
        with open(BREAKOUT_LOG_PATH) as fh:
            log = json.load(fh)
    except Exception:  # noqa: BLE001
        return None
    wins = losses = hwins = hlosses = 0
    days = sorted(log)
    for d in days:
        for r in log[d]:
            w = r.get("outcome") == "win"
            l = r.get("outcome") == "loss"
            wins += w
            losses += l
            if r.get("htf"):
                hwins += w
                hlosses += l
    n = wins + losses
    if n == 0:
        return None
    return {"wins": wins, "losses": losses, "n": n,
            "rate": round(100 * wins / n), "since": days[0] if days else "",
            "htf_wins": hwins, "htf_losses": hlosses, "htf_n": hwins + hlosses}


def track_previous_setups(diag: Optional[Diagnostics] = None,
                          data_date: Optional[str] = None) -> str:
    data_date = data_date or date.today().isoformat()
    prev_path, prev_date = _previous_setups_path(data_date)
    if not prev_path or not os.path.exists(prev_path):
        return ""
    try:
        with open(prev_path, "r") as fh:
            prev_data = json.load(fh)
    except Exception:  # noqa: BLE001
        return ""

    tracking_stocks = [
        s for s in prev_data
        if s.get("meta_score") is not None and s.get("tier", "") in ("A+", "A", "A-")
    ]
    if not tracking_stocks:
        return ""

    # --- ONE batched download for every tracked ticker (was: one call each) ---
    tickers = sorted({s["ticker"] for s in tracking_stocks})
    hist_map: Dict[str, pd.DataFrame] = {}
    try:
        raw = yf.download(tickers=tickers, period="5d", interval="1d", group_by="ticker",
                          auto_adjust=False, threads=True, progress=False)
        if raw is not None and len(raw):
            multi = isinstance(raw.columns, pd.MultiIndex)
            for t in tickers:
                try:
                    df = (raw[t] if multi else raw).dropna()
                    if len(df) >= 2:
                        hist_map[t] = df
                except Exception:  # noqa: BLE001
                    continue
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"Watchlist tracking batch download failed: {exc}")
    # Fill any missing latest bar from TradingView (Yahoo bulk feed can lag).
    _patch_stale_histories(hist_map, label="tracking(5d)")

    # FRESHNESS GUARD: outcomes may only be recorded when the grading bar is
    # provably NEWER than the picks being graded. Grading picks against their
    # own (or an older) session — which happens when Yahoo's bulk feed lags —
    # would fabricate win/loss records (happened 2026-06-11).
    bar_date = None
    if hist_map:
        try:
            bar_date = max(df.index[-1] for df in hist_map.values()).date().isoformat()
        except Exception:  # noqa: BLE001
            bar_date = None
    may_log = bool(prev_date) and bool(bar_date) and bar_date > prev_date
    if not may_log and diag:
        diag.warn(f"Watchlist outcomes NOT logged: picks of {prev_date or 'unknown date'} "
                  f"vs bars ending {bar_date or '?'} — need a strictly newer session bar")

    rows = []
    log_records = []
    n_eval = 0
    n_win = n_loss = 0
    coiled = {"A+": 0, "A": 0, "A-": 0}   # still-coiled (not triggered, not stopped)
    coiled_total = 0
    for s in tracking_stocks:
        ticker = s["ticker"]
        y_entry, y_stop = s["entry"], s["stop"]
        y_score, y_tier = s["meta_score"], s.get("tier", "A")
        hist = hist_map.get(ticker)
        if hist is None or len(hist) < 2:
            continue
        n_eval += 1

        today = hist.iloc[-1]
        t_high, t_low, t_close = today["High"], today["Low"], today["Close"]
        if pd.isna(t_high) or pd.isna(t_low) or pd.isna(t_close):
            continue                         # bad bar — don't mis-classify as coiling (audit)
        prev_close = hist.iloc[-2]["Close"]
        t_change = (t_close - prev_close) / prev_close * 100 if prev_close else 0.0

        triggered = t_high >= y_entry
        stopped_out = t_low < y_stop
        # PER-TICKER freshness: only record an outcome when THIS ticker's own last
        # bar is strictly newer than the picks' session (a batch-wide may_log let a
        # single fresh ticker green-light grading stale ones — audit H1).
        try:
            this_bar = hist.index[-1].date().isoformat()
        except Exception:  # noqa: BLE001
            this_bar = None
        ticker_fresh = bool(prev_date) and bool(this_bar) and this_bar > prev_date
        # Log the breakout outcome for the rolling win-rate (untriggered = skip).
        if triggered and ticker_fresh:
            outcome = "win" if (t_close >= y_entry and not stopped_out) else "loss"
            log_records.append({"ticker": ticker, "tier": y_tier, "outcome": outcome,
                                "htf": bool(s.get("is_htf"))})
            n_win += outcome == "win"
            n_loss += outcome == "loss"

        # Collapse the display: only ACTIONABLE outcomes (triggered or stopped) get
        # a row; everything still coiling is rolled into a single summary line.
        if not (triggered or stopped_out):
            coiled[y_tier] = coiled.get(y_tier, 0) + 1
            coiled_total += 1
            continue

        if triggered:
            if stopped_out:
                status_text, status_style = "📉 Failed (Triggered & Stopped Out)", "color:#ff7b72;font-weight:bold;"
            elif t_close >= y_entry:
                status_text, status_style = f"🚀 Triggered & Holding (+{t_change:.1f}%)", "color:#3fb950;font-weight:bold;"
            else:
                status_text, status_style = "🔄 Triggered & Pulling Back", "color:#f2cc60;font-weight:bold;"
        else:
            status_text, status_style = "📉 Stopped Out (No Trigger)", "color:#ff7b72;font-weight:bold;"

        change_col = "good" if t_change > 0 else "bad"
        tier_badge_style = {
            "A+": "background:var(--tint-green);color:#3fb950;border:1px solid #3fb950;",
            "A":  "background:var(--tint-yellow);color:#f2cc60;border:1px solid #f2cc60;",
            "A-": "background:#21262d;color:#8b949e;border:1px solid #8b949e;",
        }.get(y_tier, "background:#21262d;color:#8b949e;border:1px solid #8b949e;")
        rows.append(f"""
            <tr style="border-bottom:1px solid #30363d;text-align:center;">
                <td style="padding:10px;font-weight:bold;" class="ticker"><a href="https://tradingview.com/symbols/{esc(ticker)}" target="_blank">{esc(ticker)}</a></td>
                <td style="padding:10px;"><span style="font-size:var(--fs-table);font-weight:bold;padding:3px 6px;border-radius:4px;{tier_badge_style}">Tier {esc(y_tier)}</span></td>
                <td style="padding:10px;"><span class="score" style="border-color:#79c0ff;color:#79c0ff;background:var(--tint-accent);">{y_score}</span></td>
                <td style="padding:10px;color:#3fb950;font-weight:bold;">${y_entry:.2f}</td>
                <td style="padding:10px;color:#ff7b72;font-weight:bold;">${y_stop:.2f}</td>
                <td style="padding:10px;font-size:var(--fs-body);color:#8b949e;">${t_low:.2f} - ${t_high:.2f}</td>
                <td style="padding:10px;" class="{change_col}">{t_change:+.2f}%</td>
                <td style="padding:10px;{status_style}">{status_text}</td>
            </tr>""")

    if log_records:                          # each record already passed per-ticker freshness
        _append_breakout_log(log_records, data_date)

    if n_eval == 0:
        return ""

    # ---- summary only (full grading still logged to the win-rate; the per-name
    #      table is intentionally omitted to keep the report short) ----
    src_bit = (f" <span style='color:#8b949e;'>(picks of {prev_date})</span>"
               if prev_date else "")
    if not may_log:
        win_bit = (f"<span style='color:#f2cc60;'>⚠️ outcomes withheld — bars end "
                   f"{esc(bar_date or '?')}, need a session newer than {esc(prev_date or 'the picks')}</span>")
    elif n_win or n_loss:
        win_bit = (f"<span style='color:#3fb950;'>✅ {n_win} win</span> / "
                   f"<span style='color:#ff7b72;'>❌ {n_loss} loss</span> "
                   f"<span style='color:#8b949e;'>this session</span>")
    else:
        win_bit = "<span style='color:#8b949e;'>0 triggered this session</span>"
    cum = breakout_cumulative()
    cum_bit = ""
    if cum:
        ccol = "#3fb950" if cum["rate"] > 55 else ("#f2cc60" if cum["rate"] >= 40 else "#ff7b72")
        cum_bit = (f" &nbsp;|&nbsp; <span style='color:{ccol};font-weight:bold;'>📊 Accumulated win-rate "
                   f"{cum['rate']}%</span> <span style='color:#8b949e;'>({cum['wins']}W/{cum['losses']}L "
                   f"over {cum['n']} · since {esc(cum['since'])})</span>")
        if cum.get("htf_n"):
            hr = round(100 * cum["htf_wins"] / cum["htf_n"])
            hcol = "#3fb950" if hr > 55 else ("#f2cc60" if hr >= 40 else "#ff7b72")
            cum_bit += (f" <span style='color:{hcol};'>🚩 HTF {hr}%</span> "
                        f"<span style='color:#8b949e;'>({cum['htf_wins']}W/{cum['htf_losses']}L)</span>")
    coiled_bit = ""
    if coiled_total:
        cb = " · ".join(f"{t} {coiled[t]}" for t in ("A+", "A", "A-") if coiled.get(t))
        coiled_bit = f" &nbsp;|&nbsp; <span style='color:#8b949e;'>🌀 {coiled_total} still coiling ({cb})</span>"
    summary = (f"<span style='color:#c9d1d9;'>Graded <b>{n_eval}</b> prior picks{src_bit}</span> "
               f"&nbsp;|&nbsp; {win_bit}{cum_bit}{coiled_bit}")

    return f"""
    <div style="background-color:#161b22;border-radius:8px;padding:12px 15px;margin-bottom:25px;box-shadow:0 0 15px rgba(0,0,0,0.5);border-left:4px solid #79c0ff;">
        <span style="color:#79c0ff;font-weight:bold;font-size:var(--fs-body);text-transform:uppercase;">🔄 Yesterday's Watchlist:</span>
        <span style="font-size:var(--fs-table);">&nbsp;{summary}</span>
    </div>
    """


# ----------------------------------------------------------------------------
# SCANNERS
# ----------------------------------------------------------------------------
def scan_coil(rs_map: dict, market_modifier: float, diag: Diagnostics):
    """Two-pass coil scan: cheap server filter, then batched yfinance enrichment."""
    payload_coil = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": "close", "operation": "egreater", "right": 10},
            {"left": "volume", "operation": "egreater", "right": 500000},
            {"left": "average_volume_30d_calc", "operation": "egreater", "right": 500000},
            {"left": "average_volume_60d_calc", "operation": "egreater", "right": 500000},
            {"left": "close", "operation": "egreater", "right": "SMA200"},
            {"left": "market_cap_basic", "operation": "egreater", "right": 2000000000},
        ],
        # NOTE on the 52-week bands (within 0-20% of the 52w high, >=50% above the
        # 52w low): TradingView's /scan API rejects arithmetic on the RHS
        # (price_52_week_low * 1.5 -> HTTP 400) and exposes no precomputed % field,
        # so those two gates can't live in the server filter. They are enforced
        # below in the parse loop from price_52_week_high / price_52_week_low —
        # the net universe is identical to doing it server-side.
        "columns": [
            "name", "close", "open", "volume", "average_volume_30d_calc",
            "EMA9", "EMA21", "SMA50", "SMA200",
            "ADRP", "market_cap_basic", "Perf.1M", "Perf.3M", "Perf.6M", "Perf.Y",
            "sector", "industry", "high", "low", "change", "price_52_week_high",
            "float_shares_outstanding", "price_52_week_low",
        ],
        "sort": {"sortBy": "ADRP", "sortOrder": "desc"},
        "range": [0, 5000],
    }

    tier_a_plus: List[dict] = []
    tier_a: List[dict] = []
    tier_a_minus: List[dict] = []
    tier_a_minus_full: List[dict] = []   # uncapped A- pool (for win-rate tracking)
    coil_candidates: List[dict] = []

    # --- funnel instrumentation: exact survivor count after each filter stage ---
    n_universe = None          # TradingView totalCount matching the Stage-1 filter
    n_stage1 = 0               # rows actually fetched (range-capped at 5000)
    drop_missing = drop_proximity = drop_52w = drop_dead = 0
    n_aplus_raw = n_a_raw = n_aminus_raw = n_aminus_total = 0

    try:
        time.sleep(2)
        data = tv_post(payload_coil, label="coil", diag=diag)
        _rows = data.get("data", []) or []
        n_stage1 = len(_rows)
        _univ = data.get("totalCount")
        n_universe = _univ if isinstance(_univ, int) else None
        for row in _rows:
            d = row.get("d")
            if not d or len(d) < 23:
                drop_missing += 1
                continue
            (ticker, close, opn, vol, avg_vol, ema9, ema21, sma50, sma200, adr,
             mcap, perf_1m, perf_3m, perf_6m, perf_y, sector, industry,
             high, low, change, high_52w, float_shares, low_52w) = d

            if any(x is None for x in (close, vol, avg_vol, ema9, ema21, sma50,
                                       sma200, adr, perf_1m, perf_3m, high, low,
                                       high_52w, low_52w)):
                drop_missing += 1
                continue
            if opn is None:
                opn = close

            p6 = perf_6m or 0
            py = perf_y or 0
            # (momentum floor removed 2026-06-28 per user — no longer gates the candidate pool)
            # (ADR >=2.0% floor removed 2026-06-30 per user — ADR no longer gates the pool;
            # `adr` is still TradingView's native ADRP and is used for tier discrimination below.)

            # 52-week position band (couldn't go in the server filter — see payload
            # note). Keep names within 0-20% BELOW their 52-week high.
            pct_below_high = ((high_52w - close) / high_52w * 100) if high_52w else None
            if pct_below_high is None or not (0.0 <= pct_below_high <= 20.0):
                drop_52w += 1
                continue

            ma_cluster_pct = abs(ema9 - ema21) / ema21 * 100

            dist_fast = abs(close - ema9) / ema9
            dist_slow = abs(close - ema21) / ema21
            min_dist = min(dist_fast, dist_slow)
            vol_pct = (vol / avg_vol) * 100
            dist_52w = ((high_52w - close) / high_52w) * 100
            day_range = high - low
            day_range_pct = ((high / low) - 1.0) * 100 if low else 0.0   # same basis as ADRP (High/Low)
            is_premium_cluster = ma_cluster_pct <= 2.5

            is_squat = False
            if day_range > 0:
                close_range = (close - low) / day_range
                upper_wick = (high - max(close, opn)) / day_range
                body_pct = abs(close - opn) / day_range
                if body_pct < 0.3 and upper_wick > 0.5 and close_range < 0.35:
                    is_squat = True

            power_score = (perf_1m * 0.4) + (perf_3m * 0.3) + (p6 * 0.2) + (py * 0.1)
            theme = get_theme(ticker, industry)
            hugging_ma = "9EMA" if dist_fast <= dist_slow else "21EMA"
            ma_val = ema9 if hugging_ma == "9EMA" else ema21

            entry_price = round(high + 0.10, 2)
            stop_price = round(min(low, ma_val) - 0.05, 2)
            if stop_price >= entry_price:
                stop_price = round(ma_val - 0.05, 2)
            if stop_price >= entry_price:
                stop_price = round(entry_price * 0.95, 2)
            risk_pct = round(((entry_price - stop_price) / entry_price) * 100, 1)
            stop_reason = "PDL" if abs(stop_price - round(low - 0.05, 2)) < 0.001 else hugging_ma
            # Tight single bar: today's range ≤ the stock's 20-day ADR (real now that `adr`
            # is ADRP, not the old self-referential 1-day value). Grades A vs A- below; it is
            # NOT a universe cut (a stock with one wide bar is still worth watching).
            is_tight_1d = day_range_pct <= adr * 1.0

            # Martin "pullback plan" (additive, a DIFFERENT entry from the
            # breakout): buy on a pullback INTO the rising 9/21 EMA the stock is
            # hugging, with a buffered stop ~3% under the 9 EMA. This is a lower
            # entry than the breakout and gives a tight (~3%) risk by design.
            ma_pull = ema9 if hugging_ma == "9EMA" else ema21
            pb_entry = round(ma_pull, 2)
            pb_stop = round(ema9 * 0.97, 2)
            if pb_stop >= pb_entry:                       # e.g. hugging 21<9 EMA
                pb_stop = round(pb_entry * 0.97, 2)
            pb_risk = (round((pb_entry - pb_stop) / pb_entry * 100, 1)
                       if pb_entry > pb_stop > 0 else None)

            # The volume test is deferred to the enrichment pass: the "today <=
            # 70% of previous day" branch needs daily history, which the cheap
            # TradingView feed doesn't carry. So Gate 2 no longer pre-filters on
            # volume — that way a day-over-day dry-up is never dropped early.
            # Universe gate = near a key MA (ADR floor and the 52w-high distance are
            # handled above). Tightness (is_tight_1d / 3-day flag) is intentionally NOT
            # here — it discriminates the A+/A/A- tiers below, rather than hiding names
            # that had one wide session.
            if not (min_dist <= 0.10):
                drop_proximity += 1
                continue

            status_labels = []
            if is_squat:
                status_labels.append("⚠️ Squat (Wait for tightening)")
            if risk_pct > 6.0:
                status_labels.append("⚠️ Wide Stop (Size Down!)")
            if is_premium_cluster:
                status_labels.append("🛡️ MA Cluster (Premium)")

            coil_candidates.append({
                "ticker": ticker, "close": round(close, 2), "adr": round(adr, 2),
                "perf_1m": round(perf_1m, 2), "perf_3m": perf_3m,
                "perf_6m": round(p6, 2), "perf_12m": round(py, 2),
                "power_score": round(power_score, 1), "hugging": hugging_ma,
                "dist_pct": round(min_dist * 100, 1), "min_dist": min_dist,
                "vol_pct_int": int(vol_pct), "vol_pct_raw": vol_pct,
                "mcap": round((mcap or 0) / 1e9, 2),
                "float_shares_raw": float_shares or 0,
                "float_shares": round((float_shares or 0) / 1e6, 2) if float_shares else 0,
                "sector": sector or "N/A", "theme": theme,
                "entry": entry_price, "stop": stop_price, "risk_pct": risk_pct,
                "stop_reason": stop_reason, "status_labels": status_labels,
                "dist_52w": round(dist_52w, 1),
                "ema9": ema9, "ema21": ema21, "day_range_pct": day_range_pct,
                "is_tight_1d": is_tight_1d, "pb_entry": pb_entry, "pb_stop": pb_stop, "pb_risk": pb_risk,
            })
    except Exception as exc:  # noqa: BLE001
        diag.error(f"Coil scan (fetch/parse): {exc}")

    if coil_candidates:
        hist_map = fetch_histories_batch([c["ticker"] for c in coil_candidates], period="1y")
        for c in coil_candidates:
            hist_df = hist_map.get(c["ticker"])
            # Auto dead-stock screen: drop M&A-pinned / halted / pending-delist names
            # (recent daily range flat-lined). Catches deal pins the ADR floor can't.
            if is_dead_pinned(hist_df):
                drop_dead += 1
                continue
            # ADR is already TradingView's native ADRP (real 20-day ADR%) from the scan row —
            # no history recompute needed. History below is for the sparkline, the 3-day tight
            # flag, the volume dry-up test and the trendline.
            meta_input = {
                "perf_1m": c["perf_1m"], "perf_3m": c["perf_3m"], "adr": c["adr"],
                "close": c["close"], "sma10": c["ema9"], "sma20": c["ema21"],
                "vol_pct": c["vol_pct_raw"], "risk_pct": c["risk_pct"],
                "day_range_pct": c["day_range_pct"], "dist_52w": c["dist_52w"],
                "mcap": c["mcap"], "float_shares": c["float_shares_raw"],
            }
            meta_score_data = calculate_meta_momentum_score(meta_input, hist_df)
            meta_score = _ranking_meta_score(hist_df, meta_score_data["score"], market_modifier)
            c["status_labels"].extend(meta_score_data["badges"])
            trendline_data = calculate_trendline_analysis(c["ticker"], hist_df)

            # Sparkline reuses the history we already downloaded — no extra calls.
            spark = ""
            if hist_df is not None and len(hist_df) >= 2:
                spark = make_price_spark(hist_df["Close"].tolist(), 40)

            # Day-over-day volume dry-up: today's volume <= 70% of the prior
            # session. Needs daily history (the cheap TV feed can't supply it),
            # so it is evaluated here and used as an OR-branch in the A- gate.
            vol_dried_vs_prev = False
            vol_vs_prev_pct = None      # today's volume as % of the PREVIOUS day
            vol_pct_50 = None           # today's volume as % of the 50-day avg volume
            if hist_df is not None and len(hist_df) >= 2:
                today_vol = float(hist_df["Volume"].iloc[-1])
                prev_vol = float(hist_df["Volume"].iloc[-2])
                if prev_vol > 0:
                    vol_vs_prev_pct = today_vol / prev_vol * 100
                    vol_dried_vs_prev = vol_vs_prev_pct <= 65.0
                if len(hist_df) >= 50:
                    vol_50ma = float(hist_df["Volume"].iloc[-50:].mean())
                    if vol_50ma > 0:
                        vol_pct_50 = today_vol / vol_50ma * 100
            if vol_dried_vs_prev and c["vol_pct_raw"] > 55:
                c["status_labels"].append(f"💧 Dry vs Prev Day ({vol_vs_prev_pct:.0f}%)")

            # Martin playbook footprint study (base / higher-lows / coil / AVWAP)
            footprint = analyze_footprint(hist_df)

            # Leader-rollover stats (for the regime "downward momentum" signal)
            below_20dma = below_50dma = False
            days_since_high = None
            if hist_df is not None and len(hist_df) >= 50:
                cl = hist_df["Close"]
                px = float(cl.iloc[-1])
                below_20dma = px < float(cl.iloc[-20:].mean())
                below_50dma = px < float(cl.iloc[-50:].mean())
                recent = cl.iloc[-21:].values
                days_since_high = int(len(recent) - 1 - int(recent.argmax()))

            _rs_val, _rs_asof = resolve_rs(c["ticker"], rs_map)
            stock_data = {
                "ticker": c["ticker"], "close": c["close"], "adr": c["adr"],
                "perf_1m": c["perf_1m"], "perf_6m": c["perf_6m"], "perf_12m": c["perf_12m"],
                "power_score": c["power_score"], "rs_rating": _rs_val, "rs_asof": _rs_asof,
                "hugging": c["hugging"], "dist_pct": c["dist_pct"], "vol_pct": c["vol_pct_int"],
                "mcap": c["mcap"], "float_shares": c["float_shares"], "sector": c["sector"],
                "theme": c["theme"], "entry": c["entry"], "stop": c["stop"], "risk_pct": c["risk_pct"],
                "stop_reason": c["stop_reason"], "status_labels": c["status_labels"],
                "dist_52w": c["dist_52w"], "meta_score": meta_score,
                "meta_details": meta_score_data["details"], "trendline_data": trendline_data,
                "spark": spark, "footprint": footprint,
                "_ma_dist": (_ma_dist_data(hist_df["Close"].tolist())
                             if (hist_df is not None and len(hist_df) >= 2) else None),
                "pb_entry": c["pb_entry"], "pb_stop": c["pb_stop"], "pb_risk": c["pb_risk"], "ema9": c["ema9"],
                "below_20dma": below_20dma, "below_50dma": below_50dma, "days_since_high": days_since_high,
            }

            is_tight_flag_3d = meta_score_data.get("is_flag", False)
            is_tight_1d = c["is_tight_1d"]
            # 2-day tight bar: each of the last 2 daily bars has range% <= ADR
            # (same per-bar basis as is_tight_1d, over the last two sessions).
            is_tight_2d = False
            if hist_df is not None and len(hist_df) >= 2 and c["adr"] > 0:
                last2 = hist_df.tail(2)
                rng2 = (last2["High"] - last2["Low"]) / last2["Close"] * 100
                is_tight_2d = bool(rng2.le(c["adr"]).all())
            min_dist = c["min_dist"]
            vol_pct = c["vol_pct_raw"]
            risk_pct = c["risk_pct"]
            # Volume dry-up gates (risk no longer gates tiers). The averaging window
            # matches each tier's tightness window; "prev-day" = the session right
            # before that window. Each test passes on EITHER the prev-day OR 50-day-avg
            # comparison.
            #   A+ : mean(last 3d vol) <= 50% of the pre-window day OR the 50-day avg.
            #   A  : mean(last 2d vol) <= 55% of the pre-window day OR the 50-day avg.
            #   A- : last 1d vol       <= the previous day          OR the 50-day avg (not expanding).
            vol_aplus  = _vol_window_dryup(hist_df, 3, 50.0)
            vol_a      = _vol_window_dryup(hist_df, 2, 55.0)
            vol_aminus = _vol_window_dryup(hist_df, 1, 100.0)

            is_a_plus = (is_tight_flag_3d and min_dist <= 0.01 and vol_aplus)
            is_a = (is_tight_2d and min_dist <= 0.01 and vol_a and not is_a_plus)
            is_a_minus = (is_tight_1d and min_dist <= 0.02 and vol_aminus
                          and not is_a_plus and not is_a)

            if is_a_plus:
                tier_a_plus.append(stock_data)
            elif is_a:
                tier_a.append(stock_data)
            elif is_a_minus:
                tier_a_minus.append(stock_data)

        # Raw qualifying counts BEFORE any cap (for the diag panel / sizing).
        n_aplus_raw, n_a_raw, n_aminus_raw = len(tier_a_plus), len(tier_a), len(tier_a_minus)

        sorted_aplus = sorted(tier_a_plus, key=lambda x: x["meta_score"], reverse=True)
        tier_a_plus = sorted_aplus[:25]
        aplus_overflow = sorted_aplus[25:]          # was silently DELETED (audit H2)
        for s in aplus_overflow:
            s["status_labels"].append("⭐ Elite Pool (A+ overflow, demoted by score)")
        sorted_a_pool = sorted(tier_a, key=lambda x: x["meta_score"], reverse=True)
        tier_a = sorted_a_pool[:25]
        tier_a_overflow = sorted_a_pool[25:]
        for s in tier_a_overflow:
            s["status_labels"].append("⭐ Elite Pool (Demoted by Score)")
        # A- is the catch-all "extended / messy" bucket. A+ overflow joins the tracked
        # pool too (never silently dropped). Default sort: M.E.T.A. (v4) score high->low,
        # shown UNCAPPED. (_edges is still computed for the per-row edge badges.)
        combined_aminus = tier_a_minus + tier_a_overflow + aplus_overflow
        n_aminus_total = len(combined_aminus)
        for s in combined_aminus:
            s["_edges"] = _edge_count(s)
        tier_a_minus_full = sorted(combined_aminus,
                                   key=lambda x: x["meta_score"], reverse=True)
        tier_a_minus = tier_a_minus_full          # uncapped — show all A-
        log.info("Coil qualifying (pre-cap): A+=%d A=%d A-=%d (+%d demoted from A = %d into A-, A- UNCAPPED)",
                 n_aplus_raw, n_a_raw, n_aminus_raw, len(tier_a_overflow), n_aminus_total)

    funnel = {
        "universe_total": n_universe,        # TradingView count matching the Stage-1 filter
        "stage1_fetched": n_stage1,          # rows fetched (range-capped at 5000)
        "drop_missing": drop_missing,
        "drop_52w": drop_52w,
        "drop_dead": drop_dead,
        "drop_proximity": drop_proximity,
        "stage2_candidates": len(coil_candidates),
        "stage3_aplus": n_aplus_raw,
        "stage3_a": n_a_raw,
        "stage3_aminus": n_aminus_total,
    }
    return tier_a_plus, tier_a, tier_a_minus, tier_a_minus_full, funnel


def scan_htf(rs_map: dict, market_modifier: float, diag: Diagnostics,
             data_date: Optional[str] = None) -> List[dict]:
    """HTF v2.4.1 — High Tight Flag (owner-audited spec, 2026-06-12).

    Universe (point-in-time on the signal bar): cap > $3B, close > $10, day vol
    AND 20/30/60/90-bar avg vol each > 500k, ADR20 > 4%, within 25% of 52wk high,
    RS percentile ≥ 85. Entry = ALL seven gates: 40-bar thrust > 60%; 5-bar flag
    range < 3×ADR20; flag depth < 30% of pole; quiet bars (avg ≤ 1.0×ADR, max ≤
    1.25×ADR); flag volume < 10-bar mean; plus a 5-bar cooldown per name.
    Ticket: buy NEXT session's OPEN (close proxy), hard stop = close × (1 −
    1.5×ADR20) with gap-through filling at the open; exit plan = first close
    ≥ +40% above the 21-EMA, ladder ≤ 3 legs at the trade's peak.
    Returns A+-shaped stock_data dicts (merged into Tier A+ with 🚩 badge)."""
    payload = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": "close", "operation": "greater", "right": HTF_PRICE_MIN},
            {"left": "volume", "operation": "greater", "right": HTF_VOL_MIN},
            {"left": "market_cap_basic", "operation": "greater", "right": HTF_CAP_MIN},
            {"left": "Perf.3M", "operation": "greater", "right": 30},
        ],
        "columns": [
            "name", "close", "open", "volume", "average_volume_30d_calc",
            "ADRP", "market_cap_basic", "Perf.1M", "Perf.3M", "Perf.6M", "Perf.Y",
            "sector", "industry", "high", "low", "price_52_week_high",
            "float_shares_outstanding",
        ],
        "sort": {"sortBy": "Perf.3M", "sortOrder": "desc"},
        "range": [0, 2000],
    }

    meta_map: Dict[str, dict] = {}
    tickers: List[str] = []
    try:
        time.sleep(2)
        data = tv_post(payload, label="htf", diag=diag)
        for row in data.get("data", []):
            d = row.get("d")
            if not d or len(d) < 17:
                continue
            (name, close, opn, vol, avg_vol, adr, mcap, perf_1m, perf_3m, perf_6m,
             perf_y, sector, industry, high, low, high_52w, float_shares) = d
            if any(x is None for x in (close, vol, avg_vol, adr, perf_3m, high, low)):
                continue
            tickers.append(name)
            meta_map[name] = {
                "close": close, "open": opn if opn is not None else close,
                "vol": vol, "avg_vol": avg_vol, "adr": adr,
                "mcap": mcap or 0, "perf_1m": perf_1m or 0, "perf_3m": perf_3m,
                "perf_6m": perf_6m or 0, "perf_y": perf_y or 0,
                "sector": sector or "N/A", "industry": industry or "N/A",
                "high": high, "low": low, "high_52w": high_52w or close,
                "float_shares": float_shares or 0,
            }
    except Exception as exc:  # noqa: BLE001
        diag.error(f"HTF scan (fetch/parse): {exc}")
        return []

    if not tickers:
        return []
    # Cap the heavy history fetch to the top momentum names (HTFs live here).
    # min_rows=95: 90-bar avg-volume universe gate + 5-bar cooldown lookback.
    tickers = tickers[:1200]
    hist_map = fetch_histories_batch(tickers, period="6mo", min_rows=95)

    # FRESHNESS GUARD: if the bulk feed's bars end before the session the rest
    # of the report describes, an HTF "fire" would just be the PRIOR session's
    # setup resurfacing (this happened 2026-06-11: Yahoo's batch endpoint
    # lagged the chart endpoint by a day). Better no fires than phantom fires.
    if data_date and hist_map:
        try:
            batch_asof = max(df.index[-1] for df in hist_map.values()).date().isoformat()
        except Exception:  # noqa: BLE001
            batch_asof = None
        if batch_asof and batch_asof < data_date:
            diag.warn(f"HTF skipped: history bars end {batch_asof} but session is "
                      f"{data_date} (stale bulk feed) — no fires reported")
            return []

    def _entry_gates(H, L, C, V, i):
        """The seven v2.4.1 entry gates evaluated point-in-time at bar i.
        Returns the gate metrics dict, or None if any gate fails."""
        if i < 40 or C[i - 40] <= 0:
            return None
        lo20 = L[i - 19:i + 1]
        if float(lo20.min()) <= 0:
            return None
        adr20 = float(np.mean(H[i - 19:i + 1] / lo20 - 1.0)) * 100      # ADR(20) in %
        if adr20 <= HTF_ADR_MIN:                                        # universe: ADR>4
            return None
        thrust = C[i] / C[i - 40] - 1.0
        if thrust <= HTF_THRUST_MIN:                                    # 1 thrust
            return None
        fb = HTF_FLAG_BARS
        fH, fL, fV = H[i - fb + 1:i + 1], L[i - fb + 1:i + 1], V[i - fb + 1:i + 1]
        flag_lo, flag_hi = float(fL.min()), float(fH.max())
        if flag_lo <= 0:
            return None
        flag_rng = (flag_hi - flag_lo) / flag_lo * 100
        if flag_rng >= HTF_TIGHT_ADR_X * adr20:                         # 2 tight flag
            return None
        bar_rngs = (fH / fL - 1.0) * 100
        if float(np.mean(bar_rngs)) > HTF_QAVG_ADR_X * adr20:           # 3 quiet avg bar
            return None
        if float(np.max(bar_rngs)) > HTF_QMAX_ADR_X * adr20:            # 4 quiet max bar
            return None
        pole_lo = float(L[i - 39:i - fb + 1].min())
        pole_hi = float(H[i - 39:i + 1].max())
        if pole_hi <= pole_lo:
            return None
        depth = (pole_hi - flag_lo) / (pole_hi - pole_lo)
        if depth >= HTF_FLAG_DEPTH_MAX:                                 # 5 shallow flag
            return None
        v10 = float(np.mean(V[i - 9:i + 1])) or 1.0
        vol_contr = float(np.mean(fV)) / v10
        if vol_contr >= 1.0:                                            # 6 vol contraction
            return None
        # gate 7 (RS >= 85) is checked by the caller against the RS cross-section
        return {"adr20": adr20, "thrust": thrust, "flag_rng": flag_rng,
                "depth": depth, "vol_contr": vol_contr}

    fires: List[dict] = []
    n_cooldown = 0
    for t in tickers:
        df = hist_map.get(t)
        if df is None or len(df) < 95:
            continue
        H = df["High"].values.astype(float)
        L = df["Low"].values.astype(float)
        C = df["Close"].values.astype(float)
        V = df["Volume"].values.astype(float)
        n = len(C)
        i = n - 1

        # ---- universe gates (point-in-time on the signal bar) ----------------
        if C[i] <= HTF_PRICE_MIN or V[i] <= HTF_VOL_MIN:
            continue
        if any(float(np.mean(V[i - w + 1:i + 1])) <= HTF_VOL_MIN for w in (20, 30, 60, 90)):
            continue
        m = meta_map.get(t, {})
        if (m.get("mcap") or 0) <= HTF_CAP_MIN:
            continue
        close = round(float(m.get("close", C[i])), 2)
        high_52w = float(m.get("high_52w") or close)
        dist_52w = round(((high_52w - close) / high_52w) * 100, 1) if high_52w > 0 else 0.0
        if dist_52w > HTF_OFF_HI_MAX:                                   # within 25% of hi
            continue
        _htf_rs_val, _htf_rs_asof = resolve_rs(t, rs_map)
        if not (isinstance(_htf_rs_val, int) and _htf_rs_val >= HTF_RS_MIN):   # 7 RS gate
            continue

        # ---- entry gates on the signal bar -----------------------------------
        g = _entry_gates(H, L, C, V, i)
        if not g:
            continue
        # ---- cooldown: suppress if the signal also fired in the last 5 bars --
        if any(_entry_gates(H, L, C, V, j) for j in range(max(40, i - HTF_COOLDOWN), i)):
            n_cooldown += 1
            continue

        adr = round(g["adr20"], 2)            # point-in-time ADR(20), not the TV field
        thrust, flag_rng, depth = g["thrust"], g["flag_rng"], g["depth"]
        vol = float(m.get("vol", 0)); avg_vol = float(m.get("avg_vol", 0)) or 1.0
        vol_pct = vol / avg_vol * 100
        high = float(m.get("high", close)); low = float(m.get("low", close))
        day_range_pct = (high - low) / close * 100 if close else 0.0

        # 9/21 EMA from the history we already have (for the Dist-to-MA read).
        cl = pd.Series(C)
        ema9 = float(cl.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(cl.ewm(span=21, adjust=False).mean().iloc[-1])
        dist_pct = round(abs(close - ema9) / ema9 * 100, 1) if ema9 else 0.0

        # ---- Staged ticket: buy NEXT session's OPEN, hard stop from ADR20 ----
        stop = round(close * (1 - HTF_ADR_STOP_MULT * adr / 100.0), 2)
        entry = close                              # proxy for next-session OPEN
        if entry <= stop:                          # degenerate (huge ADR) — skip
            continue
        risk_pct = round((entry - stop) / entry * 100, 1)
        per_share = entry - stop
        shares = 0
        if per_share > 0 and entry > 0:
            shares = int(min(HTF_RISK_FRAC * HTF_EQUITY / per_share,
                             HTF_MAX_POS_FRAC * HTF_EQUITY / entry))

        # M.E.T.A. score so it sorts sensibly inside Tier A+.
        meta_input = {
            "perf_1m": m.get("perf_1m", 0), "perf_3m": m.get("perf_3m", 0), "adr": adr,
            "close": close, "sma10": ema9, "sma20": ema21,
            "vol_pct": vol_pct, "risk_pct": risk_pct,
            "day_range_pct": day_range_pct, "dist_52w": dist_52w,
            "mcap": (m.get("mcap", 0) or 0) / 1e9, "float_shares": m.get("float_shares", 0),
        }
        meta_score_data = calculate_meta_momentum_score(meta_input, df)
        meta_score = _ranking_meta_score(df, meta_score_data["score"], market_modifier)

        spark = make_price_spark(df["Close"].tolist(), 40) if len(df) >= 2 else ""
        footprint = analyze_footprint(df)
        trendline_data = calculate_trendline_analysis(t, df)
        theme = get_theme(t, m.get("industry", "N/A"))

        below_20dma = below_50dma = False
        days_since_high = None
        if len(df) >= 50:
            px = float(C[-1])
            below_20dma = px < float(cl.iloc[-20:].mean())
            below_50dma = px < float(cl.iloc[-50:].mean())
            recent = C[-21:]
            days_since_high = int(len(recent) - 1 - int(recent.argmax()))

        status_labels = [
            "🚩 HTF v2.4.1 (High Tight Flag)",
            f"⚡ +{thrust * 100:.0f}% / 40d · flag {flag_rng:.1f}% ({flag_rng / adr:.1f}×ADR) · depth {depth * 100:.0f}%",
            f"💧 flag vol {g['vol_contr']:.0%} of 10-bar avg · ADR20 {adr:.1f}%",
            f"📐 Enter NEXT OPEN · stop −{HTF_ADR_STOP_MULT * adr:.1f}% (gap-through fills at open)",
            f"🎯 Exit: 1st close ≥ +{HTF_EXT_X:.0%} vs 21EMA · ladder ≤ {HTF_MAX_LEGS} legs at trade peak",
        ]
        if shares > 0:
            status_labels.append(
                f"📊 Size ≈ {shares} sh (${HTF_EQUITY / 1000:.0f}k · 0.75%R)")
        status_labels.extend(meta_score_data["badges"])

        fires.append({
            "ticker": t, "close": close, "adr": adr,
            "perf_1m": round(float(m.get("perf_1m", 0)), 2),
            "perf_6m": round(float(m.get("perf_6m", 0)), 2),
            "perf_12m": round(float(m.get("perf_y", 0)), 2),
            "power_score": round(thrust * 100, 1),
            "rs_rating": _htf_rs_val, "rs_asof": _htf_rs_asof,
            "hugging": "9EMA", "dist_pct": dist_pct, "vol_pct": int(vol_pct),
            "mcap": round((m.get("mcap", 0) or 0) / 1e9, 2),
            "float_shares": round((m.get("float_shares", 0) or 0) / 1e6, 2) if m.get("float_shares") else 0,
            "sector": m.get("sector", "N/A"), "theme": theme,
            "entry": entry, "stop": stop, "risk_pct": risk_pct,
            "stop_reason": "HTF · −1.5×ADR20", "status_labels": status_labels,
            "dist_52w": dist_52w, "meta_score": meta_score,
            "meta_details": meta_score_data["details"], "trendline_data": trendline_data,
            "spark": spark, "footprint": footprint,
            "pb_entry": None, "pb_stop": None, "pb_risk": None, "ema9": ema9,
            "below_20dma": below_20dma, "below_50dma": below_50dma,
            "days_since_high": days_since_high, "is_htf": True,
        })

    log.info("HTF v2.4.1: %d candidates scanned, %d fires, %d cooldown-suppressed",
             len(tickers), len(fires), n_cooldown)
    fires.sort(key=lambda x: x["meta_score"], reverse=True)
    return fires


def _new_high_recurrence(df: pd.DataFrame) -> dict:
    """Count how persistently a stock has printed fresh 52-week highs, computed
    straight from price history (no log needed). A 'new-high day' = that day's
    high >= the prior-252-session max high. Returns counts over the last 21/63
    sessions plus distinct calendar weeks (of the last ~13) with >=1 new high,
    and a persistence tier ('R' relentless / 'P' persistent / '')."""
    out = {"nh_1m": 0, "nh_3m": 0, "weeks_3m": 0, "tier": "", "label": "", "at_high": False}
    if df is None or len(df) < 40:
        return out
    H = df["High"]
    prior_max = H.rolling(252, min_periods=30).max().shift(1)   # trailing high, excl. today
    is_nh = (H >= prior_max) & prior_max.notna()
    out["at_high"] = bool(is_nh.iloc[-1])                       # printing a fresh high today
    out["nh_1m"] = int(is_nh.iloc[-21:].sum())
    win = is_nh.iloc[-63:]
    out["nh_3m"] = int(win.sum())
    weeks = set()
    for dt, flag in zip(df.index[-63:], win.values):
        if flag:
            try:
                iso = dt.isocalendar()
                weeks.add((iso[0], iso[1]))
            except Exception:  # noqa: BLE001
                pass
    out["weeks_3m"] = len(weeks)
    if out["weeks_3m"] >= NH_RELENTLESS_WEEKS:
        out["tier"], out["label"] = "R", "Relentless Leader"
    elif out["weeks_3m"] >= NH_PERSIST_WEEKS:
        out["tier"], out["label"] = "P", "Persistent Leader"
    return out


def attach_persistence(stocks: List[dict], diag: Optional[Diagnostics] = None) -> None:
    """Tag each coil A-list stock with its 52wk-high persistence (recurring new
    highs over the last 13 weeks) + whether it's printing a fresh high today.
    One 2y-history batch over the displayed tier members; computed from price."""
    tickers = sorted({s["ticker"] for s in stocks})
    if not tickers:
        return
    hist = fetch_histories_batch(tickers, period="2y", min_rows=60)
    for s in stocks:
        rec = _new_high_recurrence(hist.get(s["ticker"]))
        s["persist_tier"] = rec["tier"]
        s["persist_label"] = rec["label"]
        s["nh_1m"] = rec["nh_1m"]
        s["nh_3m"] = rec["nh_3m"]
        s["weeks_3m"] = rec["weeks_3m"]
        s["at_high"] = rec["at_high"]


def compute_ants(stock_df, spy_close, *, lookback=ANTS_LOOKBACK, min_up=ANTS_MIN_UP,
                 vol_pct=ANTS_VOL_PCT, price_pct=ANTS_PRICE_PCT, use_trend=ANTS_USE_TREND,
                 use_rs=ANTS_USE_RS, count_full_only=ANTS_COUNT_FULL_ONLY,
                 rs_fast=ANTS_RS_FAST, rs_slow=ANTS_RS_SLOW,
                 chain_window=ANTS_CHAIN_WINDOW):
    """David Ryan's ANTS accumulation read, point-in-time from daily OHLCV (+ a
    benchmark close series, SPY). Returns a FIXED-shape dict (callers never
    KeyError): level 0-5 (NONE/MOM/MOM+VOL/MOM+PR/FULL/ELITE), chain (trailing
    consecutive bars the ANTS condition held), label, up_count, vol_gain,
    price_gain, rs_rising, stronger, ok. NaN-safe — a missing leg can only
    SUPPRESS a level, never fabricate one. spy_close=None => levels 1-4 still
    compute, ELITE impossible. Mirrors the owner's Pine 1:1. Display-only."""
    if stock_df is None or len(stock_df) < 2 * lookback + 2:
        return dict(_ANTS_EMPTY)
    try:
        close = stock_df["Close"].astype(float)
        vol = stock_df["Volume"].astype(float)
    except Exception:  # noqa: BLE001
        return dict(_ANTS_EMPTY)

    # momentum: up-day count over the window (classic: close > prior close)
    up_day = close > close.shift(1)
    up_count_s = up_day.rolling(lookback).sum()
    momentum_ok = up_count_s >= min_up

    # volume gain: SMA(lookback) now vs the same SMA one window ago
    vol_now = vol.rolling(lookback).mean()
    vol_prev = vol_now.shift(lookback)
    vol_gain_s = (vol_now - vol_prev) / vol_prev.replace(0, np.nan)
    vol_ok = vol_gain_s >= vol_pct

    # price gain over the window (+ optional SMA10>SMA20 trend filter)
    price_start = close.shift(lookback)
    price_gain_s = (close - price_start) / price_start.replace(0, np.nan)
    if use_trend:
        trend_ok = close.rolling(10).mean() > close.rolling(20).mean()
    else:
        trend_ok = pd.Series(True, index=close.index)
    price_ok = (price_gain_s >= price_pct) & trend_ok

    # level ladder (mutually exclusive) via np.select -> 4/3/2/1/0
    full = momentum_ok & price_ok & vol_ok
    mom_pr = momentum_ok & price_ok & ~vol_ok
    mom_vol = momentum_ok & ~price_ok & vol_ok
    mom = momentum_ok & ~price_ok & ~vol_ok
    level_s = pd.Series(np.select([full, mom_pr, mom_vol, mom], [4, 3, 2, 1], default=0),
                        index=close.index)

    # relative strength: rs_line = close / benchmark. Drives the ELITE upgrade AND
    # the report's RS-Line read (trend vs SPY + RS-new-high incl. "before price").
    # Only when a benchmark series is available + alignable.
    rs_rising_s = pd.Series(False, index=close.index)
    stronger_s = pd.Series(False, index=close.index)
    rs_line_rising = rs_new_high = rs_nh_before_price = rs_nh_3m = False
    rs_spark_vals = []
    if spy_close is not None:
        spy = spy_close.reindex(close.index)
        rs_line = close / spy.replace(0, np.nan)
        rs_rising_s = rs_line > rs_line.rolling(rs_fast).mean()
        stronger_s = rs_line > rs_line.rolling(rs_slow).mean()
        rs_lb = int(min(len(close), 252))
        rs_hi = rs_line.rolling(rs_lb, min_periods=20).max()
        # "leader" = RS line at/near its 1-year high. Strict new-highs are too sparse
        # for coil/pullback setups (which are by definition off their highs), so we
        # use a small band — this is the genuine standout signal vs the market.
        rs_new_high = bool(rs_line.iloc[-1] >= ANTS_RS_HIGH_FRAC * rs_hi.iloc[-1])
        rs_nh_3m = bool((rs_line >= rs_hi).iloc[-63:].any())   # strict RS new high in last ~3 months
        rs_line_rising = bool(rs_line.iloc[-1] > rs_line.ewm(span=21, adjust=False).mean().iloc[-1])
        high = stock_df["High"].astype(float)              # RS leads while PRICE lags = stealth leader
        px_hi = float(high.rolling(rs_lb, min_periods=20).max().iloc[-1])
        rs_nh_before_price = bool(rs_new_high and px_hi > 0 and high.iloc[-1] < ANTS_PX_LAG_FRAC * px_hi)
        rs_spark_vals = [float(x) for x in rs_line.iloc[-40:].tolist() if x == x]
    is_elite = rs_rising_s if use_rs else pd.Series(False, index=close.index)
    level_s = level_s.mask((level_s >= 4) & is_elite, 5)   # ELITE only upgrades a FULL bar

    # ant_chain: trailing consecutive run of the chain condition ending at today
    chain_cond = (level_s >= 4) if count_full_only else (level_s > 0)
    tail = chain_cond.iloc[-chain_window:].to_numpy()
    chain = 0
    for v in tail[::-1]:
        if v:
            chain += 1
        else:
            break

    # 3-month accumulation history: prior ANTS over the last ~63 bars is a good
    # sign even when today is quiet (peak level reached + how many active days).
    last3m = level_s.iloc[-63:]
    ants_3m_peak = int(last3m.max()) if len(last3m) else 0
    ants_3m_days = int((last3m > 0).sum())

    def _f(x):
        try:
            xf = float(x)
            return 0.0 if xf != xf else xf       # NaN guard
        except Exception:  # noqa: BLE001
            return 0.0

    lvl = int(level_s.iloc[-1])
    uc = up_count_s.iloc[-1]
    return {
        "level": lvl,
        "chain": int(chain),
        "label": _ANTS_LABELS.get(lvl, "NONE"),
        "up_count": int(uc) if uc == uc else 0,
        "vol_gain": _f(vol_gain_s.iloc[-1]),
        "price_gain": _f(price_gain_s.iloc[-1]),
        "rs_rising": bool(rs_rising_s.iloc[-1]),
        "stronger": bool(stronger_s.iloc[-1]),
        "ok": True,
        "ants_3m_peak": ants_3m_peak,
        "ants_3m_days": ants_3m_days,
        "rs_line_rising": rs_line_rising,
        "rs_new_high": rs_new_high,
        "rs_nh_before_price": rs_nh_before_price,
        "rs_nh_3m": rs_nh_3m,
        "rs_spark_vals": rs_spark_vals,
    }


def attach_ants(stocks: List[dict], diag: Optional[Diagnostics] = None,
                hist: Optional[Dict[str, pd.DataFrame]] = None, spy_close=None) -> None:
    """Post-scan pass: tag each coil A-list stock with its ANTS read
    (ants_level / ants_chain / ants_label / ants_ok / ants_rs_rising). Mirrors
    attach_persistence — one 2y-history batch (+ SPY) over the displayed tier
    members. Decision-support only; never touches the IBKR draft plan."""
    if not stocks:
        return
    if hist is None:
        tickers = sorted({s["ticker"] for s in stocks} | {ANTS_BENCHMARK})
        hist = fetch_histories_batch(tickers, period="2y", min_rows=60)
    if spy_close is None:
        spy_df = hist.get(ANTS_BENCHMARK)
        spy_close = spy_df["Close"] if spy_df is not None else None
    for s in stocks:
        a = compute_ants(hist.get(s["ticker"]), spy_close)
        s["ants_level"] = a["level"]
        s["ants_chain"] = a["chain"]
        s["ants_label"] = a["label"]
        s["ants_ok"] = a["ok"]
        s["ants_rs_rising"] = a["rs_rising"]
        s["ants_3m_peak"] = a["ants_3m_peak"]
        s["ants_3m_days"] = a["ants_3m_days"]
        s["rs_new_high"] = a["rs_new_high"]
        s["rs_nh_before_price"] = a["rs_nh_before_price"]
        s["rs_ok"] = bool(a["rs_spark_vals"])   # RS line was computable (SPY aligned)


def _classify_new_high(fp: dict, ext50: Optional[float], p3m: float) -> Tuple[str, str]:
    """Grade a new-high name by its prior-3-month footprint.
    Returns (tag, label) with tag in {GRN, YEL, RED}."""
    ext9 = fp.get("ext9")
    bw = fp.get("base_weeks", 0.0)
    bd = fp.get("base_depth")
    hl = fp.get("higher_lows", 0)
    # ext50 was computed and passed but never read — a name far above its 50-EMA
    # is chase-risk even if its 9-EMA extension is modest (audit).
    if (fp.get("extended") or (ext9 is not None and ext9 >= 15)
            or (ext50 is not None and ext50 >= 30) or p3m >= 80):
        return "RED", "Extended / Climax (chase risk)"
    if bw >= 3 and (bd if bd is not None else 99) <= 25 and p3m < 60:
        return "GRN", "Base breakout (tight)"
    if (ext9 is not None and ext9 < 10) and hl >= 1 and fp.get("coiled"):
        return "GRN", "Steady Stage-2 (coiled)"
    if (ext9 is not None and ext9 < 12) and hl >= 1:
        return "GRN", "Steady Stage-2 trend"
    return "YEL", "Mixed / developing"


def scan_new_highs(rs_map: dict, market_modifier: float, diag: Diagnostics) -> dict:
    """Daily NEW 52-week-high leaders (ADR>3), graded by their prior-3-month
    pattern. Returns the constructive (🟢) breakouts plus the sector-cluster
    breadth read (sectors making collective new highs — the J Law leadership tell).
    Returns {green: [...], total: int, clusters: [(sector, n), ...]}."""
    payload = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": "close", "operation": "greater", "right": 10},
            {"left": "average_volume_30d_calc", "operation": "greater", "right": 500000},
            {"left": "ADRP", "operation": "greater", "right": 3},
            {"left": "high", "operation": "egreater", "right": "price_52_week_high"},
        ],
        "columns": ["name", "close", "high", "low", "ADRP", "Perf.1M", "Perf.3M",
                    "market_cap_basic", "sector", "industry", "price_52_week_high"],
        "sort": {"sortBy": "Perf.3M", "sortOrder": "desc"},
        "range": [0, 500],
    }
    meta_map: dict = {}
    tickers: List[str] = []
    try:
        time.sleep(2)
        data = tv_post(payload, label="new_highs", diag=diag)
        for row in data.get("data", []):
            d = row.get("d")
            if not d or len(d) < 11:
                continue
            (name, close, high, low, adr, p1m, p3m, mcap, sector, industry, hi52) = d
            if None in (close, adr, p3m, high):
                continue
            tickers.append(name)
            meta_map[name] = {"close": close, "high": high, "low": low, "adr": adr,
                              "p1m": p1m or 0, "p3m": p3m, "mcap": (mcap or 0) / 1e9,
                              "sector": sector or "N/A", "industry": industry or "N/A"}
    except Exception as exc:  # noqa: BLE001
        diag.error(f"New-highs scan (fetch/parse): {exc}")
        return {"green": [], "total": 0, "clusters": []}

    if not tickers:
        return {"green": [], "total": 0, "clusters": []}
    # 2y history so each of the last ~63 sessions has a full 252-bar 52wk lookback
    # for the persistence (recurring-new-high) computation.
    hist_map = fetch_histories_batch(tickers, period="2y", min_rows=60)

    rows: List[dict] = []
    confirmed: List[str] = []                 # ALL genuine new-high names (for daily monitor)
    sector_counts: Dict[str, int] = {}
    total = n_green = n_persist = 0
    for t in tickers:
        df = hist_map.get(t)
        if df is None or len(df) < 60:
            continue
        if is_dead_pinned(df):                  # auto dead-stock screen (M&A pins etc.)
            continue
        H = df["High"].values
        prior = H[-252:-1] if len(H) > 252 else H[:-1]
        if len(prior) == 0 or H[-1] < float(prior.max()):   # confirm genuine new high
            continue
        total += 1
        confirmed.append(t)
        info = meta_map[t]
        sector_counts[info["sector"]] = sector_counts.get(info["sector"], 0) + 1

        fp = analyze_footprint(df)
        cl = df["Close"]
        e21 = _ema_last(cl, 21)
        e50 = _ema_last(cl, 50)
        ext50 = round((info["close"] - e50) / e50 * 100, 1) if e50 else None
        tag, label = _classify_new_high(fp, ext50, info["p3m"])
        rec = _new_high_recurrence(df)            # persistence (computed from price)

        if tag == "GRN":
            n_green += 1
        if rec["tier"]:
            n_persist += 1
        # Show the row if it's a constructive breakout OR a proven persistent
        # leader (persistence is itself the bullish tell, even when extended).
        if tag != "GRN" and not rec["tier"]:
            continue

        close = round(float(info["close"]), 2)
        adr = round(float(info["adr"]), 2)        # TradingView native ADRP (canonical 20-day ADR%)
        high = float(info["high"]); low = float(info["low"])
        meta_in = {"perf_1m": info["p1m"], "perf_3m": info["p3m"], "adr": adr,
                   "close": close, "sma10": _ema_last(cl, 9), "sma20": e21,
                   "vol_pct": 100, "risk_pct": 5, "day_range_pct": adr,
                   "dist_52w": 0.0, "mcap": info["mcap"], "float_shares": 0}
        ms = calculate_meta_momentum_score(meta_in, df)
        entry = round(high + 0.10, 2)
        stop = round(min(low, e21) - 0.05, 2)
        if stop >= entry:
            stop = round(entry * (1 - 1.5 * adr / 100), 2)
        risk_pct = round((entry - stop) / entry * 100, 1) if entry > stop > 0 else None
        rs = rs_map.get(t.upper())
        rows.append({
            "ticker": t, "close": close, "adr": adr, "perf_1m": round(info["p1m"], 1),
            "perf_3m": round(info["p3m"], 1), "base_weeks": fp.get("base_weeks", 0.0),
            "base_depth": fp.get("base_depth"), "higher_lows": fp.get("higher_lows", 0),
            "ext9": fp.get("ext9"), "ext50": ext50, "meta_score": _ranking_meta_score(df, ms["score"], 1.0),
            "rs_rating": rs if isinstance(rs, int) else "N/A", "sector": info["sector"],
            "theme": get_theme(t, info["industry"]), "tag": tag, "label": label,
            "entry": entry, "stop": stop, "risk_pct": risk_pct,
            "spark": make_price_spark(cl.tolist(), 60) if len(cl) >= 2 else "",
            "_ma_dist": _ma_dist_data(cl.tolist()),
            "fp_badges": fp.get("badges", []),
            "persist_tier": rec["tier"], "persist_label": rec["label"],
            "nh_1m": rec["nh_1m"], "nh_3m": rec["nh_3m"], "weeks_3m": rec["weeks_3m"],
        })

    # Persistent leaders first (relentless > persistent > none), then by M.E.T.A.
    p_rank = {"R": 2, "P": 1, "": 0}
    rows.sort(key=lambda x: (p_rank.get(x["persist_tier"], 0), x["meta_score"]), reverse=True)
    clusters = sorted(((s, n) for s, n in sector_counts.items() if n >= 3 and s != "N/A"),
                      key=lambda x: -x[1])
    log.info("New 52wk highs: %d confirmed, %d constructive(🟢), %d persistent(⭐), %d clusters(>=3)",
             total, n_green, n_persist, len(clusters))
    return {"green": rows, "total": total, "clusters": clusters, "confirmed": confirmed}


# ----------------------------------------------------------------------------
# 52-WEEK-HIGH DAILY MONITOR — once a name prints a new 52wk high we watch it for
# NH52_WATCH_DAYS trading days. The tell we want surfaced: a LOW-VOLUME pullback
# (price slips below its 50-day MA OR below the prior close, while today's volume
# is below its 30-day average). Low-volume pullback = supply drying up = healthy
# continuation watch. The same scan also flags HIGH-volume breakdowns (distribution
# warning) so the monitor reads both sides.
# ----------------------------------------------------------------------------
def load_nh52_history() -> dict:
    if os.path.exists(NH52_HISTORY_PATH):
        try:
            with open(NH52_HISTORY_PATH, "r") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_nh52_history(history: dict) -> None:
    _atomic_write(NH52_HISTORY_PATH, json.dumps(history))


def _busday_age(from_date: Optional[str], to_date: str) -> int:
    """Trading-day count between two ISO dates (0 if same/missing/unparseable)."""
    if not from_date:
        return 0
    try:
        return int(np.busday_count(np.datetime64(from_date), np.datetime64(to_date)))
    except Exception:  # noqa: BLE001
        return 0


def record_nh52_highs(history: dict, tickers: List[str], data_date: str) -> dict:
    """Add/refresh today's confirmed new-high names. high_count only bumps once
    per data date so intraday re-runs don't inflate the recurrence tally."""
    for t in tickers:
        key = t.upper()
        e = history.get(key) or {"first_seen": data_date, "high_count": 0}
        if e.get("last_high") != data_date:
            e["high_count"] = int(e.get("high_count", 0)) + 1
        e["last_high"] = data_date
        history[key] = e
    return history


def cleanup_old_nh52(history: dict, days: int, data_date: str) -> dict:
    """Drop names whose last 52wk high is older than `days` trading days."""
    return {k: v for k, v in history.items()
            if _busday_age(v.get("last_high"), data_date) <= days}


def scan_nh52_pullbacks(history: dict, rs_map: dict, diag: Diagnostics,
                        data_date: str) -> Tuple[List[dict], List[dict]]:
    """Re-check every monitored 52wk-high name against today's bar.
    Returns (pullback_flags, all_monitored) — both lists of display rows.
    pullback_flags = the LOW-VOLUME pullbacks (the awareness signal)."""
    tickers = list(history.keys())
    if not tickers:
        return [], []
    try:
        hist_map = fetch_histories_batch(tickers, period="6mo", min_rows=50)
    except Exception as exc:  # noqa: BLE001
        diag.error(f"52wk monitor (history fetch): {exc}")
        return [], []

    monitored: List[dict] = []
    for t in tickers:
        df = hist_map.get(t)
        if df is None or len(df) < 50:
            continue
        cl = df["Close"]; vol = df["Volume"]
        close = float(cl.iloc[-1]); prev = float(cl.iloc[-2])
        sma50 = float(cl.iloc[-50:].mean())
        vol_today = float(vol.iloc[-1])
        avg_vol30 = float(vol.iloc[-30:].mean())
        below_50 = sma50 > 0 and close < sma50
        below_prev = close < prev
        low_vol = avg_vol30 > 0 and vol_today < avg_vol30
        pullback = below_50 or below_prev
        vol_ratio = round(vol_today / avg_vol30, 2) if avg_vol30 > 0 else None
        vs_50 = round((close - sma50) / sma50 * 100, 1) if sma50 > 0 else None
        vs_prev = round((close - prev) / prev * 100, 1) if prev > 0 else None
        e = history[t]
        if pullback and low_vol:
            status, tag = "🟢 Low-vol pullback", "GRN"
        elif pullback and not low_vol:
            status, tag = "🔴 High-vol breakdown", "RED"
        else:
            status, tag = "⚪ Holding / extending", "HOLD"
        rs = rs_map.get(t.upper())
        monitored.append({
            "ticker": t, "close": round(close, 2), "sma50": round(sma50, 2),
            "vs_50": vs_50, "vs_prev": vs_prev, "below_50": below_50,
            "below_prev": below_prev, "low_vol": low_vol, "vol_ratio": vol_ratio,
            "days_since_high": _busday_age(e.get("last_high"), data_date),
            "last_high": e.get("last_high"), "high_count": int(e.get("high_count", 0)),
            "rs_rating": rs if isinstance(rs, int) else "N/A",
            "status": status, "tag": tag,
            "spark": make_price_spark(cl.tolist(), 60) if len(cl) >= 2 else "",
        })

    order = {"GRN": 0, "RED": 1, "HOLD": 2}
    monitored.sort(key=lambda x: (order.get(x["tag"], 9), x["days_since_high"]))
    pullbacks = [m for m in monitored if m["tag"] == "GRN"]
    log.info("52wk monitor: %d watched, %d low-vol pullbacks, %d high-vol breakdowns",
             len(monitored), len(pullbacks), sum(1 for m in monitored if m["tag"] == "RED"))
    return pullbacks, monitored


def scan_hve(hve_history: dict, diag: Diagnostics) -> List[dict]:
    """HVE / Episodic Pivot scan (low-float <= 200M)."""
    payload_ep = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": "close", "operation": "greater", "right": 7},
            {"left": "volume", "operation": "greater", "right": 1000000},
            {"left": "change", "operation": "greater", "right": 4},
        ],
        "columns": [
            "name", "close", "open", "high", "low", "volume", "average_volume_30d_calc",
            "change", "gap", "market_cap_basic", "sector", "industry", "float_shares_outstanding",
        ],
        "sort": {"sortBy": "change", "sortOrder": "desc"},
        "range": [0, 2000],
    }

    ep_matches: List[dict] = []
    try:
        time.sleep(2)
        data = tv_post(payload_ep, label="hve", diag=diag)
        for row in data.get("data", []):
            d = row.get("d")
            if not d or len(d) < 13:
                continue
            ticker, close, opn, high, low, vol, avg_vol, change, gap, mcap, sector, industry, float_shares = d
            if any(x is None for x in (close, opn, high, low, vol, avg_vol, change)):
                continue
            if not avg_vol or avg_vol <= 0:               # fresh listing → ZeroDivision would
                continue                                  # abort the whole loop (audit)
            if float_shares and float_shares > 200e6:
                continue

            rel_vol = vol / avg_vol
            if rel_vol < 2.7:
                continue
            day_range = high - low
            if day_range == 0:
                continue
            close_range = (close - low) / day_range
            if close_range < 0.75:
                continue

            theme = get_theme(ticker, industry)
            entry_price = round(high + 0.10, 2)
            stop_price = round(low - 0.05, 2)
            risk_pct = round(((entry_price - stop_price) / entry_price) * 100, 1)

            ep_matches.append({
                "ticker": ticker, "close": round(close, 2), "change": round(change, 2),
                "gap": round(gap or 0, 2), "rel_vol": round(rel_vol, 1),
                "close_range": int(close_range * 100), "mcap": round((mcap or 0) / 1e9, 2),
                "float_shares": round(float_shares / 1e6, 2) if float_shares else 0,
                "sector": sector or "N/A", "theme": theme,
                "entry": entry_price, "stop": stop_price, "risk_pct": risk_pct,
                "stop_reason": "Day 1 Low",
                "day1_volume": vol, "day1_high": high, "day1_low": low,
            })
            hve_history[ticker] = {
                "date": datetime.now().isoformat(), "day1_volume": vol, "day1_high": high,
                "day1_low": low, "day1_close": close, "rel_vol": rel_vol, "change": change,
            }
        ep_matches.sort(key=lambda x: x["rel_vol"], reverse=True)
    except Exception as exc:  # noqa: BLE001
        diag.error(f"HVE scan: {exc}")
    return ep_matches


def scan_ur(hve_history: dict, ep_matches: List[dict], diag: Diagnostics) -> List[dict]:
    """Post-HVE Undercut & Rally scan."""
    ur_matches: List[dict] = []
    ep_tickers = {ep["ticker"] for ep in ep_matches}
    try:
        for hve_ticker, hve_data in list(hve_history.items()):
            if hve_ticker in ep_tickers:
                continue
            event_date = datetime.fromisoformat(hve_data["date"])
            days_since = (datetime.now() - event_date).days
            try:
                trading_days = int(np.busday_count(
                    event_date.strftime("%Y-%m-%d"), datetime.now().strftime("%Y-%m-%d")))
            except Exception:  # noqa: BLE001
                trading_days = days_since
            days_since_hve = trading_days + 1
            if days_since_hve < 2 or days_since_hve > 5:
                continue

            try:
                payload_ur = {
                    "filter": [{"left": "name", "operation": "equal", "right": hve_ticker}],
                    "columns": ["name", "close", "open", "high", "low", "volume", "change",
                                "market_cap_basic", "sector", "industry"],
                    "range": [0, 1],
                }
                time.sleep(0.3)
                data = tv_post(payload_ur, label=f"ur:{hve_ticker}", diag=diag)
                if not data.get("data"):
                    continue
                d = data["data"][0]["d"]
                _t, close, opn, high, low, vol, change, mcap, sector, industry = d
                if any(x is None for x in (close, high, low, vol)):
                    continue

                vol_contraction = (vol / hve_data["day1_volume"]) * 100
                if vol_contraction > 60:
                    continue
                day1_low = hve_data["day1_low"]
                holding = close > day1_low * 0.95

                theme = get_theme(hve_ticker, industry)
                entry_price = round(high + 0.05, 2)
                stop_price = round(low - 0.05, 2)
                risk_pct = round(((entry_price - stop_price) / entry_price) * 100, 1)
                if risk_pct > 5.0:
                    continue

                ur_matches.append({
                    "ticker": hve_ticker, "close": round(close, 2), "change": round(change or 0, 2),
                    "days_since_hve": days_since_hve, "vol_contraction": round(vol_contraction, 1),
                    "day1_high": hve_data["day1_high"], "day1_low": day1_low,
                    "mcap": round((mcap or 0) / 1e9, 2), "sector": sector or "N/A", "theme": theme,
                    "entry": entry_price, "stop": stop_price, "risk_pct": risk_pct,
                    "stop_reason": f"Day {days_since_hve} Wick", "holding_above_low": holding,
                })
            except Exception:  # noqa: BLE001
                continue
        ur_matches.sort(key=lambda x: (x["vol_contraction"], x["days_since_hve"]))
    except Exception as exc:  # noqa: BLE001
        diag.error(f"U&R scan: {exc}")
    return ur_matches


# ----------------------------------------------------------------------------
# SCAN 4: PARABOLIC SHORT  (Martin's specialty short — climax / exhaustion)
#
# Per the playbook + the reference scanner's detect_parabolic_short: a genuine
# parabolic move — price far above the 9 EMA, acceleration increasing, volume
# expanding, multiple recent gap-ups. The actual entry is intraday (break of the
# opening-range low / AVWAP retest) with a tiny stop; here we surface the daily
# candidate and show the intraday plan. SHORT side stays small and fast.
# ----------------------------------------------------------------------------
def _parabolic_short_signal(hist_df: Optional[pd.DataFrame], ticker: str,
                            theme: str, sector: str, perf_1m: float) -> Optional[dict]:
    if hist_df is None or len(hist_df) < 60:
        return None
    close, high, low, vol = hist_df["Close"], hist_df["High"], hist_df["Low"], hist_df["Volume"]
    px = float(close.iloc[-1])
    e9 = float(close.ewm(span=9, adjust=False).mean().iloc[-1])
    e21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    if e9 <= 0 or e21 <= 0:
        return None
    dist9 = (px - e9) / e9
    dist21 = (px - e21) / e21
    if dist9 < 0.15 or dist21 < 0.20:          # must be genuinely extended
        return None

    ret_recent = float(close.iloc[-1] / close.iloc[-6] - 1.0)
    ret_prior = float(close.iloc[-6] / close.iloc[-11] - 1.0)
    if not (ret_recent > ret_prior > 0):       # acceleration increasing
        return None

    avg_vol = float(vol.iloc[-20:].mean())
    vol_ratio = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 0.0
    if vol_ratio < 1.5:
        return None

    opens = hist_df["Open"].iloc[-4:].values
    prev_close = close.shift(1).iloc[-4:].values
    gap_ups = int(np.sum(opens / prev_close - 1.0 > 0.02))

    hi, lo = float(high.iloc[-1]), float(low.iloc[-1])
    entry = round(lo * 0.999, 2)               # daily proxy for ORL-break short
    stop = round(hi * 1.001, 2)                # above day high / ORH
    risk_pct = round((stop - entry) / entry * 100.0, 1) if entry > 0 else None
    target = round(e21, 2)                       # cover toward the 21 EMA
    to_target = round((entry - target) / entry * 100.0, 1) if entry > 0 else None

    return {
        "ticker": ticker, "close": round(px, 2), "dist9": round(dist9 * 100, 1),
        "dist21": round(dist21 * 100, 1), "vol_ratio": round(vol_ratio, 1),
        "gap_ups": gap_ups, "accel": round((ret_recent - ret_prior) * 100, 1),
        "perf_1m": round(perf_1m, 1), "theme": theme, "sector": sector,
        "entry": entry, "stop": stop, "risk_pct": risk_pct,
        "target": target, "to_target": to_target,
    }


def scan_parabolic_short(diag: Diagnostics) -> List[dict]:
    payload = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": "close", "operation": "greater", "right": 7},
            {"left": "volume", "operation": "greater", "right": 1000000},
            {"left": "Perf.1M", "operation": "greater", "right": 30},
        ],
        "columns": ["name", "close", "EMA9", "Perf.1M", "industry", "sector"],
        "sort": {"sortBy": "Perf.1M", "sortOrder": "desc"},
        "range": [0, 500],
    }
    candidates: List[tuple] = []
    try:
        time.sleep(2)
        data = tv_post(payload, label="parabolic", diag=diag)
        for row in data.get("data", []):
            d = row.get("d")
            if not d or len(d) < 6:
                continue
            tk, close, ema9, perf_1m, industry, sector = d
            if close is None or ema9 is None or ema9 <= 0:
                continue
            if (close - ema9) / ema9 < 0.12:       # cheap prune; refine w/ history
                continue
            candidates.append((tk, get_theme(tk, industry), sector or "N/A", perf_1m or 0))
    except Exception as exc:  # noqa: BLE001
        diag.error(f"Parabolic-short scan: {exc}")

    if not candidates:
        return []

    tickers = [c[0] for c in candidates][:60]
    meta = {c[0]: c for c in candidates}
    hist_map = fetch_histories_batch(tickers, period="6mo", min_rows=60)
    shorts: List[dict] = []
    for tk in tickers:
        _, theme, sector, perf_1m = meta[tk]
        hd = hist_map.get(tk)
        if is_dead_pinned(hd):                  # auto dead-stock screen (deal-jump pins)
            continue
        sig = _parabolic_short_signal(hd, tk, theme, sector, perf_1m)
        if sig:
            shorts.append(sig)
    shorts.sort(key=lambda x: x["dist9"], reverse=True)
    return shorts[:15]


# ----------------------------------------------------------------------------
# REPORT BUILDING
# ----------------------------------------------------------------------------
PAGE_CSS = """
    /* ---- design tokens (taste-skill pass 2026-06) ---- */
    :root {
      --bg:#0d1117; --surface:#161b22; --raised:#21262d; --line:#30363d;
      --text:#c9d1d9; --text-2:#a8b2bc; --text-3:#8b949e;
      --accent:#58a6ff; --accent-2:#79c0ff; --green:#3fb950; --yellow:#f2cc60; --red:#ff7b72;
      --bd-green:#2ea043; --bd-yellow:#9e8420; --bd-red:#da3633; --bd-accent:#1f6feb;
      --tint-green:rgba(63,185,80,.10); --tint-yellow:rgba(242,204,96,.10);
      --tint-red:rgba(218,54,51,.10); --tint-accent:rgba(88,166,255,.10);
      --mono:ui-monospace,'SF Mono','Cascadia Mono',Menlo,Consolas,monospace;
      --fs-micro:0.6875rem; --fs-caption:0.75rem; --fs-table:0.8125rem;
      --fs-body:0.875rem; --fs-title:1rem; --fs-h1:1.375rem;
    }
    body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,
                       'PingFang TC','Hiragino Sans TC','Microsoft JhengHei','Noto Sans TC',sans-serif;
           background-color:var(--bg); color:var(--text); margin:10px auto; max-width:1400px; padding:0 4px;
           line-height:1.5; font-variant-numeric:tabular-nums; -webkit-text-size-adjust:100%; }
    h1 { color:var(--text); text-align:center; font-size:var(--fs-h1); text-transform:uppercase; letter-spacing:1px; margin-bottom:5px; }
    .header-sub { text-align:center; color:var(--text-3); font-size:var(--fs-caption); margin-top:0; margin-bottom:14px; }

    .runbar { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:0 0 24px; }
    .chip { background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:5px 12px; font-size:var(--fs-caption); color:var(--text); }
    .chip b { color:var(--accent); font-family:var(--mono); }
    .chip.green b { color:var(--green); } .chip.red b { color:var(--red); } .chip.warn b { color:var(--yellow); }
    .chip.telemetry { opacity:0.65; }

    #search { display:block; width:100%; max-width:420px; margin:0 auto 24px; padding:10px 14px; border-radius:999px;
              border:1px solid var(--line); background:var(--surface); color:var(--text); font-size:16px; box-sizing:border-box;
              position:sticky; top:8px; z-index:10; }
    #search:focus { outline:none; border-color:var(--accent); }

    .market-panel { background-color:var(--surface); border-radius:8px; padding:15px; margin-bottom:24px; border:1px solid var(--line); border-left:4px solid var(--accent); }
    .market-title { color:var(--accent); font-weight:600; font-size:var(--fs-title); text-transform:uppercase; margin-bottom:10px; }
    .market-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:15px; }
    .market-card { background-color:var(--raised); padding:10px; border-radius:8px; }
    .market-card h3 { margin:0 0 5px 0; color:var(--text); font-size:var(--fs-body); }
    .val-green { color:var(--green); font-weight:600; } .val-red { color:var(--red); font-weight:600; } .val-warn { color:var(--yellow); font-weight:600; }
    .market-summary { margin-top:15px; padding:10px; background-color:var(--tint-accent); border-radius:8px; font-size:var(--fs-body); }

    .kill-ok { background:var(--tint-green); border:1px solid var(--bd-green); color:var(--green); border-radius:8px; padding:8px 14px; font-size:var(--fs-table); font-weight:600; }
    .kill-bad { background:var(--tint-red); border:1px solid var(--bd-red); color:var(--red); border-radius:8px; padding:8px 14px; font-size:var(--fs-table); font-weight:600; }
    .kill-warn { background:var(--tint-yellow); border:1px solid var(--bd-yellow); color:var(--yellow); border-radius:8px; padding:8px 14px; font-size:var(--fs-table); }

    .regime { border:1px solid; border-radius:8px; padding:10px 14px; margin:0 0 24px; }
    .reg-head { display:flex; flex-wrap:wrap; gap:6px 12px; align-items:baseline; font-weight:700; font-size:1.05em; }
    .reg-score { font-weight:normal; font-size:var(--fs-caption); color:var(--text); font-family:var(--mono); }
    .reg-sigs { margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; }
    .reg-sig { font-size:var(--fs-micro); font-weight:500; border:1px solid; border-radius:999px; padding:2px 8px; background:var(--bg); }
    .reg-note { margin-top:8px; font-size:var(--fs-caption); color:var(--text-2); line-height:1.6; }
    .dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:4px; vertical-align:1px; }
    .dot-g { background:var(--green); } .dot-y { background:var(--yellow); } .dot-r { background:var(--red); } .dot-i { background:var(--text-3); }

    .toppicks { display:flex; flex-wrap:wrap; gap:10px; margin:0 0 24px; }
    .tp-card { flex:1 1 150px; min-width:150px; background:var(--surface); border:1px solid var(--bd-green); border-radius:8px; padding:10px 12px; }
    .tp-top { display:flex; align-items:center; gap:6px; }
    .tp-top a { font-weight:600; font-size:1.15em; color:var(--accent); text-decoration:none; }
    .tp-tier { font-size:var(--fs-micro); font-weight:500; border:1px solid; border-radius:4px; padding:1px 5px; }
    .tp-edges { margin-left:auto; font-size:var(--fs-caption); font-weight:500; color:var(--yellow); background:var(--tint-yellow); border-radius:999px; padding:1px 7px; font-family:var(--mono); }
    .tp-px { margin-top:4px; } .tp-px .lp { font-size:var(--fs-title); }
    .tp-meta { font-size:var(--fs-micro); color:var(--text-3); font-weight:normal; }
    .tp-theme { margin-top:3px; font-size:var(--fs-caption); color:var(--text-3); }
    .tp-plan { margin-top:5px; color:var(--text); font-size:var(--fs-body); font-weight:700; font-family:var(--mono); }

    .hot-themes { display:flex; flex-wrap:wrap; gap:7px; margin:0 0 24px; }
    .hot-themes.scroll { flex-wrap:nowrap; overflow-x:auto; padding-bottom:7px; -webkit-overflow-scrolling:touch; }
    .hot-themes.scroll .theme-chip { flex:0 0 auto; }
    .hot-themes.scroll::-webkit-scrollbar { height:6px; }
    .hot-themes.scroll::-webkit-scrollbar-thumb { background:var(--line); border-radius:3px; }
    .theme-chip { font-size:var(--fs-caption); border:1px solid var(--line); background:var(--surface); color:var(--text); border-radius:999px; padding:5px 11px; cursor:pointer; user-select:none; transition:background .15s; }
    .theme-chip.active { background:var(--tint-accent); outline:2px solid var(--accent); }
    details.collapsis > summary { cursor:pointer; list-style:none; margin-bottom:0; }
    details.collapsis > summary::-webkit-details-marker { display:none; }
    details.collapsis > summary::after { content:' ▸'; font-size:var(--fs-caption); opacity:0.7; }
    details.collapsis[open] > summary::after { content:' ▾'; }
    details.collapsis:not([open]) > summary { border-radius:8px; }
    .idx-spark { margin:2px 0 2px; line-height:0; }

    .diag-panel { background:var(--tint-red); border:1px solid var(--bd-red); border-radius:8px; padding:12px 15px; margin-bottom:24px; }
    .diag-panel .t { color:var(--red); font-weight:600; margin-bottom:6px; }
    .diag-panel ul { margin:4px 0 0 18px; padding:0; font-size:var(--fs-table); color:var(--red); }

    .section-title { position:relative; padding:10px 15px; margin-top:32px; margin-bottom:0; border-radius:8px 8px 0 0; font-size:var(--fs-title); font-weight:600; text-transform:uppercase; letter-spacing:0.05em; background-color:var(--surface); }
    .section-title.collapsible { cursor:pointer; user-select:none; }
    .section-title.collapsible::after { content:'▾'; position:absolute; right:15px; top:10px; opacity:0.65; font-weight:400; }
    .section-title.collapsed { border-radius:8px; }
    .section-title.collapsed::after { content:'▸'; }
    .section-title.collapsed + .table-container { display:none; }
    /* tier criteria are printed inline in each section title (the gate from scan_coil) */
    .funnel { margin:24px 0 0; background:var(--surface); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    .funnel .fn-cap { padding:8px 12px; font-size:var(--fs-caption); font-weight:700; color:var(--text); background:var(--raised); border-bottom:1px solid var(--line); }
    .funnel .fn-stage { display:flex; align-items:center; gap:14px; padding:10px 12px; border-bottom:1px solid var(--line); }
    .funnel .fn-stage:last-child { border-bottom:none; }
    .funnel .fn-body { flex:1; min-width:0; }
    .funnel .fn-title { font-size:var(--fs-caption); font-weight:600; color:var(--text); }
    .funnel .fn-sub { color:var(--text-3); font-weight:400; }
    .funnel .fn-crit { font-size:var(--fs-micro); color:var(--text-3); line-height:1.45; margin-top:3px; }
    .funnel .fn-count { font-family:var(--mono); font-weight:700; font-size:var(--fs-table); color:var(--accent); white-space:nowrap; text-align:right; }
    .funnel .fn-dot { color:var(--text-3); }
    .bg-aplus { color:var(--green); border-bottom:3px solid var(--green); }
    .bg-a { color:var(--yellow); border-bottom:3px solid var(--yellow); }
    .bg-aminus { color:var(--red); border-bottom:3px solid var(--red); }
    .bg-hve { color:var(--red); border-bottom:3px solid var(--bd-red); }
    .bg-short { color:var(--red); border-bottom:3px solid var(--bd-red); }

    details.mindset { background:var(--surface); border:1px solid var(--line); border-left:4px solid var(--bd-accent); border-radius:8px; padding:10px 16px; margin:0 0 24px; }
    details.mindset summary { cursor:pointer; color:var(--accent-2); font-weight:600; font-size:var(--fs-body); list-style:none; }
    details.mindset summary::-webkit-details-marker { display:none; }
    details.mindset ul { margin:10px 0 2px; padding-left:20px; }
    details.mindset li { color:var(--text); font-size:var(--fs-body); line-height:1.7; margin-bottom:3px; }

    .table-container { width:100%; overflow:auto; max-height:82vh; border-radius:0 0 8px 8px; background-color:var(--surface); margin-bottom:24px; -webkit-overflow-scrolling:touch; }
    table { width:100%; border-collapse:collapse; min-width:650px; }
    /* expandable columns: collapsed tables show only Ticker + Price & Narrative */
    table.cols-collapsed { min-width:0; }
    .col-hidden { display:none !important; }
    .colbar { position:sticky; left:0; padding:8px 8px 4px; }
    .colbar-btn { display:inline-flex; align-items:center; gap:5px; background:var(--raised); color:var(--text-2); border:1px solid var(--line); border-radius:6px; padding:4px 11px; font-size:var(--fs-caption); font-weight:500; line-height:1.3; cursor:pointer; white-space:nowrap; }
    .colbar-btn:hover { color:var(--text); border-color:var(--accent); }
    .colmenu { display:none; flex-wrap:wrap; gap:6px; margin-top:8px; align-items:center; }
    .colmenu.open { display:flex; }
    .colchip { font-size:var(--fs-caption); border:1px solid var(--line); background:var(--surface); color:var(--text-2); border-radius:999px; padding:4px 11px; cursor:pointer; user-select:none; white-space:nowrap; }
    .colchip.on { background:var(--tint-accent); color:var(--text); border-color:var(--accent); }
    .colchip.locked { opacity:0.55; cursor:default; }
    .colact { font-size:var(--fs-caption); color:var(--accent); cursor:pointer; padding:4px 6px; }
    .colact:hover { text-decoration:underline; }
    th, td { padding:10px 8px; text-align:left; border-bottom:1px solid var(--line); font-size:var(--fs-table); }
    th.num, td.num { text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; }
    th { background-color:var(--raised); color:var(--text-2); text-transform:uppercase; letter-spacing:0.05em; font-size:var(--fs-micro); font-weight:500; position:sticky; top:0; z-index:2; cursor:pointer; user-select:none; white-space:nowrap; }
    th:first-child, td:first-child { position:sticky; left:0; z-index:3; background:var(--surface); }
    th:first-child { z-index:4; background:var(--raised); }
    th .arrow { opacity:0.7; font-size:var(--fs-body); }
    th.sorted { color:var(--accent); } th.sorted .arrow { opacity:1; }
    .colmove { display:inline-block; cursor:pointer; color:var(--text-3); opacity:0.5; font-weight:700;
               padding:0 1px; font-size:var(--fs-body); line-height:1; -webkit-user-select:none; user-select:none; }
    .colmove:hover { color:var(--accent); opacity:1; }
    td.ma-cell, td.fy-cell { font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }

    .ticker a { font-weight:600; font-size:1.2em; color:var(--accent); text-decoration:none; }
    .ep-ticker a { font-weight:600; font-size:1.2em; color:var(--red); text-decoration:none; }
    .good { color:var(--green); font-weight:600; } .warn { color:var(--yellow); font-weight:600; } .bad { color:var(--red); font-weight:600; }
    .good, .warn, .bad, .val-green, .val-red, .val-warn { font-family:var(--mono); font-variant-numeric:tabular-nums; }
    .tag { display:inline-block; padding:3px 6px; border-radius:4px; background:var(--raised); border:1px solid var(--line); font-size:var(--fs-micro); margin:2px 0; color:var(--text-3); }
    .theme-tag { display:inline-block; padding:3px 7px; border-radius:4px; background:var(--tint-accent); border:1px solid var(--bd-accent); font-size:var(--fs-caption); font-weight:500; margin:4px 0; color:var(--accent-2); }
    .score { font-size:var(--fs-body); font-weight:600; color:var(--yellow); background:var(--tint-yellow); padding:4px 8px; border-radius:4px; border:1px solid var(--yellow); font-family:var(--mono); }
    .hve-badge { font-size:var(--fs-body); font-weight:600; color:var(--red); background:var(--tint-red); border:1px solid var(--bd-red); padding:4px 8px; border-radius:4px; display:inline-block; font-family:var(--mono); }
    .squat-badge { font-size:var(--fs-micro); font-weight:500; color:var(--text-2); background:var(--raised); border:1px solid var(--line); padding:2px 6px; border-radius:4px; display:inline-block; margin-bottom:4px; }
    .fp-badge { font-size:var(--fs-micro); font-weight:500; padding:2px 6px; border-radius:4px; display:inline-block; margin:0 4px 4px 0; }
    .fp-good { color:var(--green); background:var(--tint-green); border:1px solid var(--bd-green); }
    .fp-info { color:var(--accent-2); background:var(--tint-accent); border:1px solid var(--bd-accent); }
    .fp-warn { color:var(--yellow); background:var(--tint-yellow); border:1px solid var(--bd-yellow); }
    .edge-line { font-size:var(--fs-micro); color:var(--text-3); letter-spacing:0.03em; margin-bottom:4px; }
    .edge-line .warn-flag { color:var(--yellow); }

    .entry-box { background-color:var(--tint-green); border:1px solid var(--green); border-radius:8px; padding:6px; margin-top:5px; text-align:left; font-size:var(--fs-table); }
    .entry-text { color:var(--green); font-weight:700; font-family:var(--mono); }
    .stop-text { color:var(--red); font-weight:700; font-family:var(--mono); }
    .stop-reason { color:var(--text-3); font-weight:normal; font-size:var(--fs-micro); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; }

    details.meta { margin-top:4px; }
    details.meta summary { cursor:pointer; color:var(--text-3); font-size:var(--fs-micro); list-style:none; padding:4px 4px 4px 0; }
    details.meta summary::-webkit-details-marker { display:none; }
    details.meta[open] summary { color:var(--accent); }
    details.meta ul { margin:4px 0 0; padding-left:16px; text-align:left; color:var(--text-2); font-size:var(--fs-micro); }
    /* Tap-to-expand fundamentals on the narrative cell (theme/sector/industry). */
    details.fund > summary { cursor:pointer; list-style:none; display:block; }
    details.fund > summary::-webkit-details-marker { display:none; }
    details.fund > summary::after { content:'📊'; opacity:.45; font-size:var(--fs-micro); margin-left:4px; }
    details.fund[open] > summary::after { content:'📊 ▴'; opacity:.7; }
    .fund-wrap { margin-top:6px; text-align:left; }
    /* Override the report's global table rules (width:100%, min-width:650px, td
       padding:10px / font 13px) that would otherwise stretch this nested table
       across the whole 650px row. Force it to size to its own compact content. */
    .fund-tbl { border-collapse:collapse; font-family:var(--mono);
                width:auto !important; min-width:0 !important; }
    .fund-tbl td { padding:1px 9px 1px 0 !important; text-align:right;
                white-space:nowrap; font-size:10px !important; border-bottom:none !important; }
    .fund-tbl tr.fund-head td { color:var(--text-3); font-weight:500; border-bottom:1px solid var(--line) !important; }
    .fund-tbl td:first-child { text-align:left; }
    .fund-tbl tr.fund-act td { color:var(--text); }
    .fund-tbl tr.fund-est td { color:var(--text-3); font-style:italic; }
    .fund-up { color:#3fb950; } .fund-dn { color:#ff7b72; } .fund-flat, .fund-na { color:var(--text-3); }
    .fund-src { color:var(--text-3); font-size:10px; margin-top:3px; opacity:.7; }
    .spark { margin-top:4px; }
    .livebtn { cursor:pointer; font:inherit; border:1px solid var(--bd-accent) !important; color:var(--accent-2) !important; background:var(--tint-accent) !important; min-width:9.5em; min-height:32px; }
    .livebtn:disabled { opacity:0.6; cursor:wait; }
    .lp { transition:color .25s; font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:var(--fs-title); font-weight:700; }
    .lp.flag-up::after { content:' ▲'; color:var(--green); font-size:var(--fs-caption); }
    .lp.flag-down::after { content:' ▼ STOP'; color:var(--red); font-size:var(--fs-micro); font-weight:700; }

    /* ---- interaction guards ---- */
    @media (hover:hover) {
      tr:hover { background-color:#1f242c; }
      .theme-chip:hover { background:#1f242c; }
      th.sortable:hover { color:var(--text); }
      .livebtn:hover { background:#13314f !important; }
    }
    @media (pointer:coarse) {
      .chip, .theme-chip { padding:9px 14px; }
      .livebtn { min-height:44px; }
      th { padding:13px 8px; }
      .colmove { padding:3px 6px; font-size:var(--fs-title); opacity:0.65; }
      details.meta summary { padding:8px 8px 8px 0; font-size:var(--fs-caption); }
    }
    @media (max-width:480px) {
      html { font-size:17px; }
      th, td { padding:10px 6px; }
      body { margin:6px auto; }
    }
    @media (prefers-reduced-motion:reduce) { * { transition-duration:0.01ms !important; } }

    /* ---- engine tabs (MADRRY watchlist | Minervini | Trilogy) ---- */
    .tabs { display:flex; gap:4px; flex-wrap:wrap; margin:20px 0 16px; border-bottom:1px solid var(--line); }
    .tab-btn { cursor:pointer; font:inherit; font-size:var(--fs-body); font-weight:600; color:var(--text-3);
               background:transparent; border:none; border-bottom:2px solid transparent; padding:10px 16px; margin-bottom:-1px; }
    .tab-btn:hover { color:var(--text); }
    .tab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
    .tab-btn .tab-count { color:var(--text-3); font-weight:500; font-size:var(--fs-caption); margin-left:6px; }
    .tab-btn.active .tab-count { color:var(--accent-2); }
    .tab-panel { display:none; }
    .tab-panel.active { display:block; }
    /* ---- nested sub-tabs (e.g. 52-Week High: New Highs | Pullback) ---- */
    .subtabs { display:flex; gap:4px; flex-wrap:wrap; margin:2px 0 14px; border-bottom:1px solid var(--line); }
    .subtab-btn { cursor:pointer; font:inherit; font-size:var(--fs-caption); font-weight:600; color:var(--text-3);
                  background:transparent; border:none; border-bottom:2px solid transparent; padding:7px 13px; margin-bottom:-1px; }
    .subtab-btn:hover { color:var(--text); }
    .subtab-btn.active { color:var(--accent); border-bottom-color:var(--accent); }
    .subtab-btn .tab-count { color:var(--text-3); font-weight:500; font-size:var(--fs-micro); margin-left:5px; }
    .subtab-btn.active .tab-count { color:var(--accent-2); }
    .subtab-panel { display:none; }
    .subtab-panel.active { display:block; }
    .ext-asof { color:var(--text-3); font-size:var(--fs-caption); margin:6px 0 12px; }
    .ext-asof code { color:var(--text-2); background:var(--raised); padding:1px 5px; border-radius:4px; font-family:var(--mono); }
    .grade-badge { font-family:var(--mono); font-weight:700; font-size:var(--fs-body); padding:3px 9px; border-radius:4px; display:inline-block; }
"""

PAGE_JS = """
<script>
// ---- Manual live-price refresh (button in the run-bar) ------------------
// Browsers can't call Yahoo directly (CORS). LIVE_PRICE_PROXY = your keyless
// Cloudflare Worker (?symbols=A,B,C -> JSON). If empty, falls back to a public
// CORS proxy (best-effort). Only the displayed price updates; entry/stop/M.E.T.A.
// stay from the scan.
var LIVE_PRICE_PROXY = "__LIVE_PRICE_PROXY__";
async function _lpFetchQuotes(tickers) {
  var q = {};
  if (LIVE_PRICE_PROXY) {
    // Cloudflare Worker: one call, returns {result:[{symbol,regularMarketPrice}]}
    var res = await fetch(LIVE_PRICE_PROXY.replace(/\\/+$/, "") + "/?symbols=" + tickers.join(","), { cache: 'no-store' });
    var data = await res.json();
    var arr = (data.quoteResponse && data.quoteResponse.result) || data.result || (Array.isArray(data) ? data : []);
    arr.forEach(function (r) {
      if (!r) return;
      var sym = (r.symbol || r.ticker || '').toUpperCase();
      var px = (r.regularMarketPrice != null) ? r.regularMarketPrice : (r.price != null ? r.price : r.close);
      if (sym && px != null) q[sym] = +px;
    });
    return q;
  }
  // Fallback (no worker): Yahoo's keyless chart endpoint per symbol via a public
  // CORS proxy. Best-effort — partial results are fine.
  await Promise.all(tickers.map(function (t) {
    var y = "https://query1.finance.yahoo.com/v8/finance/chart/" + encodeURIComponent(t) + "?interval=1d&range=1d";
    var url = "https://api.allorigins.win/raw?url=" + encodeURIComponent(y);
    var sig = (typeof AbortSignal !== 'undefined' && AbortSignal.timeout) ? AbortSignal.timeout(8000) : undefined;
    return fetch(url, { cache: 'no-store', signal: sig }).then(function (r) { return r.json(); }).then(function (d) {
      var m = d && d.chart && d.chart.result && d.chart.result[0] && d.chart.result[0].meta;
      if (m && m.regularMarketPrice != null) q[t.toUpperCase()] = +m.regularMarketPrice;
    }).catch(function () {});
  }));
  return q;
}
async function refreshPrices(btn) {
  var els = Array.prototype.slice.call(document.querySelectorAll('.lp[data-tkr]'));
  var tickers = Array.from(new Set(els.map(function (e) { return e.getAttribute('data-tkr'); }).filter(Boolean)));
  var stamp = document.getElementById('liveStamp');
  if (!tickers.length) return;
  if (btn) { btn.disabled = true; btn.textContent = '⏳ Fetching…'; }
  try {
    var q = await _lpFetchQuotes(tickers);
    var updated = 0;
    els.forEach(function (e) {
      var px = q[(e.getAttribute('data-tkr') || '').toUpperCase()];
      if (px == null) return;
      updated++;
      var snap = parseFloat(e.getAttribute('data-snap'));
      e.textContent = "$" + px.toFixed(2);
      e.style.color = px > snap ? "#3fb950" : (px < snap ? "#ff7b72" : "");
      e.classList.remove('flag-up', 'flag-down');
      var entry = parseFloat(e.getAttribute('data-entry'));
      var stop = parseFloat(e.getAttribute('data-stop'));
      if (!isNaN(stop) && px <= stop) e.classList.add('flag-down');
      else if (!isNaN(entry) && px >= entry) e.classList.add('flag-up');
    });
    if (stamp) {
      stamp.textContent = "🟢 LIVE " + new Date().toLocaleTimeString() + " · " + updated + "/" + tickers.length;
      stamp.style.color = "#3fb950";
    }
  } catch (err) {
    if (stamp) {
      stamp.textContent = "⚠️ fetch failed — set a proxy (see notes)";
      stamp.style.color = "#ff7b72";
    }
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = '🔄 Refresh Prices'; }
  }
}

(function () {
  // --- combined ticker search + theme filter across every table ---
  var box = document.getElementById('search');
  var activeTheme = null;
  function applyFilters() {
    var q = box ? box.value.trim().toUpperCase() : '';
    document.querySelectorAll('table tr').forEach(function (tr) {
      if (tr.querySelector('th')) return;
      var tickerCell = tr.querySelector('.ticker a, .ep-ticker a');
      var okQ = !q || (tickerCell && tickerCell.textContent.toUpperCase().indexOf(q) !== -1);
      var okT = !activeTheme || (tr.getAttribute('data-sector') === activeTheme);
      tr.style.display = (okQ && okT) ? '' : 'none';
    });
    document.querySelectorAll('.theme-chip').forEach(function (c) {
      if (c.id === 'themeClear') return;
      c.classList.toggle('active', c.getAttribute('data-sector') === activeTheme);
    });
    var clr = document.getElementById('themeClear');
    if (clr) clr.style.display = activeTheme ? '' : 'none';
  }
  if (box) box.addEventListener('input', applyFilters);
  document.querySelectorAll('.theme-chip').forEach(function (c) {
    c.addEventListener('click', function () {
      if (c.id === 'themeClear') { activeTheme = null; applyFilters(); return; }
      var t = c.getAttribute('data-sector');
      activeTheme = (activeTheme === t) ? null : t;
      applyFilters();
    });
  });

  // --- click-to-sort columns (uses data-sort when present, else text) ---
  function cellVal(row, idx) {
    var cell = row.children[idx];
    if (!cell) return '';
    var raw = cell.getAttribute('data-sort');
    if (raw === null) raw = cell.textContent;
    var num = parseFloat(String(raw).replace(/[^0-9.\\-]/g, ''));
    return isNaN(num) ? String(raw).toUpperCase() : num;
  }
  // Shared "add a tier on a plain tap" mode — the touch/mobile equivalent of
  // Shift-click (phones have no Shift key). The floating toggle built below flips it,
  // and the header click handler treats a tap as "add tier" whenever it's on.
  var multiSortMode = false;
  document.querySelectorAll('table').forEach(function (table) {
    if (table.classList.contains('fund-tbl')) return;   // nested fundamentals mini-table: not sortable
    var headRow = table.querySelector('tr');
    if (!headRow) return;

    // Tiered multi-sort state, per table. `keys` is an ordered list of active sort
    // tiers: [{th, asc}, ...]. Tier 0 is primary, tier 1 secondary, etc. A plain
    // click = single-column sort (replaces every tier). Shift-click = add/toggle a
    // tier while keeping the existing ones, so you can do e.g. primary Fwd YoY ↓
    // then secondary "closest to 10MA" ↑. Single-column behaviour is unchanged when
    // you never hold Shift.
    var keys = [];
    function keyIndex(th) {
      for (var i = 0; i < keys.length; i++) { if (keys[i].th === th) return i; }
      return -1;
    }
    function resort() {
      var body = table.tBodies[0] || table;
      var rows = Array.prototype.filter.call(body.rows, function (r) { return !r.querySelector('th'); });
      // Resolve each tier's CURRENT column position fresh — columns can be reordered
      // (see the column-reorder IIFE below), so capture-time indexes would go stale.
      var tiers = keys.map(function (k) {
        return { idx: Array.prototype.indexOf.call(headRow.children, k.th), asc: k.asc };
      });
      rows.sort(function (a, b) {
        for (var i = 0; i < tiers.length; i++) {
          var t = tiers[i];
          var va = cellVal(a, t.idx), vb = cellVal(b, t.idx);
          if (va < vb) return t.asc ? -1 : 1;
          if (va > vb) return t.asc ? 1 : -1;
        }
        return 0;
      });
      rows.forEach(function (r) { body.appendChild(r); });
      // Repaint every header's indicator. Single tier → bare ▲/▼ (unchanged look);
      // multiple tiers → a rank prefix (1▲ 2▼ …) so the precedence is visible.
      var multi = keys.length > 1;
      Array.prototype.forEach.call(headRow.children, function (h) {
        var rank = keyIndex(h);
        var a0 = h.querySelector('.arrow');
        if (rank < 0) {
          h.classList.remove('sorted');
          h.removeAttribute('aria-sort');
          if (a0) a0.textContent = '⇅';
        } else {
          var asc = keys[rank].asc;
          h.classList.add('sorted');
          h.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
          if (a0) a0.textContent = (multi ? (rank + 1) : '') + (asc ? '▲' : '▼');
        }
      });
    }

    Array.prototype.forEach.call(headRow.children, function (th, idx) {
      th.classList.add('sortable');
      th.innerHTML = th.innerHTML + ' <span class="arrow" title="Tap to sort. Shift-click (desktop) or turn on Multi-sort (mobile) to add a secondary/tertiary sort tier.">⇅</span>';
      th.addEventListener('click', function (ev) {
        var rank = keyIndex(th);
        if (ev.shiftKey || multiSortMode) {
          // Add a new tier, or flip this tier's direction if it's already active.
          if (rank < 0) keys.push({ th: th, asc: true });
          else keys[rank].asc = !keys[rank].asc;
        } else {
          // Plain click: collapse to a single-column sort. Re-clicking the lone
          // active column toggles its direction (original single-sort behaviour).
          if (keys.length === 1 && rank === 0) keys[0].asc = !keys[0].asc;
          else keys = [{ th: th, asc: true }];
        }
        resort();
      });
    });
  });

  // Floating toggle so touch users can build multi-tier sorts without a Shift key.
  // OFF (default): a tap sorts by one column, exactly as before. ON: each header tap
  // ADDS a tier (primary, then secondary…); tapping an active tier flips its arrow.
  (function () {
    var btn = document.createElement('button');
    btn.id = 'multiSortToggle';
    btn.type = 'button';
    btn.setAttribute('aria-pressed', 'false');
    btn.textContent = '⇅ Multi-sort: off';
    btn.title = 'When ON, each column tap ADDS a sort tier (primary, then secondary…) instead of replacing it. Tap an active tier again to flip its direction. Turn OFF to go back to one-column sorting.';
    function paint() {
      var on = multiSortMode;
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
      btn.textContent = on ? '⇅ Multi-sort: ON — tap columns to add' : '⇅ Multi-sort: off';
      btn.style.background = on ? 'var(--accent,#3a86ff)' : 'var(--surface,#1b1b1b)';
      btn.style.color = on ? 'var(--bg,#0b0b0b)' : 'var(--text,#eaeaea)';
      btn.style.borderColor = 'var(--accent,#3a86ff)';
    }
    btn.style.cssText = 'position:fixed;z-index:9999;right:12px;bottom:12px;padding:9px 13px;'
      + 'border-radius:20px;border:1px solid var(--accent,#3a86ff);font:600 13px system-ui,-apple-system,sans-serif;'
      + 'box-shadow:0 2px 10px rgba(0,0,0,.45);cursor:pointer;opacity:.95;-webkit-tap-highlight-color:transparent;';
    paint();
    btn.addEventListener('click', function () { multiSortMode = !multiSortMode; paint(); });
    document.body.appendChild(btn);
  })();
})();

(function () {
  // --- engine tabs: MADRRY watchlist | Minervini | Trilogy ---
  var btns = document.querySelectorAll('.tab-btn');
  var panels = document.querySelectorAll('.tab-panel');
  btns.forEach(function (b) {
    b.addEventListener('click', function () {
      var t = b.getAttribute('data-tab');
      btns.forEach(function (x) { x.classList.toggle('active', x === b); });
      panels.forEach(function (p) { p.classList.toggle('active', p.id === 'tab-' + t); });
    });
  });
})();

(function () {
  // --- nested sub-tabs (scoped to their parent .tab-panel so they never toggle
  //     the top-level engine tabs) ---
  document.querySelectorAll('.subtab-btn').forEach(function (b) {
    b.addEventListener('click', function () {
      var t = b.getAttribute('data-subtab');
      var scope = b.closest('.tab-panel') || document;
      scope.querySelectorAll('.subtab-btn').forEach(function (x) { x.classList.toggle('active', x === b); });
      scope.querySelectorAll('.subtab-panel').forEach(function (p) { p.classList.toggle('active', p.id === 'subtab-' + t); });
    });
  });
})();

(function () {
  // --- collapsible MADRRY tier sections: click a section title to fold its table ---
  document.querySelectorAll('.section-title').forEach(function (t) {
    var next = t.nextElementSibling;
    if (!next || !next.classList.contains('table-container')) return;
    t.classList.add('collapsible');
    t.setAttribute('title', 'Click to collapse / expand');
    t.addEventListener('click', function () { t.classList.toggle('collapsed'); });
  });
})();

(function () {
  // --- user-reorderable columns: tap ‹ / › on a header to move it left / right ---
  //     Order persists per column-SCHEMA in localStorage (key 'madrry.colorder.<schema>')
  //     so it survives daily report regenerations and applies to every table sharing the
  //     schema (e.g. all three MADRRY tier tables move together). The Ticker column is
  //     pinned first — its sticky-left styling is the row anchor — so nothing moves into
  //     position 0. Touch-friendly: plain tap targets, no native drag (which mobile
  //     browsers don't support).
  var PFX = 'madrry.colorder.';
  function load(s) { try { return JSON.parse(localStorage.getItem(PFX + s)); } catch (e) { return null; } }
  function save(s, o) { try { localStorage.setItem(PFX + s, JSON.stringify(o)); } catch (e) {} }
  function keysOf(table) {
    var hr = table.querySelector('tr');
    return hr ? Array.prototype.map.call(hr.children, function (th) { return th.getAttribute('data-col'); }) : [];
  }
  function apply(table, order) {
    var keys = keysOf(table);
    if (!keys.length || keys.indexOf(null) >= 0) return;        // only fully-keyed tables
    var desired = order.filter(function (k) { return keys.indexOf(k) >= 0; });
    keys.forEach(function (k) { if (desired.indexOf(k) < 0) desired.push(k); });  // unknown/new cols keep place
    if (desired.length !== keys.length) return;
    var perm = desired.map(function (k) { return keys.indexOf(k); });
    if (perm.every(function (v, i) { return v === i; })) return;   // already in order
    Array.prototype.forEach.call(table.querySelectorAll('tr'), function (row) {
      var cells = Array.prototype.slice.call(row.children);
      if (cells.length !== keys.length) return;                   // skip colspan / placeholder rows
      var frag = document.createDocumentFragment();
      perm.forEach(function (oi) { frag.appendChild(cells[oi]); });
      row.appendChild(frag);
    });
  }
  function move(schema, key, dir) {
    var tables = document.querySelectorAll("table[data-schema='" + schema + "']");
    if (!tables.length) return;
    var cur = keysOf(tables[0]).filter(Boolean);
    var i = cur.indexOf(key); if (i < 0) return;
    var j = i + dir;
    if (j < 1 || j >= cur.length) return;                          // index 0 reserved for the ticker anchor
    cur.splice(i, 1); cur.splice(j, 0, key);
    save(schema, cur);
    Array.prototype.forEach.call(tables, function (t) { apply(t, cur); });
  }
  function addControls(table) {
    var schema = table.getAttribute('data-schema'); if (!schema) return;
    var hr = table.querySelector('tr'); if (!hr) return;
    Array.prototype.forEach.call(hr.children, function (th) {
      var key = th.getAttribute('data-col');
      if (!key || key === 'tk' || th.querySelector('.colmove')) return;   // ticker stays pinned first
      var L = document.createElement('span'); L.className = 'colmove'; L.textContent = '‹'; L.title = 'Move column left';
      var R = document.createElement('span'); R.className = 'colmove'; R.textContent = '›'; R.title = 'Move column right';
      L.addEventListener('click', function (e) { e.stopPropagation(); move(schema, key, -1); });
      R.addEventListener('click', function (e) { e.stopPropagation(); move(schema, key, 1); });
      th.appendChild(L); th.appendChild(R);
    });
  }
  var schemas = {};
  document.querySelectorAll('table[data-schema]').forEach(function (t) { schemas[t.getAttribute('data-schema')] = true; });
  Object.keys(schemas).forEach(function (s) {
    var o = load(s);
    if (o && o.length) document.querySelectorAll("table[data-schema='" + s + "']").forEach(function (t) { apply(t, o); });
  });
  document.querySelectorAll('table[data-schema]').forEach(addControls);
})();

(function () {
  // --- column selector: pick exactly which columns each stock table shows. A
  //     "🛠 Columns" button opens a tray of toggle chips (one per column). Selection
  //     persists per column-SCHEMA in localStorage (key 'madrry.colsel.<schema>' = JSON
  //     array of visible data-col keys), so it survives daily regenerations and applies
  //     to every table sharing the schema (the MADRRY tier tables move together). Ticker
  //     is always shown (the sticky row anchor) and can't be toggled off. Default view =
  //     Ticker + Price & Narrative. Hiding is by data-col key, so it coexists with
  //     click-to-sort and column reorder (both resolve positions live).
  var PFX = 'madrry.colsel.';
  var LOCKED = 'tk';                       // always visible, not toggleable
  var DEFAULT = ['tk', 'price'];
  function load(s) { try { var v = JSON.parse(localStorage.getItem(PFX + s)); return (v && v.length) ? v : null; } catch (e) { return null; } }
  function save(s, a) { try { localStorage.setItem(PFX + s, JSON.stringify(a)); } catch (e) {} }
  function keysOf(table) {
    var hr = table.querySelector('tr');
    return hr ? Array.prototype.map.call(hr.children, function (th) { return th.getAttribute('data-col'); }) : [];
  }
  function labelOf(th) {                    // header text minus injected sort arrow / reorder arrows
    var c = th.cloneNode(true);
    Array.prototype.forEach.call(c.querySelectorAll('.arrow,.colmove'), function (x) { x.remove(); });
    return (c.textContent || '').trim();
  }
  function apply(table, visible) {
    var keys = keysOf(table);
    if (!keys.length || keys.indexOf(null) >= 0) return;
    var allShown = keys.every(function (k) { return visible.indexOf(k) >= 0; });
    table.classList.toggle('cols-collapsed', !allShown);   // drop the 650px min-width when trimmed
    Array.prototype.forEach.call(table.querySelectorAll('tr'), function (row) {
      if (row.children.length !== keys.length) return;     // skip colspan / placeholder rows
      Array.prototype.forEach.call(row.children, function (cell, i) {
        cell.classList.toggle('col-hidden', visible.indexOf(keys[i]) < 0);
      });
    });
  }
  // group eligible tables by schema — a table needs both a tk and a price column
  var groups = {};
  document.querySelectorAll('table[data-schema]').forEach(function (t) {
    var keys = keysOf(t);
    if (!keys.length || keys.indexOf(null) >= 0) return;
    if (keys.indexOf('tk') < 0 || keys.indexOf('price') < 0) return;
    var s = t.getAttribute('data-schema');
    (groups[s] = groups[s] || []).push(t);
  });
  Object.keys(groups).forEach(function (schema) {
    var tables = groups[schema];
    var keys = keysOf(tables[0]);
    var hr = tables[0].querySelector('tr');
    var labels = {};
    Array.prototype.forEach.call(hr.children, function (th) { labels[th.getAttribute('data-col')] = labelOf(th); });
    var visible = (load(schema) || DEFAULT.slice()).filter(function (k) { return keys.indexOf(k) >= 0; });
    if (visible.indexOf(LOCKED) < 0) visible.unshift(LOCKED);

    function refresh() {
      tables.forEach(function (t) { apply(t, visible); });
      tables.forEach(function (t) {
        var bar = t.parentNode.querySelector('.colbar'); if (!bar) return;
        Array.prototype.forEach.call(bar.querySelectorAll('.colchip'), function (chip) {
          chip.classList.toggle('on', visible.indexOf(chip.getAttribute('data-col')) >= 0);
        });
        var btn = bar.querySelector('.colbar-btn');
        if (btn) btn.firstChild.nodeValue = '🛠 Columns (' + visible.length + '/' + keys.length + ') ';
      });
    }
    function toggleCol(k) {
      if (k === LOCKED) return;
      var i = visible.indexOf(k);
      if (i >= 0) { visible.splice(i, 1); }
      else { visible.push(k); visible.sort(function (a, b) { return keys.indexOf(a) - keys.indexOf(b); }); }
      save(schema, visible); refresh();
    }
    function setAll(a) { visible = a.slice(); if (visible.indexOf(LOCKED) < 0) visible.unshift(LOCKED); save(schema, visible); refresh(); }

    tables.forEach(function (t) {
      if (t.parentNode.querySelector('.colbar')) return;   // idempotent
      var bar = document.createElement('div'); bar.className = 'colbar';
      var btn = document.createElement('button'); btn.className = 'colbar-btn'; btn.type = 'button';
      btn.setAttribute('title', 'Choose which columns to show');
      btn.appendChild(document.createTextNode('🛠 Columns '));
      var car = document.createElement('span'); car.textContent = '▾'; btn.appendChild(car);
      var menu = document.createElement('div'); menu.className = 'colmenu';
      keys.forEach(function (k) {
        var chip = document.createElement('span');
        chip.className = 'colchip' + (k === LOCKED ? ' locked' : '');
        chip.setAttribute('data-col', k);
        chip.textContent = labels[k] || k;
        if (k === LOCKED) chip.setAttribute('title', 'Ticker is always shown');
        else chip.addEventListener('click', function () { toggleCol(k); });
        menu.appendChild(chip);
      });
      var aAll = document.createElement('span'); aAll.className = 'colact'; aAll.textContent = 'All';
      aAll.addEventListener('click', function () { setAll(keys.slice()); });
      var aDef = document.createElement('span'); aDef.className = 'colact'; aDef.textContent = 'Default';
      aDef.addEventListener('click', function () { setAll(DEFAULT.filter(function (k) { return keys.indexOf(k) >= 0; })); });
      menu.appendChild(aAll); menu.appendChild(aDef);
      btn.addEventListener('click', function () { var o = menu.classList.toggle('open'); car.textContent = o ? '▴' : '▾'; });
      bar.appendChild(btn); bar.appendChild(menu);
      t.parentNode.insertBefore(bar, t);                   // colbar = first child of .table-container
    });
    refresh();
  });
})();
</script>
"""


def _ind_badge(m: dict) -> str:
    """Industry-group RS chip (Fred6725 rs_industries) for a pick; '' if untagged."""
    pct = m.get("ind_rs")
    if not isinstance(pct, int):
        return ""
    col = "#3fb950" if pct >= IND_RS_STRONG else ("#f2cc60" if pct >= 70 else "#8b949e")
    return (f"<br><span class='tag' style='border-color:{col};color:{col};' "
            f"title='Industry-group RS percentile (Fred6725)'>🏭 {esc(m.get('ind_name') or '')} {pct}</span>")


def _meta_details_block(meta_details: List[str]) -> str:
    if not meta_details:
        return ""
    items = "".join(f"<li>{esc(x)}</li>" for x in meta_details)
    return f"<details class='meta'><summary>▸ M.E.T.A. breakdown</summary><ul>{items}</ul></details>"


def _trendline_block(tl_data: dict) -> str:
    if not (tl_data and tl_data.get("has_data")):
        return ""
    parts = []
    if tl_data["utl"]["exists"] and tl_data["utl"]["distance_pct"] is not None:
        d = tl_data["utl"]["distance_pct"]
        if d <= 3:
            parts.append(f"<span style='color:#3fb950;'>🎯 UTL: {d:.1f}%</span>")
        elif d <= 8:
            parts.append(f"<span style='color:#f2cc60;'>📐 UTL: {d:.1f}%</span>")
    if tl_data["dtl"]["breakout"]:
        parts.append("<span style='color:#ff7b72;font-weight:bold;'>🔥 DTL Break</span>")
    elif tl_data["dtl"]["exists"] and tl_data["dtl"]["distance_pct"] is not None:
        d = tl_data["dtl"]["distance_pct"]
        if 0 < d <= 3:
            parts.append(f"<span style='color:#f2cc60;'>⚡ Near DTL: {d:.1f}%</span>")
    if tl_data["trl"]["exists"] and tl_data["trl"]["distance_pct"] is not None:
        d = tl_data["trl"]["distance_pct"]
        if 0 < d <= 50:
            parts.append(f"<span style='color:#79c0ff;'>🎯 TRL: +{d:.1f}%</span>")
    if not parts:
        return ""
    return "<div style='font-size:var(--fs-caption);margin-bottom:6px;font-weight:500;'>" + " | ".join(parts) + "</div>"


def build_filter_funnel(fn: dict, n_aplus: int, n_a: int, n_aminus: int) -> str:
    """How the ~10k US-stock universe narrows to the coil tiers. The survivor count
    after each stage comes from scan_coil's per-stage instrumentation (this run's real
    numbers — see the `funnel` dict it returns). Plain divs so the table JS ignores it."""
    fn = fn or {}

    def _n(v):
        return f"{v:,}" if isinstance(v, int) else "—"

    s1_total = fn.get("universe_total")
    s1_fetched = fn.get("stage1_fetched")
    s1 = _n(s1_total if isinstance(s1_total, int) else s1_fetched)
    cap_note = ""
    if isinstance(s1_total, int) and isinstance(s1_fetched, int) and s1_total > s1_fetched:
        cap_note = f" <span class='fn-sub'>(top {s1_fetched:,} by ADR fetched)</span>"
    s2 = _n(fn.get("stage2_candidates"))
    return (
        "<div class='funnel'>"
        "<div class='fn-cap'>📊 HOW THIS LIST WAS BUILT — from ~10,000+ US-listed stocks</div>"
        "<div class='fn-stage'><div class='fn-body'>"
        "<div class='fn-title'>Stage 1 · Universe filter <span class='fn-sub'>· TradingView, server-side</span></div>"
        "<div class='fn-crit'>type = stock / DR · close ≥ $10 · day vol ≥ 500k · avg 30d &amp; 60d vol ≥ 500k · close ≥ SMA200 · market cap ≥ $2B</div>"
        f"</div><div class='fn-count'>{s1}{cap_note}</div></div>"
        "<div class='fn-stage'><div class='fn-body'>"
        "<div class='fn-title'>Stage 2 · Candidate gate <span class='fn-sub'>· client-side</span></div>"
        "<div class='fn-crit'>0–20% below the 52-week high · within 10% of the 9/21 EMA</div>"
        f"</div><div class='fn-count'>{s2}</div></div>"
        "<div class='fn-stage'><div class='fn-body'>"
        "<div class='fn-title'>Stage 3 · Coil tiers <span class='fn-sub'>· 1-year history · flag tightness · volume dry-up</span></div>"
        "<div class='fn-crit'>graded into A+ / A / A− by tightness, distance to a key MA and volume contraction (criteria in each tier title)</div>"
        f"</div><div class='fn-count'><span style='color:var(--green);'>A+ {n_aplus}</span> <span class='fn-dot'>·</span> "
        f"<span style='color:var(--yellow);'>A {n_a}</span> <span class='fn-dot'>·</span> "
        f"<span style='color:var(--red);'>A− {n_aminus}</span></div></div>"
        "</div>"
    )


def generate_coil_table(matches: List[dict], title: str, bg_class: str,
                        subtitle: str = "") -> str:
    if not matches:
        return ""
    sub_html = (f"<span class='section-sub'>{esc(subtitle)}</span>"
                if subtitle else "")
    out = [
        f'<div class="section-title {bg_class}">{esc(title)}{sub_html}</div>',
        '<div class="table-container"><table data-schema="coil">',
        "<tr><th data-col='tk'>Ticker</th><th data-col='plan'>Trade Plan</th><th data-col='price'>Price &amp; Narrative</th><th class='num' data-col='adr' title='Average Daily Range — 20-day avg of (High/Low−1), % · how much it typically moves per day (TradingView ADRP)'>ADR</th>"
        "<th data-col='rs'>RS</th><th data-col='meta'>M.E.T.A.</th><th class='num' data-col='ants'>ANTS</th>"
        + _MA_YOY_HEADERS +
        "<th data-col='status'>Status (Vol &amp; MA)</th></tr>",
    ]
    for m in matches:
        risk_color = "#3fb950" if m["risk_pct"] <= 4.0 else ("#f2cc60" if m["risk_pct"] <= 6.0 else "#ff7b72")
        vol_color = "good" if m["vol_pct"] <= 55 else ("warn" if m["vol_pct"] <= 75 else "bad")
        dist_color = "good" if m["dist_pct"] <= 4.0 else ("warn" if m["dist_pct"] <= 8.0 else "bad")
        dist52_color = "good" if m["dist_52w"] <= 25 else "bad"

        meta_score = m.get("meta_score", 0)
        if meta_score >= 70:
            ms_color, ms_bg = "#ff7b72", "var(--tint-red)"
        elif meta_score >= 50:
            ms_color, ms_bg = "#f2cc60", "var(--tint-yellow)"
        else:
            ms_color, ms_bg = "#8b949e", "#21262d"

        # Binary flags collapse into ONE muted token line (taste pass: badge
        # stacks were the main row-height/noise driver). 🚩 HTF labels stay
        # boxed (genuinely distinct setup type); ⚠️ warns stay yellow inline.
        boxed = []
        edge_tokens = []
        for lbl in m.get("status_labels", []):
            if lbl.startswith("🚩"):
                boxed.append(f"<div class='squat-badge' style='color:var(--red);border-color:var(--bd-red);'>{esc(lbl)}</div>")
            elif lbl.startswith("⚠️"):
                edge_tokens.append(f"<span class='warn-flag'>{esc(_strip_lead_emoji(lbl))}</span>")
            else:
                edge_tokens.append(esc(_strip_lead_emoji(lbl)))
        for b in m.get("footprint", {}).get("badges", []):
            if b.startswith("⚠️"):
                edge_tokens.append(f"<span class='warn-flag'>{esc(_strip_lead_emoji(b))}</span>")
            else:
                edge_tokens.append(esc(_strip_lead_emoji(b)))
        status_html = "".join(boxed)
        fp_html = f"<div class='edge-line'>{' · '.join(edge_tokens)}</div>" if edge_tokens else ""
        trendline_html = _trendline_block(m.get("trendline_data", {}))
        rs_val = m.get("rs_rating", "N/A")
        rs_asof = m.get("rs_asof")
        rs_mark = (f"<span title='carried-forward from {esc(rs_asof)} — not in today&#39;s RS source' "
                   f"style='color:var(--text-3);font-size:var(--fs-micro);vertical-align:super;'>*</span>"
                   if rs_asof else "")
        spark = m.get("spark", "")
        spark_html = f"<div class='spark'>{spark}</div>" if spark else ""

        # ANTS (David Ryan accumulation): today's chip + a 3-month prior-accumulation
        # note. Sortable by today-level, then 3M peak, then chain ("—" parks last).
        a_lvl = m.get("ants_level", 0)
        a_chain = m.get("ants_chain", 0)
        a_3m_peak = m.get("ants_3m_peak", 0)
        a_3m_days = m.get("ants_3m_days", 0)
        if m.get("ants_ok") and a_lvl > 0:
            _ants_col = {1: "#79c0ff", 2: "#f2cc60", 3: "#f2cc60",
                         4: "#3fb950", 5: "#2ea043"}.get(a_lvl, "#8b949e")
            _ants_wt = "700" if a_lvl >= 4 else "500"
            _ants_suffix = (" ·%db" % a_chain) if a_chain else ""
            _ants_title = "ANTS L%d %s" % (a_lvl, m.get("ants_label", ""))
            if a_chain:
                _ants_title += " · %d consecutive bars" % a_chain
            _ants_today = ("<span style='color:%s;font-weight:%s;font-family:var(--mono);"
                           "font-size:var(--fs-caption);' title='%s'>%s%s</span>"
                           % (_ants_col, _ants_wt, esc(_ants_title),
                              esc(m.get("ants_label", "")), _ants_suffix))
        else:
            _ants_today = "<span style='color:var(--text-3);'>—</span>"
        _ants_3m_html = ""
        if m.get("ants_ok") and a_3m_peak >= 1:
            _p3lbl = _ANTS_LABELS.get(a_3m_peak, "")
            _p3col = "#3fb950" if a_3m_peak >= 4 else ("#f2cc60" if a_3m_peak >= 2 else "#8b949e")
            _ants_3m_html = ("<div style='font-size:var(--fs-micro);color:%s;' "
                             "title='Peak ANTS level in the last ~3 months over %d active days'>"
                             "3M %s·%dd</div>" % (_p3col, a_3m_days, esc(_p3lbl), a_3m_days))
        ants_html = _ants_today + _ants_3m_html
        if m.get("ants_ok") and (a_lvl > 0 or a_3m_peak > 0):
            ants_sort = a_lvl * 100000 + a_3m_peak * 1000 + min(a_chain, 999)
        else:
            ants_sort = -1

        # RS-Line STANDOUT badge: flag only the genuine leaders — the RS line
        # (close/SPY) at a new high (Minervini's "RS new high" tell), and the
        # stronger "before price" stealth variant. Non-leaders show nothing, so a
        # blue badge = this name stands out on relative strength. (No per-row line
        # trend / sparkline — that just duplicated the price spark + RS rating.)
        rs_badge = ""
        if m.get("rs_ok"):
            if m.get("rs_nh_before_price"):
                rs_badge = ("<div class='fp-badge' style='border-color:#79c0ff;color:#79c0ff;font-weight:bold;' "
                            "title='RS line near its 1-year high while price has NOT broken out — stealth relative-strength leader'>"
                            "🔵 RS Leader ‹ Px</div>")
            elif m.get("rs_new_high"):
                rs_badge = ("<div class='fp-badge' style='border-color:#79c0ff;color:#79c0ff;' "
                            "title='RS line (close/SPY) at or near its 1-year high — relative-strength leader vs the market'>🔵 RS Leader</div>")

        # 52-week-high leadership badges: at a fresh high today + persistence
        # (recurring new highs over the last 13 weeks).
        nh_html = ""
        if m.get("at_high"):
            nh_html += "<div class='fp-badge' style='border-color:#3fb950;color:#3fb950;'>🆕 At 52W High</div>"
        if m.get("persist_tier"):
            _star = "⭐⭐" if m["persist_tier"] == "R" else "⭐"
            _pc = "#f2cc60" if m["persist_tier"] == "R" else "#f2cc60"
            nh_html += (f"<div class='fp-badge' style='border-color:{_pc};color:{_pc};font-weight:bold;'>"
                        f"{_star} {esc(m.get('persist_label',''))} · {m.get('nh_3m',0)}NH/3M ({m.get('weeks_3m',0)}w)</div>")

        # Martin pullback plan (additive second trade plan)
        pb_html = ""
        pb_risk = m.get("pb_risk")
        if pb_risk is not None:
            pbc = "#3fb950" if pb_risk <= 4.0 else ("#f2cc60" if pb_risk <= 6.0 else "#ff7b72")
            pb_html = (
                "<div class='entry-box' style='border-color:var(--bd-accent);background:var(--tint-accent);margin-top:6px;'>"
                "<span style='color:var(--text-3);font-weight:500;font-size:var(--fs-micro);'>PULLBACK</span><br>"
                f"<span class='entry-text' style='color:var(--accent-2);'>Buy ≈ ${m.get('pb_entry', m['entry'])}</span> <span class='stop-reason'>(to {esc(m['hugging'])})</span><br>"
                f"<span class='stop-text'>Stop: ${m['pb_stop']} <span class='stop-reason'>(~3% under 9EMA)</span></span><br>"
                f"<span style='color:{pbc};font-size:var(--fs-caption);font-family:var(--mono);'>Risk: {pb_risk}%</span></div>"
            )

        out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
            <td class="ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a></td>
            <td data-sort="{m['risk_pct']}">
                <div class="entry-box">
                    <span style="color:var(--text-3);font-weight:500;font-size:var(--fs-micro);">BREAKOUT</span><br>
                    <span class="entry-text">Buy: ${m['entry']}</span><br>
                    <span class="stop-text">Stop: ${m['stop']} <span class="stop-reason">({esc(m['stop_reason'])})</span></span><br>
                    <span style="color:{risk_color};font-size:var(--fs-caption);font-family:var(--mono);">Risk: {m['risk_pct']}%</span>
                </div>
                {pb_html}
            </td>
            <td data-sort="{m['close']}">{_lp(m['ticker'], m['close'], entry=m['entry'], stop=m['stop'])}{spark_html}<br>{_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span><br><span class="tag">{esc(m['sector'])}</span>{_ind_badge(m)}''')}</td>
            <td class="num" data-sort="{m['adr']}">{m['adr']}%</td>
            <td data-sort="{rs_val if isinstance(rs_val, int) else 0}"><span class="score">{esc(rs_val)}</span>{rs_mark}<br><span style="font-size:var(--fs-micro);color:var(--text-3);">1M: +{m['perf_1m']}%</span></td>
            <td data-sort="{meta_score}">
                <span style="font-size:var(--fs-body);font-weight:600;font-family:var(--mono);color:{ms_color};background:{ms_bg};padding:4px 8px;border-radius:4px;border:1px solid {ms_color};">{meta_score}</span>
                {_meta_details_block(m.get('meta_details', []))}
            </td>
            <td class="num" data-sort="{ants_sort}">{ants_html}</td>
            {_ma_cells(m.get('_ma_dist'))}{_fwd_yoy_cell(m['ticker'])}{_eps_accel_cell(m['ticker'])}
            <td style="text-align:left;" data-sort="{m['vol_pct']}">
                {nh_html}{rs_badge}{status_html}{fp_html}{trendline_html}
                <span class="{vol_color}">Vol: {m['vol_pct']}%</span><br>
                <span class="{dist_color}">Dist to {esc(m['hugging'])}: {m['dist_pct']}%</span><br>
                <span class="{dist52_color}">Off 52W: -{m['dist_52w']}%</span>
            </td>
        </tr>""")
    out.append("</table></div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# External engines: Minervini + Trilogy daily lists (their own tabs).
# Each engine OWNS its trade plan (pivot/stop/risk straight from its output);
# the price / narrative / ADR columns mirror the MADRRY watchlist layout and
# link out to TradingView the same way. Reads are defensive: a missing or
# unreadable source becomes a friendly inline note, never a scan failure.
# ---------------------------------------------------------------------------
MINERVINI_DIR = os.path.expanduser("~/minervini_engine/buy_lists")
TRILOGY_RTB = os.path.expanduser("~/Downloads/Chart learning project claude/webapp/ready_to_buy.json")


def _ext_empty(msg: str) -> str:
    return ("<div class='table-container' style='padding:18px 14px;color:var(--text-3);"
            "font-size:var(--fs-table);'>" + esc(msg) + "</div>")


def _ext_ticker_cell(tk: str) -> str:
    return (f'<td class="ticker" data-sort="{esc(tk)}">'
            f'<a href="https://www.tradingview.com/chart/?symbol={esc(tk)}" target="_blank">{esc(tk)}</a></td>')


def _enrich_external_rows(rows: List[dict], *, weekly_spark: bool = False,
                          spark_field: str = "Close", spark_n: int = 40,
                          market_modifier: float = 1.0,
                          spark_ma_spec: Optional[List[Tuple[int, str]]] = None) -> None:
    """Compute the MADRRY-style indicators for external-engine rows IN PLACE:
    M.E.T.A. score (+details), price sparkline (weekly when weekly_spark),
    Martin footprint badges (incl. Young Base), plus 52wk-high persistence,
    ANTS accumulation and RS-line leadership via the same attach_* passes the
    coil tables use. Each row must carry a 'ticker'; '_risk_pct' is used for the
    M.E.T.A. risk component when present. All fetches are best-effort."""
    if not rows:
        return
    tickers = sorted({r.get("ticker") for r in rows if r.get("ticker")})
    try:
        hist_map = fetch_histories_batch(tickers, period="1y", min_rows=40)
    except Exception:  # noqa: BLE001
        hist_map = {}
    for r in rows:
        r.setdefault("_meta_score", 0)
        r.setdefault("_meta_details", [])
        r.setdefault("_spark", "")
        r.setdefault("_footprint", {})
        df = hist_map.get(r.get("ticker"))
        if df is None or len(df) < 30:
            continue
        cl = df["Close"]
        close = float(cl.iloc[-1])
        r["_ma_dist"] = _ma_dist_data(cl.tolist())   # price-to-10/20/50-SMA sort column
        ema9 = float(cl.ewm(span=9, adjust=False).mean().iloc[-1])
        ema21 = float(cl.ewm(span=21, adjust=False).mean().iloc[-1])

        def _perf(n: int) -> float:
            if len(cl) > n and float(cl.iloc[-n - 1]) > 0:
                return (close / float(cl.iloc[-n - 1]) - 1) * 100.0
            return 0.0
        adr = _adr20(df)                          # canonical ADR%: 100×(mean(High/Low,20)−1)
        _hi, _lo = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1])
        day_range_pct = ((_hi / _lo) - 1.0) * 100.0 if _lo else 0.0   # High/Low basis, same as ADR
        hi52 = float(df["High"].iloc[-252:].max())
        dist_52w = (hi52 - close) / hi52 * 100.0 if hi52 > 0 else 0.0
        v50 = float(df["Volume"].iloc[-50:].mean())
        vol_pct = float(df["Volume"].iloc[-20:].mean() / v50 * 100.0) if v50 > 0 else 100.0
        meta_input = {
            "perf_1m": _perf(21), "perf_3m": _perf(63), "adr": adr, "close": close,
            "sma10": ema9, "sma20": ema21, "vol_pct": vol_pct,
            "risk_pct": r.get("_risk_pct", 10.0), "day_range_pct": day_range_pct,
            "dist_52w": dist_52w, "mcap": 0, "float_shares": 0,
        }
        try:
            md = calculate_meta_momentum_score(meta_input, df)
            r["_meta_score"] = _ranking_meta_score(df, md["score"], market_modifier)
            r["_meta_details"] = md["details"]
        except Exception:  # noqa: BLE001
            pass
        try:
            r["_footprint"] = analyze_footprint(df)
        except Exception:  # noqa: BLE001
            pass
        try:
            src = df[spark_field] if spark_field in df.columns else cl
            vol = df["Volume"] if "Volume" in df.columns else None
            # Keep the FULL source series so the MA overlay (when requested) is valid
            # across the whole displayed window; slice the INDEX to the last spark_n
            # for the volume bars / up-day shading.
            if weekly_spark:
                src_full = src.resample("W-FRI").last().dropna()
                idx = src_full.index[-spark_n:]
                cls_s = cl.resample("W-FRI").last().reindex(src_full.index).loc[idx]
                vol_s = vol.resample("W-FRI").sum().reindex(src_full.index).loc[idx] if vol is not None else None
            else:
                src_full = src.dropna()
                idx = src_full.index[-spark_n:]
                cls_s = cl.loc[idx]
                vol_s = vol.loc[idx] if vol is not None else None
            if spark_ma_spec:
                price_svg = make_sparkline(src_full.tolist(), width=132, height=34, pad=3,
                                           window=spark_n, show_ma=True, ma_spec=spark_ma_spec,
                                           price_sw=1.5, ma_sw=0.8, ma_labels=False)
            else:
                price_svg = make_sparkline(src_full.iloc[-spark_n:].tolist())
            vol_svg = ""
            if vol_s is not None and len(vol_s) >= 2:
                ups = [True] + [float(cls_s.iloc[k]) >= float(cls_s.iloc[k - 1])
                                for k in range(1, len(cls_s))]
                vol_svg = make_volume_bars(vol_s.tolist(), ups,
                                           width=132 if spark_ma_spec else 88)
            r["_spark"] = price_svg + vol_svg
        except Exception:  # noqa: BLE001
            pass
    # 52wk-high persistence + ANTS + RS-line leadership (self-fetch 2y + SPY).
    try:
        attach_persistence(rows)
    except Exception:  # noqa: BLE001
        pass
    try:
        attach_ants(rows)
    except Exception:  # noqa: BLE001
        pass


def _ext_meta_cell(m: dict) -> str:
    """M.E.T.A. column cell (score badge + expandable details), coil-identical."""
    score = m.get("_meta_score", 0)
    if score >= 70:
        col, bg = "#ff7b72", "var(--tint-red)"
    elif score >= 50:
        col, bg = "#f2cc60", "var(--tint-yellow)"
    else:
        col, bg = "#8b949e", "#21262d"
    return (f'<td data-sort="{score}">'
            f'<span style="font-size:var(--fs-body);font-weight:600;font-family:var(--mono);'
            f'color:{col};background:{bg};padding:4px 8px;border-radius:4px;border:1px solid {col};">{score}</span>'
            f'{_meta_details_block(m.get("_meta_details", []))}</td>')


def _ext_ants_cell(m: dict) -> str:
    """ANTS column cell (today chip + 3M peak), coil-identical."""
    a_lvl = m.get("ants_level", 0)
    a_chain = m.get("ants_chain", 0)
    a_3m_peak = m.get("ants_3m_peak", 0)
    a_3m_days = m.get("ants_3m_days", 0)
    if m.get("ants_ok") and a_lvl > 0:
        _col = {1: "#79c0ff", 2: "#f2cc60", 3: "#f2cc60", 4: "#3fb950", 5: "#2ea043"}.get(a_lvl, "#8b949e")
        _wt = "700" if a_lvl >= 4 else "500"
        _suffix = (" ·%db" % a_chain) if a_chain else ""
        _title = "ANTS L%d %s" % (a_lvl, m.get("ants_label", ""))
        if a_chain:
            _title += " · %d consecutive bars" % a_chain
        today = ("<span style='color:%s;font-weight:%s;font-family:var(--mono);font-size:var(--fs-caption);'"
                 " title='%s'>%s%s</span>" % (_col, _wt, esc(_title), esc(m.get("ants_label", "")), _suffix))
    else:
        today = "<span style='color:var(--text-3);'>—</span>"
    p3 = ""
    if m.get("ants_ok") and a_3m_peak >= 1:
        _p3lbl = _ANTS_LABELS.get(a_3m_peak, "")
        _p3col = "#3fb950" if a_3m_peak >= 4 else ("#f2cc60" if a_3m_peak >= 2 else "#8b949e")
        p3 = ("<div style='font-size:var(--fs-micro);color:%s;' title='Peak ANTS level in the last ~3 months "
              "over %d active days'>3M %s·%dd</div>" % (_p3col, a_3m_days, esc(_p3lbl), a_3m_days))
    if m.get("ants_ok") and (a_lvl > 0 or a_3m_peak > 0):
        srt = a_lvl * 100000 + a_3m_peak * 1000 + min(a_chain, 999)
    else:
        srt = -1
    return f'<td class="num" data-sort="{srt}">{today}{p3}</td>'


def _ext_leader_badges(m: dict) -> str:
    """🔵 RS Leader + ⭐ Persistent/Relentless Leader + 🆕 At 52W High, coil-identical."""
    rs_badge = ""
    if m.get("rs_ok"):
        if m.get("rs_nh_before_price"):
            rs_badge = ("<div class='fp-badge' style='border-color:#79c0ff;color:#79c0ff;font-weight:bold;' "
                        "title='RS line near its 1-year high while price has NOT broken out — stealth relative-strength leader'>"
                        "🔵 RS Leader ‹ Px</div>")
        elif m.get("rs_new_high"):
            rs_badge = ("<div class='fp-badge' style='border-color:#79c0ff;color:#79c0ff;' "
                        "title='RS line (close/SPY) at or near its 1-year high — relative-strength leader vs the market'>🔵 RS Leader</div>")
    nh_html = ""
    if m.get("at_high"):
        nh_html += "<div class='fp-badge' style='border-color:#3fb950;color:#3fb950;'>🆕 At 52W High</div>"
    if m.get("persist_tier"):
        _star = "⭐⭐" if m["persist_tier"] == "R" else "⭐"
        nh_html += (f"<div class='fp-badge' style='border-color:#f2cc60;color:#f2cc60;font-weight:bold;'>"
                    f"{_star} {esc(m.get('persist_label',''))} · {m.get('nh_3m',0)}NH/3M ({m.get('weeks_3m',0)}w)</div>")
    return nh_html + rs_badge


def _ext_fp_badges(m: dict) -> str:
    """Martin footprint badges (Young Base, Higher-Lows, EMAs Coiled, Base Nw, AVWAP…)."""
    toks = []
    for b in m.get("_footprint", {}).get("badges", []):
        if b.startswith("⚠️"):
            toks.append(f"<span class='warn-flag'>{esc(_strip_lead_emoji(b))}</span>")
        else:
            toks.append(esc(_strip_lead_emoji(b)))
    return f"<div class='edge-line'>{' · '.join(toks)}</div>" if toks else ""


def _read_json_retry(path: str, attempts: int = 4, delay: float = 0.8):
    """Read + parse a JSON file, retrying on a transient OSError/ValueError.

    The external Trilogy/Minervini feeds are rewritten IN PLACE each morning ~1 min
    before our 08:16 scan (no atomic temp-then-rename), so a single read can catch the
    file mid-write — truncated -> ValueError, or momentarily absent -> OSError — which
    would zero the tab for the whole day (observed 2026-06-26: Trilogy '...not found /
    unreadable' while the file was fine seconds later). A couple of short retries ride
    the publisher's write window out. Raises the last error only if every attempt fails."""
    last: Optional[Exception] = None
    for i in range(attempts):
        try:
            with open(path, "r") as fh:
                return json.load(fh)
        except (OSError, ValueError) as exc:
            last = exc
            if i < attempts - 1:
                time.sleep(delay)
    raise last  # type: ignore[misc]


def generate_minervini_table(market_modifier: float = 1.0) -> Tuple[str, int]:
    """Minervini engine daily VCP/SEPA buy list -> MADRRY-style table."""
    try:
        files = [f for f in os.listdir(MINERVINI_DIR)
                 if f.startswith("buy_list_") and f.endswith(".json")]
    except OSError:
        files = []
    if not files:
        return _ext_empty("Minervini engine: no buy_list file found (~/minervini_engine/buy_lists)."), 0
    latest = sorted(files)[-1]
    asof = latest[len("buy_list_"):-len(".json")]
    try:
        rows = _read_json_retry(os.path.join(MINERVINI_DIR, latest))
    except (OSError, ValueError):
        return _ext_empty(f"Minervini engine: could not read {latest}."), 0
    if not rows:
        return _ext_empty(f"Minervini engine: 0 picks on {asof}."), 0

    for m in rows:
        m["_risk_pct"] = round((m.get("stop_frac") or 0.0) * 100, 1)
    _enrich_external_rows(rows, weekly_spark=False, spark_field="Close", spark_n=60,
                          market_modifier=market_modifier, spark_ma_spec=_MA_SPEC_DARK)
    _prefetch_fundamentals([m.get("ticker") for m in rows], budget_s=30.0)

    out = [
        f"<div class='ext-asof'>🏛️ Minervini engine · daily VCP/SEPA buy list · as of {esc(asof)} · "
        f"{len(rows)} names · trade plan from <code>minervini_engine</code> · "
        f"M.E.T.A./ANTS/sparkline (daily close · 60d + 10·20·50 MA + volume)/leader badges computed by MADRRY</div>",
        "<div class='table-container'><table data-schema='minervini'>",
        "<tr><th data-col='tk'>Ticker</th><th data-col='plan'>Trade Plan</th><th data-col='price'>Price &amp; Narrative</th><th class='num' data-col='adr' title='Average Daily Range — 20-day avg of (High/Low−1), % · how much it typically moves per day (TradingView ADRP)'>ADR</th>"
        "<th data-col='rs'>RS</th><th data-col='meta'>M.E.T.A.</th><th class='num' data-col='ants'>ANTS</th>"
        + _MA_YOY_HEADERS +
        "<th data-col='status'>Status (VCP &amp; Vol)</th></tr>",
    ]
    for m in rows:
        tk = m.get("ticker", "")
        pivot, stop, close = m.get("pivot"), m.get("stop"), m.get("last_close")
        risk = round((m.get("stop_frac") or 0.0) * 100, 1)
        risk_color = "#3fb950" if risk <= 4.0 else ("#f2cc60" if risk <= 8.0 else "#ff7b72")
        status = (m.get("status") or "").replace("_", " ")
        triggered = "TRIGGER" in status
        st_color = "#3fb950" if triggered else "#f2cc60"
        st_bg = "var(--tint-green)" if triggered else "var(--tint-yellow)"
        vcp = m.get("vcp_score", 0) or 0
        vc_color = "#ff7b72" if vcp >= 85 else ("#f2cc60" if vcp >= 75 else "#8b949e")
        rs = m.get("rs", "N/A")
        adr = m.get("adr", 0)
        sector = m.get("sector", "")
        foot = m.get("footprint", "")
        rev = m.get("rev_yoy")
        perf6 = m.get("perf6m")
        ptp = m.get("pct_to_pivot")
        offhi = m.get("pct_from_high")
        slope = m.get("vol50_slope")
        rev_line = (f"<br><span style='font-size:var(--fs-micro);color:var(--text-3);'>Rev YoY {rev:+.0f}%</span>"
                    if isinstance(rev, (int, float)) else "")
        perf6_line = (f"<br><span style='font-size:var(--fs-micro);color:var(--text-3);'>6M: {perf6:+.0f}%</span>"
                      if isinstance(perf6, (int, float)) else "")
        if ptp is None:
            ptp_line = ""
        elif ptp <= 0:
            ptp_line = "<span class='good'>▲ triggered / through pivot</span>"
        else:
            ptp_line = f"<span class='warn'>{ptp:.1f}% to pivot</span>"
        offhi_line = (f"<br><span class='{'good' if (offhi or 0) >= -8 else 'bad'}'>Off high: {offhi:.1f}%</span>"
                      if isinstance(offhi, (int, float)) else "")
        slope_line = (f"<br><span style='color:var(--text-3);font-size:var(--fs-micro);'>Vol50 slope: {slope:+.1f}%</span>"
                      if isinstance(slope, (int, float)) else "")
        price_cell = _lp(tk, close, entry=pivot, stop=stop) if close is not None else "—"
        spark_html = f"<div class='spark'>{m.get('_spark','')}</div>" if m.get("_spark") else ""
        leader_html = _ext_leader_badges(m)
        fp_html = _ext_fp_badges(m)
        out.append(f"""<tr data-sector="{esc(sector)}">
            {_ext_ticker_cell(tk)}
            <td data-sort="{risk}">
                <div class="entry-box">
                    <span style="color:var(--text-3);font-weight:500;font-size:var(--fs-micro);">VCP PIVOT</span><br>
                    <span class="entry-text">Buy: ${pivot}</span><br>
                    <span class="stop-text">Stop: ${stop}</span><br>
                    <span style="color:{risk_color};font-size:var(--fs-caption);font-family:var(--mono);">Risk: {risk}%</span>
                </div>
            </td>
            <td data-sort="{close if close is not None else 0}">{price_cell}{spark_html}<br>
                {_narrative(tk, f'''<span class="tag">{esc(foot)}</span>{rev_line}<br>
                <span class="tag">{esc(sector)}</span>''')}</td>
            <td class="num" data-sort="{adr}">{adr}%</td>
            <td data-sort="{rs if isinstance(rs, int) else 0}"><span class="score">{esc(rs)}</span>{perf6_line}</td>
            {_ext_meta_cell(m)}
            {_ext_ants_cell(m)}
            {_ma_cells(m.get('_ma_dist'))}{_fwd_yoy_cell(tk)}{_eps_accel_cell(tk)}
            <td style="text-align:left;" data-sort="{ptp if ptp is not None else 999}">
                {leader_html}{fp_html}
                <span class="fp-badge" style="border-color:{st_color};color:{st_color};background:{st_bg};">{esc(status)}</span>
                <span class="fp-badge" style="border-color:{vc_color};color:{vc_color};" title="Minervini VCP score">VCP {vcp}</span><br>
                {ptp_line}{offhi_line}{slope_line}
            </td>
        </tr>""")
    out.append("</table></div>")
    return "".join(out), len(rows)


def generate_trilogy_table(limit: Optional[int] = None, market_modifier: float = 1.0) -> Tuple[str, int]:
    """Trilogy nightly O'Neil reference-class buy-stop list -> MADRRY-style table.
    limit=None shows ALL candidates (default); pass an int to cap the display."""
    try:
        data = _read_json_retry(TRILOGY_RTB)
    except (OSError, ValueError):
        return _ext_empty("Trilogy: ready_to_buy.json not found / unreadable."), 0
    cands = data.get("candidates", []) or []
    asof = data.get("asof", "")
    total = len(cands)
    if not cands:
        return _ext_empty(f"Trilogy: 0 candidates on {asof}."), 0
    shown = cands if limit is None else cands[:limit]
    extra = f" · showing top {len(shown)} of {total}" if total > len(shown) else ""

    # Stash each candidate's risk for the M.E.T.A. risk component, then enrich
    # the SHOWN rows with MADRRY indicators (weekly sparkline for Trilogy).
    for c in shown:
        _piv = c.get("pivot")
        _stp = round(_piv * 0.92, 2) if isinstance(_piv, (int, float)) else None
        _ib = c.get("ideal_buy")
        c["_risk_pct"] = (round((_ib - _stp) / _ib * 100, 1)
                          if isinstance(_ib, (int, float)) and isinstance(_stp, (int, float)) and _ib else 10.0)
    _enrich_external_rows(shown, weekly_spark=True, market_modifier=market_modifier,
                          spark_ma_spec=_MA_SPEC_10W)
    _prefetch_fundamentals([c.get("ticker") for c in shown], budget_s=30.0)

    out = [
        f"<div class='ext-asof'>📚 Trilogy nightly · O'Neil reference-class buy-stop list · as of {esc(asof)} · "
        f"{total} candidates{extra} · trade plan from <code>trilogy webapp</code> · "
        f"M.E.T.A./ANTS/weekly sparkline (+10-week MA)/leader badges computed by MADRRY</div>",
        "<div class='table-container'><table data-schema='trilogy'>",
        "<tr><th data-col='tk'>Ticker</th><th data-col='plan'>Trade Plan</th><th data-col='price'>Price &amp; Narrative</th><th data-col='grade'>Grade</th>"
        "<th class='num' data-col='win20'>Win20</th><th data-col='meta'>M.E.T.A.</th><th class='num' data-col='ants'>ANTS</th>"
        + _MA_YOY_HEADERS +
        "<th data-col='status'>Status</th></tr>",
    ]
    grade_col = {"A": "#3fb950", "B": "#79c0ff", "C": "#f2cc60", "D": "#ff7b72", "F": "#ff7b72"}
    for c in shown:
        tk = c.get("ticker", "")
        ideal = c.get("ideal_buy")
        top = c.get("buy_range_top")
        pivot = c.get("pivot")
        close = c.get("last_close")
        stop = round(pivot * 0.92, 2) if isinstance(pivot, (int, float)) else None
        if isinstance(ideal, (int, float)) and isinstance(stop, (int, float)) and ideal:
            risk = round((ideal - stop) / ideal * 100, 1)
        else:
            risk = 0.0
        risk_color = "#3fb950" if risk <= 8.0 else ("#f2cc60" if risk <= 12.0 else "#ff7b72")
        grade = c.get("grade", "")
        gcol = grade_col.get(grade, "#8b949e")
        likeness = c.get("likeness_q")
        likeness_line = (f"<br><span style='font-size:var(--fs-micro);color:var(--text-3);'>likeness Q{likeness}</span>"
                         if likeness is not None else "")
        pattern = (c.get("pattern") or "").replace("_", " ")
        family = c.get("family", "")
        stage = c.get("stage")
        stage_txt = f"Stage {stage}" if stage is not None else ""
        sector = c.get("sector", "")
        rs_line = (c.get("rs_line") or "").replace("_", " ")
        rs_rank = c.get("sector_rs_rank")
        rs_rank_line = (f"<br><span style='font-size:var(--fs-micro);color:var(--text-3);'>sector RS #{rs_rank}</span>"
                        if rs_rank else "")
        win20 = c.get("win20_rate")
        win_txt = f"{win20 * 100:.0f}%" if isinstance(win20, (int, float)) else "—"
        status = c.get("status", "")
        gated = c.get("gated")
        checklist = c.get("checklist", "")
        mtail = c.get("monster_tail")
        mdec = c.get("monster_tail_decile")
        detail = c.get("detail", "")
        st_color = "#ff7b72" if gated else ("#3fb950" if "BUYING RANGE" in status else "#8b949e")
        mtail_line = (f"<br><span class='fp-badge fp-warn'>~MONSTER-TAIL d{mdec}</span>" if mtail else "")
        range_txt = f" <span class='stop-reason'>(top ${top})</span>" if top is not None else ""
        price_cell = _lp(tk, close, entry=ideal, stop=stop) if close is not None else "—"
        spark_html = f"<div class='spark'>{c.get('_spark','')}</div>" if c.get("_spark") else ""
        leader_html = _ext_leader_badges(c)
        fp_html = _ext_fp_badges(c)
        rs_line_html = ((f"<span style='color:var(--accent-2);font-size:var(--fs-micro);'>RS line: {esc(rs_line)}</span>"
                         + (f" <span style='color:var(--text-3);font-size:var(--fs-micro);'>· sector RS #{rs_rank}</span>" if rs_rank else "")
                         + "<br>") if rs_line else "")
        out.append(f"""<tr data-sector="{esc(sector)}">
            {_ext_ticker_cell(tk)}
            <td data-sort="{risk}">
                <div class="entry-box">
                    <span style="color:var(--text-3);font-weight:500;font-size:var(--fs-micro);">BUY-STOP</span><br>
                    <span class="entry-text">Buy: ${ideal}</span>{range_txt}<br>
                    <span class="stop-text">Stop: ${stop} <span class="stop-reason">(pivot −8%)</span></span><br>
                    <span style="color:{risk_color};font-size:var(--fs-caption);font-family:var(--mono);">Risk: {risk}%</span>
                </div>
            </td>
            <td data-sort="{close if close is not None else 0}">{price_cell}{spark_html}<br>
                {_narrative(tk, f'''<span class="theme-tag">{esc(pattern)}</span><br>
                <span class="tag">{esc(family)}</span> <span class="tag">{esc(stage_txt)}</span> <span class="tag">{esc(sector)}</span>''')}</td>
            <td data-sort="{esc(grade)}"><span class="grade-badge" style="color:{gcol};border:1px solid {gcol};background:rgba(0,0,0,0.18);">{esc(grade) or '—'}</span>{likeness_line}</td>
            <td class="num" data-sort="{win20 if isinstance(win20, (int, float)) else 0}">{win_txt}</td>
            {_ext_meta_cell(c)}
            {_ext_ants_cell(c)}
            {_ma_cells(c.get('_ma_dist'))}{_fwd_yoy_cell(tk)}{_eps_accel_cell(tk)}
            <td style="text-align:left;" data-sort="{esc(status)}" title="{esc(detail)}">
                {leader_html}{fp_html}{rs_line_html}
                <span class="fp-badge" style="border-color:{st_color};color:{st_color};">{esc(status)}</span>
                <span class="warn-flag" style="font-size:var(--fs-micro);"> {esc(checklist)}</span>{mtail_line}
            </td>
        </tr>""")
    out.append("</table></div>")
    return "".join(out), total


def generate_hve_table(ep_matches: List[dict]) -> str:
    out = ['<div class="section-title bg-hve">💥 HVE (EPISODIC PIVOTS) — LOW FLOAT ≤200M</div>',
           '<div class="table-container"><table>',
           "<tr><th>Ticker</th><th>Price &amp; Gap</th><th>Narrative &amp; Conviction</th><th>QM Trade Plan</th></tr>"]
    if not ep_matches:
        out.append("<tr><td colspan='4' style='color:#8b949e;'>No HVE events detected today.</td></tr>")
    else:
        for m in ep_matches:
            risk_color = "#3fb950" if m["risk_pct"] <= 4.0 else ("#f2cc60" if m["risk_pct"] <= 6.0 else "#ff7b72")
            float_txt = f"{m['float_shares']}M" if m["float_shares"] else "N/A"
            out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
                <td class="ep-ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a></td>
                <td data-sort="{m['close']}">{_lp(m['ticker'], m['close'], entry=m['entry'], stop=m['stop'])} <span class="good">(+{m['change']}%)</span><br><span style="font-size:var(--fs-caption);color:#8b949e;">Gap: {m['gap']}%</span></td>
                <td data-sort="{m['rel_vol']}">{_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span>''')}<br><br><span class="hve-badge">{m['rel_vol']}x Avg!</span><br><span style="font-size:var(--fs-caption);color:#8b949e;margin-top:4px;display:inline-block;">Float: {float_txt}</span></td>
                <td data-sort="{m['risk_pct']}">
                    <div style="font-size:var(--fs-caption);color:#a5d6ff;text-align:left;margin-bottom:4px;">✔️ Close Range {m['close_range']}%</div>
                    <div class="entry-box">
                        <span class="entry-text">Buy-Stop: ${m['entry']}</span><br>
                        <span class="stop-text">Stop: ${m['stop']} <span class="stop-reason">({esc(m['stop_reason'])})</span></span><br>
                        <span style="color:{risk_color};font-size:var(--fs-body);">Risk: {m['risk_pct']}%</span>
                    </div>
                </td>
            </tr>""")
    out.append("</table></div>")
    return "".join(out)


def generate_tier_a_study_tab() -> Tuple[str, int]:
    """Render the forward Tier-A win/loss study from tier_a_tracking.json.

    Reads the precomputed study_summary block written by
    madrry_tier_a_tracker.py — this tab renders, it does not recompute.
    Returns (html, resolved_count) so the tab button can show a count.
    """
    try:
        with open(TIER_A_TRACKING_PATH) as fh:
            db = json.load(fh)
        s = db["study_summary"]
    except Exception as exc:  # noqa: BLE001
        return (f"<div class='table-container' style='padding:16px;color:#8b949e;'>"
                f"Tier-A tracking study not available yet "
                f"({esc(str(exc))}). It is generated after the next scan by "
                f"<code>madrry_tier_a_tracker.py</code>.</div>", 0)

    def wr_color(wr):
        if wr is None:
            return "#8b949e"
        if wr >= 60:
            return "#3fb950"
        if wr >= 50:
            return "#d29922"
        if wr >= 40:
            return "#db6d28"
        return "#f85149"

    def bucket_table(title, rows, label="bucket"):
        h = [f"<div class='section-title' style='background-color:var(--surface);"
             f"color:#79c0ff;border-bottom:2px solid #30363d;'>{title}</div>",
             "<div class='table-container'><table>",
             f"<tr><th>{label}</th><th>Win</th><th>Loss</th><th>N</th><th>Win%</th></tr>"]
        if not rows:
            h.append("<tr><td colspan='5' style='color:#8b949e;'>No resolved names yet.</td></tr>")
        for r in rows:
            wr = r.get("wr")
            h.append(
                f"<tr><td style='text-align:left;font-weight:600;'>{esc(r.get('k', r.get('bucket','')))}</td>"
                f"<td style='color:#3fb950;'>{r['w']}</td>"
                f"<td style='color:#f85149;'>{r['l']}</td>"
                f"<td>{r['n']}</td>"
                f"<td style='color:{wr_color(wr)};font-weight:700;'>"
                f"{(str(wr)+'%') if wr is not None else '—'}</td></tr>")
        h.append("</table></div>")
        return "".join(h)

    ov = s["overall"]
    recal = s["recal"]

    # --- header / methodology ---
    out = [
        "<div class='section-title' style='background-color:var(--surface);"
        "color:#d2a8ff;border-bottom:3px solid #d2a8ff;'>📈 TIER-A FORWARD WIN/LOSS STUDY</div>",
        "<div class='table-container' style='padding:14px 16px;line-height:1.6;'>",
        f"<div style='font-size:var(--fs-body);'>"
        f"<b style='color:#c9d1d9;'>{ov['total']}</b> Tier-A names tracked from first appearance · "
        f"<b style='color:#3fb950;'>{ov['w']} win</b> / <b style='color:#f85149;'>{ov['l']} loss</b> resolved · "
        f"overall win-rate <b style='color:{wr_color(ov['wr'])};'>{ov['wr']}%</b> · "
        f"<span style='color:#8b949e;'>{ov['open']} still open · data as-of {esc(s.get('asof',''))}</span></div>",
        "<div style='font-size:var(--fs-caption);color:#8b949e;margin-top:6px;'>"
        "WIN = fresh 52-week high after pick · LOSS = intraday touch of entry×0.92 (−8%) · "
        "40-bar window · first event wins · one tracker per ticker (first appearance). "
        "Young sample — buckets firm up as open names resolve.</div>",
        "</div>",
    ]
    if ov["n"] < 8:
        out.append("<div class='table-container' style='padding:12px 16px;color:#d29922;'>"
                   "⚠️ Fewer than 8 resolved names — treat all figures as provisional.</div>")

    # --- headline: win-rate by META score ---
    out.append(bucket_table("WIN-RATE BY META SCORE  (the headline: does higher META → higher win-rate?)",
                            s["by_meta"], "META"))

    # --- active weights vs next-fit preview ---
    rt = ["<div class='section-title' style='background-color:var(--surface);"
          "color:#79c0ff;border-bottom:2px solid #30363d;'>"
          "M.E.T.A. WEIGHTS — active (live) vs next-fit preview</div>",
          "<div class='table-container' style='padding:8px 12px;font-size:var(--fs-caption);"
          "color:#8b949e;'>Active weights are LIVE in scoring now. ‘Preview’ = the "
          "weekend re-fit candidate; it is applied only if it separates winners "
          "strictly better than the active set (else held). Edge &gt;0 = component "
          "scored higher on winners.</div>",
          "<div class='table-container'><table>",
          "<tr><th>Component</th><th>Active</th><th>Edge</th><th>Preview</th><th>Δ</th></tr>"]
    for w in recal["weights"]:
        d = w["new"] - w["cur"]
        dcol = "#3fb950" if d > 0 else ("#f85149" if d < 0 else "#8b949e")
        ecol = "#3fb950" if w["edge"] > 0 else ("#f85149" if w["edge"] < 0 else "#8b949e")
        rt.append(f"<tr><td style='text-align:left;'>{esc(w['comp'])}</td>"
                  f"<td style='font-weight:700;'>{w['cur']}</td>"
                  f"<td style='color:{ecol};'>{w['edge']:+.2f}</td>"
                  f"<td>{w['new']}</td>"
                  f"<td style='color:{dcol};'>{d:+d}</td></tr>")
    rt.append("</table></div>")
    out.append("".join(rt))

    # active vs preview monotonicity side by side
    out.append("<div style='display:flex;flex-wrap:wrap;gap:12px;'>"
               "<div style='flex:1;min-width:280px;'>"
               + bucket_table(f"ACTIVE weights (spread {recal['spread_v1']} pts)",
                              recal["v1_buckets"], "score")
               + "</div><div style='flex:1;min-width:280px;'>"
               + bucket_table(f"Next-fit preview (spread {recal['spread_v2']} pts)",
                              recal["v2_buckets"], "score")
               + "</div></div>")

    # --- component edges ---
    ct = ["<div class='section-title' style='background-color:var(--surface);"
          "color:#79c0ff;border-bottom:2px solid #30363d;'>"
          "META COMPONENT — winner vs loser average points</div>",
          "<div class='table-container'><table>",
          "<tr><th>Component</th><th>Win avg</th><th>Loss avg</th><th>Edge</th></tr>"]
    for c in s["components"]:
        e = c["edge"]
        ecol = "#3fb950" if (e or 0) > 0 else ("#f85149" if (e or 0) < 0 else "#8b949e")
        ct.append(f"<tr><td style='text-align:left;'>{esc(c['comp'])}</td>"
                  f"<td>{c['win_avg'] if c['win_avg'] is not None else '—'}</td>"
                  f"<td>{c['loss_avg'] if c['loss_avg'] is not None else '—'}</td>"
                  f"<td style='color:{ecol};font-weight:700;'>"
                  f"{(('%+.1f' % e) if e is not None else '—')}</td></tr>")
    ct.append("</table></div>")
    out.append("".join(ct))

    # --- the other cuts ---
    out.append("<div style='display:flex;flex-wrap:wrap;gap:12px;'>")
    for title, rows, lbl in [
        ("BY RS RATING", s["by_rs"], "RS"),
        ("BY TIER", s["by_tier"], "tier"),
        ("BY HTF FLAG", s["by_htf"], "flag"),
        ("BY ANTS", s["by_ants"], "ANTS"),
        ("BY AT-HIGH STATUS", s["by_athigh"], "status"),
    ]:
        out.append("<div style='flex:1;min-width:220px;'>"
                   + bucket_table(title, rows, lbl) + "</div>")
    out.append("</div>")
    out.append(bucket_table("WIN-RATE BY SECTOR", s["by_sector"], "sector"))

    return "".join(out), ov["n"]


def generate_ur_table(ur_matches: List[dict]) -> str:
    out = ['<div class="section-title" style="background-color:var(--surface);color:#79c0ff;border-bottom:3px solid #79c0ff;">🎯 POST-HVE U&amp;R (PULLBACK &amp; UNDERCUT)</div>',
           '<div class="table-container"><table>',
           "<tr><th>Ticker</th><th>Price &amp; 1W Perf</th><th>Narrative &amp; Status</th><th>U&amp;R Trade Plan</th></tr>"]
    if not ur_matches:
        out.append("<tr><td colspan='4' style='color:#8b949e;'>No Post-HVE U&amp;R candidates. Waiting for HVE stocks to consolidate...</td></tr>")
    else:
        for m in ur_matches:
            risk_color = "#3fb950" if m["risk_pct"] <= 3.5 else ("#f2cc60" if m["risk_pct"] <= 5.0 else "#ff7b72")
            vol_color = "good" if m["vol_contraction"] <= 40 else ("warn" if m["vol_contraction"] <= 60 else "bad")
            holding_color = "good" if m["holding_above_low"] else "bad"
            out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
                <td class="ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank" style="color:#79c0ff;">{esc(m['ticker'])}</a></td>
                <td data-sort="{m['close']}">{_lp(m['ticker'], m['close'], entry=m['entry'], stop=m['stop'])} <span style="color:#8b949e;">({m['change']:+}%)</span><br><span style="font-size:var(--fs-caption);color:#79c0ff;background:var(--tint-accent);padding:2px 6px;border-radius:4px;">Day {m['days_since_hve']} since HVE</span></td>
                <td data-sort="{m['vol_contraction']}">{_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span>''')}<br><br>
                    <span class="squat-badge {vol_color}">Vol: {m['vol_contraction']:.0f}% of Day 1</span><br>
                    <span class="squat-badge {holding_color}">Above D1 Low: {'Yes ✓' if m['holding_above_low'] else 'No ✗'}</span><br>
                    <span style="font-size:var(--fs-micro);color:#8b949e;">D1 High: ${m['day1_high']}</span>
                </td>
                <td data-sort="{m['risk_pct']}">
                    <div style="font-size:var(--fs-caption);color:#79c0ff;text-align:left;margin-bottom:4px;">⚡ U&amp;R: Undercut D{m['days_since_hve']-1}L then reclaim</div>
                    <div class="entry-box" style="border-color:#79c0ff;background-color:rgba(210,168,255,0.1);">
                        <span class="entry-text" style="color:#79c0ff;">Buy: ${m['entry']}</span><br>
                        <span class="stop-text">Stop: ${m['stop']} <span class="stop-reason">({esc(m['stop_reason'])})</span></span><br>
                        <span style="color:{risk_color};font-size:var(--fs-body);">Risk: {m['risk_pct']}%</span>
                    </div>
                </td>
            </tr>""")
    out.append("</table></div>")
    out.append("""
    <div style="background-color:var(--tint-accent);border-left:4px solid #79c0ff;padding:15px;margin:20px 0;border-radius:0 8px 8px 0;">
        <div style="color:#79c0ff;font-weight:bold;margin-bottom:8px;">📖 Post-HVE U&amp;R Strategy (Like MXL Apr 28)</div>
        <div style="font-size:var(--fs-table);color:#c9d1d9;line-height:1.6;">
            <strong>Setup:</strong> Day 2-5 after massive HVE breakout. Volume dries to &lt;50% of Day 1 (institutional holding).<br>
            <strong>Trigger:</strong> Price undercuts Previous Day Low, flushes stops, then violently reclaims.<br>
            <strong>Entry:</strong> On reclaim of the undercut level with volume expansion.<br>
            <strong>Stop:</strong> 1% below the wick low of the undercut day (typically 2-4% risk).<br>
            <strong>Example:</strong> MXL Day 3 — vol dried from 29M to 5.6M (19%), undercut $50.40 to $49.60, reclaimed for 2.6% risk, then +30% next day!
        </div>
    </div>""")
    return "".join(out)


_EXT_PCTILES: Optional[dict] = None


def _load_ext_pctiles() -> dict:
    """Load spy_qqq_extension_percentiles.csv -> {TICKER: {MA_COL: [(value, pctile)...]}}."""
    global _EXT_PCTILES
    if _EXT_PCTILES is not None:
        return _EXT_PCTILES
    out: dict = {}
    try:
        with open(EXT_PCTILE_PATH, newline="") as f:
            for row in csv.DictReader(f):
                tk = row["ticker"].strip().upper()
                p = float(row["pctile"])
                d = out.setdefault(tk, {})
                for col in ("SMA10", "SMA20", "SMA50", "EMA10", "EMA20", "EMA50"):
                    d.setdefault(col, []).append((float(row[col]), p))
        for tk in out:
            for col in out[tk]:
                out[tk][col].sort()
    except Exception:  # noqa: BLE001
        out = {}
    _EXT_PCTILES = out
    return out


def ext_percentile(ticker: str, ma_col: str, ext_value: float) -> Optional[float]:
    """Map a current extension % to its historical percentile (interpolated)."""
    tbl = _load_ext_pctiles().get((ticker or "").upper(), {}).get(ma_col)
    if not tbl:
        return None
    if ext_value <= tbl[0][0]:
        return tbl[0][1]
    if ext_value >= tbl[-1][0]:
        return tbl[-1][1]
    for i in range(1, len(tbl)):
        v0, p0 = tbl[i - 1]
        v1, p1 = tbl[i]
        if v0 <= ext_value <= v1:
            return p1 if v1 == v0 else p0 + (ext_value - v0) / (v1 - v0) * (p1 - p0)
    return None


# ---- forward base rates: "if this state, what has SPY/QQQ/IWM done next?" ----
# Built offline by build_forward_baserates.py from full inception history with the
# SAME metric definitions as the live cards. OOS validation (predictive_power.py)
# showed regime(200MA) × distribution-days is the most predictive combination, so
# that is the primary lookup; below-all-3-MAs extension is a secondary context line.
_FORWARD_BR: Optional[dict] = None


def _load_forward_baserates() -> dict:
    global _FORWARD_BR
    if _FORWARD_BR is not None:
        return _FORWARD_BR
    out: dict = {}
    try:
        with open(FORWARD_BASERATE_PATH) as f:
            out = json.load(f)
    except Exception:  # noqa: BLE001
        out = {}
    _FORWARD_BR = out
    return out


def _dist_bucket(dist: float) -> str:
    d = int(dist or 0)
    if d <= 2:
        return "0-2"
    if d <= 4:
        return "3-4"
    if d <= 6:
        return "5-6"
    if d <= 8:
        return "7-8"
    return "9+"


def forward_baserate(ticker: str, dist_days: float, above_200: Optional[bool],
                     min_n: int = 30) -> Optional[dict]:
    """Most-specific reliable forward base-rate cell for the live state.
    Fallback ladder (each needs the 4w sample n>=min_n):
        regime × dist bucket  ->  dist bucket (regime-agnostic)  ->  baseline."""
    blk = _load_forward_baserates().get((ticker or "").upper())
    if not blk:
        return None
    bucket = _dist_bucket(dist_days)

    def ok(cell):
        return bool(cell and cell.get("f4w") and cell["f4w"].get("n", 0) >= min_n)

    if above_200 is not None:
        rk = "bull" if above_200 else "bear"
        cell = blk.get("regime_dist", {}).get(rk, {}).get(bucket)
        if ok(cell):
            tag = "&gt;200MA" if above_200 else "&lt;200MA"
            return {"scope": "regime_dist", "label": f"{tag} · {bucket} dist days",
                    "stats": cell}
    cell = blk.get("dist", {}).get(bucket)
    if ok(cell):
        return {"scope": "dist", "label": f"{bucket} dist days (all regimes)", "stats": cell}
    cell = blk.get("baseline")
    if ok(cell):
        return {"scope": "baseline", "label": "any day (baseline)", "stats": cell}
    return None


def forward_ext_baserate(ticker: str, ext10: float, ext20: float, ext50: float,
                         above_200: Optional[bool], min_n: int = 30) -> Optional[dict]:
    """Secondary extension-only context: the below-all-3-MAs base rate, regime-aware
    when available. Only returned when price is actually below all three MAs."""
    blk = _load_forward_baserates().get((ticker or "").upper())
    if not blk or not (ext10 < 0 and ext20 < 0 and ext50 < 0):
        return None

    def ok(cell):
        return bool(cell and cell.get("f4w") and cell["f4w"].get("n", 0) >= min_n)

    if above_200 is not None:
        cell = blk.get("regime_ext_belowall3", {}).get("bull" if above_200 else "bear")
        if ok(cell):
            return {"label": f"below all 3 MAs · {'bull' if above_200 else 'bear'}", "stats": cell}
    cell = blk.get("ext_belowall3")
    if ok(cell):
        return {"label": "below all 3 MAs", "stats": cell}
    return None


def _fwd_num(cell: Optional[dict]) -> str:
    """Format one horizon cell as 'med +2.3%·71%' colored by sign of the median."""
    if not cell:
        return "<span style='color:#6e7681;'>—</span>"
    med, win = cell["median"], cell["win"]
    col = "val-green" if med > 0 else ("val-red" if med < 0 else "#c9d1d9")
    return f"<span class='{col}'>{med:+.1f}%·{win:.0f}%</span>"


def _forward_block(md: dict) -> str:
    """Compact 'if this state → forward 1w/4w/8w' card line for one index."""
    br = forward_baserate(md.get("ticker", ""), md.get("dist_days", 0), md.get("above_200"))
    if not br:
        return ""
    s = br["stats"]
    n = (s.get("f4w") or {}).get("n", 0)
    rows = (f"<div style='margin-top:6px;border-top:1px solid #21262d;padding-top:5px;"
            f"font-size:var(--fs-table);color:#8b949e;'>"
            f"<span title=\"Historical forward price return after days in the SAME state "
            f"(median · win-rate). Conditioner: 200-day-MA regime × O'Neil distribution-day "
            f"bucket — the most predictive combination in out-of-sample testing. "
            f"Built from full history since inception.\">"
            f"📊 If this state → forward <span style='color:#6e7681;'>({br['label']}, n={n})</span></span><br>"
            f"&nbsp;&nbsp;1w {_fwd_num(s.get('f1w'))} · "
            f"4w {_fwd_num(s.get('f4w'))} · "
            f"8w {_fwd_num(s.get('f8w'))}")
    ext = forward_ext_baserate(md.get("ticker", ""), md.get("ext_10", 0.0),
                               md.get("ext_20", 0.0), md.get("ext_50", 0.0), md.get("above_200"))
    if ext:
        es = ext["stats"]
        rows += (f"<br><span style='color:#6e7681;'>&nbsp;&nbsp;{ext['label']}:</span> "
                 f"4w {_fwd_num(es.get('f4w'))}")
    return rows + "</div>"


def breadth_day_over_day(br50: float, br200: float) -> Tuple[Optional[float], Optional[float]]:
    """Persist today's breadth and return the percentage-point change vs the most
    recent PRIOR day (None on first-ever run). Multiple same-day runs overwrite
    today's entry, so the delta is always vs a genuine previous day."""
    hist = {}
    if os.path.exists(BREADTH_HISTORY_PATH):
        try:
            with open(BREADTH_HISTORY_PATH) as fh:
                hist = json.load(fh)
        except Exception:  # noqa: BLE001
            hist = {}
    today = date.today().isoformat()
    prior_keys = [k for k in sorted(hist) if k < today]
    prev = hist[prior_keys[-1]] if prior_keys else None
    d50 = round(br50 - prev["br50"], 1) if prev else None
    d200 = round(br200 - prev["br200"], 1) if prev else None
    hist[today] = {"br50": round(br50, 1), "br200": round(br200, 1)}
    hist = {k: hist[k] for k in sorted(hist)[-40:]}   # keep ~40 days
    try:
        _atomic_write(BREADTH_HISTORY_PATH, json.dumps(hist))
    except Exception:  # noqa: BLE001
        pass
    return d50, d200


def _ext_line(label: str, ext: float, pct: Optional[float]) -> str:
    """One extension row: '+X.X% · P##' colored by historical percentile."""
    base = f"{ext:+.1f}%"
    if pct is None:
        return f"{label}: <span style='color:#c9d1d9;'>{base}</span>"
    pcol = "val-red" if pct >= 90 else ("val-warn" if pct >= 75 else "val-green")
    flag = " ⚠️ stretched" if pct >= 90 else ("" if pct < 75 else " hot")
    return f"{label}: <span class='{pcol}'>{base} · P{pct:.0f}{flag}</span>"


def _bd_chip(d: Optional[float]) -> str:
    if d is None:
        return ""
    arr = "▲" if d > 0 else ("▼" if d < 0 else "▬")
    col = "val-green" if d > 0 else ("val-red" if d < 0 else "")
    return f" <span class='{col}' style='font-size:var(--fs-table);'>{arr} {d:+.1f}pp</span>"


def build_market_section(market_data: List[dict], breadth: dict,
                         regime: str = "GREEN", allow_breakouts: bool = True) -> Tuple[str, str]:
    """Returns (html, overall_trend)."""
    out = ['<div class="market-panel"><div class="market-title">🌐 MARKET OVERVIEW (The Bibles + QM)</div><div class="market-grid">']
    overall_trend = "GREEN"
    for md in market_data:
        tk = md["ticker"]
        trend_col = "val-green" if md["trend"] == "GREEN" else "val-red"
        dist_col = "val-red" if md["dist_days"] >= 6 else ("val-warn" if md["dist_days"] >= 4 else "val-green")  # align to regime thresholds (audit)
        if tk == "QQQ" and md["trend"] == "RED":
            overall_trend = "RED"
        chg = md.get("change_pct", 0.0)
        chg_pt = md.get("change_pt", 0.0)
        chg_col = "val-green" if chg >= 0 else "val-red"
        p10 = ext_percentile(tk, "SMA10", md.get("ext_10", 0.0))
        p20 = ext_percentile(tk, "SMA20", md.get("ext_20", 0.0))
        p50 = ext_percentile(tk, "SMA50", md.get("ext_50", 0.0))
        spark = md.get("spark", "")
        out.append(f"""
            <div class="market-card">
                <h3>{esc(tk)} {_lp(tk, md['close'], style="float:right", fmt="{:.2f}")}</h3>
                <div style="font-size:var(--fs-caption);margin-bottom:4px;"><span class="{chg_col}">{chg:+.2f}% ({chg_pt:+.2f})</span> <span style="color:#8b949e;">vs prev close</span></div>
                <div class="idx-spark">{spark}</div>
                <div style="font-size:var(--fs-table);color:#8b949e;margin:5px 0;">10SMA/21SMA: <span class="{trend_col}">{md['trend']}</span></div>
                <div style="font-size:var(--fs-table);color:#8b949e;">
                    {_ext_line("Above 10MA", md.get('ext_10', 0.0), p10)}<br>
                    {_ext_line("Above 20MA", md.get('ext_20', 0.0), p20)}<br>
                    {_ext_line("Above 50MA", md.get('ext_50', 0.0), p50)}<br>
                    Dist Days (O'Neil): <span class="{dist_col}">{md['dist_days']} days</span>
                </div>
                {_forward_block(md)}
            </div>""")

    if breadth.get("ok"):
        def brow(label, val, chg):
            col = "val-green" if val > 50 else "val-red"
            return f"&gt; {label}: <span class='{col}'>{val:.1f}%</span>{_bd_chip(chg)}"
        asof = breadth.get("asof", "")
        breadth_html = (
            f"""<div class="market-card">
                <h3>S&amp;P 500 Breadth</h3>
                <div style="font-size:var(--fs-table);color:#8b949e;">
                    {brow("20MA (S5TW)", breadth['above20'], breadth.get('chg20'))}<br>
                    {brow("50MA (S5FI)", breadth['above50'], breadth.get('chg50'))}<br>
                    {brow("200MA (S5TH)", breadth['above200'], breadth.get('chg200'))}<br>
                    <span style="font-size:var(--fs-caption);">% of S&amp;P 500 members · vs prev day · Barchart {esc(asof)}</span>
                </div>
            </div>""")
    else:
        breadth_html = ('<div class="market-card"><h3>S&amp;P 500 Breadth</h3>'
                        '<div style="font-size:var(--fs-table);color:#8b949e;">Unavailable (Barchart fetch failed).</div></div>')

    out.append(f"""
            {breadth_html}
        </div>
    </div>""")
    return "".join(out), overall_trend


def build_runbar(counts: Dict[str, int], market_modifier: float, runtime: float,
                 regime: str, allow_breakouts: bool) -> str:
    cls = {"GREEN": "green", "YELLOW": "warn", "RED": "red"}.get(regime, "")
    # No emoji here: a red 🚫 inside a YELLOW chip reads as mixed signals.
    bo = ("<span style='color:var(--green);'>Breakouts ON</span>" if allow_breakouts
          else "<span style='color:var(--red);'>Breakouts OFF</span>")
    return f"""
    <div class="runbar">
        <span class="chip {cls}"><b>{regime}</b> Regime · {bo}</span>
        <span class="chip">A+ <b>{counts['a_plus']}</b></span>
        <span class="chip">A <b>{counts['a']}</b></span>
        <span class="chip">A- <b>{counts['a_minus']}</b></span>
        <span class="chip">HVE <b>{counts['hve']}</b></span>
        <span class="chip">U&amp;R <b>{counts['ur']}</b></span>
        <span class="chip red">Short <b>{counts.get('short', 0)}</b></span>
        <span class="chip">Mkt Mod <b>{market_modifier}x</b></span>
        <span class="chip">Run <b>{runtime:.1f}s</b></span>
        <button class="chip livebtn" onclick="refreshPrices(this)">🔄 Refresh Prices</button>
        <span class="chip" id="liveStamp" style="color:#8b949e;">prices frozen at scan — tap 🔄 for live</span>
    </div>"""


def build_diag_panel(diag: Diagnostics) -> str:
    if not diag.errors and not diag.warnings:
        return ""
    items = "".join(f"<li>❌ {esc(e)}</li>" for e in diag.errors)
    items += "".join(f"<li>⚠️ {esc(w)}</li>" for w in diag.warnings)
    return f'<div class="diag-panel"><div class="t">⚠️ DATA DIAGNOSTICS ({len(diag.errors)} errors, {len(diag.warnings)} warnings)</div><ul>{items}</ul></div>'


# Martin's core mindset (心法) — distilled from the playbook's Philosophy +
# Non-Negotiables. Bilingual, kept short on purpose.
_MINDSET_BULLETS = [
    "環境優先，型態其次 — 再完美的型態，遇到惡劣大盤都是輸的交易。(Environment &gt; setup, always.)",
    "每筆交易獨立 — 可以更高價買回（$90停損→$91可進），但絕不追高（→$100不買）。",
    "緊停損 × 高頻再進場 — 接受常被洗出；邊際在於再進場與找下一檔強勢股，不是放寬停損。",
    "不怕錯過 — 被洗後續漲是好訊號（環境健康），去找下一檔即可。",
    "開盤別急（前15–30分鐘）— 多數虧損是跌破開盤低點後反轉；等「跌破ORL→收回」再進，勝率更高。",
    "槓桿是緊停損的副產品 — 0.4%停損的300%槓桿 = 1%風險。複製『風險』，不要複製『槓桿』。",
    "避開乖離過大的延伸股 — 9/21/50 EMA一旦大幅張開、價格遠離9EMA，就出局等下一個底部。",
    "賣在強勢、停損在弱勢 — 沿9/21 EMA移動停損；過度延伸（離9EMA約15–20%）一律分批了結。",
    "調整到適合自己 — 時間框架、停損幅度、單筆風險、周轉率，微調到符合自己的心理與作息。",
]


def build_mindset_panel() -> str:
    items = "".join(f"<li>{b}</li>" for b in _MINDSET_BULLETS)
    return (
        '<details class="mindset"><summary>📌 精簡摘要 — Martin 心法 '
        '(Trading Mindset) — tap to expand</summary>'
        f'<ul>{items}</ul></details>'
    )


def _edge_count(m: dict) -> int:
    """Light Multiple-Edge tally from fields the scanner already computed
    (preview of J Law's M.E.T.A. stacking) — used to rank Top Picks."""
    fp = m.get("footprint", {}) or {}
    tl = m.get("trendline_data", {}) or {}
    e = 0
    _vp = m.get("vol_pct")                                # `or` default would treat 0 as missing
    _dp = m.get("dist_pct")                               # (audit: 0.0% = best-hugging, not absent)
    if _vp is not None and _vp <= 55:
        e += 1                                            # 💧 VooDoo / low-vol
    if _dp is not None and _dp <= 4:
        e += 1                                            # 🎯 hugging 9/21-EMA
    if fp.get("avwap_holding"):
        e += 1                                            # 📍 AVWAP hold
    if fp.get("higher_lows", 0) >= 1 and fp.get("coiled"):
        e += 1                                            # 🏗️ higher-lows + coiled
    if (tl.get("utl", {}).get("score", 0) >= 10) or tl.get("dtl", {}).get("breakout"):
        e += 1                                            # 📐 trendline / DTL
    if (m.get("meta_score") or 0) >= 60:
        e += 1                                            # ⚡ strong momentum
    rs = m.get("rs_rating")
    if (isinstance(rs, int) and rs >= 80) or (m.get("dist_52w", 99) <= 5):
        e += 1                                            # 🔥 RS leadership
    return e


def _pick_overlay_score(s: dict) -> float:
    """Composite TOP-PICK quality from the 2026-06-15 point-in-time separation
    study (`~/Downloads/qmwork/madrry_asof`, validated 89% recall / 0.94 meta-corr
    vs the live scanner). The only overlay that beat SPY out-of-sample (+6.5pts/3m,
    +20pts/1y, 64-69% win, holds on the untouched last third) was
        coiled & RS>=90 & within 10% of 52wk-high & risk<=3.5%
    with LOW ADR (low volatility) the single strongest separator and `coiled` the
    leave-one-out keystone. The scanner's OWN tier->edges->meta ranking measured
    ~= the market (no SPY edge), so this REPLACES the old edge-count primary sort.
    Soft preference (never a hard gate) so the daily plan never starves."""
    fp = s.get("footprint", {}) or {}
    sc = 0.0
    if fp.get("coiled"):                       # keystone: EMAs 9/21/50 clustered, not extended
        sc += 3.0
    adr = s.get("adr") or 0.0
    if 0 < adr <= 4.0:                         # low volatility wins (adr is the top NEGATIVE separator)
        sc += 1.0
    elif adr > 6.0:                            # adr>6 lagged SPY ~5pts
        sc -= 1.5
    rs = s.get("rs_rating")
    if isinstance(rs, int):
        if rs >= 90:
            sc += 1.5
        elif rs >= 80:
            sc += 0.5
    d52 = s.get("dist_52w", 99)               # near the 52wk high
    if d52 <= 5:
        sc += 1.0
    elif d52 <= 10:
        sc += 0.5
    if (s.get("risk_pct") or 99) <= 3.5:      # tight asymmetric stop
        sc += 1.0
    return round(sc, 2)


def _is_high_conviction(s: dict) -> bool:
    """All four legs of the validated best rule (study 2026-06-15): coiled & RS>=90
    & within 10% of the 52wk high & risk<=3.5%. Surfaced as a ★ badge; far too
    sparse (~1/day, 0 most days) to be a hard gate, so it is a FLAG, not a filter."""
    fp = s.get("footprint", {}) or {}
    rs = s.get("rs_rating")
    return bool(fp.get("coiled") and isinstance(rs, int) and rs >= 90
                and s.get("dist_52w", 99) <= 10 and (s.get("risk_pct") or 99) <= 3.5)


def _hc_legs(s: dict) -> int:
    """How many of the four high-conviction legs a pick meets (0-4): coiled,
    RS>=90, within 10% of the 52wk high, risk<=3.5%. 4/4 == _is_high_conviction.
    Replaces the (display-only, non-predictive) additive overlay score on the
    cards — the validated edge is binary at 4-of-4 (2026-06-15 logic audit)."""
    fp = s.get("footprint", {}) or {}
    rs = s.get("rs_rating")
    return (int(bool(fp.get("coiled"))) + int(isinstance(rs, int) and rs >= 90)
            + int(s.get("dist_52w", 99) <= 10) + int((s.get("risk_pct") or 99) <= 3.5))


def build_regime(market_data: List[dict], breadth: dict,
                 t2108: Optional[dict] = None, vix: Optional[dict] = None,
                 sector_rs: Optional[dict] = None, leader_stats: Optional[dict] = None,
                 winrate: Optional[dict] = None) -> Tuple[str, str, bool]:
    """Market-top early-warning grid: 10 scored tells (+ VIX info) rolled into a
    GREEN/YELLOW/RED verdict with hard 🔴 overrides. Returns (html, regime, allow_breakouts)."""
    md = {m["ticker"]: m for m in market_data}
    qqq, spy = md.get("QQQ"), md.get("SPY")
    br50 = breadth.get("above50", 50.0)
    br200 = breadth.get("above200", 50.0)
    dist_max = max((m.get("dist_days", 0) for m in market_data), default=0)
    ext_max = max((m.get("ext_50", 0) for m in market_data), default=0.0)

    sigs = []   # (state, label)  state ∈ g/y/r/i ; i = info (not scored)

    # 1) Trend
    if qqq and spy and qqq["trend"] == "GREEN" and spy["trend"] == "GREEN":
        sigs.append(("g", "Trend ✓ (QQQ/SPY 10&gt;21)"))
    elif qqq and spy and qqq["trend"] == "RED" and spy["trend"] == "RED":
        sigs.append(("r", "Trend ✗ (QQQ &amp; SPY 10&lt;21)"))
    else:
        sigs.append(("y", "Trend mixed (QQQ/SPY)"))
    # 2/3) S&P breadth > 50/200DMA — but a FAILED breadth fetch returns the 50.0
    #      sentinel, which would score as two phantom YELLOW tells (audit H3).
    #      Guard on the ok flag like every other failable input does.
    if breadth.get("ok"):
        sigs.append((("g" if br50 > 50 else "y" if br50 >= 40 else "r"), f"Breadth &gt;50MA {br50:.0f}%"))
        sigs.append((("g" if br200 > 50 else "y" if br200 >= 40 else "r"), f"&gt;200MA {br200:.0f}%"))
    else:
        sigs.append(("i", "Breadth &gt;50MA n/a"))
        sigs.append(("i", "&gt;200MA n/a"))
    # 4) Distribution days
    sigs.append((("g" if dist_max < 4 else "y" if dist_max < 6 else "r"), f"Distribution {dist_max}d"))
    # 5) Climax extension — calibrated to history via the percentile table
    #    (same P## as the QQQ/SPY cards; take the more-stretched of the two).
    climax = []
    for tk in ("QQQ", "SPY"):
        m = md.get(tk)
        if m is not None:
            p = ext_percentile(tk, "SMA50", m.get("ext_50", 0.0))
            if p is not None:
                climax.append((p, tk, m.get("ext_50", 0.0)))
    if climax:
        cp, ctk, cext = max(climax)
        cst = "r" if cp >= 90 else ("y" if cp >= 75 else "g")
        cflag = " ⚠️stretched" if cp >= 90 else (" hot" if cp >= 75 else "")
        sigs.append((cst, f"Climax {ctk} +{cext:.0f}% · P{cp:.0f}{cflag}"))
    else:
        sigs.append((("g" if ext_max <= 10 else "y" if ext_max <= 15 else "r"),
                     f"Climax ext {ext_max:.0f}%"))
    # 6) Downward momentum — leaders rolling over
    if leader_stats and leader_stats.get("n"):
        b50 = leader_stats["below50"]
        st = "g" if b50 < 30 else ("y" if b50 <= 50 else "r")
        sigs.append((st, f"Leaders &lt;50DMA {b50:.0f}% · noNH {leader_stats['no_new_high']:.0f}%"))
    else:
        sigs.append(("i", "Leader momentum n/a"))
    # 7) Topping-range breakdown (QQQ/SPY)
    idx_below = [m for m in (qqq, spy) if m and m.get("close_below_range")]
    idx_pos = [m.get("range_pos", 1.0) for m in (qqq, spy) if m]
    if idx_below:
        sigs.append(("r", f"Topping: {'/'.join(m['ticker'] for m in idx_below)} broke range"))
    elif idx_pos and min(idx_pos) < 0.2:
        sigs.append(("y", "Topping: near range low"))
    else:
        sigs.append(("g", "Range: holding"))
    # 8) T2108 breadth + divergence override
    diverge = False
    if t2108 and t2108.get("ok"):
        tv = t2108["t2108"]
        st = "g" if tv > 50 else ("y" if tv >= 40 else "r")
        dchg = t2108.get("chg")
        darr = (f" {'▲' if dchg > 0 else '▼'}{abs(dchg):.0f}" if dchg is not None else "")
        diverge = bool(t2108.get("divergence"))
        lbl = f"T2108 {tv:.0f}%{darr}" + (" ⚠️DIVERGE" if diverge else "")
        sigs.append((("r" if diverge else st), lbl))
    else:
        sigs.append(("i", "T2108 n/a"))
    # 9) Breakout win-rate
    if winrate and winrate.get("n", 0) >= 8:
        wr = winrate["winrate"]
        st = "g" if wr > 55 else ("y" if wr >= 40 else "r")
        sigs.append((st, f"Breakout win {wr}% (n{winrate['n']})"))
        wr_red = wr < 35
    else:
        nlog = winrate["n"] if winrate else 0
        sigs.append(("i", f"Breakout win: building · {nlog} trades"))
        wr_red = False
    # 10) Sector-RS weakening
    if sector_rs and sector_rs.get("n"):
        lw, wt, n = sector_rs["lead_weak"], sector_rs["weak_total"], sector_rs["n"]
        st = "r" if lw >= 2 else ("y" if (lw == 1 or wt >= n / 2) else "g")
        sigs.append((st, f"Sectors: {lw}/{sector_rs['lead_n']} leaders weak · {wt}/{n} soft"))
    else:
        sigs.append(("i", "Sector RS n/a"))
    # VIX — SCORED 🟡 caution above 20 (or a >15% 1-day spike); info-only when calm.
    if vix:
        vix_warn = vix["level"] > 20 or vix["chg_pct"] > 15
        if vix["level"] > 20:
            vix_lbl = f"VIX {vix['level']} ({vix['chg_pct']:+.0f}%) ⚠️ &gt;20 caution"
        elif vix["chg_pct"] > 15:
            vix_lbl = f"VIX {vix['level']} ({vix['chg_pct']:+.0f}%) ⚠️ spike"
        else:
            vix_lbl = f"VIX {vix['level']} ({vix['chg_pct']:+.0f}%) · calm &lt;20 不計分"
        sigs.append(("y" if vix_warn else "i", vix_lbl))
    else:
        sigs.append(("i", "VIX n/a"))

    reds = sum(1 for s, _ in sigs if s == "r")
    yellows = sum(1 for s, _ in sigs if s == "y")
    hard_override = diverge or wr_red
    if hard_override or reds >= 3:
        regime = "RED"
    elif reds >= 1 or yellows >= 3:
        regime = "YELLOW"
    else:
        regime = "GREEN"
    allow_breakouts = regime != "RED" and dist_max < 4 and not hard_override

    color = {"GREEN": "#3fb950", "YELLOW": "#f2cc60", "RED": "#ff7b72"}[regime]
    bg = {"GREEN": "rgba(63,185,80,0.10)", "YELLOW": "rgba(242,204,96,0.10)",
          "RED": "rgba(218,54,51,0.12)"}[regime]
    if allow_breakouts:
        bo_txt = "Breakouts ALLOWED — trade leaders at multiple-edge pivots."
    elif regime != "RED":
        bo_txt = f"Breakouts CAUTION — {reds} red / {yellows} amber tells; be selective, trim size."
    else:
        why = "breadth divergence (index up, T2108 down)" if diverge else (
            "breakout win-rate collapsing" if wr_red else f"{reds} red tells")
        bo_txt = f"Breakouts SUPPRESSED — {why}; raise cash / wait for the right side."

    # State carried by a CSS dot + border/text color (was triple-encoded emoji).
    dot = {"g": "<span class='dot dot-g'></span>", "y": "<span class='dot dot-y'></span>",
           "r": "<span class='dot dot-r'></span>", "i": "<span class='dot dot-i'></span>"}
    cmap = {"g": "#3fb950", "y": "#f2cc60", "r": "#ff7b72", "i": "#79c0ff"}
    grid = "".join(
        f"<span class='reg-sig' style='border-color:{cmap[s]};color:{cmap[s]};'>{dot[s]}{lbl}</span>"
        for s, lbl in sigs
    )
    html = (
        f"<div class='regime' style='border-color:{color};background:{bg};'>"
        f"<div class='reg-head'><span style='color:{color};'>🚦 REGIME: {regime}</span>"
        f"<span class='reg-score'>{reds} red / {yellows} amber · {esc(bo_txt)}</span></div>"
        f"<div class='reg-sigs'>{grid}</div>"
        f"<div class='reg-note'>Early-warning tells: leaders rolling over, topping breakdown, T2108 + divergence, "
        f"breakout win-rate, sector RS. A perfect setup in a RED tape is a losing trade — 環境優先，型態其次.</div>"
        f"</div>"
    )
    return html, regime, allow_breakouts


def _ants_edge_bonus(s: dict) -> int:
    """A strong ANTS read counts as an extra Multiple-Edge for TOP-PICKS ranking
    ONLY (FULL +1, ELITE +2). Never folded into _edges (which drives the ⚡ chip
    + A- sort) and never seen by the IBKR order plan."""
    lvl = s.get("ants_level", 0) if s.get("ants_ok") else 0
    return 2 if lvl >= 5 else (1 if lvl >= 4 else 0)


def _rank_top_picks(a_plus: List[dict], a: List[dict],
                    a_minus: List[dict], ants_boost: bool = False) -> List[Tuple[str, dict]]:
    """Shared TOP-PICKS ranking — (tier, Multiple-Edge count, M.E.T.A.). Returns
    [(tier_label, stock), ...] best-first. Used by the dashboard AND the order
    plan. ants_boost (dashboard only) floats FULL/ELITE ANTS names up WITHOUT
    mutating _edges — the order plan calls with the default so IBKR is unaffected."""
    # Include A- too, so a HIGH-CONVICTION (4-of-4) name in A- can rank first —
    # the validated edge is TIER-AGNOSTIC (2026-06-15 decision). The HC-first->tier
    # sort keeps NON-HC A- at the bottom (so they're drafted only when <3 A+/A
    # exist, as before); this promotes ONLY genuine high-conviction A- names.
    pool = [(0, s) for s in a_plus] + [(1, s) for s in a] + [(2, s) for s in a_minus]
    if not pool:
        return []
    for _, s in pool:
        s["_edges"] = _edge_count(s)
        s["_overlay"] = _pick_overlay_score(s)        # DISPLAY ONLY (see 2026-06-15 logic audit)
        s["_high_conviction"] = _is_high_conviction(s)
    # 2026-06-15 LOGIC-AUDIT FIX: the only validated SPY-beating edge is the 4-of-4
    # CONJUNCTION (_is_high_conviction), NOT the additive overlay. By leg count,
    # r_3m = -1.8 / -0.9 / +0.6 / +2.5 / +12.9 — the edge lives ONLY at 4-of-4; an
    # additive score would draft unseparated 3-of-4 names on the ~81% of days with
    # no 4-leg name. So draft high-conviction names FIRST, then fall back to the
    # scanner's original tier->edges->meta. _overlay no longer sorts (display only).
    if ants_boost:
        pool.sort(key=lambda r: (not r[1]["_high_conviction"], r[0],
                                 -(r[1]["_edges"] + _ants_edge_bonus(r[1])),
                                 -(r[1].get("meta_score") or 0)))
    else:
        pool.sort(key=lambda r: (not r[1]["_high_conviction"], r[0], -r[1]["_edges"],
                                 -(r[1].get("meta_score") or 0)))
    tier_label = {0: "A+", 1: "A", 2: "A-"}
    return [(tier_label.get(r, "A"), s) for r, s in pool]


def _append_dedup(path: str, rec: dict, key: str) -> None:
    """Append rec to a JSONL file, replacing any existing row with the same key
    (so same-session re-runs overwrite, never double-count). Atomic rewrite."""
    rows = []
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if o.get(key) != rec.get(key):
                    rows.append(o)
            except Exception:  # noqa: BLE001
                pass
    rows.append(rec)
    rows.sort(key=lambda r: str(r.get(key)))
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def _log_toppicks_forward(plan: dict, data_date: str, regime: str,
                          allow_breakouts: bool) -> None:
    """Persist today's regime + TOP PICKS to append-only forward logs so the
    (dropped) regime gate can be RE-TUNED later on the REAL build_regime sequence.
    One row per session, deduped by date. Reviewed weekly (Sat) then monthly."""
    now = datetime.now(timezone.utc).isoformat()
    _append_dedup(REGIME_HISTORY_PATH,
                  {"date": data_date, "regime": regime,
                   "allow_breakouts": bool(allow_breakouts), "logged_at": now},
                  key="date")
    keep = ("ticker", "limit_price", "stop_reference", "risk_per_share",
            "meta_score", "overlay_score", "high_conviction", "is_htf", "sector")
    _append_dedup(TOP_PICKS_HISTORY_PATH,
                  {"plan_id": plan.get("plan_id"), "date": data_date,
                   "regime": regime, "logged_at": now, "size": "FULL (gate dropped)",
                   "picks": [{k: p.get(k) for k in keep} for p in plan.get("picks", [])]},
                  key="date")


def write_order_plan(a_plus: List[dict], a: List[dict], a_minus: List[dict],
                     regime: str, allow_breakouts: bool, data_date: str) -> dict:
    """Write top_picks_orders.json — the deterministic INTENT for the IBKR
    draft-staging step. NO live orders, NO account data. All sizing/validation is
    done HERE in Python (the staging agent does one multiply: shares = floor(
    equity * shares_per_equity)). Regime-gated (fail-closed allowlist). The
    staging step alone touches IBKR and only ever creates DRAFTS the user
    reviews + submits — it never transmits. Hardened per safety audit 2026-06-14."""
    # GATE DROPPED 2026-06-15 (owner): always FULL size, draft in EVERY regime.
    # The regime is still recorded + forward-logged so a gate can be re-tuned on
    # the real build_regime later. (Was: GREEN full / YELLOW half / RED no-draft.)
    size_mult = 1.0
    gated = False
    ranked = _rank_top_picks(a_plus, a, a_minus)

    picks: List[dict] = []
    seen_sectors: set = set()
    cum_frac = 0.0
    if not gated:
        for tier, s in ranked:
            if len(picks) >= IBKR_TOP_N:
                break
            entry, stop = s.get("entry"), s.get("stop")
            sector = (s.get("sector") or "N/A")
            try:
                entry = float(entry); stop = float(stop)
            except (TypeError, ValueError):
                continue
            # --- fail-closed per-pick validation (skip, never draft, on junk) ---
            if not (entry > 0 and stop > 0 and entry > stop):
                continue
            limit_price = round(entry, 2)
            rps = round(entry - stop, 4)
            if limit_price < IBKR_MIN_PRICE:
                continue
            if rps <= 0 or rps < IBKR_MIN_RPS_PCT * limit_price:   # 1-cent-stop / cap-junk
                continue
            if IBKR_ONE_PER_SECTOR and sector in seen_sectors:
                continue
            # equity unknown at scan time -> emit FRACTION the agent multiplies once.
            # Regime size multiplier (GREEN 1.0 / YELLOW 0.5) scales the whole position.
            risk_spe = IBKR_RISK_FRAC / rps
            cap_spe = IBKR_MAX_POS_FRAC / limit_price
            shares_per_equity = min(risk_spe, cap_spe) * size_mult
            pos_frac = shares_per_equity * limit_price          # this pick's deployment frac
            if cum_frac + pos_frac > IBKR_MAX_SESSION_FRAC:     # portfolio deployment cap
                continue
            cum_frac += pos_frac
            seen_sectors.add(sector)
            picks.append({
                "ticker": s["ticker"], "tier": tier, "sector": sector,
                "side": "BUY", "order_type": "LIMIT", "time_in_force": "DAY",
                "limit_price": limit_price,
                "stop_reference": round(stop, 2),               # USER's manual stop, NOT an order
                "risk_per_share": rps,
                "shares_per_equity": round(shares_per_equity, 8),
                "cap_bound": cap_spe < risk_spe,   # True => position-cap binds; actual risk < risk_frac
                "max_pos_frac": IBKR_MAX_POS_FRAC,
                "meta_score": s.get("meta_score"), "edges": s.get("_edges"),
                "overlay_score": s.get("_overlay"),
                "high_conviction": bool(s.get("_high_conviction")),
                "is_htf": bool(s.get("is_htf")),
            })

    plan_id = data_date + ":" + hashlib.sha1(
        json.dumps([(p["ticker"], p["limit_price"]) for p in picks],
                   sort_keys=True).encode()).hexdigest()[:10]
    plan = {
        "plan_id": plan_id,
        "generated_for_session": data_date,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_session": _expected_session_date(),
        "regime": regime, "allow_breakouts": allow_breakouts, "gated_out": gated,
        "regime_size_mult": size_mult,
        "gate": "DROPPED 2026-06-15 — always full size, regime recorded for forward re-tune",
        "breakouts_suppressed_note": (None if allow_breakouts else
                                      "regime flags breakouts suppressed (distribution/RED tells) — "
                                      "gate dropped so drafted anyway at full size; review carefully"),
        "reason": f"top {len(picks)} picks at full size (regime {regime or '?'}, gate dropped)",
        "risk_frac": IBKR_RISK_FRAC, "max_pos_frac": IBKR_MAX_POS_FRAC,
        "max_session_frac": IBKR_MAX_SESSION_FRAC, "top_n": IBKR_TOP_N,
        "sizing": "shares = floor(equity * shares_per_equity)  # agent does ONLY this multiply+floor",
        "safety": "DRAFTS ONLY via create_order_instruction — user reviews & submits; NEVER transmit.",
        "picks": picks,
    }
    try:
        _atomic_write(IBKR_ORDER_PLAN_PATH, json.dumps(plan, indent=2))
    except Exception as exc:  # noqa: BLE001
        # Quarantine: overwrite with a do-not-stage gate so a stale plan is never
        # read as today's. Do not crash the report (the runbook freshness gate is
        # the second backstop).
        log.error("write_order_plan failed (%s) — quarantining plan file", exc)
        try:
            _atomic_write(IBKR_ORDER_PLAN_PATH, json.dumps({
                "gated_out": True, "picks": [], "plan_id": data_date + ":WRITEFAIL",
                "generated_for_session": data_date,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "reason": "PLAN WRITE FAILED — do not stage"}))
        except Exception:  # noqa: BLE001
            pass
    log.info("Order plan: %s (%d picks, plan_id %s)",
             "GATED — no drafts" if gated else "staged", len(picks), plan_id)
    try:                                    # forward-tracking log (non-fatal)
        _log_toppicks_forward(plan, data_date, regime, allow_breakouts)
    except Exception as exc:  # noqa: BLE001
        log.warning("top-picks forward log failed (non-fatal): %s", exc)
    return plan


def build_top_picks(a_plus: List[dict], a: List[dict], a_minus: List[dict],
                    drafted: Optional[List[str]] = None) -> str:
    """Above-the-fold 'what do I act on today' dashboard: best A+/A names ranked by
    HIGH-CONVICTION (4-of-4) first, then tier -> edges -> M.E.T.A. (2026-06-15 fix).
    `drafted` = tickers the IBKR plan actually staged (top-3, one-per-sector), so
    each card shows its 🎯 N/4 legs + whether it was ✅ drafted or ⤷ skipped."""
    ranked = _rank_top_picks(a_plus, a, a_minus, ants_boost=True)
    if not ranked:
        return ""
    picks = [s for _, s in ranked[:5]]
    rank_of = {id(s): t for t, s in ranked}
    drafted = drafted or []
    draft_pos = {t: i + 1 for i, t in enumerate(drafted)}
    _sec_of = {x["ticker"]: (x.get("sector") or "N/A") for x in (a_plus + a + a_minus)}
    drafted_sectors = {_sec_of.get(t) for t in drafted}

    cards = []
    for s in picks:
        tr = rank_of.get(id(s), "A")
        _legs = _hc_legs(s)
        _sec = s.get("sector") or "N/A"
        if s["ticker"] in draft_pos:
            _ds = (f'<span class="tp-tier" style="color:#3fb950;border-color:#3fb950;" '
                   f'title="drafted to IBKR — top-3 by high-conviction then tier/edges/M.E.T.A., one per sector">✅ DRAFT #{draft_pos[s["ticker"]]}</span>')
        elif _sec in drafted_sectors:
            _ds = ('<span class="tp-tier" style="color:#8b949e;border-color:#8b949e;" '
                   'title="not drafted — one-per-sector rule: this sector is already taken by a higher-ranked pick">⤷ sector taken</span>')
        else:
            _ds = ('<span class="tp-tier" style="color:#8b949e;border-color:#8b949e;" '
                   'title="shown for context — outside the top-3 drafted">— watch</span>')
        _legcol = "#d2a8ff" if _legs == 4 else ("#79c0ff" if _legs == 3 else "var(--text-3)")
        _score = (f'<span class="tp-meta" style="color:{_legcol};" title="high-conviction legs met (of 4): '
                  f'coiled · RS≥90 · within 10% of 52wk high · risk≤3.5%. 4/4 = ★ the validated SPY-beating edge; '
                  f'3/4 = one leg away (watch); ≤2/4 = no measured edge.">🎯 {_legs}/4</span>')
        tcol = {"A+": "#3fb950", "A": "#f2cc60", "A-": "#ff7b72"}.get(tr, "#8b949e")
        rc = "#3fb950" if s["risk_pct"] <= 4 else ("#f2cc60" if s["risk_pct"] <= 6 else "#ff7b72")
        _atp = ""
        if s.get("ants_ok") and s.get("ants_level", 0) >= 1:
            _atp_s = ("·%db" % s.get("ants_chain", 0)) if s.get("ants_chain") else ""
            _atp = "<span class=\"tp-meta\">🐜 %s%s</span>" % (esc(s.get("ants_label", "")), _atp_s)
        elif s.get("ants_ok") and s.get("ants_3m_peak", 0) >= 4:
            _atp = ("<span class=\"tp-meta\" style=\"color:var(--text-3);\">🐜 3M %s</span>"
                    % esc(_ANTS_LABELS.get(s.get("ants_3m_peak", 0), "")))
        _rsl = ""
        if s.get("rs_ok") and s.get("rs_nh_before_price"):
            _rsl = "<span class=\"tp-meta\" style=\"color:#79c0ff;\">🔵 RS Lead‹Px</span>"
        elif s.get("rs_ok") and s.get("rs_new_high"):
            _rsl = "<span class=\"tp-meta\" style=\"color:#79c0ff;\">🔵 RS Leader</span>"
        cards.append(f"""
        <div class="tp-card">
            <div class="tp-top"><a href="https://www.tradingview.com/chart/?symbol={esc(s['ticker'])}" target="_blank">{esc(s['ticker'])}</a>
                <span class="tp-tier" style="color:{tcol};border-color:{tcol};">{tr}</span>
                <span class="tp-edges">⚡{s.get('_edges',0)}</span>{('<span class="tp-tier" style="color:#d2a8ff;border-color:#d2a8ff;" title="coiled · RS≥90 · within 10% of 52wk high · risk≤3.5% — the validated SPY-beating overlay (study 2026-06-15)">★ HI-CONV</span>') if s.get('_high_conviction') else ''} {_ds}</div>
            <div class="tp-px">{_lp(s['ticker'], s['close'], style='', entry=s.get('entry'), stop=s.get('stop'))} {_score} <span class="tp-meta">M.E.T.A. {s.get('meta_score',0)}</span> {_atp} {_rsl}</div>
            <div class="tp-theme">{esc(s['theme'])}</div>
            <div class="tp-plan"><span class="entry-text">Buy ${s['entry']}</span> · <span class="stop-text">Stop ${s['stop']}</span> · <span style="color:{rc};">{s['risk_pct']}%</span></div>
        </div>""")
    return (f"<div class='section-title' style='background-color:var(--surface);color:#3fb950;border-bottom:3px solid #3fb950;'>"
            f"⭐ TOP PICKS — TODAY'S BEST MULTIPLE-EDGE SETUPS</div>"
            f"<div class='toppicks'>{''.join(cards)}</div>")


def build_hot_industries(ind: dict, pool: List[dict]) -> str:
    """Fred6725 industry-group RS leaderboard (144 groups). Shows the top groups
    by RS percentile, with a 🎯 marker on groups that today's coil leaders sit in
    (your watchlist meeting the strongest industries = the J Law confluence)."""
    rows = (ind or {}).get("rows", [])
    if not rows:
        return ""
    held = {}
    for s in pool:
        nm = s.get("ind_name")
        if nm:
            held[nm] = held.get(nm, 0) + 1
    chips = []
    for r in rows[:12]:
        col = "#3fb950" if r["pct"] >= IND_RS_STRONG else ("#f2cc60" if r["pct"] >= 70 else "#8b949e")
        n = held.get(r["industry"], 0)
        mark = f" <b style='color:#56d364;'>🎯{n}</b>" if n else ""
        chips.append(
            f"<span class='theme-chip' data-sector='{esc(r['sector'])}' style='border-color:{col};'>"
            f"🏭 {esc(r['industry'])} <b style='color:{col};'>{r['pct']}</b>"
            f"<span style='color:#8b949e;'> · {esc(r['sector'])}</span>{mark}</span>")
    n_strong = sum(1 for r in rows if r["pct"] >= IND_RS_STRONG)
    return (
        "<details class='collapsis'><summary class='section-title' style='background-color:var(--surface);"
        "color:#56d364;border-bottom:3px solid #56d364;'>"
        f"🏭 HOT INDUSTRY GROUPS — Fred6725 RS · {n_strong} groups ≥{IND_RS_STRONG} "
        f"(top 12 of {len(rows)}; 🎯 = your picks' group)</summary>"
        f"<div class='hot-themes scroll'>{''.join(chips)}</div></details>")


def build_hot_themes(pool: List[dict]) -> str:
    """Aggregate today's leaders by SECTOR → 'Hot Sectors' strip (J Law: leadership
    is a sector phenomenon; watch for sectors making collective new highs)."""
    agg: Dict[str, dict] = {}
    for s in pool:
        th = (s.get("sector") or "").strip()
        if not th or th == "N/A":
            continue
        a = agg.setdefault(th, {"n": 0, "meta": [], "near": 0, "lead": 0, "t": []})
        a["n"] += 1
        a["meta"].append(s.get("meta_score") if s.get("meta_score") is not None else 50)
        d52 = s.get("dist_52w")
        if d52 is not None and d52 <= 5:
            a["near"] += 1
        if s.get("tier") in ("A+", "A"):
            a["lead"] += 1
        a["t"].append(s["ticker"])
    if not agg:
        return ""
    rows = []
    for th, a in agg.items():
        avg = sum(a["meta"]) / len(a["meta"])
        strength = 0.5 * avg + 8 * a["lead"] + 4 * a["near"] + 2 * a["n"]
        rows.append((strength, th, a, avg))
    # Rank by SIZE first (how many of today's leaders sit in the sector — the
    # collective-strength read), strength score only as the tiebreak.
    rows.sort(key=lambda r: (r[2]["n"], r[0]), reverse=True)
    multi = [r for r in rows if r[2]["n"] >= 2]
    rows = (multi or rows)[:8]
    if not rows:
        return ""

    smax = max(r[0] for r in rows) or 1
    chips = []
    for strength, th, a, avg in rows:
        ratio = strength / smax
        col = "#3fb950" if ratio >= 0.66 else ("#f2cc60" if ratio >= 0.4 else "#8b949e")
        nh = f" · 🔥{a['near']} NH" if a["near"] >= 2 else ""
        chips.append(
            f"<span class='theme-chip' data-sector='{esc(th)}' style='border-color:{col};'>"
            f"{esc(th)} <b style='color:{col};'>{a['n']}</b> · avg {avg:.0f}{nh}</span>"
        )
    chips.append("<span class='theme-chip' id='themeClear' style='border-color:#8b949e;display:none;'>✕ Show all</span>")
    return (f"<div class='section-title' style='background-color:var(--surface);color:#f2cc60;border-bottom:3px solid #f2cc60;'>"
            f"🔥 HOT SECTORS — 今日強勢板塊 (top 3 · scroll → for more · tap to filter)</div>"
            f"<div class='hot-themes scroll'>{''.join(chips)}</div>")


def generate_new_highs_section(nh: dict) -> str:
    """New 52-week-high leaders: constructive (🟢) breakouts + persistent leaders
    (⭐ recurring new highs) + the sector clusters making collective new highs."""
    green = nh.get("green", [])
    clusters = nh.get("clusters", [])
    total = nh.get("total", 0)
    if total == 0:
        return ""

    n_persist = sum(1 for m in green if m.get("persist_tier"))
    summary = (f'<summary class="section-title" style="background-color:var(--tint-green);color:#3fb950;border-bottom:3px solid #3fb950;">'
               f'🆕 NEW 52-WEEK HIGHS — LEADERSHIP · {total} new highs · '
               f'{n_persist}⭐ persistent · {len(clusters)} sector clusters</summary>')
    out = ['<details class="collapsis nh">', summary]

    # --- sector-cluster breadth strip ---
    if clusters:
        chips = []
        for s, n in clusters:
            chips.append(f"<span class='theme-chip' data-sector='{esc(s)}' style='border-color:#3fb950;'>"
                         f"{esc(s)} <b style='color:#3fb950;'>×{n}</b></span>")
        out.append(
            "<div style='background:var(--tint-green);border-left:4px solid #3fb950;padding:10px 14px;margin:0 0 14px;border-radius:0 8px 8px 0;'>"
            "<div style='color:#3fb950;font-weight:bold;font-size:var(--fs-body);margin-bottom:6px;'>"
            f"🔥 Sectors making COLLECTIVE new highs today ({total} new highs total · ≥3 = cluster)</div>"
            f"<div class='hot-themes' style='margin:0;'>{''.join(chips)}</div></div>")
    else:
        out.append(f"<div style='color:#8b949e;font-size:var(--fs-table);margin-bottom:12px;'>"
                   f"{total} new 52-wk highs today · no single sector reached a ≥3 cluster.</div>")

    # --- leaders table: constructive (🟢) + persistent (⭐) names ---
    if not green:
        out.append("<div style='color:#8b949e;font-size:var(--fs-body);padding:6px 0;'>"
                   "No 🟢 constructive or ⭐ persistent names today — "
                   "the rest of the new highs are extended or still developing.</div></details>")
        return "".join(out)

    out.append('<div class="table-container"><table data-schema="newhighs">')
    out.append("<tr><th data-col='tk'>Ticker</th><th data-col='price'>Price &amp; Narrative</th><th data-col='adr' title='Average Daily Range — 20-day avg of (High/Low−1), % · how much it typically moves per day (TradingView ADRP)'>ADR</th><th data-col='rs'>RS</th>"
               "<th data-col='pattern'>3-Month Pattern &amp; Persistence</th><th data-col='meta'>M.E.T.A.</th>"
               + _MA_YOY_HEADERS +
               "<th data-col='plan'>Continuation Plan</th></tr>")
    _tag_style = {"GRN": ("var(--tint-green)", "#3fb950"), "YEL": ("var(--tint-yellow)", "#f2cc60"), "RED": ("var(--tint-red)", "#ff7b72")}
    for m in green:
        spark_html = f"<div class='spark'>{m['spark']}</div>" if m.get("spark") else ""
        rs_val = m.get("rs_rating", "N/A")
        base = (f"{m['base_weeks']:.0f}w / {m['base_depth']:.0f}% deep"
                if m.get("base_depth") is not None else f"{m['base_weeks']:.0f}w base")
        ext9 = f"{m['ext9']:.0f}%" if m.get("ext9") is not None else "–"
        ext50 = f"{m['ext50']:.0f}%" if m.get("ext50") is not None else "–"
        ext50_col = "good" if (m.get("ext50") or 0) <= 15 else ("warn" if (m.get("ext50") or 0) <= 25 else "bad")
        fp_html = "".join(
            f"<div class='fp-badge {('fp-good' if ('Higher-Low' in b or 'Coiled' in b) else 'fp-info')}'>{esc(b)}</div>"
            for b in m.get("fp_badges", []))
        risk = m.get("risk_pct")
        rc = "#3fb950" if (risk or 9) <= 4 else ("#f2cc60" if (risk or 9) <= 6 else "#ff7b72")
        risk_txt = f"{risk}%" if risk is not None else "n/a"
        # pattern badge colored by grade
        dot = {"GRN": "🟢", "YEL": "🟡", "RED": "🔴"}.get(m.get("tag"), "🟢")
        pbg, pcol = _tag_style.get(m.get("tag"), ("var(--tint-green)", "#3fb950"))
        pattern_badge = f"<div class='squat-badge' style='background:{pbg};color:{pcol};border-color:{pcol};'>{dot} {esc(m['label'])}</div>"
        # persistence badge (⭐ / ⭐⭐) from recurring new highs
        persist_badge = ""
        if m.get("persist_tier"):
            star = "⭐⭐" if m["persist_tier"] == "R" else "⭐"
            pc = "#f2cc60" if m["persist_tier"] == "R" else "#f2cc60"
            persist_badge = (f"<div class='squat-badge' style='background:#221d08;color:{pc};border-color:{pc};font-weight:bold;'>"
                             f"{star} {esc(m['persist_label'])} · {m['nh_3m']} NH-days/3M ({m['weeks_3m']} wks) · {m['nh_1m']}/1M</div>")
        meta_disp_col = "#3fb950" if m.get("tag") == "GRN" else pcol
        out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
            <td class="ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a></td>
            <td data-sort="{m['close']}">{_lp(m['ticker'], m['close'], entry=m['entry'], stop=m['stop'])}{spark_html}<br>{_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span><br><span class="tag">{esc(m['sector'])}</span>{_ind_badge(m)}''')}</td>
            <td data-sort="{m['adr']}">{m['adr']}%</td>
            <td data-sort="{rs_val if isinstance(rs_val,int) else 0}"><span class="score">{esc(rs_val)}</span><br><span style="font-size:var(--fs-micro);color:#8b949e;">1M:+{m['perf_1m']}% · 3M:+{m['perf_3m']}%</span></td>
            <td style="font-size:var(--fs-table);text-align:left;">
                {persist_badge}{pattern_badge}
                {fp_html}
                <div style="margin-top:4px;color:#8b949e;">🧱 {esc(base)} · 🏗️ {m['higher_lows']} HL</div>
                <div><span class="good">+{ext9} vs 9EMA</span> · <span class="{ext50_col}">+{ext50} vs 50EMA</span></div>
            </td>
            <td data-sort="{m['meta_score']}"><span style="font-size:var(--fs-title);font-weight:bold;color:{meta_disp_col};">{m['meta_score']}</span></td>
            {_ma_cells(m.get('_ma_dist'))}{_fwd_yoy_cell(m['ticker'])}{_eps_accel_cell(m['ticker'])}
            <td data-sort="{risk if risk is not None else 999}">
                <div class="entry-box" style="border-color:#3fb950;background:rgba(86,211,100,0.07);">
                    <span style="color:#3fb950;font-weight:bold;font-size:var(--fs-table);">Buy &gt; ${m['entry']}</span><br>
                    <span class="stop-text">Stop: ${m['stop']} <span class="stop-reason">(21EMA / −1.5×ADR)</span></span><br>
                    <span style="color:{rc};font-size:var(--fs-body);">Risk: {risk_txt}</span>
                </div>
            </td>
        </tr>""")
    out.append("</table></div></details>")
    return "".join(out)


def generate_nh52_monitor_section(pullbacks: List[dict], monitored: List[dict]) -> str:
    """Dedicated tab: every name that printed a 52wk high in the last
    NH52_WATCH_DAYS trading days, re-checked daily. Low-volume pullbacks (the
    awareness signal) are sorted to the top and highlighted."""
    n_pull = len(pullbacks)
    n_break = sum(1 for m in monitored if m.get("tag") == "RED")
    head = (f"<h2 style='margin:4px 0 2px;'>📉 52-Week-High Pullback Monitor</h2>"
            f"<p class='header-sub' style='margin:0 0 14px;'>Names that printed a new "
            f"52wk high in the last {NH52_WATCH_DAYS} trading days, re-checked each run · "
            f"<b style='color:#3fb950;'>{n_pull}</b> low-vol pullback"
            f"{'' if n_pull == 1 else 's'} · "
            f"<b style='color:#ff7b72;'>{n_break}</b> high-vol breakdown"
            f"{'' if n_break == 1 else 's'} · {len(monitored)} watched</p>")
    if not monitored:
        return (head + "<div style='color:#8b949e;font-size:var(--fs-body);padding:8px 0;'>"
                "No names on the 52wk-high monitor yet — they accumulate as the daily "
                "scan prints fresh new highs, then stay here for "
                f"{NH52_WATCH_DAYS} trading days.</div>")

    out = [head]
    if pullbacks:
        out.append("<div style='background:var(--tint-green);border-left:4px solid #3fb950;"
                   "padding:10px 14px;margin:0 0 14px;border-radius:0 8px 8px 0;color:#3fb950;"
                   "font-size:var(--fs-table);'>🟢 <b>Low-volume pullback</b> = price slipped below "
                   "its 50-day MA or the prior close while volume dried up below its 30-day average "
                   "— supply exhausting, a constructive continuation watch.</div>")
    out.append('<div class="table-container"><table>')
    out.append("<tr><th>Ticker</th><th>Status</th><th>Price</th>"
               "<th>vs 50-MA</th><th>vs Prev Close</th><th>Volume vs 30d Avg</th>"
               "<th>RS</th><th>Watch</th></tr>")
    _tag_col = {"GRN": "#3fb950", "RED": "#ff7b72", "HOLD": "#8b949e"}
    for m in monitored:
        col = _tag_col.get(m["tag"], "#8b949e")
        spark_html = f"<div class='spark'>{m['spark']}</div>" if m.get("spark") else ""
        rs_val = m.get("rs_rating", "N/A")
        vs50 = m.get("vs_50"); vsprev = m.get("vs_prev"); vr = m.get("vol_ratio")
        vs50_col = "#ff7b72" if (vs50 is not None and vs50 < 0) else "#3fb950"
        vsprev_col = "#ff7b72" if (vsprev is not None and vsprev < 0) else "#3fb950"
        vr_col = "#3fb950" if (vr is not None and vr < 1) else "#ff7b72"
        vs50_txt = f"{vs50:+.1f}%" if vs50 is not None else "–"
        vsprev_txt = f"{vsprev:+.1f}%" if vsprev is not None else "–"
        vr_txt = f"{vr:.2f}×" if vr is not None else "–"
        out.append(f"""<tr>
            <td class="ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a>{spark_html}</td>
            <td><span class="squat-badge" style="background:rgba(0,0,0,0.12);color:{col};border-color:{col};font-weight:bold;">{esc(m['status'])}</span></td>
            <td data-sort="{m['close']}">${m['close']}</td>
            <td data-sort="{vs50 if vs50 is not None else 0}"><span style="color:{vs50_col};">{vs50_txt}</span><br><span style="font-size:var(--fs-micro);color:#8b949e;">50MA ${m['sma50']}</span></td>
            <td data-sort="{vsprev if vsprev is not None else 0}"><span style="color:{vsprev_col};">{vsprev_txt}</span></td>
            <td data-sort="{vr if vr is not None else 9}"><span style="color:{vr_col};font-weight:bold;">{vr_txt}</span><br><span style="font-size:var(--fs-micro);color:#8b949e;">{'below' if (vr is not None and vr < 1) else 'above'} avg</span></td>
            <td data-sort="{rs_val if isinstance(rs_val,int) else 0}"><span class="score">{esc(rs_val)}</span></td>
            <td data-sort="{m['days_since_high']}"><span style="font-size:var(--fs-table);">{m['days_since_high']}d since high</span><br><span style="font-size:var(--fs-micro);color:#8b949e;">{m['high_count']}× NH · last {esc(m.get('last_high') or '–')}</span></td>
        </tr>""")
    out.append("</table></div>")
    return "".join(out)


def generate_short_table(shorts: List[dict]) -> str:
    out = [
        '<div class="section-title bg-short">🔻 PARABOLIC SHORT — CLIMAX / EXHAUSTION '
        '(乖離過大 · 拋物線見頂)</div>',
        '<div class="table-container"><table>',
        "<tr><th>Ticker</th><th>Price &amp; Extension</th><th>Climax Stats</th>"
        "<th>Short Plan (intraday)</th></tr>",
    ]
    if not shorts:
        out.append("<tr><td colspan='4' style='color:#8b949e;'>No parabolic-short "
                   "candidates — nothing is climactically extended right now.</td></tr>")
    else:
        for m in shorts:
            risk = m.get("risk_pct")
            risk_txt = f"{risk}%" if risk is not None else "n/a"
            tt = m.get("to_target")
            tt_txt = f"+{tt}% to 21EMA" if tt is not None else ""
            out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
                <td class="ep-ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a></td>
                <td data-sort="{m['dist9']}">{_lp(m['ticker'], m['close'])}<br><span class="bad">+{m['dist9']}% above 9EMA</span><br><span style="font-size:var(--fs-caption);color:#8b949e;">+{m['dist21']}% above 21EMA</span><br>{_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span>''')}</td>
                <td data-sort="{m['vol_ratio']}" style="font-size:var(--fs-table);text-align:left;">
                    <span class="bad">🔥 Vol {m['vol_ratio']}x</span><br>
                    <span class="warn">📈 {m['gap_ups']} recent gap-up{'s' if m['gap_ups'] != 1 else ''}</span><br>
                    <span style="color:#8b949e;">⚡ accel +{m['accel']}%</span><br>
                    <span style="color:#8b949e;">1M: +{m['perf_1m']}%</span>
                </td>
                <td data-sort="{risk if risk is not None else 999}">
                    <div class="entry-box" style="border-color:#da3633;background:rgba(218,54,51,0.08);">
                        <span style="color:#ff7b72;font-weight:bold;font-size:var(--fs-table);">🔻 Short Setup</span><br>
                        <span class="stop-text">Trigger: break of ORL / AVWAP retest</span><br>
                        <span class="stop-reason">Daily proxy entry ${m['entry']} · stop &gt; day-high ${m['stop']} ({risk_txt})</span><br>
                        <span style="color:#3fb950;">Cover → 21EMA ${m['target']} <span class="stop-reason">({tt_txt})</span></span>
                    </div>
                    <div style="font-size:var(--fs-micro);color:#8b949e;margin-top:4px;">⚠️ Intraday stop is far tighter (above ORH, ~0.4–2%). Best when it gaps UP (exhaustion). Stand aside if it reclaims AVWAP.</div>
                </td>
            </tr>""")
    out.append("</table></div>")
    return "".join(out)


def fetch_intraday_5m(ticker: str) -> Optional[dict]:
    """5-minute intraday snapshot: running VWAP, the first-30-min opening range
    (ORB), and current vs 5-bar volume. Ported from madrry_trade_plan.py."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?interval=5m&range=1d"
    try:
        data = _request_json(url, headers={"User-Agent": "Mozilla/5.0"},
                             label=f"intraday:{ticker}", retries=2, timeout=15)
        result = data["chart"]["result"][0]
        q = result["indicators"]["quote"][0]
        highs, lows, closes, vols = q["high"], q["low"], q["close"], q["volume"]
    except Exception:  # noqa: BLE001
        return None

    valid: List[dict] = []
    cum_vol = 0.0
    cum_tv = 0.0
    orb_high = -1.0
    orb_low = float("inf")
    for i in range(len(closes)):
        if closes[i] is None or vols[i] is None or highs[i] is None or lows[i] is None:
            continue
        tp = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_vol += vols[i]
        cum_tv += tp * vols[i]
        vwap = cum_tv / cum_vol if cum_vol > 0 else closes[i]
        if len(valid) < 6:                       # first 6 valid 5m bars = ORB (30m)
            orb_high = max(orb_high, highs[i])
            orb_low = min(orb_low, lows[i])
        valid.append({"close": closes[i], "vol": vols[i], "vwap": vwap})

    if not valid:
        return None
    avg5 = sum(x["vol"] for x in valid[-5:]) / min(5, len(valid))
    return {
        "current_price": valid[-1]["close"], "vwap": valid[-1]["vwap"],
        "orb_high": orb_high, "orb_low": orb_low,
        "current_vol": valid[-1]["vol"], "avg_5_vol": avg5, "bars": len(valid),
    }


def build_intraday_action_plan(setups_pool: List[dict], diag: Diagnostics,
                               risk_dollar: float = 500.0) -> str:
    """LIVE intraday execution plan (ORB + VWAP + volume + QQQ kill-switch),
    amended from madrry_trade_plan.py and rendered into the report. Operates on
    the current scan's high-conviction names (A+ VCP power + HVE rel-vol)."""
    targets = [s for s in setups_pool
               if (s.get("power_score") or 0) > 100 or (s.get("rel_vol") or 0) > 3.0]
    targets.sort(key=lambda x: (x.get("power_score") or (x.get("rel_vol", 0) * 100)),
                 reverse=True)
    targets = targets[:6]

    # Fetch QQQ (kill-switch) + every target's 5m snapshot concurrently.
    need = ["QQQ"] + [t["ticker"] for t in targets]
    intra: Dict[str, Optional[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futs = {pool.submit(fetch_intraday_5m, tk): tk for tk in dict.fromkeys(need)}
        for fut in concurrent.futures.as_completed(futs):
            try:
                intra[futs[fut]] = fut.result()
            except Exception:  # noqa: BLE001
                intra[futs[fut]] = None

    # --- QQQ intraday VWAP kill-switch banner ---
    qqq = intra.get("QQQ")
    market_is_safe = True
    if qqq:
        safe = qqq["current_price"] >= qqq["vwap"]
        market_is_safe = safe
        if safe:
            banner = (f'<div class="kill-ok">🟢 MARKET GREEN LIGHT — QQQ above intraday VWAP '
                      f'(${qqq["current_price"]:.2f} ≥ ${qqq["vwap"]:.2f}). Breakouts have tailwind.</div>')
        else:
            banner = (f'<div class="kill-bad">🚨 MARKET RED LIGHT — QQQ BELOW intraday VWAP '
                      f'(${qqq["current_price"]:.2f} &lt; ${qqq["vwap"]:.2f}). Breakouts fail in weak tape — '
                      f'cancel longs or trade tiny.</div>')
    else:
        banner = ('<div class="kill-warn">⚠️ Could not fetch QQQ intraday data (market may be closed). '
                  'Plan uses the last available session.</div>')

    rows = []
    for s in targets:
        tk = s["ticker"]
        is_hve = "rel_vol" in s
        kind = "💥 HVE" if is_hve else "🏆 A+ VCP"
        kind_color = "#ff7b72" if is_hve else "#3fb950"
        it = intra.get(tk)
        if not it or it["bars"] < 5:
            rows.append(f"""<tr>
                <td class="ticker" data-sort="{esc(tk)}"><a href="https://www.tradingview.com/chart/?symbol={esc(tk)}" target="_blank">{esc(tk)}</a><br><span style="font-size:var(--fs-micro);font-weight:bold;color:{kind_color};">{kind}</span></td>
                <td colspan="3" style="color:#8b949e;text-align:left;">⏳ Waiting for intraday data to populate (market closed or pre-open).</td>
            </tr>""")
            continue

        cp, vwap = it["current_price"], it["vwap"]
        orb_h, orb_l = it["orb_high"], it["orb_low"]
        cur_v, avg_v = it["current_vol"], it["avg_5_vol"]
        above_vwap = cp >= vwap
        vol_ok = cur_v > avg_v

        trigger = orb_h if is_hve else (s.get("entry") or orb_h)
        stop = (orb_l - 0.05) if is_hve else (s.get("stop") or orb_l)
        risk_dist = trigger - stop
        if risk_dist <= 0:
            continue
        risk_pct = risk_dist / trigger * 100.0
        shares = int(risk_dollar // risk_dist)

        vwap_cls = "good" if above_vwap else "bad"
        vwap_txt = "✔️ Above VWAP (buyers)" if above_vwap else "❌ Below VWAP (CANCEL)"
        vol_cls = "good" if vol_ok else "warn"
        risk_color = "#3fb950" if risk_pct <= 5.0 else "#ff7b72"

        live = (f'<span style="font-weight:bold;">${cp:.2f}</span> '
                f'<span class="stop-reason">VWAP ${vwap:.2f}</span><br>'
                f'<span class="stop-reason">ORB ${orb_l:.2f} – ${orb_h:.2f}</span><br>'
                f'<span class="{vwap_cls}">{vwap_txt}</span>')

        if not above_vwap:
            protocol = ('<span class="bad">❌ STAND DOWN — price under VWAP, sellers in control.</span>'
                        '<br><span class="stop-reason">Re-evaluate only on a reclaim of VWAP.</span>')
        else:
            steps = [
                f'1️⃣ <b>不要盲掛 buy-stop.</b> 等 5m/30m K棒<b>收盤站上</b> ${trigger:.2f}.',
                f'2️⃣ 突破K棒量必須 &gt; 5根均量 '
                f'<span class="{vol_cls}">(now {cur_v:,.0f} vs {avg_v:,.0f})</span>. 無量=陷阱.',
            ]
            if is_hve:
                steps.append(f'3️⃣ <b>較佳:</b> 等回測 ${trigger:.2f} 守住再進 (避免追高).')
            if not market_is_safe:
                steps.insert(0, '⚠️ <span class="bad">QQQ 在 VWAP 下，假突破機率高.</span>')
            protocol = "<br>".join(steps)

        size_block = (f'<span class="entry-text">Trigger ${trigger:.2f}</span><br>'
                      f'<span class="stop-text">Stop ${stop:.2f}</span><br>'
                      f'<span style="color:{risk_color};">Risk {risk_pct:.1f}% · {shares} sh (${risk_dollar:.0f})</span>')

        rows.append(f"""<tr>
            <td class="ticker" data-sort="{esc(tk)}"><a href="https://www.tradingview.com/chart/?symbol={esc(tk)}" target="_blank">{esc(tk)}</a><br><span style="font-size:var(--fs-micro);font-weight:bold;color:{kind_color};">{kind}</span></td>
            <td style="text-align:left;font-size:var(--fs-caption);" data-sort="{cp:.2f}">{live}</td>
            <td style="text-align:left;font-size:var(--fs-caption);">{protocol}</td>
            <td style="text-align:left;font-size:var(--fs-caption);" data-sort="{risk_pct:.1f}">{size_block}</td>
        </tr>""")

    if not rows:
        return ""

    return f"""
    <div class="section-title" style="background-color:#10243a;color:#79c0ff;border-bottom:3px solid #1f6feb;">⚔️ INTRADAY EXECUTION PLAN — 盤中執行 (ORB + VWAP · Live · Risk ${risk_dollar:.0f})</div>
    <div style="margin:0 0 4px;">{banner}</div>
    <div class="table-container"><table>
        <thead><tr><th>Ticker</th><th>Live (Price / VWAP / ORB)</th><th>Execution Protocol</th><th>Trigger / Stop / Size</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table></div>
    <div style="background:#10243a;border-left:4px solid #1f6feb;padding:12px 15px;margin:0 0 20px;border-radius:0 8px 8px 0;font-size:var(--fs-table);color:#c9d1d9;line-height:1.6;">
        <b style="color:#79c0ff;">紀律：</b> 開盤前 15–30 分鐘別急；等 5m/30m <b>收盤</b>確認、量過均，再進。價在 VWAP 下一律不做多。<br>
        賣在強勢、沿 9/21 EMA 移動停損；風險每筆固定 ${risk_dollar:.0f}（部位是緊停損的副產品）。
    </div>
    """


# ----------------------------------------------------------------------------
# ORCHESTRATION
# ----------------------------------------------------------------------------
def run_scanners_and_generate_html() -> str:
    diag = Diagnostics()
    t0 = time.time()

    with timed(diag, "rs_scores"):
        rs_map = fetch_and_load_rs_scores(diag)
    with timed(diag, "industry_rs"):
        industry_rs = fetch_and_load_industry_rs(diag)

    with timed(diag, "market_health"):
        market_data, breadth = fetch_market_health(diag)

    with timed(diag, "regime_data"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as rpool:
            f_vix = rpool.submit(fetch_vix, diag)
            f_t2108 = rpool.submit(fetch_t2108, diag)
            f_sect = rpool.submit(fetch_sector_rs, diag)
            vix = f_vix.result()
            t2108 = f_t2108.result()
            sector_rs = f_sect.result()

    qqq_trend = next((m["trend"] for m in market_data if m["ticker"] == "QQQ"), "GREEN")
    spy_trend = next((m["trend"] for m in market_data if m["ticker"] == "SPY"), "GREEN")
    above_50_pct = breadth.get("above50", 50.0)
    if qqq_trend == "GREEN" and spy_trend == "GREEN":
        market_modifier = 1.2
    elif qqq_trend == "RED" and spy_trend == "RED":
        market_modifier = 0.4 if above_50_pct < 40.0 else 0.7
    else:
        market_modifier = 1.0
    log.info("QQQ=%s SPY=%s Above50=%.1f%% MarketMod=%s",
             qqq_trend, spy_trend, above_50_pct, market_modifier)

    # 10 calendar days > the 5 TRADING-day U&R window, so weekend-spanning day-4/5
    # setups aren't purged before scan_ur can see them (audit H4).
    hve_history = cleanup_old_hve(load_hve_history(), days=10)

    # Data date = trading date of the last daily bar (from QQQ). Computed BEFORE
    # the scans so they can reject stale history feeds (Yahoo's bulk endpoint
    # can lag the chart endpoint by a session during EOD consolidation).
    data_date = next((m.get("asof") for m in market_data if m.get("ticker") == "QQQ" and m.get("asof")),
                     None) or date.today().isoformat()
    # Vintage check vs the wall clock: if even the chart feed is a session
    # behind (it happens during Yahoo's EOD consolidation window), say so
    # loudly in the report instead of presenting old bars as today's scan.
    stale_banner = ""
    expected_session = _expected_session_date()
    if expected_session and data_date < expected_session:
        diag.warn(f"Data vintage {data_date} is behind expected session {expected_session} "
                  f"(vendor still consolidating) — report marked provisional")
        stale_banner = (f"<div class='kill-warn' style='margin:0 0 16px;text-align:center;'>"
                        f"⚠️ DATA VINTAGE: bars end <b>{esc(data_date)}</b> but the last completed "
                        f"session is <b>{esc(expected_session)}</b> — vendor data still consolidating; "
                        f"treat tiers/fires as provisional (or re-run later).</div>")

    with timed(diag, "scan_coil"):
        tier_a_plus, tier_a, tier_a_minus, tier_a_minus_full, coil_funnel = scan_coil(rs_map, market_modifier, diag)
    with timed(diag, "scan_htf"):
        htf_matches = scan_htf(rs_map, market_modifier, diag, data_date)
        # Merge HTF fires INTO Tier A+ (user's chosen placement). Dedup against
        # any name already surfaced by the coil scan; re-sort A+ by M.E.T.A.
        _coil_names = {s["ticker"] for s in tier_a_plus + tier_a + tier_a_minus_full}
        new_htf = [h for h in htf_matches if h["ticker"] not in _coil_names]
        if new_htf:
            tier_a_plus = sorted(tier_a_plus + new_htf,
                                 key=lambda x: x["meta_score"], reverse=True)
    with timed(diag, "scan_hve"):
        ep_matches = scan_hve(hve_history, diag)
    with timed(diag, "scan_ur"):
        ur_matches = scan_ur(hve_history, ep_matches, diag)
    with timed(diag, "scan_short"):
        short_matches = scan_parabolic_short(diag)
    with timed(diag, "scan_new_highs"):
        nh_data = scan_new_highs(rs_map, market_modifier, diag)

    # ---- exclusion list: drop M&A / pending-delist names report-wide (see
    # excluded_tickers.txt). Applied once here, after every scan, so a single
    # list governs all tabs (coil tiers, HVE, U&R, parabolic, new highs). ----
    if EXCLUDED_TICKERS:
        def _drop(lst):
            return [d for d in lst if (d.get("ticker") or "").upper() not in EXCLUDED_TICKERS]
        _before = (len(tier_a_plus) + len(tier_a) + len(tier_a_minus_full)
                   + len(ep_matches) + len(ur_matches) + len(short_matches)
                   + len(nh_data.get("green", [])))
        tier_a_plus = _drop(tier_a_plus)
        tier_a = _drop(tier_a)
        tier_a_minus = _drop(tier_a_minus)
        tier_a_minus_full = _drop(tier_a_minus_full)
        ep_matches = _drop(ep_matches)
        ur_matches = _drop(ur_matches)
        short_matches = _drop(short_matches)
        nh_data["green"] = _drop(nh_data.get("green", []))
        nh_data["confirmed"] = [t for t in nh_data.get("confirmed", [])
                                if (t or "").upper() not in EXCLUDED_TICKERS]
        nh_data["total"] = len(nh_data["confirmed"])
        _after = (len(tier_a_plus) + len(tier_a) + len(tier_a_minus_full)
                  + len(ep_matches) + len(ur_matches) + len(short_matches)
                  + len(nh_data.get("green", [])))
        if _before != _after:
            log.info("Excluded %d row(s) via excluded_tickers.txt (%s)",
                     _before - _after, ", ".join(sorted(EXCLUDED_TICKERS)))
    # 52wk-high daily monitor: record today's new highs, prune to the watch window,
    # then re-check every watched name for a low-volume pullback (awareness signal).
    with timed(diag, "nh52_monitor"):
        nh52_hist = load_nh52_history()
        nh52_hist = record_nh52_highs(nh52_hist, nh_data.get("confirmed", []), data_date)
        nh52_hist = cleanup_old_nh52(nh52_hist, NH52_WATCH_DAYS, data_date)
        nh52_pullbacks, nh52_monitored = scan_nh52_pullbacks(nh52_hist, rs_map, diag, data_date)
        save_nh52_history(nh52_hist)
    # Tag the displayed coil A-list with 52wk-high persistence + at-new-high.
    # Enrich the FULL A- pool (not just the top 25) so every displayed A- row has badges.
    with timed(diag, "persistence"):
        attach_persistence(tier_a_plus + tier_a + tier_a_minus_full, diag)
    # ANTS (David Ryan accumulation) — tags coil A-list before Top-Picks/order-plan
    # so the display column + Top-Picks boost see them. Does NOT affect IBKR drafts.
    with timed(diag, "ants"):
        attach_ants(tier_a_plus + tier_a + tier_a_minus_full, diag)
    # Industry-group RS (Fred6725 rs_industries) — display-only leadership tag.
    attach_industry_rs(tier_a_plus + tier_a + tier_a_minus_full
                       + nh_data.get("green", []) + ep_matches + ur_matches, industry_rs["by_ticker"])

    save_hve_history(hve_history)

    # ---- assemble report ----
    # Tag coil tiers up front so Hot-Themes / Top-Picks can read them.
    for s in tier_a_plus:
        s["tier"] = "A+"
    for s in tier_a:
        s["tier"] = "A"
    for s in tier_a_minus_full:        # tags the displayed 25 + the overflow
        s["tier"] = "A-"
    # Tracking writes the session's breakout-outcome log; win-rate after it.
    # (data_date computed above, before the scans.)
    tracking_html = track_previous_setups(diag, data_date)
    winrate = breakout_winrate()
    _lp = tier_a_plus + tier_a + tier_a_minus
    _ln = len(_lp) or 1
    leader_stats = {
        "n": len(_lp),
        "below20": 100 * sum(1 for s in _lp if s.get("below_20dma")) / _ln,
        "below50": 100 * sum(1 for s in _lp if s.get("below_50dma")) / _ln,
        "no_new_high": 100 * sum(1 for s in _lp if (s.get("days_since_high") or 0) >= 20) / _ln,
    }
    regime_html, regime, allow_breakouts = build_regime(
        market_data, breadth, t2108, vix, sector_rs, leader_stats, winrate)
    market_html, overall_trend = build_market_section(market_data, breadth, regime, allow_breakouts)
    # Deterministic IBKR draft-order INTENT (top_picks_orders.json) FIRST, so the
    # dashboard cards can show which picks were actually drafted. No orders, no
    # account data — the staging step alone touches IBKR (drafts only).
    _plan = write_order_plan(tier_a_plus, tier_a, tier_a_minus, regime, allow_breakouts, data_date)
    _drafted = [p.get("ticker") for p in (_plan.get("picks") or [])]
    top_picks_html = build_top_picks(tier_a_plus, tier_a, tier_a_minus, drafted=_drafted)
    hot_themes_html = build_hot_themes(tier_a_plus + tier_a + tier_a_minus + ep_matches)
    hot_industries_html = build_hot_industries(industry_rs, tier_a_plus + tier_a + tier_a_minus)

    # Warm the fundamentals cache for EVERY narrative-bearing MADRRY ticker BEFORE the
    # first table renders. Must precede generate_new_highs_section (below): its 🟢 green
    # rows also tap fundamentals and were previously omitted from the batch, forcing one
    # synchronous TradingView POST + cache flush per new-high name during HTML assembly.
    # Batched + disk-cached + time-boxed; never fatal (Minervini/Trilogy warm their own).
    _prefetch_fundamentals(
        [s.get("ticker") for s in (tier_a_plus + tier_a + tier_a_minus_full
                                   + ep_matches + ur_matches + short_matches)]
        + [m.get("ticker") for m in nh_data.get("green", [])])
    # Tier 3 — estimate-revision counts (per-ticker yfinance) for the TOP PICKS only.
    _prefetch_revisions([s.get("ticker") for _, s in
                         _rank_top_picks(tier_a_plus, tier_a, tier_a_minus_full)[:REVISIONS_TOP_N]])

    new_highs_html = generate_new_highs_section(nh_data)
    nh52_monitor_html = generate_nh52_monitor_section(nh52_pullbacks, nh52_monitored)
    counts = {
        "a_plus": len(tier_a_plus), "a": len(tier_a), "a_minus": len(tier_a_minus_full),
        "hve": len(ep_matches), "ur": len(ur_matches), "short": len(short_matches),
    }
    runtime = time.time() - t0

    # External-engine tabs (own trade plans; MADRRY-style price/narrative columns)
    minervini_html, minervini_n = generate_minervini_table(market_modifier)
    trilogy_html, trilogy_n = generate_trilogy_table(market_modifier=market_modifier)
    tier_a_study_html, tier_a_study_n = generate_tier_a_study_tab()
    tabs_bar = (
        "<div class='tabs' role='tablist'>"
        "<button class='tab-btn active' data-tab='madrry'>📋 MADRRY Watchlist</button>"
        f"<button class='tab-btn' data-tab='minervini'>🏛️ Minervini<span class='tab-count'>{minervini_n}</span></button>"
        f"<button class='tab-btn' data-tab='trilogy'>📚 Trilogy<span class='tab-count'>{trilogy_n}</span></button>"
        f"<button class='tab-btn' data-tab='pivots'>💥 Pivots &amp; U&amp;R<span class='tab-count'>{len(ep_matches) + len(ur_matches)}</span></button>"
        f"<button class='tab-btn' data-tab='short'>🔻 Short<span class='tab-count'>{len(short_matches)}</span></button>"
        f"<button class='tab-btn' data-tab='hi52'>📈 52-Week High<span class='tab-count'>{nh_data.get('total', 0)}</span></button>"
        f"<button class='tab-btn' data-tab='tracking'>📈 Tracking<span class='tab-count'>{tier_a_study_n}</span></button>"
        "</div>"
    )

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>MADRRY Ultimate Scanner Report</title>",
        f"<style>{PAGE_CSS}</style></head><body>",
        "<h1>MADRRY Watchlist</h1>",
        f"<p class='header-sub'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        build_runbar(counts, market_modifier, runtime, regime, allow_breakouts),
        stale_banner,
        regime_html,
        market_html,
        # ---- engine tabs: switch between MADRRY watchlist, Minervini, Trilogy ----
        tabs_bar,
        "<div class='tab-panel active' id='tab-madrry'>",
        hot_themes_html,
        hot_industries_html,
        # action zone: filter control + picks + tables come BEFORE long context
        "<input id='search' type='search' placeholder='🔎 Filter by ticker (e.g. NVDA)…' autocomplete='off'>",
        top_picks_html,
        tracking_html,
        build_filter_funnel(coil_funnel, len(tier_a_plus), len(tier_a), len(tier_a_minus_full)),
        generate_coil_table(tier_a_plus, "🏆 TIER A+ (strict 3-day flag · ≤1% from EMA · 3-day vol ≤50% of prev-day or 50-day avg · incl. 🚩 HTF) — TRIGGER READY", "bg-aplus"),
        generate_coil_table(tier_a, "🔥 TIER A (2-day tight candle · ≤1% from EMA · 2-day vol ≤55% of prev-day or 50-day avg) — DEVELOPING", "bg-a"),
        generate_coil_table(tier_a_minus_full, "🚀 TIER A- (1-day tight candle · ≤2% from EMA · 1-day vol ≤ prev-day or 50-day avg) — EXTENDED / MESSY", "bg-aminus"),
        "</div>",  # /tab-madrry
        f"<div class='tab-panel' id='tab-minervini'>{minervini_html}</div>",
        f"<div class='tab-panel' id='tab-trilogy'>{trilogy_html}</div>",
        # ---- Episodic Pivots (HVE) + Post-HVE U&R — own tab ----
        "<div class='tab-panel' id='tab-pivots'>",
        generate_hve_table(ep_matches),
        generate_ur_table(ur_matches),
        "</div>",
        # ---- Parabolic Short — own tab ----
        f"<div class='tab-panel' id='tab-short'>{generate_short_table(short_matches)}</div>",
        # ---- 52-Week High — New Highs + Pullback as two sub-tabs ----
        "<div class='tab-panel' id='tab-hi52'>",
        "<div class='subtabs' role='tablist'>",
        f"<button class='subtab-btn active' data-subtab='nh'>🆕 New 52wk Highs<span class='tab-count'>{nh_data.get('total', 0)}</span></button>",
        f"<button class='subtab-btn' data-subtab='pull'>📉 52wk Pullback<span class='tab-count'>{len(nh52_pullbacks)}</span></button>",
        "</div>",
        f"<div class='subtab-panel active' id='subtab-nh'>{new_highs_html}</div>",
        f"<div class='subtab-panel' id='subtab-pull'>{nh52_monitor_html}</div>",
        "</div>",
        f"<div class='tab-panel' id='tab-tracking'>{tier_a_study_html}</div>",
        build_mindset_panel(),
        build_diag_panel(diag),
        PAGE_JS.replace("__LIVE_PRICE_PROXY__", LIVE_PRICE_PROXY),
        "</body></html>",
    ]
    html = "".join(parts)

    # ---- persist outputs (atomic) ----
    for s in tier_a_plus:
        s["tier"] = "A+"
    for s in tier_a:
        s["tier"] = "A"
    for s in tier_a_minus_full:
        s["tier"] = "A-"
    for s in ep_matches:
        s["tier"] = "HVE"
    for s in ur_matches:
        s["tier"] = "UR"
    # Persist the FULL uncapped A- pool (not just the displayed 25) so EVERY pick
    # is graded for the breakout win-rate the next session. Picks are filed under
    # the scan's DATA date — re-running later the same day overwrites the same
    # file (the day's final run wins), so tomorrow's tracking always grades the
    # genuine previous SESSION, never an intermediate intraday run.
    _setups_payload = json.dumps(tier_a_plus + tier_a + tier_a_minus_full + ep_matches + ur_matches)
    _atomic_write(LATEST_SETUPS_PATH, _setups_payload)   # legacy/stable alias
    _save_dated_setups(_setups_payload, data_date)
    _atomic_write(HTML_REPORT_PATH, html)
    # Stable "latest" alias so the preview / bookmarks always show the newest run
    _atomic_write(os.path.join(WORKSPACE, "madrry_report.html"), html)

    # ---- also generate Markdown report for cron delivery ----
    md_path = MD_REPORT_PATH
    md_content = build_markdown_report(
        tier_a_plus, tier_a, tier_a_minus, ep_matches, ur_matches, short_matches,
        market_data, breadth, overall_trend, market_modifier, runtime, diag
    )
    _atomic_write(md_path, md_content)

    log.info("DONE in %.1fs | A+=%d A=%d A-=%d HVE=%d U&R=%d SHORT=%d | errors=%d warnings=%d",
             runtime, counts["a_plus"], counts["a"], counts["a_minus"],
             counts["hve"], counts["ur"], counts["short"], len(diag.errors), len(diag.warnings))
    return HTML_REPORT_PATH


# ----------------------------------------------------------------------------
# MARKDOWN REPORT BUILDER (for cron delivery)
# ----------------------------------------------------------------------------
def build_markdown_report(
    tier_a_plus, tier_a, tier_a_minus, ep_matches, ur_matches, short_matches,
    market_data, breadth, overall_trend, market_modifier, runtime, diag
) -> str:
    """Generate a plain-text Markdown report for cron delivery."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# MADRRY Scanner Report — {now}",
        "",
        f"**Market Trend:** {overall_trend} | **Modifier:** {market_modifier:.1f}x",
        "",
        "## Market Health",
        "",
        "| Ticker | Price | Trend |",
        "|--------|-------|-------|",
    ]
    for m in market_data:
        lines.append(f"| {m['ticker']} | {m.get('close', m.get('price', 0)):.2f} | {m.get('trend', 'N/A')} |")

    if breadth.get("ok"):
        lines.append(f"\n**S&P 500 Breadth:** {breadth['above20']:.1f}% &gt;20MA · "
                     f"{breadth['above50']:.1f}% &gt;50MA · {breadth['above200']:.1f}% &gt;200MA (Barchart)")

    lines.extend(["", "## Summary", ""])
    lines.append(f"- **Tier A+ (Trigger Ready):** {len(tier_a_plus)}")
    lines.append(f"- **Tier A (Developing):** {len(tier_a)}")
    lines.append(f"- **Tier A- (Extended):** {len(tier_a_minus)}")
    lines.append(f"- **HVE (Episodic Pivot):** {len(ep_matches)}")
    lines.append(f"- **U&R (Undercut & Rally):** {len(ur_matches)}")
    lines.append(f"- **Short (Parabolic):** {len(short_matches)}")
    lines.append(f"- **Runtime:** {runtime:.1f}s")

    def _tier_section(title, setups):
        if not setups:
            return []
        out = ["", f"## {title}", ""]
        out.append("| Ticker | Price | Score | ANTS | RS Lead | Setup | Entry | Stop | Risk% |")
        out.append("|--------|-------|-------|------|---------|-------|-------|------|-------|")
        for s in setups[:25]:
            tk = s.get("ticker", "N/A")
            price = s.get("close", s.get("price", 0))
            score = s.get("meta_score", s.get("score", 0))
            setup = s.get("setup", "VCP")
            entry = s.get("entry", "N/A")
            stop = s.get("stop", "N/A")
            risk = s.get("risk_pct", 0)
            if s.get("ants_ok"):
                ants = s.get("ants_label", "—") + (("·%db" % s.get("ants_chain", 0)) if s.get("ants_chain") else "")
                if s.get("ants_3m_peak", 0) >= 1:
                    ants += " (3M %s)" % _ANTS_LABELS.get(s.get("ants_3m_peak", 0), "")
            else:
                ants = "—"
            if s.get("rs_ok") and s.get("rs_nh_before_price"):
                rsl = "🔵 Lead‹Px"
            elif s.get("rs_ok") and s.get("rs_new_high"):
                rsl = "🔵 Lead"
            else:
                rsl = "—"
            out.append(f"| {tk} | {price:.2f} | {score:.0f} | {ants} | {rsl} | {setup} | {entry} | {stop} | {risk:.1f}% |")
        return out

    lines.extend(_tier_section("Tier A+ — TRIGGER READY", tier_a_plus))
    lines.extend(_tier_section("Tier A — Developing", tier_a))
    lines.extend(_tier_section("Tier A- — Extended", tier_a_minus))
    lines.extend(_tier_section("HVE — Episodic Pivot", ep_matches))
    lines.extend(_tier_section("U&R — Undercut & Rally", ur_matches))
    lines.extend(_tier_section("Short — Parabolic", short_matches))

    if diag.errors:
        lines.extend(["", "## Errors", ""])
        for e in diag.errors[:10]:
            lines.append(f"- ⚠️ {e}")

    lines.extend(["", "---", "*Generated by MADRRY Scanner v2*", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    run_scanners_and_generate_html()
