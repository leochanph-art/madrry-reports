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

import bisect
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

# S/R zone engine (entry-quality shadow mode). Guarded import: a broken module
# must never take the daily report down with it.
try:
    import madrry_sr_zones as _srz
except Exception:  # noqa: BLE001
    _srz = None

# Pullback-recovery detector (tutorial #2, shadow mode). Same guard, same deal.
try:
    import madrry_pullback_buy as _pbb
except Exception:  # noqa: BLE001
    _pbb = None

# Trendline engine v2 (tutorial #3, shadow mode). The legacy
# calculate_trendline_analysis block stays untouched.
try:
    import madrry_trendlines as _tlv2
except Exception:  # noqa: BLE001
    _tlv2 = None

# Parallel-channel engine (tutorial #4, 2+1 construction, shadow mode).
# NOT part of the Stage-4 support gate - needs its own user sign-off.
try:
    import madrry_channels as _chv
except Exception:  # noqa: BLE001
    _chv = None

# ----------------------------------------------------------------------------
# CONFIG  (paths unchanged — centralised so they live in one place)
# ----------------------------------------------------------------------------
WORKSPACE = "/Users/boundbythese/.openclaw/workspace"
LATEST_SETUPS_PATH = os.path.join(WORKSPACE, "latest_setups.json")
HVE_HISTORY_PATH = os.path.join(WORKSPACE, "hve_history.json")
BREADTH_HISTORY_PATH = os.path.join(WORKSPACE, "breadth_history.json")
HEADLINE_METER_HISTORY_PATH = os.path.join(WORKSPACE, "headline_meter_history.json")
MARKET_INTERNALS_PATH = os.path.join(WORKSPACE, "market_internals.json")
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
    from madrry_meta_v4 import meta_v4_score_prob as _meta_v4_score_prob
except Exception:  # noqa: BLE001 — missing model/module => legacy score
    def _meta_v4_score(_df):
        return None

    def _meta_v4_score_prob(_df):
        return None

# ---- Fundamentals (past-2Q + next-2Q revenue/EPS) for the tap-to-expand narrative.
# Self-contained, disk-cached, never fatal. If the module is missing the helper is a
# no-op that returns the plain narrative unchanged.
try:
    import madrry_fundamentals as _fund
except Exception:  # noqa: BLE001
    _fund = None


# ETF tickers seen by this run's coil scan (populated in scan_coil). Funds have
# no fundamentals: without this short-circuit the row cells below would each
# trigger a synchronous per-ticker TradingView POST + Yahoo scrape + cache flush
# at render time (madrry_fundamentals.get() is not cache-only on a miss), and
# the not-ok fund record expires daily so the cost would recur EVERY run.
_ETF_TICKERS: set = set()


def _narrative(ticker: str, inner_html: str) -> str:
    """Wrap a row's narrative (theme/sector/industry) so tapping it reveals the
    fundamentals panel. Falls back to the bare narrative if data/module unavailable."""
    if _fund is None or ticker in _ETF_TICKERS:
        return inner_html
    try:
        return _fund.details_html(ticker, inner_html)
    except Exception:
        return inner_html


# ---- shared row cells (2026-07-05 layout: chart-centric table / mobile cards) ----
def _lesson_badge(m: dict) -> str:
    """Compact lesson-confluence chip for the ticker cell; '' below 3/4."""
    try:
        ls = m.get("lesson_confluence")
        if not ls or len(ls) < 3:
            return ""
        return (f"<span class='lesson-ct' title='{len(ls)} of the 4 tutorial lessons "
                f"agree on this entry ({esc(' + '.join(str(x) for x in ls))})'>"
                f"{len(ls)}/4 LESSONS</span>")
    except Exception:  # noqa: BLE001
        return ""


def _tk_cell(m: dict, *, entry=None, stop=None, cls: str = "ticker") -> str:
    """Sticky ticker cell — the card anchor on mobile: link + live price +
    lesson-confluence chip."""
    tk = str(m.get("ticker", ""))
    lp = _lp(tk, m.get("close"), entry=entry, stop=stop) if m.get("close") is not None else ""
    return (f'<td class="{cls}" data-sort="{esc(tk)}">'
            f'<a href="https://www.tradingview.com/chart/?symbol={esc(tk)}" target="_blank">{esc(tk)}</a>'
            + (f"<div>{lp}</div>" if lp else "")
            + _lesson_badge(m) + "</td>")


def _chart_cell(chart_html: str, sort_val) -> str:
    """The Chart column: the candlestick chart ALONE (sorts by price)."""
    return f'<td class="c-chart" data-sort="{sort_val}">{chart_html or ""}</td>'


def _narr_cell(ticker: str, inner_html: str) -> str:
    """Narrative column (theme/sector/industry, tap-to-expand fundamentals)."""
    return f'<td class="c-narr">{_narrative(ticker, inner_html)}</td>'


def _edge_details(m: dict, lines: List[str]) -> str:
    """Collapsible wrapper for the engine detail lines. The summary strip shows
    which engines have a read (+ the lesson count); the full text lives inside
    so rows stay short. '' when no engine produced a line."""
    lines = [x for x in lines if x]
    if not lines:
        return ""
    toks = []
    ls = m.get("lesson_confluence") or []
    if len(ls) >= 3:
        toks.append(f"<span class='lesson-ct'>{len(ls)}/4</span>")
    for key, has in (("SR", m.get("sr_grade")), ("PB", m.get("pb2_state")),
                     ("TL", m.get("tl_sup_kind") or m.get("tl_res_kind")),
                     ("CH", m.get("ch_dir") or m.get("ch_flags"))):
        if has:
            toks.append(f"<span class='lbl'>{key}</span>")
    summary = " ".join(toks) if toks else "<span class='lbl'>ENGINES</span>"
    return ("<details class='lessons'><summary>" + summary
            + " <span class='sumhint'>details</span></summary>"
            + "".join(lines) + "</details>")


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


def _risk_cls(pct, lo: float = 4.0, hi: float = 6.0) -> str:
    """Risk%% -> utility class (site-specific thresholds preserved via lo/hi)."""
    try:
        v = float(pct)
    except (TypeError, ValueError):
        return "risk-md"
    return "risk-lo" if v <= lo else ("risk-md" if v <= hi else "risk-hi")


def _plan_kicker(m: dict) -> str:
    """Trade-plan cell kicker — names the LESSON the printed plan came from
    (2026-07-06 USER: the plan follows the 4 tutorial lessons where validated)."""
    src = m.get("plan_src")
    if src == "PB":
        return ("<span class='kicker' title='Plan from the pullback-recovery lesson (拉回買入法): "
                "buy the break of the mini downtrend line, stop under the prior low.'>PB PULLBACK</span>")
    if src == "SR":
        return ("<span class='kicker' title='Breakout entry; stop from the S&amp;R lesson — "
                "just outside the graded protecting zone instead of a generic offset.'>BREAKOUT · SR STOP</span>")
    return "<span class='kicker'>BREAKOUT</span>"


def _plan_jump(tk: str) -> str:
    """'→ IBKR draft' chip linking the plan cell to this ticker's draft card in
    TOP PICKS. Rendered on every coil row; a loader JS prunes chips whose target
    card doesn't exist (only the drafted top-3 carry the id)."""
    return (f"<a class='plan-jump' href='#tp-{esc(tk)}' "
            f"title='jump to this ticker&#39;s IBKR draft order card'>→ IBKR draft</a>")


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
            cells.append(f"<td class='num ma-cell c-stat' data-label='Δ{p}MA' data-sort='9999'>—</td>")
        else:
            ab = abs(v)
            arrow, acls = ("▲", "arr-up") if v >= 0 else ("▼", "arr-dn")
            cells.append(
                f"<td class='num ma-cell c-stat' data-label='Δ{p}MA' data-sort='{ab:.4f}'>{ab:.1f}"
                f"<span class='{acls}'>{arrow}</span></td>")
    return "".join(cells)


def _fwd_yoy_cell(ticker: str) -> str:
    """Sortable <td> — the FIRST forward (estimate) revenue quarter's YoY growth, read
    from the fundamentals cache (warmed by _prefetch_fundamentals). data-sort = the %
    so clicking descending puts the fastest-growing names first; missing → '—' with
    data-sort −999 (parks last on descending). Never raises."""
    y, lbl = None, ""
    if _fund is not None and ticker not in _ETF_TICKERS:   # funds: no fetch, no junk cache
        try:
            rec = _fund.get(ticker)
            if rec:
                for r in rec.get("rev", []):
                    if r.get("est"):                 # the NEXT forward quarter, whatever its YoY
                        y, lbl = r.get("yoy"), r.get("lbl", "")
                        break                        # y may be None -> renders '—' (truthful to header)
        except Exception:
            y = None
    if y is None:
        return "<td class='num fy-cell c-stat' data-label='Fwd YoY' data-sort='-999'>—</td>"
    pct = y * 100.0
    col = "var(--green)" if pct > 0.5 else ("var(--red)" if pct < -0.5 else "var(--text-3)")
    sign = "+" if pct >= 0 else ""
    sub = (f"<br><span class='sub'>{esc(lbl)}</span>"
           if lbl else "")
    return (f"<td class='num fy-cell c-stat' data-label='Fwd YoY' data-sort='{pct:.2f}'>"
            f"<span style='color:{col};font-weight:600;'>{sign}{pct:.0f}%</span>{sub}</td>")


def _eps_accel_cell(ticker: str) -> str:
    """Sortable <td> — O'Neil earnings ACCELERATION: the TREND in quarterly EPS YoY growth
    (a rising growth RATE = accelerating, the CANSLIM 'C' refined). Headline = an
    acceleration arrow + the latest reported quarter's EPS YoY%; sub-line = the recent YoY
    path (+ a ✦TTM mark when trailing-12-month EPS is at a new high). data-sort = the change
    in YoY rate in pp, so clicking descending puts the fastest accelerators first; missing →
    '—' parked last (data-sort −9999). Reads the fundamentals cache; never raises."""
    a = None
    if _fund is not None and ticker not in _ETF_TICKERS:   # funds: no fetch, no junk cache
        try:
            rec = _fund.get(ticker)
            a = rec.get("eps_accel") if rec else None
        except Exception:
            a = None
    score = a.get("accel_score") if a else None
    verdict = a.get("verdict") if a else None
    if score is None or verdict is None:
        return "<td class='num accel-cell c-stat' data-label='EPS Acc' data-sort='-9999'>—</td>"
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
        sub = (f"<br><span class='sub'>"
               f"{esc(path)}%{ttm}</span>")
    elif ttm:
        sub = f"<br><span style='font-size:var(--fs-micro);'>{ttm}</span>"
    else:
        sub = ""
    return (f"<td class='num accel-cell c-stat' data-label='EPS Acc' data-sort='{score:.2f}'>"
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


def _ranking_meta_score_ex(hist_df, legacy_score, market_modifier):
    """Like _ranking_meta_score, but ALSO returns the raw scores the ledger/Phase-3
    calibration needs — computed from a SINGLE v4 feature pass (no extra work vs the
    plain ranking call). Returns (display_score, {legacy_score_raw, v4_prob_raw}).
      display_score  = v4 percentile (or legacy×modifier fallback) — identical to
                       _ranking_meta_score, so display/ranking is unchanged.
      legacy_score_raw = the raw legacy M.E.T.A. component score (pre-modifier).
      v4_prob_raw    = the raw P(+2ADR win) in [0,1] BEFORE percentile calibration."""
    try:
        legacy_raw = round(float(legacy_score), 1)
    except (TypeError, ValueError):
        legacy_raw = None
    sp = _meta_v4_score_prob(hist_df)
    if sp is not None:
        pctile, prob = sp
        return pctile, {"legacy_score_raw": legacy_raw, "v4_prob_raw": prob}
    disp = round(min(legacy_score * market_modifier, 100.0), 1)
    return disp, {"legacy_score_raw": legacy_raw, "v4_prob_raw": None}


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
# support only; does NOT affect the IBKR draft plan. ANTS rs_line (close/^GSPC vs
# its own MA) is DISTINCT from the Fred6725 RS percentile (resolve_rs).
ANTS_LOOKBACK = 15            # window for up-count / vol-gain / price-gain
ANTS_MIN_UP = 12             # >= this many up-days in the window => momentum_ok
ANTS_PRICE_PCT = 0.20        # close up >= 20% over the window
ANTS_VOL_PCT = 0.20          # avg volume up >= 20% vs the prior window
ANTS_USE_TREND = True        # require SMA10 > SMA20 for the price leg
ANTS_USE_RS = True           # enable the ELITE upgrade (rs_line rising vs ^GSPC)
ANTS_COUNT_FULL_ONLY = False # chain counts any level>0 (True = FULL+ only)
ANTS_RS_FAST = 20            # is_rs_rising: rs_line > SMA(rs_line, 20)
ANTS_RS_SLOW = 50            # isStronger (info): rs_line > SMA(rs_line, 50)
ANTS_CHAIN_WINDOW = 60       # bars scanned for the trailing chain run
ANTS_BENCHMARK = "^GSPC"     # RS line vs the S&P 500 INDEX (2026-07-06 USER: was SPY)
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
# ---- ETF coil leg (USER 2026-07-15: "screen ETFs as well — ETF charts trade by
# our method"). ETFs run the SAME coil pipeline (52w band, RS 80+ gate, RS-line
# swing gate, tiers, entry engines); only the data plumbing differs:
#   · size gate = AUM ≥ $2B (funds have no market_cap_basic; mirrors the $2B cap)
#   · Stage-2 RS 80+: funds are absent from the Fred6725 stock/industry RS feeds,
#     so the SAME score is computed locally — verified formula (0.005 median err
#     vs the source CSV): 100·(1+strength)/(1+strength_SPY), strength =
#     0.4·q1+0.2·q2+0.2·q3+0.2·q4 over trailing 63-bar windows — then percentile-
#     ranked against the stock cross-section's raw scores (RS cache CSV). An ETF
#     must rank RS ≥ 80 AMONG STOCKS to pass; not computable ⇒ dropped (hard gate).
#   · ETF rows are REPORT-ONLY for the learning loops: tier-A tracker, v4 tracker
#     and calibration skip is_etf rows so stock-fit models never train on fund
#     label geometry (low-ADR ETFs hit the +2·ADR win bar far too easily).
ETF_AUM_MIN = 2_000_000_000   # $2B AUM — fund-side analogue of the $2B cap gate
ETF_THEME_MAX = 44            # fund-name chars shown as the row's theme

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

# v9 chart-first layout (2026-07-18): card deck on narrow screens, master-detail
# "Desk" on wide (foldable unfolded / desktop). False restores the legacy
# 8-tab table report — legacy generators and their JS stay in this file.
LAYOUT_V9 = True
# REV 10 (USER 2026-07-18: "make the chart has 6 months to 1y history 'i can
# select'"). Daily chart payloads now carry ~1 trading year so CANDLE_JS can
# slice a 3M/6M/1Y window client-side. Weekly sparks keep their own length.
CHART_WINDOW = 252


def _lp(ticker: Any, close: Any, *, style: str = "",
        entry: Any = None, stop: Any = None, fmt: str = "{:.2f}") -> str:
    """A live-price <span>: tagged so the Refresh-Prices button can update it in
    place. data-snap holds the scan-time price (for up/down colouring).
    Index symbols (caret-prefixed, e.g. ^IXIC) are points, not dollars, so the
    '$' prefix is dropped — the refresh JS keys off the same caret to stay consistent."""
    extra = ""
    if entry is not None:
        extra += f" data-entry='{entry}'"
    if stop is not None:
        extra += f" data-stop='{stop}'"
    unit = "" if str(ticker).startswith("^") else "$"
    style_attr = f" style='{style}'" if style else ""
    return (f"<span class='lp' data-tkr='{esc(str(ticker))}' data-snap='{close}'{extra}"
            f"{style_attr}>{unit}{fmt.format(close)}</span>")


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
                _n = 0
                for t in (row.get("Tickers") or "").split(","):
                    t = t.strip().upper().replace(".", "-")
                    if t:
                        by_ticker[t] = rec
                        _n += 1
                rec["n_tickers"] = _n   # sector-wave denominator (additive)
        except Exception as exc:  # noqa: BLE001
            if diag:
                diag.warn(f"industry RS parse failed: {exc}")
    rows.sort(key=lambda r: r["pct"], reverse=True)
    log.info("Industry RS: %d groups, %d tickers mapped", len(rows), len(by_ticker))
    return {"rows": rows, "by_ticker": by_ticker}


def attach_industry_rs(stocks: List[dict], by_ticker: Dict[str, dict]) -> None:
    """Tag each pick with its industry-group RS percentile + name (display-only;
    never touches the IBKR draft plan). Also attaches the weekly Weinstein
    group-stage read (group_stage_map.json, 14-day staleness suppressor) as
    additive grp_* keys for the GRP chip."""
    gmap, gwaves = _load_group_stage_map(), _sector_wave_index()
    for s in stocks:
        rec = by_ticker.get((s.get("ticker") or "").upper().replace(".", "-"))
        s["ind_rs"] = rec["pct"] if rec else None
        s["ind_name"] = rec["industry"] if rec else None
        try:
            g = gmap.get(s["ind_name"]) if (gmap and s.get("ind_name")) else None
            if g:
                s["grp_stage"] = g.get("stage")
                s["grp_above"] = g.get("pct_above")
            w = gwaves.get(s.get("ind_name")) if gwaves else None
            if w:
                s["grp_wave_n"] = w.get("n")
                s["grp_wave_size"] = w.get("size")
        except Exception:  # noqa: BLE001
            pass


_group_stage_cache: Optional[dict] = None
_sector_wave_cache: Optional[dict] = None


def _load_group_stage_map() -> Optional[dict]:
    """group_stage_map.json (weekly, build_group_stage.py) with a 14-day
    staleness suppressor. Cached per run. Never raises."""
    global _group_stage_cache
    if _group_stage_cache is not None:
        return _group_stage_cache or None
    out = {}
    try:
        with open(os.path.join(WORKSPACE, "group_stage_map.json")) as fh:
            d = json.load(fh)
        asof = datetime.strptime(d.get("asof", "1970-01-01"), "%Y-%m-%d").date()
        if (date.today() - asof).days <= 14:
            out = d.get("groups") or {}
        else:
            log.warning("group_stage_map.json stale (asof %s) — GRP stage suppressed", d.get("asof"))
    except Exception:  # noqa: BLE001
        out = {}
    _group_stage_cache = out
    return out or None


def _sector_wave_index() -> dict:
    """{industry: wave-dict} from the run's sector-wave computation (cached by
    compute_sector_waves). Empty when unavailable."""
    return {w["industry"]: w for w in (_sector_wave_cache or [])}


def compute_sector_waves(by_ticker: Dict[str, dict]) -> List[dict]:
    """Group-ignition detector (Weinstein ch.3 bottom-up tally, calibrated
    2026-07-17): distinct non-ETF tickers with outcome=='win' in the last 5
    breakout-log dates, per industry; fires on (n>=3 AND n/group_size>=20%)
    OR n>=10. Informational banner only — thresholds are cadence-calibrated
    (28 log days), not proven edge. Never raises."""
    global _sector_wave_cache
    try:
        with open(BREAKOUT_LOG_PATH) as fh:
            bl = json.load(fh)
        dates = sorted(bl)[-5:]
        prev_dates = sorted(bl)[-6:-1]
        if not dates:
            _sector_wave_cache = []
            return []

        def _tally(dts):
            seen: Dict[str, set] = {}
            for dkey in dts:
                for r in bl.get(dkey, []):
                    if r.get("outcome") != "win" or r.get("is_etf"):
                        continue
                    tk = (r.get("ticker") or "").upper().replace(".", "-")
                    rec = by_ticker.get(tk)
                    if not rec or not rec.get("industry"):
                        continue
                    seen.setdefault(rec["industry"], set()).add(tk)
            return seen

        cur, prev = _tally(dates), _tally(prev_dates)

        def _fires(ind, tks):
            size = None
            for rec in by_ticker.values():
                if rec.get("industry") == ind:
                    size = rec.get("n_tickers")
                    break
            n = len(tks)
            share = (n / size) if size else 0.0
            return (n >= 3 and share >= 0.20) or n >= 10, size, share

        waves = []
        for ind, tks in cur.items():
            ok, size, share = _fires(ind, tks)
            if not ok:
                continue
            was, _s, _sh = _fires(ind, prev.get(ind, set()))
            waves.append({"industry": ind, "n": len(tks), "size": size,
                          "share": round(share * 100), "members": sorted(tks),
                          "is_new": not was})
        waves.sort(key=lambda w: -w["n"])
        _sector_wave_cache = waves
        return waves
    except Exception:  # noqa: BLE001
        _sector_wave_cache = []
        return []


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


def fetch_histories_batch_intraday(tickers: List[str], period: str = "60d",
                                   interval: str = "1h", min_rows: int = 80) -> Dict[str, pd.DataFrame]:
    """Batch intraday history (SHADOW multi-timeframe lessons). Separate from
    fetch_histories_batch on purpose: _patch_stale_histories appends a DAILY
    bar and would corrupt an hourly frame."""
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers=tickers, period=period, interval=interval,
                          group_by="ticker", auto_adjust=False, threads=True, progress=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("Intraday batch download failed: %s", exc)
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
    return out


def _pct_off_52wk_of(valid, c) -> Optional[float]:
    """% distance of close c from the 52-week high of the (close, vol) series.
    Feeds the headline-meter complacency check. Never raises."""
    try:
        hi = max(x[0] for x in valid[-252:])
        return round((c / hi - 1.0) * 100.0, 2) if hi > 0 else None
    except Exception:  # noqa: BLE001
        return None


def _fetch_one_index(ticker: str) -> Optional[dict]:
    """Fetch + compute one market-health card (used in a thread pool)."""
    # 14mo (not 3mo) so we carry ≥200 bars for the SMA200 bull/bear regime that
    # the forward-base-rate lookup conditions on. The 10/20/50-SMA, dist-days and
    # asof/TV-patch logic all read only the tail, so the longer range is inert to them.
    from urllib.parse import quote
    # Index symbols carry a caret (^IXIC) — percent-encode the path segment so the
    # Yahoo v8 request is well-formed (plain tickers like IWM pass through unchanged).
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='')}?interval=1d&range=14mo"
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

    # ---- Big Picture extras (ADDITIVE keys — nothing above changes) ----
    # Session volume vs prior session + vs 50-day average. TV-patched bars can
    # carry vol=0 -> report None rather than a bogus comparison (same audit rule
    # as the dist-day loop above).
    vol_today = valid[-1][1]
    vol_prev = valid[-2][1] if len(valid) >= 2 else 0
    vol_vs_prev = ((vol_today / vol_prev - 1) * 100) if (vol_today > 0 and vol_prev > 0) else None
    _v50 = [v for _, v in valid[-50:] if v > 0]
    vol_vs_avg50 = ((vol_today / (sum(_v50) / len(_v50)) - 1) * 100) if (vol_today > 0 and len(_v50) >= 20) else None

    # IBD-spec distribution count: trailing 25 sessions, close down >=0.2% on
    # volume above the prior session; a day EXPIRES once the index closes >=5%
    # above that day's close (O'Neil's rally-erasure rule). Kept separate from
    # `dist_days` above, which the regime tells already consume (additive-only).
    window25 = valid[-26:]
    closes25 = [c2 for c2, _ in window25]
    dist_days_ibd = 0
    for i in range(1, len(window25)):
        pc, pv = window25[i - 1]
        cc, cv = window25[i]
        if not pc or cv <= 0 or pv <= 0:
            continue
        if (pc - cc) / pc * 100 >= 0.2 and cv > pv:
            later = closes25[i + 1:]
            if later and max(later) >= cc * 1.05:
                continue                      # expired: rallied 5% past the dist close
            dist_days_ibd += 1

    # Rally-attempt day count + follow-through-day flag (O'Neil). Bottom = lowest
    # close of the last 60 bars; day 1 = first higher close after it; an FTD is a
    # day-4+ gain >=1.2% on volume above the prior session. Informational only —
    # the Big Picture prints it when the regime is RED.
    tail60 = valid[-60:]
    lows60 = [c2 for c2, _ in tail60]
    low_i = min(range(len(lows60)), key=lambda j: lows60[j])
    rally_day = 0
    ftd_today = False
    start = None
    for j in range(low_i + 1, len(tail60)):
        if lows60[j] > lows60[low_i]:
            start = j
            break
    if start is not None:
        # day 1 = the bar at `start` (first higher close after the low); today is
        # index len-1, so today's day number = (len-1) - start + 1 = len - start.
        rally_day = len(tail60) - start
        lc, lv = tail60[-1]
        pc2, pv2 = tail60[-2]
        if (rally_day >= 4 and pc2 and lv > 0 and pv2 > 0
                and (lc - pc2) / pc2 * 100 >= 1.2 and lv > pv2):
            ftd_today = True

    # Weinstein stage read for the Market Overview banner (30-week ≈ 150-day MA
    # on the daily series; slope over ~5 weeks = 25 sessions; churn window 50
    # sessions ≈ 10 weeks). ADDITIVE keys — nothing above changes.
    wk_stage = wk_ma_slope = None
    try:
        arr = np.asarray([x[0] for x in valid], dtype=float)
        if arr.size >= 176:
            ma150 = np.convolve(arr, np.full(150, 1.0 / 150.0), mode="valid")
            if ma150.size >= 26:
                slope_raw = float(ma150[-1] / ma150[-26] - 1.0) * 100.0
                if np.isfinite(slope_raw):
                    above = bool(arr[-1] > ma150[-1])
                    tail = min(50, ma150.size)
                    # Churn on ~weekly points (every 5th session) so the index
                    # read matches the stock version's 10-weekly-bar basis —
                    # daily bars whipsaw a flat MA far more often (review fix).
                    diff = (arr[-tail:] - ma150[-tail:])[::-1][::5]
                    sgn = np.sign(diff)
                    churn = int(np.sum(sgn[1:] * sgn[:-1] < 0)) if sgn.size >= 2 else 0
                    # classify on the UNROUNDED slope (parity with _stage_read)
                    wk_stage = _stage_label(above, slope_raw, churn)
                    wk_ma_slope = round(slope_raw, 2)
    except Exception:  # noqa: BLE001
        wk_stage = wk_ma_slope = None

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
        "vol_vs_prev": vol_vs_prev,
        "vol_vs_avg50": vol_vs_avg50,
        "dist_days_ibd": dist_days_ibd,
        "rally_day": rally_day,
        "ftd_today": ftd_today,
        "wk_stage": wk_stage,
        "wk_ma_slope": wk_ma_slope,
        "pct_off_52wk": _pct_off_52wk_of(valid, c),
    }


def _bc_num(v):
    """Parse a Barchart numeric that may be 'unch' (its literal for a ZERO
    day-over-day change), 'N/A', '', '+1.60', '-0.60', etc. 'unch' -> 0.0;
    anything non-numeric -> None. This is the fix for the 2026-07 outage where
    'unch' on ONE symbol's priceChange threw and killed the whole breadth block."""
    if v is None:
        return None
    s = str(v).strip().replace("+", "").replace(",", "")
    if s.lower() in ("unch", "unchanged"):
        return 0.0
    if s.lower() in ("", "n/a", "na", "null", "-", "--"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def _breadth_from_history() -> Optional[dict]:
    """Last persisted breadth (breadth_history.json) as a STALE fallback — far
    better than 'Unavailable'. Change columns are unknown for a carried reading."""
    try:
        hist = json.load(open(BREADTH_HISTORY_PATH))
        if not hist:
            return None
        k = max(hist)
        e = hist[k]
        b50 = float(e.get("br50", 50.0))
        return {"ok": True, "stale": True, "asof": k,
                "above20": float(e.get("br20", b50)),
                "above50": b50, "above200": float(e.get("br200", 50.0)),
                "chg20": None, "chg50": None, "chg200": None}
    except Exception:  # noqa: BLE001
        return None


def fetch_sp_breadth(diag: Optional[Diagnostics] = None, attempts: int = 4) -> dict:
    """S&P 500 breadth straight from Barchart (TradingView's EOD source for these
    indices): % of members above the 20/50/200-day MA = $S5TW / $S5FI / $S5TH,
    with the day-over-day percentage-point change.

    Retries the full cookie+XSRF+API handshake up to `attempts` times with linear
    backoff (transient 401s / cold-cache misses are common), parses 'unch' safely,
    and on total failure carries the last stored reading (stale) rather than
    showing 'Unavailable'. Only returns {'ok': False} if history is empty too."""
    import http.cookiejar
    import time as _time
    import urllib.parse as _up
    need = ("$S5TW", "$S5FI", "$S5TH")
    last_exc = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            cj = http.cookiejar.CookieJar()
            op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
            op.addheaders = [("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"),
                             ("Accept", "text/html")]
            op.open("https://www.barchart.com/stocks/quotes/$S5FI/overview", timeout=15).read()
            xsrf = next((_up.unquote(c.value) for c in cj if c.name == "XSRF-TOKEN"), "")
            if not xsrf:
                raise RuntimeError("no XSRF token (cold cache)")
            url = ("https://www.barchart.com/proxies/core-api/v1/quotes/get?"
                   "symbols=$S5TW,$S5FI,$S5TH&fields=symbol,lastPrice,priceChange,tradeTime")
            req = urllib.request.Request(url, headers={
                "x-xsrf-token": xsrf, "Accept": "application/json",
                "User-Agent": "Mozilla/5.0", "Referer": "https://www.barchart.com/"})
            data = json.loads(op.open(req, timeout=15).read())
            m = {r["symbol"]: r for r in data.get("data", [])}
            missing = [s for s in need if s not in m]
            if missing:
                raise RuntimeError(f"missing symbols {missing}")

            def lvl(sym):
                v = _bc_num(m[sym].get("lastPrice"))
                if v is None:
                    raise RuntimeError(f"bad lastPrice for {sym}: {m[sym].get('lastPrice')!r}")
                return v

            a20, a50, a200 = lvl("$S5TW"), lvl("$S5FI"), lvl("$S5TH")
            # priceChange 'unch' -> 0.0; genuinely absent -> None (chip just hides)
            c20 = _bc_num(m["$S5TW"].get("priceChange"))
            c50 = _bc_num(m["$S5FI"].get("priceChange"))
            c200 = _bc_num(m["$S5TH"].get("priceChange"))
            if diag and attempt > 1:
                diag.warn(f"S&P breadth: recovered on attempt {attempt}")
            return {"ok": True, "above20": a20, "above50": a50, "above200": a200,
                    "chg20": c20, "chg50": c50, "chg200": c200,
                    "asof": m.get("$S5FI", {}).get("tradeTime", "")}
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < max(1, attempts):
                _time.sleep(1.5 * attempt)   # linear backoff: 1.5s, 3s, 4.5s
    # every attempt failed -> carry the last stored reading (stale) if we have one
    stale = _breadth_from_history()
    if stale is not None:
        if diag:
            diag.warn(f"S&P breadth (Barchart) failed after {attempts} attempts "
                      f"({last_exc}); showing stale reading from {stale['asof']}")
        return stale
    if diag:
        diag.warn(f"S&P breadth (Barchart) fetch failed after {attempts} attempts "
                  f"and no history: {last_exc}")
    return {"ok": False, "above50": 50.0, "above200": 50.0}


def fetch_market_health(diag: Optional[Diagnostics] = None) -> Tuple[List[dict], dict]:
    """Index health (parallel Yahoo) + S&P-500 breadth (Barchart)."""
    # ^IXIC (Nasdaq Composite) leads — it drives the GREEN/YELLOW/RED regime and
    # the market-modifier. The S&P 500 and Nasdaq-100 cards use the INDICES
    # ^GSPC/^NDX (2026-07-06 USER: was the SPY/QQQ ETFs) for deeper history; IWM
    # stays an ETF so it remains the TradingView stale-bar freshness anchor (the
    # TV scan API can't patch caret indices — see the data_date note in run()).
    tickers = ["^IXIC", "^NDX", "^GSPC", "IWM"]
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
    """Per-sector RS vs the S&P 500 index ^GSPC (1-month relative perf) + short-
    term momentum (price vs its own 50-DMA). Flags how many leadership sectors
    are rolling over. (2026-07-06 USER: benchmark was the SPY ETF.)"""
    try:
        raw = yf.download(tickers=SECTOR_ETFS + ["^GSPC"], period="3mo", interval="1d",
                          group_by="ticker", auto_adjust=False, threads=True, progress=False)
        if raw is None or len(raw) == 0:
            return None
        multi = isinstance(raw.columns, pd.MultiIndex)

        def closes(t):
            df = (raw[t] if multi else raw).dropna()
            return df["Close"] if len(df) >= 50 else None

        bench = closes("^GSPC")
        if bench is None:
            return None
        spy_1m = bench.iloc[-1] / bench.iloc[-21] - 1.0
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


# ---- palette constants ("calm editorial dark" 2026-07-05) ------------------
# Single source of truth for every colour drawn OUTSIDE the CSS cascade
# (inline SVG fill/stroke attributes can't read CSS variables when the SVG is
# built server-side). The :root block in PAGE_CSS mirrors these values —
# keep the two in sync.
C_UP = "#54b87f"       # bullish / up-day
C_DOWN = "#e06c6a"     # bearish / down-day
C_WARN = "#d3a04d"     # caution amber
C_ACCENT = "#8cb4d6"   # the one UI accent
C_MUTED = "#4a4a52"    # volume bars, quiet fills
C_TEXT3 = "#82827c"    # caption gray (SVG labels, doji candles)

# MA overlay colours + the periods we draw.
# 50-MA deliberately the most muted (slowest line, should recede).
_MA_SPEC = [(10, C_ACCENT), (20, C_WARN), (50, "#6b6b74")]   # accent / amber / gray
# Darker, thinner variants for the small external-engine sparklines (Minervini /
# Trilogy) so the MA lines sit clearly BEHIND the price line.
_MA_SPEC_DARK = [(10, "#567a99"), (20, "#8a6a36"), (50, "#4d4d55")]
_MA_SPEC_10W = [(10, "#567a99")]                             # single 10-week MA (Trilogy)


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

    color = C_UP if disp[-1] >= disp[0] else C_DOWN
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


# ----------------------------------------------------------------------------
# CANDLESTICK CHART (2026-07-05 layout upgrade)
#
# The chart cell ships a compact JSON payload in a data attribute; ONE shared
# client-side renderer (CANDLE_JS) lazily draws HOLLOW candles (light grey =
# up, light red = down — USER 2026-07-06 round 2, minimal design) + hollow
# volume bars with a 50-day volume MA + labelled 10/20/50 MAs + a faint grid
# with a right-side price scale and last-price marker, plus a decluttered set
# of levels (USER 2026-07-06): the SR zone band, plan entry/stop, and the two
# SALIENT trendlines — no text tags on any line except the MA period. Server-
# side SVG candles were rejected: ~16KB/chart × ~400 would triple the size.
# ----------------------------------------------------------------------------
def _cfin(x, nd: int = 2) -> Optional[float]:
    """Finite float rounded to nd, else None (payload hygiene)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return round(v, nd)


def _candle_overlays(plan: Optional[dict]) -> dict:
    """Whitelisted chart levels, short-keyed to keep the payload small. v9
    (2026-07-18): the CLEAN chart draws only the SR zone band + the two SALIENT
    trendlines (tl_draw_* — most-touched/recent/near-price, chosen by the
    engine's salience pass). The plan entry/stop are NO LONGER drawn on the
    canvas (user picked a clean chart; the levels read as text in the card and
    live on the .lp span's data-entry/data-stop). Channel rails, PB lines and
    the SR-stop line are not drawn either (still in the collapsible text)."""
    if not plan:
        return {}
    ov: Dict[str, Any] = {}

    def put(k, v, nd=2):
        f = _cfin(v, nd)
        if f is not None:
            ov[k] = f

    # draw the last-60-bar-anchored zone when the engine produced one, else the
    # trade/gate protecting zone (USER 2026-07-06 progressive-lookback drawing)
    put("srl", plan.get("sr_draw_lo", plan.get("sr_prot_lo")))
    put("srh", plan.get("sr_draw_hi", plan.get("sr_prot_hi")))
    # salient trendlines picked for the chart; fall back to the nearest trade
    # line only if the engine produced no salient line.
    put("tsn", plan.get("tl_draw_sup_now", plan.get("tl_sup_now")))
    put("tsd", plan.get("tl_draw_sup_slope_d", plan.get("tl_sup_slope_d")), 4)
    put("trn", plan.get("tl_draw_res_now", plan.get("tl_res_now")))
    put("trd", plan.get("tl_draw_res_slope_d", plan.get("tl_res_slope_d")), 4)
    # a diagonal without its slope (or vice versa) is undrawable — drop the orphan
    for a, b in (("tsn", "tsd"), ("trn", "trd")):
        if (a in ov) != (b in ov):
            ov.pop(a, None)
            ov.pop(b, None)
    if "srl" in ov and "srh" in ov and ov["srl"] > ov["srh"]:
        ov.pop("srl"), ov.pop("srh")
    return ov


def make_candle_chart(hist_df: Optional[pd.DataFrame], plan: Optional[dict] = None,
                      window: int = 60, weekly: bool = False) -> str:
    """<div class='cchart' data-c='{...}'> rendered client-side by CANDLE_JS.

    Payload: t0 (first bar date) + dt (calendar-day gaps), OHLC (2dp, 4dp
    under $1), volume in thousands, up to 49 pre-window closes AND volumes
    (so the 50-MA and the 50-day volume MA are valid at the left edge),
    w=1 for weekly bars, ov = overlay levels.
    Never raises; '' when history is unusable."""
    try:
        if hist_df is None or len(hist_df) < 2:
            return ""
        df = hist_df
        if weekly:
            vol = df["Volume"] if "Volume" in df.columns else pd.Series(np.nan, index=df.index)
            df = pd.DataFrame({
                "Open": df["Open"].resample("W-FRI").first(),
                "High": df["High"].resample("W-FRI").max(),
                "Low": df["Low"].resample("W-FRI").min(),
                "Close": df["Close"].resample("W-FRI").last(),
                "Volume": vol.resample("W-FRI").sum(),
            }).dropna(subset=["Open", "High", "Low", "Close"])
        df = df[df["Close"].notna()]
        if len(df) < 2:
            return ""
        win = df.tail(window)
        pre_lo, pre_hi = max(0, len(df) - len(win) - 49), len(df) - len(win)
        pre = df["Close"].iloc[pre_lo:pre_hi]
        nd = 4 if float(win["Close"].iloc[-1]) < 1.0 else 2
        payload: Dict[str, Any] = {
            "t0": str(win.index[0].date()),
            "dt": [0] + [int((win.index[i] - win.index[i - 1]).days)
                         for i in range(1, len(win))],
            "o": [_cfin(x, nd) for x in win["Open"].tolist()],
            "h": [_cfin(x, nd) for x in win["High"].tolist()],
            "l": [_cfin(x, nd) for x in win["Low"].tolist()],
            "c": [_cfin(x, nd) for x in win["Close"].tolist()],
            "v": [int(round((f or 0.0) / 1000.0)) for f in
                  (_cfin(x, 0) for x in win["Volume"].tolist())] if "Volume" in win.columns
                 else [0] * len(win),
            "p": [_cfin(x, nd) for x in pre.tolist()],
            # pre-window volumes (thousands) so the 50-day volume MA has a full
            # 50-bar tail at the left edge of the 60-bar window
            "pv": [int(round((f or 0.0) / 1000.0)) for f in
                   (_cfin(x, 0) for x in df["Volume"].iloc[pre_lo:pre_hi].tolist())]
                  if "Volume" in df.columns else [],
            "w": 1 if weekly else 0,
        }
        ov = _candle_overlays(plan)
        if ov:
            payload["ov"] = ov
        js = json.dumps(payload, separators=(",", ":")).replace("'", "&#39;")
        return f"<div class='cchart' data-c='{js}'></div>"
    except Exception:                                    # never kill a scan
        return ""


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
            # Weinstein VOL read (S1 study 2026-07-17): trigger-day volume vs the
            # scan-time 5-day base stamped on the pick. Display/log ONLY — never a
            # filter (94% of monsters trigger on <2x volume).
            dvol = None
            try:
                vb = s.get("vol_base5")
                tv = float(today.get("Volume") or 0)
                if vb and tv > 0:
                    dvol = round(tv / float(vb), 2)
            except Exception:  # noqa: BLE001
                dvol = None
            # Loss post-mortem factor labels (Weinstein ch.10 diary; F4 spec) —
            # descriptive booleans computed from the FROZEN pick row; None when
            # the source field is missing. f_lowvol is a LABEL only: the quiet-
            # volume FILTER is refuted (pb2 study), this never gates anything.
            _vp = s.get("vol_pct")
            _rs = s.get("rs_rating")
            _ir = s.get("ind_rs")
            _st = s.get("wk_stage")
            log_records.append({"ticker": ticker, "tier": y_tier, "outcome": outcome,
                                "htf": bool(s.get("is_htf")),
                                # ETF picks (2026-07-15) ARE graded — they're displayed picks and
                                # the win-rate chip reflects what the report showed — but the tag
                                # keeps the series separable from the stock breakout stats.
                                "is_etf": bool(s.get("is_etf")),
                                # regime marker so the rolling win-rate series isn't misread across
                                # the 2026-07-06 stop-geometry boundary (graded on THIS pick's y_stop).
                                "stop_version": s.get("stop_version", "tight_3day"),
                                "dvol": dvol,
                                "f_lowvol": (bool(_vp < 50) if isinstance(_vp, (int, float)) else None),
                                "f_rs_low": (bool(_rs < 80) if isinstance(_rs, (int, float)) else None),
                                "f_ind_weak": (bool(_ir < 50) if isinstance(_ir, (int, float)) else None),
                                "f_not_s2": (not str(_st).startswith("S2")) if _st else None})
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
                status_text, status_style = "✗ Failed (Triggered & Stopped Out)", "color:#e06c6a;font-weight:bold;"
            elif t_close >= y_entry:
                status_text, status_style = f"✓ Triggered & Holding (+{t_change:.1f}%)", "color:#54b87f;font-weight:bold;"
            else:
                status_text, status_style = "~ Triggered & Pulling Back", "color:#d3a04d;font-weight:bold;"
        else:
            status_text, status_style = "✗ Stopped Out (No Trigger)", "color:#e06c6a;font-weight:bold;"

        change_col = "good" if t_change > 0 else "bad"
        tier_badge_style = {
            "A+": "background:var(--tint-green);color:#54b87f;border:1px solid #54b87f;",
            "A":  "background:var(--tint-yellow);color:#d3a04d;border:1px solid #d3a04d;",
            "A-": "background:#1f1f23;color:#82827c;border:1px solid #82827c;",
        }.get(y_tier, "background:#1f1f23;color:#82827c;border:1px solid #82827c;")
        rows.append(f"""
            <tr style="border-bottom:1px solid #26262b;text-align:center;">
                <td style="padding:10px;font-weight:bold;" class="ticker"><a href="https://tradingview.com/symbols/{esc(ticker)}" target="_blank">{esc(ticker)}</a></td>
                <td style="padding:10px;"><span style="font-size:var(--fs-table);font-weight:bold;padding:3px 6px;border-radius:4px;{tier_badge_style}">Tier {esc(y_tier)}</span></td>
                <td style="padding:10px;"><span class="score" style="border-color:#aecfe8;color:#aecfe8;background:var(--tint-accent);">{y_score}</span></td>
                <td style="padding:10px;color:#54b87f;font-weight:bold;">${y_entry:.2f}</td>
                <td style="padding:10px;color:#e06c6a;font-weight:bold;">${y_stop:.2f}</td>
                <td style="padding:10px;font-size:var(--fs-body);color:#82827c;">${t_low:.2f} - ${t_high:.2f}</td>
                <td style="padding:10px;" class="{change_col}">{t_change:+.2f}%</td>
                <td style="padding:10px;{status_style}">{status_text}</td>
            </tr>""")

    if log_records:                          # each record already passed per-ticker freshness
        _append_breakout_log(log_records, data_date)

    if n_eval == 0:
        return ""

    # ---- summary only (full grading still logged to the win-rate; the per-name
    #      table is intentionally omitted to keep the report short) ----
    src_bit = (f" <span style='color:#82827c;'>(picks of {prev_date})</span>"
               if prev_date else "")
    if not may_log:
        win_bit = (f"<span style='color:#d3a04d;'>⚠️ outcomes withheld — bars end "
                   f"{esc(bar_date or '?')}, need a session newer than {esc(prev_date or 'the picks')}</span>")
    elif n_win or n_loss:
        win_bit = (f"<span style='color:#54b87f;'>✓ {n_win} win</span> / "
                   f"<span style='color:#e06c6a;'>✗ {n_loss} loss</span> "
                   f"<span style='color:#82827c;'>this session</span>")
    else:
        win_bit = "<span style='color:#82827c;'>0 triggered this session</span>"
    cum = breakout_cumulative()
    cum_bit = ""
    if cum:
        ccol = "#54b87f" if cum["rate"] > 55 else ("#d3a04d" if cum["rate"] >= 40 else "#e06c6a")
        cum_bit = (f" &nbsp;|&nbsp; <span style='color:{ccol};font-weight:bold;'>Accumulated win-rate "
                   f"{cum['rate']}%</span> <span style='color:#82827c;'>({cum['wins']}W/{cum['losses']}L "
                   f"over {cum['n']} · since {esc(cum['since'])})</span>")
        if cum.get("htf_n"):
            hr = round(100 * cum["htf_wins"] / cum["htf_n"])
            hcol = "#54b87f" if hr > 55 else ("#d3a04d" if hr >= 40 else "#e06c6a")
            cum_bit += (f" <span style='color:{hcol};'>🚩 HTF {hr}%</span> "
                        f"<span style='color:#82827c;'>({cum['htf_wins']}W/{cum['htf_losses']}L)</span>")
    coiled_bit = ""
    if coiled_total:
        cb = " · ".join(f"{t} {coiled[t]}" for t in ("A+", "A", "A-") if coiled.get(t))
        coiled_bit = f" &nbsp;|&nbsp; <span style='color:#82827c;'>{coiled_total} still coiling ({cb})</span>"
    # Weinstein VOL aggregate for today's triggered picks (display/log only).
    vol_bit = ""
    _dv = [r["dvol"] for r in log_records if r.get("dvol")]
    if _dv:
        _n2 = sum(1 for v in _dv if v >= 2.0)
        _tip = ("Trigger-day volume vs the pick's prior-5-day average (Weinstein ch.4 "
                "2x rule). Backtest 2026-07-17, 181,212 triggered coil episodes: >=2x -> "
                "65%/69% +1R rate (pre-2026/2026+) vs 49.5%/46% under 1x - but 94% of "
                "monsters trigger under 2x volume, so informational only, never a filter.")
        vol_bit = (f" &nbsp;|&nbsp; <span style='color:#82827c;' title='{esc(_tip)}'>"
                   f"VOL≥2x on {_n2}/{len(_dv)} triggered · avg {sum(_dv)/len(_dv):.1f}x</span>")
    # Regime footnote (continuity marker, §3a) — self-activating: shows ONLY once the log carries
    # atr_5day outcomes, so it's invisible on pre-switch reports and marks the boundary once live.
    regime_bit = ""
    try:
        if os.path.exists(BREAKOUT_LOG_PATH):
            with open(BREAKOUT_LOG_PATH) as _fh:
                _bl = json.load(_fh)
            if any(r.get("stop_version") == "atr_5day" for _d in _bl for r in _bl[_d]):
                regime_bit = (" &nbsp;|&nbsp; <span class='sub'>"
                              "⚙️ coil stop regime: 1.5×ADR + 5-day validity, effective session "
                              "2026-07-06 (first printed on the 2026-07-07 report); earlier picks "
                              "used the tight stop</span>")
    except Exception:  # noqa: BLE001 — a footnote must never break the report
        regime_bit = ""
    summary = (f"<span style='color:#ececea;'>Graded <b>{n_eval}</b> prior picks{src_bit}</span> "
               f"&nbsp;|&nbsp; {win_bit}{cum_bit}{coiled_bit}{vol_bit}{regime_bit}")

    return f"""
    <div style="background-color:#18181b;border-radius:8px;padding:12px 15px;margin-bottom:25px;box-shadow:0 0 15px rgba(0,0,0,0.5);">
        <span style="color:#aecfe8;font-weight:bold;font-size:var(--fs-body);text-transform:uppercase;">Yesterday's Watchlist:</span>
        <span style="font-size:var(--fs-table);">&nbsp;{summary}</span>
    </div>
    """


# ----------------------------------------------------------------------------
# GEOMETRY RATIFICATION (user-approved 2026-07-04): coil A+/A/A- switch to the
# 1.5×ADR stop + 5 trading-day validity, effective for sessions on/after the
# switch session. DATE-GATED so pre-switch sessions (incl. a holiday re-run of
# older data) are byte-identical — the switch cannot fire on stale data. The
# M.E.T.A. score/tier are already computed on the STRUCTURAL risk (continuity
# preserved, no re-calibration); this pass only overrides the PRINTED stop, its
# risk_pct (→ IBKR sizing follows: wider stop → smaller position), and stamps
# valid_until + stop_version. The structural level is kept as stop_tight (+reason)
# for the report's "support" line and as a stable ledger field.
# ----------------------------------------------------------------------------
STOP_REGIME_SWITCH_SESSION = "2026-07-06"   # first session under ATR+5d (first printed on the 2026-07-07 run)
_ATR_STOP_MULT = 1.5
_VALIDITY_TRADING_DAYS = 5


def _sr_quality(hist_df, entry, direction):
    """S/R entry-quality features (madrry_sr_zones). INFORMATIONAL SHADOW MODE
    per WINNER_RADAR_CONTINUATION.md 2.7 — the sr_* keys are displayed and logged
    to the ledger but filter nothing and change no printed entry/stop. Never
    raises; returns {} when the module or history is unavailable."""
    if _srz is None or hist_df is None or entry is None:
        return {}
    try:
        out = _srz.analyze(hist_df, entry, direction)
        return out if isinstance(out, dict) and "sr_error" not in out else {}
    except Exception:  # noqa: BLE001
        return {}


def _pb2_quality(hist_df):
    """Pullback-recovery features (madrry_pullback_buy). INFORMATIONAL SHADOW
    MODE per WINNER_RADAR_CONTINUATION.md 2.7 — displayed and logged, filters
    nothing, changes no printed entry/stop. Never raises."""
    if _pbb is None or hist_df is None:
        return {}
    try:
        out = _pbb.analyze_pullback(hist_df)
        return out if isinstance(out, dict) and "pb2_error" not in out else {}
    except Exception:  # noqa: BLE001
        return {}


def _edge_support(m: dict) -> List[str]:
    """Which verified entry engines back this pick — the USER-RATIFIED Stage-3
    support gate (2026-07-04): a coil pick must be backed by at least one of
      SR = S/R zone structure grade A/B (protecting zone, flip/retest read),
      PB = a valid 8-rule pullback-recovery (setup or recovery, vetoes clear),
      TL = a governing diagonal supports the entry (at the UTL/TSL, a fresh
           break-up, or diagonal support within 1.5 ATR).
    Each engine was verified point-in-time against its tutorial's own trades
    (27/27, 23/23, 16/16 — see VERIFICATION.md). Never raises."""
    out: List[str] = []
    try:
        if m.get("sr_grade") in ("A", "B"):
            out.append("SR")
        if m.get("pb2_state") in ("setup", "recovery"):
            out.append("PB")
        tf = [str(f) for f in (m.get("tl_flags") or [])]
        d = m.get("tl_sup_dist_atr")
        if (any(f.startswith(("at_UTL", "at_TSL", "fresh_break_up")) for f in tf)
                or (isinstance(d, (int, float)) and d <= 1.5)):
            out.append("TL")
    except Exception:  # noqa: BLE001
        pass
    return out


def _lesson_plan(m: dict) -> None:
    """2026-07-06 USER: the PRINTED trade plan follows the tutorial lessons when
    an entry engine produced a validated plan — the lesson geometry (PB trigger
    over the mini-DTL with the stop under the prior low; SR stop just outside
    the protecting zone) is more accurate than the generic breakout math
    (high+0.10 / min(low, MA)−0.05). Mutates entry/stop IN PLACE so the chart
    overlays, the IBKR order plan and the forward trackers all see ONE plan;
    the originals are kept as entry_raw/stop_raw and the source is tagged in
    plan_src for the cell. Bounded: risk must stay ≤10% (tutorial hard limit)
    or the refinement is skipped. Never raises."""
    try:
        e0, s0 = m.get("entry"), m.get("stop")
        m["entry_raw"], m["stop_raw"] = e0, s0
        m["stop_reason_raw"] = m.get("stop_reason")
        # L2 pullback-recovery = the entry-TIMING lesson: a live trigger replaces
        # the whole plan (its stop is the tutorial's under-the-prior-low stop).
        # The trigger must sit ABOVE the current close (adversarial review
        # 2026-07-07): a 'recovery' trigger is below close by construction, and
        # printing an already-cleared Buy stages below-market limit drafts and
        # auto-wins next-session grading — inflating the breakout win-rate tell.
        t, st = m.get("pb2_trigger"), m.get("pb2_stop")
        if (m.get("pb2_state") in ("setup", "recovery") and t and st
                and 0 < float(st) < float(t)
                and float(t) > float(m.get("close") or 0)
                and (float(t) - float(st)) / float(t) * 100.0 <= 10.0):
            m["entry"] = round(float(t), 2)
            m["stop"] = round(float(st), 2)
            m["plan_src"] = "PB"
            m["stop_reason"] = "PB prior-low"
        # L1 S&R = the stop-PLACEMENT lesson: with a graded protecting zone the
        # stop belongs just OUTSIDE the zone, not at a generic low/EMA offset.
        elif (m.get("sr_grade") in ("A", "B") and m.get("sr_stop_suggest") and e0
              and 0 < float(m["sr_stop_suggest"]) < float(e0)
              and (float(e0) - float(m["sr_stop_suggest"])) / float(e0) * 100.0 <= 10.0):
            m["stop"] = round(float(m["sr_stop_suggest"]), 2)
            m["plan_src"] = "SR"
            m["stop_reason"] = "outside SR zone"
        else:
            return
        e, s = float(m["entry"]), float(m["stop"])
        m["risk_pct"] = round((e - s) / e * 100.0, 1)
    except Exception:  # noqa: BLE001 — plan refinement must never kill a scan
        m.pop("plan_src", None)


def _lesson_confluence(m: dict) -> List[str]:
    """Which of the four tutorial lessons' ENTRY criteria this pick meets with
    QUALITY (USER-DIRECTED 2026-07-05: surface these as important). Stricter
    than the Stage-4 gate on purpose - the gate asks "is the pick backed",
    this asks "is it a textbook example":
      L1 S&R       - protecting zone graded A (flip/shakeout/confluence live
                     inside the grade),
      L2 pullback  - valid setup/recovery with risk <= 6% (the tutorial's
                     ideal band; vetoes already cleared by the state),
      L3 trendline - at the governing UTL/TSL or a fresh line break-up,
      L4 channel   - inside an UP channel whose read carries quality
                     (fresh 2+1 projection or a top confluent with a
                     horizontal zone - both era-consistent in ch_study).
    Display + dashboard-ranking only; never a filter; the IBKR draft plan
    ranking is untouched. Never raises."""
    out: List[str] = []
    try:
        if m.get("sr_grade") == "A":
            out.append("S&R")
        r = m.get("pb2_risk_pct")
        if m.get("pb2_state") in ("setup", "recovery") \
                and isinstance(r, (int, float)) and r <= 6.0:
            out.append("PB")
        tf = [str(f) for f in (m.get("tl_flags") or [])]
        if any(f.startswith(("at_UTL", "at_TSL", "fresh_break_up")) for f in tf):
            out.append("TL")
        cf = [str(f) for f in (m.get("ch_flags") or [])]
        if m.get("ch_dir") == "up" and (
                "fresh_projection" in cf or "top_sr_confluence" in cf):
            out.append("CH")
    except Exception:  # noqa: BLE001
        pass
    return out


def _lessons_on_frame(hist_df, entry, include_pb: bool = True):
    """Lesson-confluence tokens on an ARBITRARY bar frame (weekly / 1h).
    SHADOW ONLY: never gates tiers, never touches entry/stop/IBKR plan.
    None when the frame is unusable (caller omits the timeframe chip)."""
    if hist_df is None or len(hist_df) < 60 or entry is None:
        return None
    try:
        feats: dict = {}
        feats.update(_sr_quality(hist_df, entry, "long"))
        if include_pb:
            feats.update(_pb2_quality(hist_df))
        feats.update(_tl_quality(hist_df, entry, "long"))
        feats.update(_ch_quality(hist_df, entry, "long"))
        return _lesson_confluence(feats)
    except Exception:  # noqa: BLE001
        return None


def _lessons_line(m: dict) -> str:
    """Gold 🎓 badge when >=3 of the four lessons agree on the entry - the
    instructor's multiple-edge trading area, surfaced per the user's
    2026-07-05 direction. Never raises - '' when below the bar."""
    try:
        ls = m.get("lesson_confluence")
        if not ls or len(ls) < 3:
            return ""
        label = " + ".join(esc(str(x)) for x in ls)
        tip = (f"LESSON CONFLUENCE {len(ls)}/4 - this entry is a textbook case of "
               f"{len(ls)} tutorial lessons at once: "
               + ", ".join({"S&R": "grade-A protecting zone (flip/shakeout/confluence)",
                            "PB": "valid pullback-recovery, risk <= 6%",
                            "TL": "at the governing trendline or fresh break-up",
                            "CH": "up channel with a quality 2+1 rail read"}.get(str(x), str(x))
                           for x in ls)
               + ". Multiple independent edges stacking on one entry - the "
                 "multiple-edge trading area. Boosts dashboard Top-Picks ranking; "
                 "filters nothing; draft order plan unaffected.")
        return (f"<div class='edge-line' title='{esc(tip)}'>"
                f"<span class='lbl lbl-hot'>LESSONS {len(ls)}/4</span> "
                f"<span style='color:#d3a04d;'>{label}</span></div>")
    except Exception:  # noqa: BLE001
        return ""


def _line_break_watch(hist_df, close):
    """USER-TAUGHT resistance-line rule (2026-07-08, backtested same day):
    line across two swing highs (strict max of +-3 bars, confirmed 3 later),
    span 40-240 td, slope <= +0.15%/day of P1, no intermediate high above
    line*1.01; BREAK = first close > line*1.005 (high > line*1.06 first
    invalidates). Returns additive display keys: lbw_state 'break' (closed
    above within the last 3 bars) or 'watch' (ceiling within 10% overhead),
    lbw_line_at, lbw_span. SHADOW/display-only. Never raises."""
    try:
        if hist_df is None or len(hist_df) < 60 or not close:
            return {}
        H = hist_df["High"].values.astype(float)
        C = hist_df["Close"].values.astype(float)
        n = len(H)
        piv = [i for i in range(3, n - 3)
               if H[i] == H[i - 3:i + 4].max() and H[i - 3:i + 4].argmax() == 3]
        best_watch, best_break = None, None
        for a in range(len(piv)):
            i1 = piv[a]
            for b2 in range(a + 1, len(piv)):
                i2 = piv[b2]
                span = i2 - i1
                if span < 40:
                    continue
                if span > 240:
                    break
                slope = (H[i2] - H[i1]) / span
                if slope > 0.0015 * H[i1]:
                    continue
                mid = H[i1 + 1:i2]
                line_mid = H[i1] + slope * np.arange(1, span)
                if np.any(mid > line_mid * 1.01):
                    continue
                state, brk_bar = "active", None
                for bb in range(i2 + 3, n):
                    lv = H[i1] + slope * (bb - i1)
                    if lv <= 0:
                        state = "dead"
                        break
                    if C[bb] > lv * 1.005:
                        state, brk_bar = "broken", bb
                        break
                    if H[bb] > lv * 1.06:
                        state = "dead"
                        break
                lv_now = H[i1] + slope * (n - 1 - i1)
                if state == "broken" and brk_bar >= n - 3:
                    if best_break is None or span > best_break[1]:
                        best_break = (lv_now, span)
                elif state == "active" and lv_now > 0 and \
                        (lv_now * 0.90) <= close <= (lv_now * 1.005):
                    if best_watch is None or span > best_watch[1]:
                        best_watch = (lv_now, span)
        if best_break:
            return {"lbw_state": "break", "lbw_line_at": round(float(best_break[0]), 2),
                    "lbw_span": int(best_break[1])}
        if best_watch:
            return {"lbw_state": "watch", "lbw_line_at": round(float(best_watch[0]), 2),
                    "lbw_span": int(best_watch[1])}
        return {}
    except Exception:  # noqa: BLE001
        return {}


def _lbw_line(m: dict) -> str:
    """Display row for the line-break watch. '' when no line is relevant."""
    try:
        st = m.get("lbw_state")
        if st not in ("break", "watch"):
            return ""
        lv, span = m.get("lbw_line_at"), m.get("lbw_span")
        tip = ("Resistance line across two swing highs (span %s td, flat-to-gently-rising, "
               "no pierce in between). Backtest 2026-07-08 (53,788 episodes): adding the "
               "line-break as a re-entry trigger cut missed +50%% runs from 6.5%% to 2.9%% "
               "of episodes. Informational only - no tier, plan or draft impact." % span)
        if st == "break":
            body = ("<span style='color:#d3a04d;font-weight:600;'>BREAK</span> · "
                    "closed above the $%s ceiling (span %sd)" % (lv, span))
        else:
            body = ("<span style='color:var(--text-2);'>watch · ceiling $%s overhead "
                    "(span %sd)</span>" % (lv, span))
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>LINE</span> " + body + "</div>")
    except Exception:  # noqa: BLE001
        return ""


# ---- Weinstein Phase-1 chips (2026-07-16 book study): Mansfield RS zero-cross,
# overhead-supply zone, weekly stage read. ALL additive display keys, SHADOW/
# display-only — no tier, plan, gate or draft impact. Verdicts + backtest numbers
# recorded in memory note `weinstein-study-verdicts`.

_STOOQ_ROOT = os.path.expanduser("~/Downloads/data/daily/us")
_stooq_files_cache: Optional[Dict[str, str]] = None


def _stooq_file(ticker: str) -> Optional[str]:
    """Path of the local Stooq daily file for `ticker` (one os.walk, cached).
    Stooq names are lowercase with '-' for '.' (BRK.B -> brk-b.us.txt)."""
    global _stooq_files_cache
    if _stooq_files_cache is None:
        idx: Dict[str, str] = {}
        try:
            for base, _dirs, files in os.walk(_STOOQ_ROOT):
                for f in files:
                    if f.endswith(".us.txt"):
                        idx[f[:-len(".us.txt")].upper()] = os.path.join(base, f)
        except Exception:  # noqa: BLE001
            idx = {}
        if not idx:
            # Loud, once: under launchd the 08:16 run may be TCC-blocked from
            # ~/Downloads (known Mac behavior in this project) — OH chips then
            # silently run on the 2y basis. Make that visible in the log.
            log.warning("Stooq dump at %s unreadable/empty (TCC on launchd?) — "
                        "OH chips fall back to the 2y basis", _STOOQ_ROOT)
        _stooq_files_cache = idx
    return _stooq_files_cache.get((ticker or "").upper().replace(".", "-"))


def _overhead_read(ticker: str, close, hist_df) -> dict:
    """Overhead-supply zone vs the prior multi-year high. Reference = max high
    over the past 5y EXCLUDING the last ~3 months (local Stooq file; falls back
    to the 2y yfinance history excluding the last 63 bars). Zones from the
    2026-07-16 backtest (5,307 breakouts): DEEP <0.67x / SUPPLY 0.67-0.85 /
    CLEAR 0.85-1.09 (best median+failure zone) / EXT >1.09. Heavy-supply names
    showed ~2x the MONSTER rate, so this must never become a veto. Additive
    display keys only. Never raises."""
    try:
        if (not close) and hist_df is not None and len(hist_df):
            close = float(hist_df["Close"].iloc[-1])
        if not close or close != close or close <= 0:  # None / NaN / non-positive
            return {}
        # 2y yfinance reference (always computed when possible): fallback when
        # no Stooq file exists, and blended in when the local dump has gone
        # stale enough (>100d) that the [now-5y, now-92d] window would silently
        # miss recent shelf highs.
        yf_ref = None
        if hist_df is not None and len(hist_df) > 130:
            v = float(hist_df["High"].iloc[:-63].max())
            if v == v and v > 0:                       # NaN-safe
                yf_ref = v
        ref = basis = None
        p = _stooq_file(ticker)
        if p:
            try:
                raw = pd.read_csv(p)
                dts = pd.to_datetime(raw.iloc[:, 2].astype(str), format="%Y%m%d",
                                     errors="coerce")
                his = pd.to_numeric(raw.iloc[:, 5], errors="coerce")
                now = pd.Timestamp.today()
                w = (dts >= now - pd.Timedelta(days=5 * 365)) & \
                    (dts <= now - pd.Timedelta(days=92))
                if int(w.sum()) >= 150 and his[w].notna().any():
                    ref, basis = float(his[w].max()), "5y"
                    # Post-dump split guard (2026-07-17 review): the dump is a
                    # frozen snapshot, so a split AFTER its last bar leaves the
                    # 5y highs in pre-split terms while the live close is post-
                    # split. Compare closes on overlapping dates; a persistent
                    # ratio != 1 is the split factor — rescale the reference.
                    try:
                        if hist_df is not None and len(hist_df):
                            stq = pd.Series(
                                pd.to_numeric(raw.iloc[:, 7], errors="coerce").values,
                                index=dts).dropna()
                            stq = stq[stq.index.notna()]
                            yfc = hist_df["Close"].copy()
                            yfc.index = yfc.index.normalize()
                            common = stq.index.intersection(yfc.index)[-10:]
                            if len(common) >= 3:
                                f = float((yfc.loc[common] / stq.loc[common]).median())
                                if f > 0 and abs(f - 1.0) > 0.02:
                                    ref *= f
                    except Exception:  # noqa: BLE001
                        pass
                    # Stale-dump blend: past ~100d the [now-5y, now-92d] window
                    # loses recent shelf highs — hand over to the live 2y ref
                    # when it is the higher one (and label it honestly).
                    if (dts.max() < now - pd.Timedelta(days=100)
                            and yf_ref is not None and yf_ref > ref):
                        ref, basis = yf_ref, "2y"
            except Exception:  # noqa: BLE001
                ref = None
        if ref is None and yf_ref is not None:
            ref, basis = yf_ref, "2y"
        if not ref or ref != ref or ref <= 0:          # None / NaN / non-positive
            return {}
        ratio = close / ref
        zone = ("DEEP" if ratio < 0.67 else "SUPPLY" if ratio < 0.85
                else "CLEAR" if ratio <= 1.09 else "EXT")
        return {"oh_ratio": round(ratio, 3), "oh_zone": zone,
                "oh_basis": basis, "oh_ref": round(ref, 2)}
    except Exception:  # noqa: BLE001
        return {}


def _mansfield_read(hist_df, spy_close) -> dict:
    """Weinstein/Mansfield RS: weekly (W-FRI) stock/S&P ratio vs its own 52-week
    SMA (the 'zero line'). mans_val = ratio/SMA52 - 1 for the latest week;
    mans_cross_wks = full weekly bars since the last neg->pos zero cross while
    still positive today (0 = this week), None if never/uncomputable/crossed
    before the 2y window. Backtest 2026-07-16 (69,512 breakouts, 10y Stooq):
    a cross <=8wk before a 52w-high breakout catches 71.6% of the monsters the
    RS-80 gate misses (median lead 3wk, era-consistent) at ~19 non-monsters per
    monster -> monster-FINDER chip, never a gate. Never raises."""
    try:
        if hist_df is None or spy_close is None or not len(hist_df):
            return {}
        wk_s = hist_df["Close"].resample("W-FRI").last().dropna()
        wk_b = spy_close.resample("W-FRI").last().dropna()
        ratio = (wk_s / wk_b).dropna()
        if len(ratio) < 56:                    # 52wk MA + a little cross context
            return {}
        mans = (ratio / ratio.rolling(52).mean() - 1.0).dropna()
        if len(mans) < 2:
            return {}
        vals = mans.values
        cross = None
        if vals[-1] > 0:
            k = len(vals) - 1
            while k > 0 and vals[k - 1] > 0:
                k -= 1
            if k > 0:                          # vals[k-1] <= 0 -> genuine cross
                cross = len(vals) - 1 - k
        return {"mans_val": round(float(vals[-1]), 4),
                "mans_cross_wks": (int(cross) if cross is not None else None)}
    except Exception:  # noqa: BLE001
        return {}


def _stage_read(hist_df) -> dict:
    """Weinstein weekly stage classifier (ch.2 quiz algorithm): 30-week MA of
    W-FRI closes; slope over ~5 weeks (rising > +0.5%, falling < -0.5%); churn
    = >=3 MA crossings in the last 10 weeks (Stage-3 whipsaw tell). Additive
    display keys only. Never raises."""
    try:
        if hist_df is None or len(hist_df) < 190:
            return {}
        wk = hist_df["Close"].resample("W-FRI").last().dropna()
        if len(wk) < 36:
            return {}
        ma = wk.rolling(30).mean().dropna()
        if len(ma) < 6:
            return {}
        slope = float(ma.iloc[-1] / ma.iloc[-6] - 1.0) * 100.0
        above = bool(float(wk.iloc[-1]) > float(ma.iloc[-1]))
        diff = (wk.reindex(ma.index) - ma).dropna().tail(10)
        sgn = np.sign(diff.values)
        churn = int(np.sum(sgn[1:] * sgn[:-1] < 0)) if len(sgn) >= 2 else 0
        st = _stage_label(above, slope, churn)
        return {"wk_stage": st, "wk_ma_slope": round(slope, 2),
                "wk_above_ma": above}
    except Exception:  # noqa: BLE001
        return {}


def _stage_label(above: bool, slope: float, churn: int) -> str:
    """Shared Weinstein stage classification (stocks + index cards).
    S1 branch (review 2026-07-17): price below a FLAT MA is a Stage-1 base,
    not Stage 4 — ch.2 reserves Stage 4 for a genuinely declining MA."""
    if above and slope > 0.5:
        return "S2"
    if above and slope >= -0.5:
        return "S3? churn" if churn >= 3 else "S2 flat-MA"
    if above:
        return "S3? MA↘"
    if slope > 0.5:
        return "S2 dip"
    if slope >= -0.5:
        return "S1 base"
    return "S4⚠"


def _weinstein_keys(s: dict, df, spy_close) -> None:
    """Attach the three Weinstein Phase-1 chip key sets to one row (additive;
    each helper returns {} on any problem so a partial fetch can never poison
    the row)."""
    s.update(_mansfield_read(df, spy_close))
    s.update(_stage_read(df))
    s.update(_overhead_read(s.get("ticker"), s.get("close"), df))
    s.update(_tc_read(df, s.get("entry")))


def attach_weinstein(stocks: List[dict], diag=None, hist=None, spy_close=None) -> None:
    """Standalone Weinstein-chip attachment for displayed rows that do NOT go
    through attach_ants (NH-52wk green rows, Lesson Radar). Reuses a caller-
    provided history batch when available; otherwise fetches its own (2y +
    benchmark). Additive keys only; never raises."""
    try:
        if not stocks:
            return
        if hist is None:
            tickers = sorted({s["ticker"] for s in stocks if s.get("ticker")}
                             | {ANTS_BENCHMARK})
            hist = fetch_histories_batch(tickers, period="2y", min_rows=60)
        if spy_close is None:
            spy_df = hist.get(ANTS_BENCHMARK)
            spy_close = spy_df["Close"] if spy_df is not None else None
        for s in stocks:
            _weinstein_keys(s, hist.get(s.get("ticker")), spy_close)
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"Weinstein chip attachment skipped: {exc}")


def _etf_note(m: dict) -> str:
    """Cohort disclosure for ETF rows: every Weinstein study cohort was US
    common stocks (the backtests excluded ETF dirs), but ETFs ride the same
    scan_coil path and render the same chips (2026-07-18 review)."""
    return (" NOTE: this row is an ETF - the cited study cohort was US common "
            "stocks only, so treat the numbers as context, not a measured rate."
            ) if m.get("is_etf") else ""


def _pba_advance(hist_df, entry) -> dict:
    """Pre-breakout advance (Weinstein ch.5 TC signal C): entry vs the lowest
    LOW of the last 126 trading days (inclusive). Backtest 2026-07-17 (214,286
    full-window coil episodes 2016-2025): monster rate (+50% in 12m) by bucket
    8.6% (<15) / 12.5% (15-40) / 26.3% (40-100) / 45.5% (>=100), era-consistent
    and monotone every year — but win% at entry is FLAT and the risk-adjusted
    net premium era-flips, so this is monster-CONTEXT only, never a score or
    gate. Additive display key. Never raises."""
    try:
        if hist_df is None or len(hist_df) < 126 or not entry:
            return {}
        lo = float(hist_df["Low"].iloc[-126:].min())
        if lo != lo or lo <= 0:
            return {}
        return {"pba_pct": round((float(entry) / lo - 1.0) * 100.0, 1)}
    except Exception:  # noqa: BLE001
        return {}


def _pba_line(m: dict) -> str:
    """PBA chip — renders only at >=40% (mobile-first, no noise); amber at
    >=100%. '' otherwise."""
    try:
        pba = m.get("pba_pct")
        if pba is None or pba < 40:
            return ""
        tip = ("Pre-breakout advance: entry vs the lowest low of the last 126 trading "
               "days. Backtest 2026-07-17, 214,286 full-window coil episodes 2016-2025: "
               "+50%-within-12m rate by PBA bucket 8.6% (<15%) / 12.5% (15-40) / 26.3% "
               "(40-100) / 45.5% (>=100) - era-consistent, monotone every year. Two-sided: "
               "the same names also fail big more often (-33% rate 14.8% -> 35.8%) and "
               "win% at entry is flat (~51-54%). The risk-adjusted net premium FLIPS SIGN "
               "between backtest halves, so no edge is claimed. Measured on the coil board. "
               "Monster-context only - no tier, plan or draft impact." + _etf_note(m))
        # Visible text stays two-sided on its own (mobile has no hover, so the
        # tooltip's caveats can't be the only place they live — 2026-07-18
        # review). Full numbers remain in the tooltip for desktop.
        if pba >= 100:
            body = ("<span style='color:#d3a04d;font-weight:600;'>+%.0f%%</span> off 6-mo "
                    "low · big prior advance · more monsters AND more big failures" % pba)
        else:
            body = ("+%.0f%% off 6-mo low · stored energy · more monsters AND more "
                    "big failures" % pba)
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>PBA</span> " + body + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _tc_read(hist_df, entry) -> dict:
    """Weinstein Triple-Confirmation state (ch.5). Conditions at scan time:
    B = Mansfield fresh zero-cross (read from the row's mans_cross_wks by the
    caller), C = entry >= 1.40x the min CLOSE of the last 126 td (the tested
    S6 definition — distinct from the PBA chip's min-LOW basis). A (volume) is
    only knowable at/after the trigger via the validated week-to-date PACE
    rule: week-to-date volume x5/elapsed-days >= 2x trailing-4wk avg AND > any
    single week of the last 26. Returns additive keys tc_pba_ok / tc_vol_pace
    (today's pace ratio vs 4wk avg, informational). Backtest 2026-07-17
    (181,444 triggered episodes): B+C = 40% monster proxy vs 24% board avg;
    full TC (pace variant) = 50.5% monster, +0.62R vs +0.05R avg, ~34/yr.
    Never raises."""
    try:
        if hist_df is None or len(hist_df) < 140 or not entry:
            return {}
        cl = hist_df["Close"].iloc[-126:]
        cmin = float(cl.min())
        if cmin != cmin or cmin <= 0:
            return {}
        pba_ok = bool(float(entry) / cmin >= 1.40)
        out = {"tc_pba_ok": pba_ok}
        # week-to-date volume pace vs the trailing 4 full weeks (Fri-anchored)
        try:
            vol = hist_df["Volume"].resample("W-FRI").sum().dropna()
            days = hist_df["Volume"].resample("W-FRI").count().dropna()
            if len(vol) >= 6:
                elapsed = max(1, int(days.iloc[-1]))
                pace = float(vol.iloc[-1]) * 5.0 / elapsed
                base4 = float(vol.iloc[-5:-1].mean())
                wk26max = float(vol.iloc[-27:-1].max()) if len(vol) >= 27 else float(vol.iloc[:-1].max())
                if base4 > 0:
                    out["tc_vol_pace"] = round(pace / base4, 2)
                    out["tc_vol_over26"] = bool(pace > wk26max)
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception:  # noqa: BLE001
        return {}


def _tc_line(m: dict) -> str:
    """TC chip: 'setup 2/3' when MRS fresh cross + PBA>=40% co-fire (asof-
    knowable), gold 'TC 3/3' when the week-to-date volume pace also clears the
    tested bar. '' otherwise."""
    try:
        wks = m.get("mans_cross_wks")
        if not m.get("tc_pba_ok") or wks is None or wks > 8:
            return ""
        tip = ("Weinstein triple-confirmation (backtest 2026-07-17, 181k triggered coil "
               "episodes, 10y): Mansfield fresh cross + prior advance >=40% = 40% monster "
               "proxy vs 24% board avg; if the breakout week then prints volume >=2x its "
               "4-week average AND above every single week of the last 26 -> full TC: "
               "~50% monster proxy (2.1x), +0.6R vs +0.05R avg, ~34/yr. Median TC trade "
               "still stops out - the edge is the ~24% that run >=2R. Numbers measured on "
               "the COIL board only. Informational only - no tier, plan or draft impact." + _etf_note(m))
        pace = m.get("tc_vol_pace")
        if pace is not None and pace >= 2.0 and m.get("tc_vol_over26"):
            body = ("<span style='color:var(--yellow);font-weight:600;'>TC ✓ 3/3</span> "
                    "vol pace %.1fx 4wk-avg · MRS cross %sw · PBA ok" % (pace, wks))
        else:
            body = ("setup ◆ 2/3 · MRS cross %sw · PBA ok%s"
                    % (wks, (" · vol pace %.1fx" % pace) if pace is not None else ""))
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>TC</span> " + body + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _group_line(m: dict) -> str:
    """GRP chip: the pick's industry group's weekly Weinstein stage + breadth
    (+ sector-wave membership when firing). '' when no group data."""
    try:
        st = m.get("grp_stage")
        ind = m.get("ind_name")
        if not ind or (not st and not m.get("grp_wave_n")):
            return ""
        bits = []
        if st:
            col = ("var(--red)" if str(st).startswith("S4")
                   else "var(--yellow)" if str(st).startswith("S3")
                   else "var(--text-2)")
            bits.append("%s <span style='color:%s;font-weight:600;'>%s</span>"
                        % (esc(ind), col, esc(st)))
            if m.get("grp_above") is not None:
                bits.append("%.0f%%&gt;150d" % m["grp_above"])
        else:
            bits.append(esc(ind))
        if m.get("grp_wave_n"):
            bits.append("<span style='color:var(--yellow);'>wave %d/%s</span>"
                        % (m["grp_wave_n"], m.get("grp_wave_size") or "?"))
        tip = ("Industry-group Weinstein stage (weekly map, build_group_stage.py: stage from "
               "median 150d-SMA slope + % members above their own 150d SMA; ch.3: never buy "
               "into a Stage-3/4 group, the same chart gains 50-75% in a strong group vs "
               "5-10% in a weak one). 'wave' = distinct 5-day breakout winners in the group "
               "(ch.3 group-ignition tally). The 50-75%/5-10% figures are Weinstein's book "
               "numbers, NOT measured here; wave thresholds are cadence-calibrated on 28 "
               "days of live log - no edge claim. Informational only - no tier, plan or "
               "draft impact.")
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>GRP</span> " + " · ".join(bits) + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _mans_line(m: dict) -> str:
    """MRS chip: fresh zero-cross (gold) or below-zero warning; '' for the
    common long-positive case and when uncomputable."""
    try:
        v = m.get("mans_val")
        if v is None:
            return ""
        wks = m.get("mans_cross_wks")
        tip = ("Weinstein/Mansfield RS: weekly stock/S&P-500 ratio vs its own 52-week "
               "average (zero line). Backtest 2026-07-16 (69,512 breakouts, 10y): a fresh "
               "negative-to-positive cross within 8 weeks of breakout catches 71.6% of the "
               "monsters the RS-80 gate misses (median lead 3 weeks) at ~19 non-monsters "
               "per monster - monster-FINDER, not a gate. Latest week may be partial "
               "intraweek. Informational only - no tier, plan or draft impact." + _etf_note(m))
        if wks is not None and wks <= 8:
            when = "this week" if wks == 0 else "%dw ago" % wks
            body = ("<span style='color:var(--yellow);font-weight:600;'>zero-cross ⊕</span> "
                    "%s · MRS %+.1f%%" % (when, v * 100))
        elif v <= 0:
            body = ("<span style='color:var(--text-2);'>below zero (MRS %+.1f%%) - "
                    "52w RS base not reclaimed</span>" % (v * 100))
        else:
            return ""
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>MRS</span> " + body + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _oh_line(m: dict) -> str:
    """Overhead-supply zone chip. '' when uncomputable."""
    try:
        zone = m.get("oh_zone")
        if not zone:
            return ""
        ratio = m.get("oh_ratio") or 0.0
        basis = m.get("oh_basis") or "multi-year"
        ref = m.get("oh_ref")
        tip = ("Overhead supply vs the prior %s high (excluding the last ~3 months). "
               "Backtest 2026-07-16 (5,307 breakouts, 10y): CLEAR zone (0.85-1.09x the old "
               "high) beat heavy-supply breakouts by +6-7pp median 12m return and 8-12pp "
               "fewer -20%%-first failures, era-consistent - BUT heavy-supply names showed "
               "~2x the monster rate, so this is a zone label, never a veto. Informational "
               "only - no tier, plan or draft impact." % basis + _etf_note(m))
        # "BLUE SKY" (not "EXTENDED") — the report's EXTENDED vocabulary means
        # chase-risk distance above the pivot; OH >1.09x means no overhead
        # supply at all, which the backtest shows is merely a THINNER edge than
        # CLEAR, not a chase warning (2026-07-17 review).
        style = {"CLEAR": "var(--green)", "EXT": "var(--yellow)",
                 "SUPPLY": "var(--yellow)", "DEEP": "var(--red)"}[zone]
        note = {"CLEAR": "no meaningful overhead",
                "EXT": "%.0f%% above the old high — thinner edge than CLEAR" % ((ratio - 1) * 100),
                "SUPPLY": "shelf overhead near $%s" % ref,
                "DEEP": "heavy supply up to $%s" % ref}[zone]
        body = ("<span style='color:%s;font-weight:600;'>%s</span> · %.2fx %s high · %s"
                % (style, "BLUE SKY" if zone == "EXT" else zone, ratio, basis, note))
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>OH</span> " + body + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _stage_line(m: dict) -> str:
    """Weekly Weinstein stage chip. '' when uncomputable."""
    try:
        st = m.get("wk_stage")
        if not st:
            return ""
        slope = m.get("wk_ma_slope")
        tip = ("Weinstein weekly stage read (ch.2 quiz algorithm): 30-week MA slope over "
               "~5 weeks (rising > +0.5%, falling < -0.5%), price vs the MA, and MA-crossing "
               "churn (>=3 crossings/10wk = Stage-3 whipsaw). Live spot-check 2026-07-15: "
               "5 A- coil picks had a still-declining 30-week MA. Latest week may be "
               "partial intraweek. Informational only - no tier, plan or draft impact.")
        col = ("var(--red)" if st.startswith("S4")
               else "var(--yellow)" if st.startswith("S3")
               else "var(--text-2)")
        sl_txt = (" · 30wk MA %+.1f%%/5wk" % slope) if slope is not None else ""
        body = ("<span style='color:%s;font-weight:600;'>%s</span>%s"
                % (col, esc(st), sl_txt))
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>STAGE</span> " + body + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _mtf_line(m: dict) -> str:
    """Compact multi-timeframe lesson row: 'TF D 4/4 · W 2/4 · H 3/4'. The
    denominator is ALWAYS 4 (PB auto-fails on weekly: needs >=120 weekly
    bars, more than the 2y fetch holds). Shadow read — filters nothing,
    drafts nothing. '' when no higher-timeframe read exists."""
    try:
        if m.get("mtf_w") is None and m.get("mtf_h") is None:
            return ""
        d = m.get("lesson_confluence") or []
        bits = ["<span style='color:var(--text-2);'>D %d/4</span>" % len(d)]
        tips = ["D: " + (" + ".join(str(x) for x in d) if d else "-")]
        for key, tag in (("mtf_w", "W"), ("mtf_h", "H")):
            v = m.get(key)
            if v is None:
                continue
            # weekly >=3/4 promoted to a gold chip: the 2026-07-07 board backtest's
            # pre-registered SURFACE rule passed era-consistently (daily-miss/weekly-hit
            # cohort beat the daily>=3 baseline both eras; W 4/4 strongest bucket).
            # Display-only — still filters nothing, drafts nothing.
            if key == "mtf_w" and len(v) >= 3:
                bits.append("<span class='lbl lbl-hot'>W %d/4</span>" % len(v))
            else:
                bits.append("<span style='color:var(--text-2);'>%s %d/4</span>" % (tag, len(v)))
            tips.append(tag + ": " + (" + ".join(str(x) for x in v) if v else "-")
                        + (" (PB n/a on weekly)" if key == "mtf_w" else ""))
        tip = ("Lesson read per timeframe: Daily / Weekly (2y resample) / 1-Hour (60d). "
               "Weekly >=3/4 highlighted per the 2026-07-07 backtest (era-consistent). "
               "Informational only - no tier, plan or draft impact. " + " | ".join(tips))
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>TF</span> " + " · ".join(bits) + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _enrich_mtf_lessons(rows: List[dict], diag: "Diagnostics") -> None:
    """SHADOW weekly + 1h lesson read for DISPLAYED names only (<=80 tickers:
    A+ then A then Lesson Radar then A-). Two batch downloads; adds ONLY
    additive keys mtf_w / mtf_h (lists of str, json-safe). Never raises."""
    try:
        seen, subset = set(), []
        for m in rows:
            tk = m.get("ticker")
            if tk and tk not in seen:
                seen.add(tk)
                subset.append(m)
            if len(subset) >= 80:
                break
        if not subset:
            return
        tks = [m["ticker"] for m in subset]
        d2 = fetch_histories_batch(tks, period="2y", min_rows=400)   # weekly needs >=60 wk bars
        h1 = fetch_histories_batch_intraday(tks, period="60d", interval="1h", min_rows=80)
        for m in subset:
            tk, entry = m["ticker"], m.get("entry")
            df2 = d2.get(tk)
            if df2 is not None:
                try:
                    wk = pd.DataFrame({
                        "Open": df2["Open"].resample("W-FRI").first(),
                        "High": df2["High"].resample("W-FRI").max(),
                        "Low": df2["Low"].resample("W-FRI").min(),
                        "Close": df2["Close"].resample("W-FRI").last(),
                        "Volume": df2["Volume"].resample("W-FRI").sum(),
                    }).dropna(subset=["Open", "High", "Low", "Close"])
                    toks = _lessons_on_frame(wk, entry, include_pb=False)  # PB MIN_BARS=120 wk > 2y
                    if toks is not None:
                        m["mtf_w"] = toks
                except Exception:  # noqa: BLE001
                    pass
            toks_h = _lessons_on_frame(h1.get(tk), entry, include_pb=True)
            if toks_h is not None:
                m["mtf_h"] = toks_h
    except Exception as exc:  # noqa: BLE001
        diag.warn(f"MTF lesson enrichment skipped: {exc}")


def _tl_quality(hist_df, entry, direction):
    """Trendline-v2 features (madrry_trendlines). INFORMATIONAL SHADOW MODE
    per WINNER_RADAR_CONTINUATION.md 2.7 — displayed and logged, filters
    nothing, changes no printed plan. Never raises."""
    if _tlv2 is None or hist_df is None or entry is None:
        return {}
    try:
        out = _tlv2.analyze_lines(hist_df, entry, direction)
        return out if isinstance(out, dict) and "tl_error" not in out else {}
    except Exception:  # noqa: BLE001
        return {}


def _ch_quality(hist_df, entry, direction):
    """Parallel-channel features (madrry_channels, tutorial #4). INFORMATIONAL
    SHADOW MODE per WINNER_RADAR_CONTINUATION.md 2.7 — displayed and logged,
    filters nothing, changes no printed plan, and does NOT count toward the
    Stage-4 support gate. Never raises."""
    if _chv is None or hist_df is None or entry is None:
        return {}
    try:
        out = _chv.analyze_channels(hist_df, entry, direction)
        return out if isinstance(out, dict) and "ch_error" not in out else {}
    except Exception:  # noqa: BLE001
        return {}


def _chart_plan(m: dict, hist_df, *, direction: str = "long") -> dict:
    """REV 10 (USER 2026-07-18: "you didnt draw lines in 52 week pullback and all
    others… draw at all charts. thats the whole point").

    Returns a COPY of the row enriched with the S/R-zone + salient-trendline
    draw keys, purely so make_candle_chart()/_candle_overlays() can put the same
    lesson levels on THIS section's chart that coil cards already get. Only the
    engine's own reads are used — sr_draw_lo/hi and tl_draw_sup/res_now+slope_d
    from _sr_quality/_tl_quality/_ch_quality — never an invented line.

    Deliberately NON-MUTATING: the caller's row (and therefore the ledger
    snapshot written to latest_setups_*.json) is untouched, so this is
    display-only and cannot perturb tracking or the M.E.T.A. recalibration.
    Keys a section already computed win. Never raises — a chart is decoration."""
    plan = dict(m) if isinstance(m, dict) else {}
    if hist_df is None:
        return plan
    if plan.get("sr_draw_lo") is not None and plan.get("tl_draw_sup_now") is not None:
        return plan                                  # section already has a read
    entry = next((plan.get(k) for k in
                  ("entry", "pivot", "ideal_buy", "last_close", "close")
                  if plan.get(k) is not None), None)
    if entry is None:
        return plan
    try:
        for _fn in (_sr_quality, _tl_quality, _ch_quality):
            for k, v in (_fn(hist_df, entry, direction) or {}).items():
                plan.setdefault(k, v)
    except Exception:  # noqa: BLE001 — chart decoration must never kill a scan
        pass
    return plan


def _atr_stop(entry, adr, direction):
    """1.5×ADR stop, direction-aware. Long: below entry; short: above. None if unusable."""
    if entry is None or adr is None:
        return None
    try:
        entry = float(entry)
        adr = float(adr)
    except (TypeError, ValueError):
        return None
    if adr <= 0 or entry <= 0:
        return None
    if direction == "short":
        return round(entry * (1 + _ATR_STOP_MULT * adr / 100.0), 2)
    return round(entry * (1 - _ATR_STOP_MULT * adr / 100.0), 2)


def _valid_until(data_date, n=_VALIDITY_TRADING_DAYS):
    """The date `n` TRADING days after `data_date` (weekend + US-holiday aware). Returns None
    if the calendar can't answer (out-of-range year) so the report simply omits the field —
    never guesses a validity that could count a holiday."""
    try:
        import us_market_calendar as _cal
        d = data_date
        for _ in range(int(n)):
            d = _cal.next_trading_day(d).isoformat()
        return d
    except Exception:
        return None


def _geo_line(m):
    """Secondary info line for an atr_5day coil card: the structural level as "support" (real
    market structure, no longer the printed stop) + the 5-day validity. Empty string for
    tight_3day / pre-switch picks so their cards render byte-identically. All-`.get()` (no KeyError)."""
    if m.get("stop_version") not in ("atr_5day", "lesson_v1"):
        return ""
    bits = []
    if m.get("stop_tight") is not None:
        bits.append(f"support: {esc(m.get('stop_structural_reason') or 'struct')} ${m['stop_tight']}")
    if m.get("valid_until"):
        bits.append(f"valid until {esc(m['valid_until'])}")
    if not bits:
        return ""
    return "<br><span class='stop-reason' style='color:var(--text-3);'>" + " · ".join(bits) + "</span>"


def _apply_stop_regime(picks, data_date):
    """Switch coil picks to the ratified ATR+5d geometry for sessions >= the switch session.
    Idempotent (skips a pick already carrying stop_version) and mutates dicts IN PLACE so the
    HTML cards, the IBKR order plan, and the ledger snapshot all see the same switched stop.
    Pre-switch it only stamps stop_version='tight_3day' + the stable stop_tight/stop_atr fields,
    changing nothing the report prints."""
    switched = bool(data_date) and str(data_date) >= STOP_REGIME_SWITCH_SESSION
    vu = _valid_until(data_date) if switched else None
    for p in picks:
        if not isinstance(p, dict) or p.get("stop_version"):
            continue
        direction = p.get("direction")
        # A validated LESSON plan takes precedence over the ratified ATR geometry
        # (2026-07-06 USER: the plan follows the 4 tutorial lessons — _lesson_plan
        # already replaced entry/stop in scan_coil, and the chart drew THAT plan).
        # Keep the ledger's structural fields from the pre-refinement originals and
        # stamp an honest stop_version so the atr_5day calibration cohort stays pure.
        if p.get("plan_src"):
            p["stop_tight"] = p.get("stop_raw", p.get("stop"))
            p["stop_structural_reason"] = p.get("stop_reason_raw") or p.get("stop_reason")
            a_ls = _atr_stop(p.get("entry"), p.get("adr"), direction)
            if a_ls is not None:
                p["stop_atr"] = a_ls
            p["stop_version"] = "lesson_v1"
            continue
        # always preserve the structural stop + expose the ATR stop (stable ledger fields)
        p["stop_tight"] = p.get("stop")
        p["stop_structural_reason"] = p.get("stop_reason")
        a_stop = _atr_stop(p.get("entry"), p.get("adr"), direction)
        if a_stop is not None:
            p["stop_atr"] = a_stop
        if switched and a_stop is not None:
            entry = float(p["entry"])
            p["stop"] = a_stop
            p["risk_pct"] = (round((a_stop - entry) / entry * 100, 1) if direction == "short"
                             else round((entry - a_stop) / entry * 100, 1))
            p["stop_reason"] = "1.5×ADR"
            p["valid_until"] = vu
            p["stop_version"] = "atr_5day"
        else:
            p["stop_version"] = "tight_3day"
    return picks


# ----------------------------------------------------------------------------
# SCANNERS
# ----------------------------------------------------------------------------
# Stage-2 RS gate floor (USER 2026-07-15): a name qualifies only if the STOCK's
# RS percentile is 80+ OR its 144-industry-group RS percentile is 80+. NOTE:
# the 2026-06 backtest ruled stock-only RS80+ a NO-GO hard filter (~25% monster
# loss) — the industry-group OR-branch is the user's explicit revision; watch
# the tier hit-rate and monster-miss rate as live data accumulates.
RS_GATE_MIN = 80
# A carried-forward stock RS older than this many calendar days can't pass the
# GATE (display keeps rendering it with its asof). Without a cap, a dead source
# would keep gating the whole universe on arbitrarily old percentiles forever
# (adversarial review 2026-07-15).
RS_GATE_MAX_STALE_DAYS = 7


def _rs_carry_fresh(asof: Optional[str]) -> bool:
    """resolve_rs stamps carried-forward values with their asof date; fresh
    (today's) values come back with asof=None. The gate accepts a carry only
    within RS_GATE_MAX_STALE_DAYS; an unparseable stamp counts as stale."""
    if not asof:
        return True
    try:
        return (date.today() - date.fromisoformat(str(asof))).days <= RS_GATE_MAX_STALE_DAYS
    except Exception:  # noqa: BLE001
        return False


def _date_indexed(s: pd.Series) -> pd.Series:
    """Re-key a daily series on tz-naive normalized dates. yfinance's BATCH feed
    returns tz-naive indexes while single-ticker history is tz-aware
    (America/New_York) — reindexing one onto the other yields all-NaN, which
    would make the RS-line gate silently stand down for every name (caught by
    the 2026-07-15 probe: drop_rsline=0 across 421 candidates)."""
    idx = s.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    out = pd.Series(s.to_numpy(), index=idx.normalize())
    return out[~out.index.duplicated(keep="last")]


def _zigzag_swing_highs(vals: np.ndarray, tol: float) -> Tuple[List[int], Optional[float]]:
    """Swing highs via a zigzag filter: a high only counts once the series has
    retraced >= `tol` (relative) below it. Micro-pivots inside a tight coil
    never confirm — the naive N-bar local-max pivot whipsawed exactly there
    (adversarial review 2026-07-15: ~85% false rejection of coiled leaders).
    One pivot per leg, so exact-tie flat tops can't double-count either.
    Returns (confirmed swing-high indices, current rising-leg max or None)."""
    highs: List[int] = []
    trend = 1                      # treat the window start as a rising leg
    ext, ext_i = float(vals[0]), 0
    for i in range(1, len(vals)):
        v = float(vals[i])
        if trend == 1:
            if v > ext:
                ext, ext_i = v, i
            elif v <= ext * (1.0 - tol):
                highs.append(ext_i)
                trend, ext, ext_i = -1, v, i
        else:
            if v < ext:
                ext, ext_i = v, i
            elif v >= ext * (1.0 + tol):
                trend, ext, ext_i = 1, v, i
    return highs, (ext if trend == 1 else None)


def _rs_line_swings_up(hist_df, bench_close, lookback: int = 126,
                       tol: float = 0.03) -> Tuple[bool, bool]:
    """Stage-2 RS-line slope gate (USER 2026-07-15): the RS line (close/^GSPC)
    must be sloping UP swing-over-swing — each recent CONFIRMED swing high
    (zigzag, >= `tol` retracement) above the previous one, judged on the last
    (up to) 3 swings within `lookback` sessions. A current rising leg that has
    already cleared the last confirmed swing counts as the newest swing.
    Guards (adversarial review 2026-07-15):
      · collapse guard — ascending old swings can't mask a fresh RS collapse:
        the line must still hold within 15% of its latest swing high;
      · <2 swings (no retracement = straight-line RS) passes if the line is
        above its 21-day mean OR within 5% of its window high, so a leader in
        a quiet coil is never coin-flipped out by day noise.
    Returns (ok, evaluated) — evaluated=False means the gate STOOD DOWN
    (missing/short/unalignable data) and ok is forced True, so a feed outage
    can never empty the report; the caller counts stand-downs and warns loudly
    if nothing was actually evaluated."""
    if hist_df is None or bench_close is None or len(hist_df) < 40:
        return True, False
    try:
        close = _date_indexed(hist_df["Close"].astype(float))
        bench = _date_indexed(bench_close.astype(float)).replace(0, np.nan)
        rs = (close / bench.reindex(close.index).ffill()).dropna()
        if len(rs) < 40:
            return True, False
        vals = rs.iloc[-lookback:].to_numpy()
        highs, cur_leg_hi = _zigzag_swing_highs(vals, tol)
        hi_vals = [float(vals[i]) for i in highs]
        if cur_leg_hi is not None and (not hi_vals or cur_leg_hi > hi_vals[-1]):
            hi_vals.append(cur_leg_hi)
        last = hi_vals[-3:]
        if len(last) >= 2:
            ascending = all(b > a for a, b in zip(last, last[1:]))
            return bool(ascending and vals[-1] >= last[-1] * 0.85), True
        # <2 confirmed swings: either a straight-line RS or a downtrend whose
        # rallies never reach tol. Pass only when the line is pinned near its
        # window high, or is above its 21-day mean AND net-up over the window
        # (the bare 21d-mean test alone let a grinding downtrend sneak through
        # on a terminal up-wiggle — caught by synthetic check #2).
        return bool(vals[-1] >= 0.95 * vals.max()
                    or (vals[-1] > vals[-21:].mean() and vals[-1] > vals[0])), True
    except Exception:  # noqa: BLE001
        return True, False


# ---- Stage-4 short leg helpers (Weinstein ch.7, 2026-07-17) -----------------
# Per §2.6 NO backtest verdict exists or is claimed for short outcomes
# (survivorship: the local dump lacks the delisted blowups shorts profit on);
# every outcome-quality question is WAIT-FOR-DATA via the live ledger.

def _rs_line_swings_down(hist_df, bench_close, lookback: int = 126,
                         tol: float = 0.03) -> Tuple[bool, bool]:
    """Mirror of _rs_line_swings_up for the SHORT leg: the RS line must slope
    DOWN swing-over-swing — each recent CONFIRMED swing LOW below the previous
    (swing lows of rs == swing highs of -rs, so _zigzag_swing_highs is reused
    verbatim on the negated series). Collapse-guard mirror: a fresh RS rip
    >15% above the latest swing low disqualifies. Returns (ok, evaluated) —
    CRITICAL INVERSION vs the long gate: the CALLER must treat evaluated=False
    as NOT passed. A short REQUIREMENT fails CLOSED — missing data must never
    qualify a short; the independent rs_pct<=25 OR-branch keeps the leg
    outage-resilient."""
    if hist_df is None or bench_close is None or len(hist_df) < 40:
        return False, False
    try:
        close = _date_indexed(hist_df["Close"].astype(float))
        bench = _date_indexed(bench_close.astype(float)).replace(0, np.nan)
        rs = (close / bench.reindex(close.index).ffill()).dropna()
        if len(rs) < 40:
            return False, False
        vals = rs.iloc[-lookback:].to_numpy()
        lows_idx, cur_leg_neg = _zigzag_swing_highs(-vals, tol)
        lo_vals = [float(vals[i]) for i in lows_idx]
        cur_leg_lo = (-cur_leg_neg) if cur_leg_neg is not None else None
        if cur_leg_lo is not None and (not lo_vals or cur_leg_lo < lo_vals[-1]):
            lo_vals.append(cur_leg_lo)
        last = lo_vals[-3:]
        if len(last) >= 2:
            descending = all(b < a for a, b in zip(last, last[1:]))
            return bool(descending and vals[-1] <= last[-1] * 1.15), True
        # <2 confirmed swing lows: pass only when the line is pinned near its
        # window low, or below its 21-day mean AND net-down over the window.
        return bool(vals[-1] <= 1.05 * vals.min()
                    or (vals[-1] < vals[-21:].mean() and vals[-1] < vals[0])), True
    except Exception:  # noqa: BLE001
        return False, False


def _round_buy_stop(raw: float) -> float:
    """Short protective buy-stop placed just ABOVE the next round number
    (ch.7: covering orders cluster at round figures; below $20 every half-
    point counts). Grid $1.00/$0.10 nudge >= $20, $0.50/$0.05 below."""
    grid, nudge = (1.0, 0.10) if raw >= 20 else (0.5, 0.05)
    return round(math.ceil(raw / grid - 1e-9) * grid + nudge, 2)


def _swing_rule_target(hist_df, support_low: float) -> Optional[dict]:
    """Downside swing-rule target (ch.7 measured move): top-area window =
    weekly bars since the last W-FRI close above the 30-week MA (capped 52w;
    26w when never above); target = support_low - (peak - support_low)."""
    try:
        wk_c = hist_df["Close"].resample("W-FRI").last().dropna()
        wk_h = hist_df["High"].resample("W-FRI").max().dropna()
        ma = wk_c.rolling(30).mean()
        above = (wk_c > ma).fillna(False)
        idx = None
        for i in range(len(above) - 1, -1, -1):
            if bool(above.iloc[i]):
                idx = i
                break
        n_since = (len(above) - 1 - idx) if idx is not None else 26
        window = max(4, min(n_since, 52))
        peak = float(wk_h.iloc[-window:].max())
        target = support_low - (peak - support_low)
        out = {"peak": round(peak, 2)}
        if target > 0.05 * support_low:
            out["target_swing"] = round(target, 2)
        else:
            out["target_swing"] = None       # measured move exceeds price
        return out
    except Exception:  # noqa: BLE001
        return None


_RS_STOCKS_CACHE_CSV = os.path.join(WORKSPACE, "memory", "rs_stocks_cache.csv")


def _dtc_lookup(tickers: List[str], diag=None, budget_s: float = 25.0) -> Dict[str, dict]:
    """Days-to-cover for the (bounded, <=15) final short survivors. Primary =
    yfinance Ticker.info shortRatio (FINRA bi-monthly, worst-case ~3.5wk
    stale); fallback = the already-cached Fred CSV: ShortFloatPct (fraction)
    x Float / AvgVol10. Missing data flags, never excludes."""
    csv_map: Dict[str, dict] = {}
    try:
        with open(_RS_STOCKS_CACHE_CSV) as fh:
            for row in csv.DictReader(fh):
                tk = (row.get("Ticker") or "").strip().upper()
                if tk:
                    csv_map[tk] = row
    except Exception:  # noqa: BLE001
        csv_map = {}
    out: Dict[str, dict] = {}
    t0 = time.time()
    for t in tickers:
        rec = {"dtc": None, "dtc_src": None, "short_pct_float": None, "dtc_asof": None}
        try:
            # Wall-clock budget (2026-07-18 review): yf .info is a serial network
            # call with no internal deadline — a hung endpoint would stall the
            # whole run. Past the budget we skip the slow leg and fall through to
            # the free CSV, so every row still gets stamped.
            if time.time() - t0 > budget_s:
                raise TimeoutError("dtc budget exhausted")
            info = yf.Ticker(t).info or {}
            sr = info.get("shortRatio")
            if sr:
                rec["dtc"] = round(float(sr), 2)
                rec["dtc_src"] = "yf"
                spf = info.get("shortPercentOfFloat")
                rec["short_pct_float"] = round(float(spf) * 100, 1) if spf else None
                dsi = info.get("dateShortInterest")
                if dsi:
                    try:
                        rec["dtc_asof"] = datetime.fromtimestamp(int(dsi)).strftime("%Y-%m-%d")
                    except Exception:  # noqa: BLE001
                        pass
        except Exception:  # noqa: BLE001
            pass
        if rec["dtc"] is None:
            row = csv_map.get(t.upper().replace(".", "-")) or csv_map.get(t.upper())
            if row:
                try:
                    spf = float(row.get("ShortFloatPct") or 0)
                    flt = float(row.get("Float") or 0)
                    av = float(row.get("AvgVol10") or 0)
                    if spf > 0 and flt > 0 and av > 0:
                        rec["dtc"] = round(spf * flt / av, 2)
                        rec["dtc_src"] = "csv"
                        rec["short_pct_float"] = round(spf * 100, 1)
                except Exception:  # noqa: BLE001
                    pass
        out[t] = rec
    return out


def _dtc_line(m: dict) -> str:
    """DTC chip for short cards: red >=5x (squeeze fuel), plain otherwise,
    'n/a · flag' when uncomputable."""
    try:
        # Rows that never went through _dtc_lookup (parabolic-short leg) carry no
        # 'dtc' key at all — stay silent there rather than printing a misleading
        # "n/a" on every card. Stage-4 rows always get the key (possibly None),
        # so a genuine lookup failure still flags (2026-07-18).
        if "dtc" not in m:
            return ""
        dtc = m.get("dtc")
        tip = ("Days-to-cover = short interest / avg daily volume (FINRA bi-monthly "
               "settlement, published ~T+7bd — worst-case ~3.5 weeks stale%s). Weinstein "
               "ch.7: 3-4x is normal for heavily-shorted names, >=5x is squeeze fuel "
               "(Bowmar ~10x squeezed 20->45; HSN 31x squeezed 18->282), >=10x excluded. "
               "Informational chip; the 10x exclusion is the only gate."
               % (f"; asof {m['dtc_asof']}" if m.get("dtc_asof") else ""))
        if dtc is None:
            body = "<span style='color:var(--text-2);'>n/a · flag (both sources failed)</span>"
        elif dtc >= 5:
            spf = m.get("short_pct_float")
            body = ("<span style='color:var(--red);font-weight:600;'>%.1fx — squeeze risk</span>%s"
                    % (dtc, f" · {spf}% of float short" if spf else ""))
        else:
            body = "%.1fx (%s)" % (dtc, m.get("dtc_src") or "?")
        return ("<div class='edge-line' title='" + esc(tip) + "'>"
                "<span class='lbl'>DTC</span> " + body + "</div>")
    except Exception:  # noqa: BLE001
        return ""


# ---- ETF RS percentile (Stage-2 ETF leg, 2026-07-15) ------------------------
_ETF_RS_SCORES: Optional[List[float]] = None    # sorted stock raw RS scores (lazy)


def _stock_rs_score_dist() -> Optional[List[float]]:
    """Sorted raw 'Relative Strength' scores of the full stock cross-section,
    read from the RS cache CSV (fetch_and_load_rs_scores keeps only Percentile
    in memory but always writes the full raw CSV to RS_CACHE_PATH). Used to
    percentile-rank locally computed ETF scores. None ⇒ unavailable (the ETF
    RS gate then stands down LOUDLY, mirroring the stock-side outage rule)."""
    global _ETF_RS_SCORES
    if _ETF_RS_SCORES is not None:
        return _ETF_RS_SCORES
    try:
        scores: List[float] = []
        with open(RS_CACHE_PATH, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    scores.append(float(row["Relative Strength"]))
                except (KeyError, TypeError, ValueError):
                    continue
        if len(scores) >= 500:          # sanity: a real cross-section, not a stub
            scores.sort()
            _ETF_RS_SCORES = scores
    except Exception:  # noqa: BLE001
        _ETF_RS_SCORES = None
    return _ETF_RS_SCORES


def _ibd_strength(closes) -> Optional[float]:
    """Fred6725/skyte weighted strength: 0.4·q1 + 0.2·q2 + 0.2·q3 + 0.2·q4,
    where qN = cumulative return over the last N·63 bars (overlapping windows,
    exactly `closes.tail(N*63)` last/first − 1 — verified 0.005 median error
    against the source CSV on 2026-07-15). None ⇒ not computable."""
    try:
        c = closes.dropna().astype(float)
        if len(c) < 64:                 # need at least one full quarter
            return None
        qs = []
        for n in (1, 2, 3, 4):
            w = c.tail(min(len(c), n * 63))
            qs.append(float(w.iloc[-1] / w.iloc[0]) - 1.0)
        return 0.4 * qs[0] + 0.2 * qs[1] + 0.2 * qs[2] + 0.2 * qs[3]
    except Exception:  # noqa: BLE001
        return None


def _etf_rs_percentile(hist_df, bench_strength: Optional[float]) -> Optional[int]:
    """IBD-style RS percentile for an ETF, comparable to the Fred6725 stock
    percentiles: score = 100·(1+strength)/(1+strength_bench), ranked into the
    stock cross-section's raw-score distribution. None ⇒ not computable for
    THIS name (missing/short history) — the caller treats that as a hard-gate
    fail, same as a stock absent from a live RS feed."""
    dist = _stock_rs_score_dist()
    if hist_df is None or bench_strength is None or not dist:
        return None
    s = _ibd_strength(hist_df["Close"])
    if s is None or (1.0 + bench_strength) == 0:
        return None
    score = 100.0 * (1.0 + s) / (1.0 + bench_strength)
    pos = bisect.bisect_left(dist, score)
    return max(0, min(99, int(pos * 100 / len(dist))))


def scan_coil(rs_map: dict, market_modifier: float, diag: Diagnostics,
              industry_by_ticker: Optional[Dict[str, dict]] = None):
    """Two-pass coil scan: cheap server filter, then batched yfinance enrichment.
    Stage-2 gates (2026-07-15): near the 52w high + stock/industry RS 80+ +
    RS line rising swing-over-swing (the 9/21-EMA proximity UNIVERSE gate was
    removed; the ≤1%/≤2% EMA-hug conditions remain inside the A+/A/A- tiers)."""
    payload_coil = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": "close", "operation": "egreater", "right": 10},
            {"left": "volume", "operation": "egreater", "right": 500000},
            {"left": "average_volume_30d_calc", "operation": "egreater", "right": 500000},
            {"left": "average_volume_60d_calc", "operation": "egreater", "right": 500000},
            {"left": "close", "operation": "egreater", "right": "SMA200"},
            {"left": "market_cap_basic", "operation": "egreater", "right": 2000000000},
            # ADR floor (USER 2026-07-06): drop dead / illiquid names that barely
            # move (e.g. RAMP ~0.5% ADR). Server-side ADRP gate — TradingView
            # accepts it (the 52wk-high scan uses the same field). Re-introduces a
            # floor at 1.5% (the earlier 2.0% floor was removed 2026-06-30).
            {"left": "ADRP", "operation": "egreater", "right": 1.5},
        ],
        # NOTE on the 52-week band (within 0-20% of the 52w high): TradingView's
        # /scan API rejects arithmetic on the RHS (price_52_week_high * 0.8 ->
        # HTTP 400) and exposes no precomputed % field, so the gate can't live in
        # the server filter; it is enforced below in the parse loop. (An earlier
        # version of this note also claimed a ">=50% above the 52w low" gate —
        # AUDIT 2026-07-04: no such gate exists anywhere in the loop and the
        # report never advertised one; price_52_week_low is fetched but unused.)
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

    # ETF leg (USER 2026-07-15): same technical filters, fund plumbing. `aum`
    # takes market_cap_basic's SLOT so the positional 23-tuple unpack below is
    # unchanged (mcap var simply carries AUM for fund rows); `description` is
    # appended as column 24 so the row can show the fund's real name as its
    # theme. All 23 required columns verified populated for type=fund (probe
    # 2026-07-15: 73/73 rows, zero nulls).
    _etf_cols = list(payload_coil["columns"])
    _etf_cols[_etf_cols.index("market_cap_basic")] = "aum"
    payload_etf = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["fund"]},
            {"left": "typespecs", "operation": "has", "right": ["etf"]},
            {"left": "close", "operation": "egreater", "right": 10},
            {"left": "volume", "operation": "egreater", "right": 500000},
            {"left": "average_volume_30d_calc", "operation": "egreater", "right": 500000},
            {"left": "average_volume_60d_calc", "operation": "egreater", "right": 500000},
            {"left": "close", "operation": "egreater", "right": "SMA200"},
            {"left": "aum", "operation": "egreater", "right": ETF_AUM_MIN},
            {"left": "ADRP", "operation": "egreater", "right": 1.5},
        ],
        "columns": _etf_cols + ["description"],
        "sort": {"sortBy": "ADRP", "sortOrder": "desc"},
        "range": [0, 1000],
    }

    tier_a_plus: List[dict] = []
    tier_a: List[dict] = []
    tier_a_minus: List[dict] = []
    tier_a_minus_full: List[dict] = []   # uncapped A- pool (for win-rate tracking)
    coil_candidates: List[dict] = []
    lesson_radar: List[dict] = []   # >=3/4 lessons but NO tier — display-only radar

    # --- funnel instrumentation: exact survivor count after each filter stage ---
    n_universe = None          # TradingView totalCount matching the Stage-1 filter
    n_stage1 = 0               # rows actually fetched (range-capped at 5000)
    drop_missing = drop_proximity = drop_52w = drop_dead = 0
    # drop_proximity stays declared (always 0 since the 2026-07-15 gate removal)
    # so the funnel dict keeps its shape — additive-only rule.
    drop_rs = 0                # failed the stock/industry RS 80+ gate
    drop_rsline = 0            # RS line not rising swing-over-swing
    drop_unsupported = 0       # met a tier but no verified entry engine backs it
    # ETF-leg instrumentation (2026-07-15, additive-only keys in the funnel)
    n_universe_etf = None      # TradingView fund count matching the ETF Stage-1 filter
    n_stage1_etf = 0           # ETF rows actually fetched
    drop_rs_etf = 0            # ETF subset of drop_rs (locally computed RS < 80 / uncomputable)
    n_aplus_raw = n_a_raw = n_aminus_raw = n_aminus_total = 0

    industry_by_ticker = industry_by_ticker or {}
    stock_rs_alive = bool(rs_map or _RS_HISTORY)
    rs_gate_active = bool(stock_rs_alive or industry_by_ticker)
    if not rs_gate_active:
        diag.warn("Stage-2 RS 80+ gate STOOD DOWN — stock RS map, RS history "
                  "and industry RS are ALL empty (source outage); universe not gated on RS")
    elif not industry_by_ticker:
        # Partial outages must be LOUD (adversarial review 2026-07-15): losing
        # the industry branch silently reshapes the gate into stock-only RS80+,
        # the exact configuration the 2026-06 backtest ruled NO-GO.
        diag.warn("Stage-2 RS gate DEGRADED to stock-only RS80+ — industry RS "
                  "map is empty (feed/parse failure); this is the "
                  "backtest-rejected NO-GO shape, investigate the industry feed")
    elif not stock_rs_alive:
        diag.warn("Stage-2 RS gate DEGRADED to industry-group-only — stock RS "
                  "map AND history are empty (source outage)")

    try:
        time.sleep(2)
        data = tv_post(payload_coil, label="coil", diag=diag)
        _rows = data.get("data", []) or []
        n_stage1 = len(_rows)
        _univ = data.get("totalCount")
        n_universe = _univ if isinstance(_univ, int) else None
        # ETF leg — its OWN try: a fund-scan outage must never take down the
        # stock scan (the report just runs stock-only that day, warned loudly).
        # diag=None on purpose: _request_json would log a diag.ERROR before
        # raising, which trips the morning script's errors=0 publish gate — a
        # transient fund-feed outage must degrade to a WARNING, not an alert.
        _etf_rows: List[dict] = []
        try:
            time.sleep(2)
            _etf_data = tv_post(payload_etf, label="coil_etf", diag=None)
            _etf_rows = _etf_data.get("data", []) or []
            _etf_univ = _etf_data.get("totalCount")
            n_universe_etf = _etf_univ if isinstance(_etf_univ, int) else None
        except Exception as exc:  # noqa: BLE001
            diag.warn(f"ETF coil scan failed — today's coil universe is stock-only ({exc})")
        n_stage1_etf = len(_etf_rows)
        for _src_etf, row in ([(False, r) for r in _rows]
                              + [(True, r) for r in _etf_rows]):
            d = row.get("d")
            if not d or len(d) < (24 if _src_etf else 23):
                drop_missing += 1
                continue
            etf_desc = None
            if _src_etf:
                etf_desc = str(d[23] or "").strip()
                d = d[:23]              # aum sits in the mcap slot — same unpack
            (ticker, close, opn, vol, avg_vol, ema9, ema21, sma50, sma200, adr,
             mcap, perf_1m, perf_3m, perf_6m, perf_y, sector, industry,
             high, low, change, high_52w, float_shares, low_52w) = d
            if _src_etf:
                # funds: TradingView's sector/industry are boilerplate
                # ("Miscellaneous" / "Investment Trusts/Mutual Funds") — the
                # sector chip becomes "ETF" and the fund NAME becomes the theme.
                sector, industry = "ETF", None
                _ETF_TICKERS.add(ticker)   # render cells skip fundamentals for these

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
            # (ADR floor: server-side ADRP >= 1.5% in payload_coil per USER 2026-07-06 —
            # keeps out dead/illiquid names like RAMP; `adr` is TradingView's native ADRP
            # and is still used for tier discrimination below.)

            # 52-week position band (couldn't go in the server filter — see payload
            # note). Keep names within 0-20% BELOW their 52-week high.
            pct_below_high = ((high_52w - close) / high_52w * 100) if high_52w else None
            if pct_below_high is None or not (0.0 <= pct_below_high <= 20.0):
                drop_52w += 1
                continue

            # Stage-2 RS gate (USER 2026-07-15): only RS leaders qualify —
            # stock RS percentile 80+ OR the name's industry-group RS 80+.
            # resolve_rs carries forward the last known value for names that
            # flicker out of the source (ADRs / spinoffs), so a one-day source
            # gap doesn't drop a leader.
            # ETF rows are NOT exempt — their RS check needs price history the
            # cheap TV feed doesn't carry, so it is DEFERRED to the enrichment
            # pass below (same 80+ bar, computed via _etf_rs_percentile).
            if rs_gate_active and not _src_etf:
                _rs_v, _rs_asof = resolve_rs(ticker, rs_map)
                if not isinstance(_rs_v, int):
                    # class shares: TradingView prints BRK.B, the RS sources
                    # may key the dash form — try it before giving up
                    _rs_v, _rs_asof = resolve_rs((ticker or "").replace(".", "-"), rs_map)
                _ind_rec = industry_by_ticker.get((ticker or "").upper().replace(".", "-"))
                _ind_pct = _ind_rec.get("pct") if _ind_rec else None
                _stock_ok = (isinstance(_rs_v, int) and _rs_v >= RS_GATE_MIN
                             and _rs_carry_fresh(_rs_asof))
                _ind_ok = isinstance(_ind_pct, int) and _ind_pct >= RS_GATE_MIN
                if not (_stock_ok or _ind_ok):
                    drop_rs += 1
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
            theme = (etf_desc[:ETF_THEME_MAX] if (_src_etf and etf_desc)
                     else get_theme(ticker, industry))
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
            # (9/21-EMA proximity UNIVERSE gate (min_dist <= 0.10) REMOVED
            # 2026-07-15 per USER: "remove stage 2 'close to the 9/21 EMA'" —
            # near-highs + the RS gates now define Stage 2. min_dist still
            # grades the A+/A/A- tiers below (user kept those conditions) and
            # feeds display + entry-plan context.)

            status_labels = []
            if is_squat:
                status_labels.append("⚠️ Squat (Wait for tightening)")
            if risk_pct > 6.0:
                status_labels.append("⚠️ Wide Stop (Size Down!)")
            if is_premium_cluster:
                status_labels.append("🛡️ MA Cluster (Premium)")
            if _src_etf:
                # cap column shows AUM for funds — the badge makes that explicit
                status_labels.append(f"🧺 ETF · AUM ${round((mcap or 0) / 1e9)}B")

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
                "is_etf": _src_etf, "etf_desc": etf_desc,
            })
    except Exception as exc:  # noqa: BLE001
        diag.error(f"Coil scan (fetch/parse): {exc}")

    if coil_candidates:
        hist_map = fetch_histories_batch([c["ticker"] for c in coil_candidates], period="1y")
        # Benchmark for the RS-line slope gate. Fetched SEPARATELY on purpose:
        # the batch path TV-patches stale daily bars and TradingView's scan API
        # has no caret indices, so ^GSPC must not go through it.
        bench_close = None
        try:
            _bench_df = fetch_stock_history(ANTS_BENCHMARK, period="1y")
            bench_close = _bench_df["Close"] if _bench_df is not None else None
        except Exception:  # noqa: BLE001
            bench_close = None
        if bench_close is None:
            diag.warn(f"Stage-2 RS-line swing gate STOOD DOWN — {ANTS_BENCHMARK} "
                      "history unavailable; universe not gated on RS-line slope")
        # ETF RS-gate prerequisites (2026-07-15): benchmark strength (SPY — the
        # source CSV's verified reference; ^GSPC fallback carries a ~0.7-point
        # score bias, acceptable) + the stock raw-score distribution. If either
        # is unavailable the ETF leg of the RS gate STANDS DOWN — LOUDLY, per
        # the house rule that silent gate reshaping is never allowed.
        etf_gate_ready = False
        spy_strength = None
        if rs_gate_active and any(c.get("is_etf") for c in coil_candidates):
            try:
                _spy_df = fetch_stock_history("SPY", period="1y")
                spy_strength = (_ibd_strength(_spy_df["Close"])
                                if _spy_df is not None else None)
            except Exception:  # noqa: BLE001
                spy_strength = None
            if spy_strength is None and bench_close is not None:
                spy_strength = _ibd_strength(bench_close)
                if spy_strength is not None:
                    # the fallback silently shifts scores ~0.7pt (^GSPC is a
                    # price index) — enough to flip a borderline ETF across the
                    # hard 80 bar, so it must be LOUD like every other
                    # degradation path.
                    diag.warn("ETF RS gate benchmark DEGRADED to ^GSPC — SPY "
                              "history unavailable; ETF scores carry a ~0.7-pt "
                              "upward bias vs the verified SPY reference")
            etf_gate_ready = bool(spy_strength is not None and _stock_rs_score_dist())
            if not etf_gate_ready:
                diag.warn("Stage-2 RS gate STOOD DOWN for ETFs — benchmark "
                          "strength or stock RS-score distribution unavailable; "
                          "ETF rows not RS-gated this run")
        rsline_checked = 0     # names the RS-line gate actually EVALUATED
        for c in coil_candidates:
            hist_df = hist_map.get(c["ticker"])
            # Auto dead-stock screen: drop M&A-pinned / halted / pending-delist names
            # (recent daily range flat-lined). Catches deal pins the ADR floor can't.
            if is_dead_pinned(hist_df):
                drop_dead += 1
                continue
            # Stage-2 RS gate, ETF leg (deferred from the parse loop — needs
            # history): same 80+ bar as stocks, percentile computed locally.
            # HARD gate: uncomputable (young ETF, <200-bar history ⇒ hist_df
            # None) counts as a fail, exactly like a stock absent from a live
            # RS feed. Young funds also lack the history every entry engine
            # needs, so this doubles as the "scanned but can never tier" fix.
            if c.get("is_etf") and rs_gate_active and etf_gate_ready:
                _etf_pct = _etf_rs_percentile(hist_df, spy_strength)
                if _etf_pct is None or _etf_pct < RS_GATE_MIN:
                    drop_rs += 1
                    drop_rs_etf += 1
                    continue
                c["etf_rs_pct"] = _etf_pct
            # Stage-2 RS-line slope gate (USER 2026-07-15): RS line must be
            # making higher swing highs vs ^GSPC. Runs BEFORE the heavy
            # enrichment so rejected names never pay for charts/engines.
            _rsl_ok, _rsl_eval = _rs_line_swings_up(hist_df, bench_close)
            if _rsl_eval:
                rsline_checked += 1
            if not _rsl_ok:
                drop_rsline += 1
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
            meta_score, _raw_scores = _ranking_meta_score_ex(hist_df, meta_score_data["score"], market_modifier)
            c["status_labels"].extend(meta_score_data["badges"])
            trendline_data = calculate_trendline_analysis(c["ticker"], hist_df)

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
            if c.get("is_etf") and c.get("etf_rs_pct") is not None:
                _rs_val, _rs_asof = c["etf_rs_pct"], None   # computed fresh this run
            stock_data = {
                "ticker": c["ticker"], "close": c["close"], "adr": c["adr"],
                "perf_1m": c["perf_1m"], "perf_6m": c["perf_6m"], "perf_12m": c["perf_12m"],
                "power_score": c["power_score"], "rs_rating": _rs_val, "rs_asof": _rs_asof,
                "hugging": c["hugging"], "dist_pct": c["dist_pct"], "vol_pct": c["vol_pct_int"],
                "mcap": c["mcap"], "float_shares": c["float_shares"], "sector": c["sector"],
                "theme": c["theme"], "entry": c["entry"], "stop": c["stop"], "risk_pct": c["risk_pct"],
                "stop_reason": c["stop_reason"], "status_labels": c["status_labels"],
                "dist_52w": c["dist_52w"], "meta_score": meta_score, "section": "coil", **_raw_scores,
                "meta_details": meta_score_data["details"], "trendline_data": trendline_data,
                "spark": "", "footprint": footprint,
                "_ma_dist": (_ma_dist_data(hist_df["Close"].tolist())
                             if (hist_df is not None and len(hist_df) >= 2) else None),
                "pb_entry": c["pb_entry"], "pb_stop": c["pb_stop"], "pb_risk": c["pb_risk"], "ema9": c["ema9"],
                "is_etf": bool(c.get("is_etf")), "etf_desc": c.get("etf_desc"),
                "below_20dma": below_20dma, "below_50dma": below_50dma, "days_since_high": days_since_high,
                **_sr_quality(hist_df, c["entry"], "long"),
                **_pb2_quality(hist_df),
                **_tl_quality(hist_df, c["entry"], "long"),
                **_ch_quality(hist_df, c["entry"], "long"),
                **_line_break_watch(hist_df, c["close"]),
                **_pba_advance(hist_df, c["entry"]),
                # scan-time 5-day volume base for the next-morning VOL grading
                # read (S1 study 2026-07-17: trigger-day vol >= 2x prior-5-td avg
                # -> +15.6/+23.0pp win-1R both eras; display/log only, 94% of
                # monsters trigger under 2x so this must NEVER gate).
                "vol_base5": (round(float(hist_df["Volume"].iloc[-5:].mean()), 0)
                              if len(hist_df) >= 5 and float(hist_df["Volume"].iloc[-5:].mean()) > 0
                              else None),
            }
            # Lesson-refined plan BEFORE the chart, so entry/stop overlays, the
            # IBKR order plan and the trackers all carry the SAME refined plan.
            _lesson_plan(stock_data)
            if stock_data.get("plan_src"):
                # the Wide-Stop badge was set from the ORIGINAL risk — re-judge it
                # against the refined plan (adversarial review 2026-07-07)
                _labs = [x for x in stock_data["status_labels"] if "Wide Stop" not in x]
                if float(stock_data.get("risk_pct") or 0) > 6.0:
                    _labs.append("⚠️ Wide Stop (Size Down!)")
                stock_data["status_labels"] = _labs
            # Candlestick chart AFTER the engine merge so the payload can draw
            # the lesson levels (SR zone, PB trigger/stop, TL/CH diagonals).
            stock_data["spark"] = make_candle_chart(hist_df, stock_data, CHART_WINDOW)

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

            # Tier gates (2026-07-15 USER, second ruling): the ≤1%/≤2%-from-EMA
            # (min_dist) conditions STAY in the tiers — only the Stage-2
            # universe proximity gate (min_dist ≤ 10%) was removed. Tiers =
            # tightness duration + EMA hug + volume dry-up, on the RS-gated pool.
            is_a_plus = (is_tight_flag_3d and min_dist <= 0.01 and vol_aplus)
            is_a = (is_tight_2d and min_dist <= 0.01 and vol_a and not is_a_plus)
            is_a_minus = (is_tight_1d and min_dist <= 0.02 and vol_aminus
                          and not is_a_plus and not is_a)
            # tier met on structure alone, BEFORE the support gate zeroes the
            # flags — recorded so the Lesson Radar can say what was missed
            _pre_gate_tier = "A+" if is_a_plus else ("A" if is_a else ("A-" if is_a_minus else None))

            # Stage-3 support gate (USER-RATIFIED 2026-07-04): structural tier
            # criteria alone no longer suffice — the pick must ALSO be backed by
            # at least one verified entry engine (SR zones / pullback-recovery /
            # trendlines v2). Names that met a tier but lack support are counted
            # for the funnel and dropped from the tiers.
            stock_data["edge_support"] = _edge_support(stock_data)
            stock_data["lesson_confluence"] = _lesson_confluence(stock_data)
            if (is_a_plus or is_a or is_a_minus) and not stock_data["edge_support"]:
                drop_unsupported += 1
                is_a_plus = is_a = is_a_minus = False

            if is_a_plus:
                tier_a_plus.append(stock_data)
            elif is_a:
                tier_a.append(stock_data)
            elif is_a_minus:
                tier_a_minus.append(stock_data)
            elif len(stock_data["lesson_confluence"]) >= 3:
                # Lesson Radar: full lesson stacking but no tier — display-only
                stock_data["radar_missed_tier"] = _pre_gate_tier or "none"
                stock_data["radar_reason"] = ("support gate"
                                              if (_pre_gate_tier and not stock_data["edge_support"])
                                              else "tightness / vol dry-up")
                lesson_radar.append(stock_data)

        # Silent-stand-down detector: the benchmark was fetched but the gate
        # never actually evaluated anyone → alignment/data failure (this is how
        # the 2026-07-15 tz-index bug was caught). Warn loudly; never drop.
        if bench_close is not None and rsline_checked == 0:
            diag.warn(f"Stage-2 RS-line swing gate evaluated 0 of "
                      f"{len(coil_candidates)} candidates — alignment/data "
                      "failure; gate effectively stood down")

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
        lesson_radar.sort(key=lambda x: (-len(x.get("lesson_confluence") or []),
                                         -(x.get("meta_score") or 0.0)))
        lesson_radar = lesson_radar[:30]

    funnel = {
        "universe_total": n_universe,        # TradingView count matching the Stage-1 filter
        "stage1_fetched": n_stage1,          # rows fetched (range-capped at 5000)
        "drop_missing": drop_missing,
        "drop_52w": drop_52w,
        "drop_dead": drop_dead,
        "drop_proximity": drop_proximity,    # always 0 since 2026-07-15 (gate removed; key kept additive-only)
        "drop_rs": drop_rs,                  # failed stock/industry RS 80+ (2026-07-15)
        "drop_rsline": drop_rsline,          # RS line not rising swing-over-swing (2026-07-15)
        "stage2_candidates": len(coil_candidates),
        # Stage-2 survivors AFTER the enrichment-side drops (dead pins + RS-line
        # gate + the DEFERRED ETF RS gate, which also fires on rows already in
        # coil_candidates) — what the funnel card displays. stage2_candidates
        # keeps its old meaning (parse-loop survivors) for downstream compat.
        "stage2_final": max(0, len(coil_candidates) - drop_dead - drop_rsline - drop_rs_etf),
        "drop_unsupported": drop_unsupported,
        "lesson_radar": len(lesson_radar),
        "stage3_aplus": n_aplus_raw,
        "stage3_a": n_a_raw,
        "stage3_aminus": n_aminus_total,
        # ETF-leg breakdown (2026-07-15, additive-only). NOTE: universe_total /
        # stage1_fetched above stay STOCK-ONLY (unchanged semantics per the
        # additive-only rule); the funnel card sums both legs for display.
        "etf_universe_total": n_universe_etf,
        "etf_stage1_fetched": n_stage1_etf,
        "drop_rs_etf": drop_rs_etf,          # ETF subset of drop_rs
    }
    return tier_a_plus, tier_a, tier_a_minus, tier_a_minus_full, lesson_radar, funnel


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
        day_range_pct = ((high / low) - 1.0) * 100 if low else 0.0   # High/Low basis, same as ADRP

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
        meta_score, _raw_scores = _ranking_meta_score_ex(df, meta_score_data["score"], market_modifier)

        spark = (make_candle_chart(df, _chart_plan({"entry": entry, "stop": stop}, df), CHART_WINDOW)
                 if len(df) >= 2 else "")
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
            "dist_52w": dist_52w, "meta_score": meta_score, "section": "coil", **_raw_scores,
            "meta_details": meta_score_data["details"], "trendline_data": trendline_data,
            "spark": spark, "footprint": footprint,
            "_ma_dist": _ma_dist_data(cl.tolist()),   # HTF fires merge into A+; wire the vs-MA columns
            "pb_entry": None, "pb_stop": None, "pb_risk": None, "ema9": ema9,
            "below_20dma": below_20dma, "below_50dma": below_50dma,
            "days_since_high": days_since_high, "is_htf": True,
            # HTF fires merge straight into Tier A+ and render with the coil card
            # spec, so they need the same two scan-time stamps coil rows get
            # (2026-07-18 review): without vol_base5 their next-session VOL read
            # is silently None, and without pba_pct the PBA chip can never fire.
            **_pba_advance(df, entry),
            "vol_base5": (round(float(V[-5:].mean()), 0)      # V is a numpy array here
                          if len(V) >= 5 and float(V[-5:].mean()) > 0 else None),
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
    benchmark close series, ^GSPC). Returns a FIXED-shape dict (callers never
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
    # the report's RS-Line read (trend vs SPX + RS-new-high incl. "before price").
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
    attach_persistence — one 2y-history batch (+ ^GSPC) over the displayed tier
    members. Also attaches the Weinstein Phase-1 chips (2026-07-16): Mansfield
    zero-cross (mans_*), weekly stage (wk_*) and overhead-supply zone (oh_*) —
    all additive display keys. Decision-support only; never touches the IBKR
    draft plan."""
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
        s["rs_ok"] = bool(a["rs_spark_vals"])   # RS line was computable (benchmark aligned)
        # Weinstein Phase-1 chips (additive keys via the shared helper — the
        # same attachment runs on NH-52wk and Lesson Radar rows through
        # attach_weinstein, per the 2026-07-17 review).
        _weinstein_keys(s, hist.get(s["ticker"]), spy_close)


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
    # for the persistence (recurring-new-high) computation. ^GSPC rides along for
    # the Weinstein MRS chip (2026-07-17 fix: NH rows never pass attach_ants).
    hist_map = fetch_histories_batch(tickers + [ANTS_BENCHMARK], period="2y", min_rows=60)

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
        nh_meta, nh_raw = _ranking_meta_score_ex(df, ms["score"], 1.0)
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
            "ext9": fp.get("ext9"), "ext50": ext50, "meta_score": nh_meta, "section": "nh52", **nh_raw,
            "rs_rating": rs if isinstance(rs, int) else "N/A", "sector": info["sector"],
            "theme": get_theme(t, info["industry"]), "tag": tag, "label": label,
            "entry": entry, "stop": stop, "risk_pct": risk_pct,
            "spark": "",
            "_ma_dist": _ma_dist_data(cl.tolist()),
            "fp_badges": fp.get("badges", []),
            "persist_tier": rec["tier"], "persist_label": rec["label"],
            "nh_1m": rec["nh_1m"], "nh_3m": rec["nh_3m"], "weeks_3m": rec["weeks_3m"],
            **_sr_quality(df, entry, "long"),
            **_pb2_quality(df),
            **_tl_quality(df, entry, "long"),
            **_ch_quality(df, entry, "long"),
        })
        _lesson_plan(rows[-1])          # refined plan before chart (one plan everywhere)
        rows[-1]["lesson_confluence"] = _lesson_confluence(rows[-1])
        rows[-1]["spark"] = make_candle_chart(df, _chart_plan(rows[-1], df), CHART_WINDOW)

    # Weinstein chips for the displayed NH rows (2026-07-17 review fix: these
    # rows never pass attach_ants, so the chips were dead code here before).
    _spy_df = hist_map.get(ANTS_BENCHMARK)
    attach_weinstein(rows, diag, hist=hist_map,
                     spy_close=_spy_df["Close"] if _spy_df is not None else None)

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
        # 1y (was 6mo): every existing metric here is a trailing .iloc[-k<=60:]
        # window and the spark takes the last 60 closes, so the display is
        # byte-identical — the deeper frame exists so the pb2 read applies the
        # SAME Stage-2 200MA gate as the coil/NH52 surfaces (a 6mo frame
        # silently skipped it) and clears the detector's 120-bar floor.
        hist_map = fetch_histories_batch(tickers, period="1y", min_rows=50)
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
            status, tag = "● Low-vol pullback", "GRN"
        elif pullback and not low_vol:
            status, tag = "● High-vol breakdown", "RED"
        else:
            status, tag = "● Holding / extending", "HOLD"
        rs = rs_map.get(t.upper())
        monitored.append({
            "ticker": t, "close": round(close, 2), "sma50": round(sma50, 2),
            "vs_50": vs_50, "vs_prev": vs_prev, "below_50": below_50,
            "below_prev": below_prev, "low_vol": low_vol, "vol_ratio": vol_ratio,
            "days_since_high": _busday_age(e.get("last_high"), data_date),
            "last_high": e.get("last_high"), "high_count": int(e.get("high_count", 0)),
            "rs_rating": rs if isinstance(rs, int) else "N/A",
            "status": status, "tag": tag,
            "spark": "",
            # REV 10b: sortable numbers for the Screener. df is already in
            # memory here, so this is arithmetic only — no extra fetches.
            "perf_1m": (round((close / float(cl.iloc[-22]) - 1) * 100, 1)
                        if len(cl) > 22 and float(cl.iloc[-22]) > 0 else None),
            "perf_6m": (round((close / float(cl.iloc[-127]) - 1) * 100, 1)
                        if len(cl) > 127 and float(cl.iloc[-127]) > 0 else None),
            "adr": round(_adr20(df), 2),
            "dist_52w": (round((float(df["High"].iloc[-252:].max()) - close)
                               / float(df["High"].iloc[-252:].max()) * 100, 1)
                         if float(df["High"].iloc[-252:].max()) > 0 else None),
            "vol_pct": (round(vol_today / avg_vol30 * 100) if avg_vol30 > 0 else None),
            **_pb2_quality(df),
        })
        monitored[-1]["spark"] = make_candle_chart(df, _chart_plan(monitored[-1], df), CHART_WINDOW)

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
        # v9: HVE cards carry a candle chart. Screener rows hold no history, so
        # fetch a small batch here (list is tiny; recent IPOs may return few
        # bars — min_rows low, and a missing frame just means no chart).
        if ep_matches:
            try:
                _hmap = fetch_histories_batch([m["ticker"] for m in ep_matches],
                                              period="1y", min_rows=30)
                for m in ep_matches:
                    _hd = _hmap.get(m["ticker"])
                    if _hd is not None:
                        m["spark"] = make_candle_chart(_hd, _chart_plan(m, _hd), CHART_WINDOW)
            except Exception as exc:  # noqa: BLE001 — chart is decoration, never fatal
                diag.warn(f"HVE sparks skipped: {exc}")
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
        # v9: U&R cards carry a candle chart (same tiny-batch pattern as HVE)
        if ur_matches:
            try:
                _hmap = fetch_histories_batch([m["ticker"] for m in ur_matches],
                                              period="1y", min_rows=30)
                for m in ur_matches:
                    _hd = _hmap.get(m["ticker"])
                    if _hd is not None:
                        m["spark"] = make_candle_chart(_hd, _chart_plan(m, _hd), CHART_WINDOW)
            except Exception as exc:  # noqa: BLE001 — chart is decoration, never fatal
                diag.warn(f"U&R sparks skipped: {exc}")
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
        **_sr_quality(hist_df, entry, "short"),
        **_tl_quality(hist_df, entry, "short"),
        **_ch_quality(hist_df, entry, "short"),
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
    shorts = shorts[:15]
    # v9: short cards carry a candle chart — history already in memory
    for s in shorts:
        _hd = hist_map.get(s["ticker"])
        if _hd is not None:
            s["spark"] = make_candle_chart(_hd, _chart_plan(s, _hd, direction="short"), CHART_WINDOW)
    return shorts


def _stage4_short_signal(hist_df, ticker: str, meta: dict, rs_pct,
                         ind_rec: Optional[dict], bench_close) -> Optional[dict]:
    """Weinstein ch.7 Stage-4 breakdown candidate. Gates (NO volume condition
    anywhere — ch.2/ch.7 asymmetry: breakdowns are valid on light volume):
    1. Stage 4: price below the 30wk MA, MA flat-to-declining (slope <= +0.5).
    2. Inverted RS: rs_pct <= 25 OR RS-line falling swing-over-swing
       (fail-CLOSED on missing data — see _rs_line_swings_down).
    3. Weak group: industry RS percentile <= 25 (unmapped = excluded).
    4. Entry near the shelf: at/just breaking the 45-bar support low, not
       already collapsed (anti-chase).
    Returns the card dict or None."""
    try:
        if hist_df is None or len(hist_df) < 190:
            return None
        st = _stage_read(hist_df)
        _slope = st.get("wk_ma_slope") if st else None
        if not st or st.get("wk_above_ma") is not False or _slope is None or _slope > 0.5:
            return None
        rs_ok = isinstance(rs_pct, int) and rs_pct <= 25
        rs_line_down = None
        if not rs_ok:
            down_ok, evaluated = _rs_line_swings_down(hist_df, bench_close)
            rs_line_down = bool(down_ok and evaluated)
            if not rs_line_down:
                return None
        if not ind_rec or ind_rec.get("pct") is None or ind_rec["pct"] > 25:
            return None
        close = float(hist_df["Close"].iloc[-1])
        support_low = float(hist_df["Low"].iloc[-45:-5].min())
        if support_low != support_low or support_low <= 0:
            return None
        if not (support_low * 0.97 <= close <= support_low * 1.08):
            return None                       # not at the shelf / already gone
        entry = round(support_low - 0.10, 2)
        swing = _swing_rule_target(hist_df, support_low) or {}
        # ch.7 pattern filter 1: demand a substantial prior advance above the
        # breakdown (a top, not a flat Stage-1 base — Weinstein: a base is
        # where you COVER; a top after a flat stage 2 only retraces to its
        # nearby base). Peak of the top window must be >=30% above the shelf.
        if not swing.get("peak") or swing["peak"] < support_low * 1.30:
            return None
        # protective buy-stop: nearest confirmed rally peak above close in the
        # last 60 sessions (same +-3 pivot form as the LINE chip); fallback =
        # the 30wk-MA value.
        H = hist_df["High"].values.astype(float)
        n = len(H)
        raw_stop = None
        piv = [i for i in range(max(3, n - 60), n - 3)
               if H[i] == H[i - 3:i + 4].max() and H[i - 3:i + 4].argmax() == 3]
        cand = [H[i] for i in piv if H[i] > close]
        if cand:
            raw_stop = float(min(cand))
            basis = "rally peak $%.2f" % raw_stop
        else:
            wk_ma = hist_df["Close"].resample("W-FRI").last().rolling(30).mean().dropna()
            if not len(wk_ma):
                return None
            raw_stop = float(wk_ma.iloc[-1])
            basis = "30wk MA $%.2f" % raw_stop
        buy_stop = _round_buy_stop(raw_stop)
        risk_pct = round((buy_stop - entry) / entry * 100.0, 1)
        if risk_pct > 20:                     # ch.7 stop-distance screen
            return None
        tgt = swing.get("target_swing")
        return {
            "ticker": ticker, "close": round(close, 2),
            "adr": meta.get("adr"), "perf_1m": meta.get("p1m"),
            "perf_6m": meta.get("p6m"), "theme": meta.get("theme"),
            "sector": meta.get("sector") or "N/A",
            "short_style": "stage4",
            "wk_stage": st.get("wk_stage"), "wk_ma_slope": st.get("wk_ma_slope"),
            "rs_pct": (rs_pct if isinstance(rs_pct, int) else None),
            "rs_line_down": rs_line_down,
            "ind_rs": ind_rec["pct"], "ind_name": ind_rec.get("industry"),
            "support_low": round(support_low, 2),
            "peak": swing.get("peak"), "target_swing": tgt,
            "to_target_pct": (round((entry - tgt) / entry * 100.0, 1) if tgt else None),
            "entry": entry, "stop": buy_stop, "buy_stop_basis": basis,
            "risk_pct": risk_pct,
            **_sr_quality(hist_df, entry, "short"),
            **_tl_quality(hist_df, entry, "short"),
            **_ch_quality(hist_df, entry, "short"),
        }
    except Exception:  # noqa: BLE001
        return None


def scan_stage4_short(diag: Diagnostics, rs_map: dict,
                      industry_by_ticker: Dict[str, dict]) -> List[dict]:
    """Weinstein ch.7 STAGE-4 BREAKDOWN short leg (2026-07-17). Display-only:
    rows carry section='short' so every learning loop keeps skipping them and
    they can never reach the IBKR order plan (write_order_plan consumes coil
    tiers only)."""
    payload = {
        "filter": [
            {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
            {"left": "close", "operation": "greater", "right": 10},
            {"left": "average_volume_30d_calc", "operation": "greater", "right": 1000000},
            {"left": "market_cap_basic", "operation": "greater", "right": 2000000000},
            {"left": "close", "operation": "less", "right": "SMA200"},
            {"left": "ADRP", "operation": "egreater", "right": 1.5},
        ],
        "columns": ["name", "close", "ADRP", "Perf.1M", "Perf.6M", "industry", "sector"],
        "sort": {"sortBy": "Perf.6M", "sortOrder": "asc"},   # weakest first
        "range": [0, 800],
    }
    meta_map: Dict[str, dict] = {}
    tickers: List[str] = []
    try:
        time.sleep(2)
        data = tv_post(payload, label="stage4_short", diag=diag)
        for row in data.get("data", []):
            d = row.get("d")
            if not d or len(d) < 7:
                continue
            tk, close, adr, p1m, p6m, industry, sector = d
            if close is None:
                continue
            # weak-group pre-filter saves history fetches (gate 3 re-checked in
            # the signal builder)
            rec = industry_by_ticker.get((tk or "").upper().replace(".", "-"))
            if not rec or rec.get("pct") is None or rec["pct"] > 25:
                continue
            tickers.append(tk)
            meta_map[tk] = {"adr": adr, "p1m": p1m, "p6m": p6m,
                            "theme": get_theme(tk, industry), "sector": sector}
    except Exception as exc:  # noqa: BLE001
        diag.warn(f"Stage-4 short scan (fetch/parse): {exc}")
        return []
    if not tickers:
        return []
    tickers = tickers[:120]   # stated cap: first 120 weak-group survivors
    hist_map = fetch_histories_batch(tickers + [ANTS_BENCHMARK], period="2y", min_rows=190)
    bench_df = hist_map.get(ANTS_BENCHMARK)
    bench_close = bench_df["Close"] if bench_df is not None else None
    rows: List[dict] = []
    for tk in tickers:
        hd = hist_map.get(tk)
        if hd is None or is_dead_pinned(hd):
            continue
        sig = _stage4_short_signal(hd, tk, meta_map.get(tk) or {},
                                   rs_map.get(tk.upper()),
                                   industry_by_ticker.get(tk.upper().replace(".", "-")),
                                   bench_close)
        if sig:
            rows.append(sig)
    rows.sort(key=lambda r: (r.get("ind_rs") or 99, r.get("rs_pct") if r.get("rs_pct") is not None else 99))
    rows = rows[:10]          # display cap (stated)
    # days-to-cover only for the bounded final survivors (yf .info is slow)
    dtc = _dtc_lookup([r["ticker"] for r in rows], diag)
    keep: List[dict] = []
    for r in rows:
        r.update(dtc.get(r["ticker"], {}))
        if r.get("dtc") is not None and r["dtc"] >= 10:
            continue                          # ch.7 exclusion: >=10x squeeze fuel
        keep.append(r)
    # v9: stage-4 cards carry a candle chart — 2y history already in memory
    for r in keep:
        _hd = hist_map.get(r["ticker"])
        if _hd is not None:
            r["spark"] = make_candle_chart(_hd, _chart_plan(r, _hd, direction="short"), CHART_WINDOW)
    return keep


# ----------------------------------------------------------------------------
# REPORT BUILDING
# ----------------------------------------------------------------------------
PAGE_CSS = """
    /* ---- design tokens ("calm editorial dark" 2026-07-05) ----
       Hex values mirror the Python palette constants next to _MA_SPEC —
       keep the two blocks in sync. */
    :root {
      /* neutrals — v9: the ratified dark skin (rev-8 artifacts). Cool graphite-
         blue ground; same token NAMES so every legacy var() keeps working. */
      --bg:#0b0e12; --surface:#12161c; --raised:#161b22; --hover:#1a2027;
      --line:#1a2027; --line-2:#232a33;
      --text:#e6edf3; --text-2:#b7c0c9; --text-3:#8b949e;
      /* structural accent — desaturated sky (NOT the act cyan) */
      --accent:#8cb4d6; --accent-2:#aecfe8; --tint-accent:rgba(140,180,214,.08);
      --tint-accent-2:rgba(140,180,214,.16);
      /* act cyan — strictly interactive / "you are here" (v9 discipline) */
      --act:#4cc2ff; --act-bd:#2b5a75; --tint-act:rgba(76,194,255,.10);
      --tier:#3fb68b; --flag-ink:#7da9c4; --flag-bd:#31404d;
      /* semantic trio — trading load-bearing only */
      --up:#33d17a;   --tint-up:rgba(51,209,122,.08);    --bd-up:#2f6b4b;
      --down:#ff5c5c; --tint-down:rgba(255,92,92,.08);   --bd-down:#8a4341;
      --warn:#d9a83c; --tint-warn:rgba(217,168,60,.08);  --bd-warn:#7d6231;
      /* migration aliases — every legacy var() reference keeps working */
      --green:var(--up); --red:var(--down); --yellow:var(--warn);
      --bd-green:var(--bd-up); --bd-red:var(--bd-down); --bd-yellow:var(--bd-warn);
      --bd-accent:var(--accent); --tint-green:var(--tint-up);
      --tint-red:var(--tint-down); --tint-yellow:var(--tint-warn);
      /* candlestick chart — TRUE 4-state hollow candles (v9): neutral slate up,
         red down; colour = close vs prev close, hollow/filled = close vs open */
      --candle-up:#9aa7b3; --candle-down:#ff5c5c; --vol-ma:#9aa4ae;
      --chart-grid:#1f2630; --chart-axis:#8b949e;
      /* MA colours unchanged — the learned 10=blue / 20=gold / 50=grey mapping */
      --ma-fast:#8cb4d6; --ma-mid:#d3a04d; --ma-slow:#6b6b74;
      --mono:ui-monospace,'SF Mono','Cascadia Mono',Menlo,Consolas,monospace;
      --fs-micro:0.6875rem; --fs-caption:0.75rem; --fs-table:0.8125rem;
      --fs-body:0.875rem; --fs-title:1rem; --fs-h1:1.125rem;
      --r-chip:3px; --r-card:6px;
    }
    body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,
                       'PingFang TC','Hiragino Sans TC','Microsoft JhengHei','Noto Sans TC',sans-serif;
           background-color:var(--bg); color:var(--text); margin:10px auto; max-width:1400px; padding:0 4px;
           line-height:1.5; font-variant-numeric:tabular-nums; -webkit-text-size-adjust:100%; }
    h1 { color:var(--text-2); text-align:center; font-size:var(--fs-h1); font-weight:600; text-transform:uppercase; letter-spacing:.14em; margin-bottom:5px; }
    .header-sub { text-align:center; color:var(--text-3); font-size:var(--fs-caption); margin-top:0; margin-bottom:14px; }

    .runbar { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin:0 0 24px; }
    .chip { background:var(--surface); border:1px solid var(--line); border-radius:999px; padding:5px 12px; font-size:var(--fs-caption); color:var(--text); }
    .chip b { color:var(--accent); font-family:var(--mono); }
    .chip.green b { color:var(--green); } .chip.red b { color:var(--red); } .chip.warn b { color:var(--yellow); }

    #search { display:block; width:100%; max-width:420px; margin:0 auto 24px; padding:10px 14px; border-radius:999px;
              border:1px solid var(--line); background:var(--surface); color:var(--text); font-size:16px; box-sizing:border-box;
              position:sticky; top:8px; z-index:10; }
    #search:focus { outline:none; border-color:var(--accent); }

    .market-panel { background-color:var(--surface); border-radius:8px; padding:15px; margin-bottom:24px; border:1px solid var(--line); }
    .market-title { color:var(--accent); font-weight:600; font-size:var(--fs-title); text-transform:uppercase; margin-bottom:10px; }
    .market-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:15px; }
    .market-card { background-color:var(--raised); padding:10px; border-radius:8px; }
    .market-card h3 { margin:0 0 5px 0; color:var(--text); font-size:var(--fs-body); }
    .val-green { color:var(--green); font-weight:600; } .val-red { color:var(--red); font-weight:600; } .val-warn { color:var(--yellow); font-weight:600; }

    .kill-warn { background:var(--tint-yellow); border:1px solid var(--bd-yellow); color:var(--yellow); border-radius:8px; padding:8px 14px; font-size:var(--fs-table); }

    .regime { border:1px solid; border-radius:8px; padding:10px 14px; margin:0 0 24px; }
    .reg-head { display:flex; flex-wrap:wrap; gap:6px 12px; align-items:baseline; font-weight:700; font-size:1.05em; }
    .reg-score { font-weight:normal; font-size:var(--fs-caption); color:var(--text); font-family:var(--mono); }
    .reg-sigs { margin-top:8px; display:flex; flex-wrap:wrap; gap:6px; }
    .reg-sig { font-size:var(--fs-micro); font-weight:500; border:1px solid; border-radius:999px; padding:2px 8px; background:var(--bg); display:inline-block; }
    /* click-to-expand tell meaning (2026-07-06): native details, no marker */
    .reg-sig-w > summary { list-style:none; cursor:pointer; }
    .reg-sig-w > summary::-webkit-details-marker { display:none; }
    .reg-sig-w[open] .reg-sig { background:var(--raised); }
    .reg-exp { font-size:var(--fs-caption); color:var(--text-3); line-height:1.5; max-width:420px;
               margin:4px 2px 4px 8px; padding:4px 8px; }
    .reg-note { margin-top:8px; font-size:var(--fs-caption); color:var(--text-2); line-height:1.6; }
    .dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:4px; vertical-align:1px; }
    .dot-g { background:var(--green); } .dot-y { background:var(--yellow); } .dot-r { background:var(--red); } .dot-i { background:var(--text-3); }

    .toppicks { display:flex; flex-wrap:wrap; gap:10px; margin:0 0 24px; }
    .tp-card { flex:1 1 190px; min-width:180px; background:var(--surface); border:1px solid var(--line);
               border:1px solid var(--line); border-radius:var(--r-card); padding:11px 13px; }
    .tp-card.t-aplus { border-top-color:var(--up); }
    .tp-card.t-a { border-top-color:var(--warn); }
    .tp-card.t-aminus { border-top-color:var(--down); }
    .tp-top { display:flex; align-items:center; gap:6px; }
    .tp-top a { font-weight:700; font-size:1.2em; color:var(--text); text-decoration:none; }
    .tp-top a:hover { color:var(--accent); }
    .tp-tier { font-size:10px; font-weight:600; border:1px solid; border-radius:var(--r-chip); padding:1px 5px; }
    .tp-edges { font-size:var(--fs-micro); font-weight:500; color:var(--text-2); border:1px solid var(--line-2);
                border-radius:var(--r-chip); padding:1px 6px; font-family:var(--mono); }
    .tp-draft { margin-left:auto; font-size:10px; color:var(--text-3); white-space:nowrap; }
    .plan-jump { display:inline-block; margin-top:3px; font-size:var(--fs-micro); color:var(--accent-2);
                 text-decoration:none; border:1px solid var(--accent); border-radius:3px; padding:0 5px; }
    .plan-jump:hover { background:var(--tint-accent); }
    .tp-px { margin-top:7px; display:flex; align-items:baseline; gap:8px; }
    .tp-px .lp { font-size:1.35rem; }
    .tp-legs { font-size:var(--fs-micro); color:var(--text-3); }
    .tp-signals { margin-top:6px; display:flex; flex-wrap:wrap; gap:3px 5px; align-items:center; }
    .tp-meta { font-size:var(--fs-micro); color:var(--text-3); font-weight:normal; }
    .tp-theme { margin-top:6px; font-size:var(--fs-caption); color:var(--text-3); }

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

    .section-title { position:relative; display:flex; align-items:baseline; gap:8px; flex-wrap:wrap;
                     padding:10px 15px; margin-top:32px; margin-bottom:0; border-radius:8px 8px 0 0;
                     font-size:var(--fs-caption); font-weight:600; text-transform:uppercase; letter-spacing:0.08em;
                     color:var(--text); background-color:var(--surface); border:1px solid var(--line); border-bottom:none; }
    .section-title .tdot { width:8px; height:8px; border-radius:50%; align-self:center; flex:none; background:var(--text-3); }
    .section-sub { font-weight:400; text-transform:none; letter-spacing:0; color:var(--text-3); font-size:var(--fs-micro); }
    .section-title.collapsible { cursor:pointer; user-select:none; }
    .section-title.collapsible::after { content:'▾'; position:absolute; right:15px; top:10px; opacity:0.65; font-weight:400; }
    .section-title.collapsed { border-radius:8px; }
    .section-title.collapsed::after { content:'▸'; }
    .section-title.collapsed + .table-container { display:none; }
    /* tier criteria are printed inline in each section title (the gate from scan_coil) */
    .funnel { margin:24px 0 0; background:var(--surface); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    /* funnel is a collapsed <details> now (2026-07-06): summary = the caption */
    .funnel > summary.fn-cap { cursor:pointer; list-style:none; }
    .funnel > summary.fn-cap::-webkit-details-marker { display:none; }
    .funnel .fn-cap { padding:8px 12px; font-size:var(--fs-caption); font-weight:700; color:var(--text); background:var(--raised); border-bottom:1px solid var(--line); }
    .funnel .fn-stage { display:flex; align-items:center; gap:14px; padding:10px 12px; border-bottom:1px solid var(--line); }
    .funnel .fn-stage:last-child { border-bottom:none; }
    .funnel .fn-body { flex:1; min-width:0; }
    .funnel .fn-title { font-size:var(--fs-caption); font-weight:600; color:var(--text); }
    .funnel .fn-sub { color:var(--text-3); font-weight:400; }
    .funnel .fn-crit { font-size:var(--fs-micro); color:var(--text-3); line-height:1.45; margin-top:3px; }
    .funnel .fn-count { font-family:var(--mono); font-weight:700; font-size:var(--fs-table); color:var(--accent); white-space:nowrap; text-align:right; }
    .funnel .fn-dot { color:var(--text-3); }
    .bg-aplus .tdot { background:var(--up); }
    .bg-a .tdot { background:var(--warn); }
    .bg-aminus .tdot, .bg-hve .tdot, .bg-short .tdot { background:var(--down); }

    details.mindset { background:var(--surface); border:1px solid var(--line); border-radius:8px; padding:10px 16px; margin:0 0 24px; }
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
    /* utility classes replacing repeated inline styles (2026-07-05 cleanup) */
    .sub { font-size:var(--fs-micro); color:var(--text-3); }
    .arr-up { color:var(--up); font-size:var(--fs-micro); }
    .arr-dn { color:var(--down); font-size:var(--fs-micro); }
    .risk-lo { color:var(--up); } .risk-md { color:var(--warn); } .risk-hi { color:var(--down); }
    .risk-lo, .risk-md, .risk-hi { font-size:var(--fs-caption); font-family:var(--mono); }
    .meta-pill { font-size:var(--fs-body); font-weight:600; font-family:var(--mono);
                 padding:4px 8px; border-radius:var(--r-chip); border:1px solid; display:inline-block; }
    .meta-hi { color:var(--down); background:var(--tint-down); border-color:var(--down); }
    .meta-md { color:var(--warn); background:var(--tint-warn); border-color:var(--warn); }
    .meta-lo { color:var(--text-3); background:var(--raised); border-color:var(--text-3); }
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
    details.fund > summary::after { content:' fin ▸'; opacity:.45; font-size:var(--fs-micro); margin-left:4px; }
    details.fund[open] > summary::after { content:' fin ▴'; opacity:.7; }
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
    .fund-up { color:#54b87f; } .fund-dn { color:#e06c6a; } .fund-flat, .fund-na { color:var(--text-3); }
    .fund-src { color:var(--text-3); font-size:10px; margin-top:3px; opacity:.7; }
    .spark { margin-top:4px; }
    /* candlestick chart cell: JSON payload rendered lazily by CANDLE_JS.
       v9: chart height is width-independent (262px deck geometry), so the
       lazy placeholder is a fixed height — no layout jump on render. The
       OHLCV readout lives in the chart's reserved band (no floating tip). */
    .cchart { margin-top:4px; width:100%; max-width:340px; }
    .cchart:empty { height:262px; background:var(--raised); border-radius:var(--r-card); opacity:.45; }
    .cchart svg { display:block; width:100%; height:auto; touch-action:pan-y; }
    .livebtn { cursor:pointer; font:inherit; border:1px solid var(--bd-accent) !important; color:var(--accent-2) !important; background:var(--tint-accent) !important; min-width:9.5em; min-height:32px; }
    .livebtn:disabled { opacity:0.6; cursor:wait; }
    .lp { transition:color .25s; font-family:var(--mono); font-variant-numeric:tabular-nums; font-size:var(--fs-title); font-weight:700; }
    .lp.flag-up::after { content:' ▲'; color:var(--green); font-size:var(--fs-caption); }
    .lp.flag-down::after { content:' ▼ STOP'; color:var(--red); font-size:var(--fs-micro); font-weight:700; }

    /* ---- interaction guards ---- */
    @media (hover:hover) {
      tr:hover { background-color:var(--hover); }
      .theme-chip:hover { background:var(--hover); }
      th.sortable:hover { color:var(--text); }
      .livebtn:hover { background:var(--tint-accent-2) !important; }
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

    /* ---- chart-centric rows + mobile cards (2026-07-05 layout) ---- */
    .kicker { display:block; color:var(--text-3); font-size:var(--fs-micro); font-weight:600;
              letter-spacing:.08em; text-transform:uppercase; }
    .lbl { display:inline-block; font-size:10px; font-weight:600; letter-spacing:.05em; line-height:1.4;
           border:1px solid var(--line-2); border-radius:var(--r-chip); padding:0 4px; color:var(--text-2); }
    .lbl-hot { color:var(--warn); border-color:var(--bd-warn); }
    .lesson-ct { display:inline-block; font-size:10px; font-weight:600; letter-spacing:.04em;
                 color:var(--warn); border:1px solid var(--bd-warn); border-radius:var(--r-chip);
                 padding:1px 5px; margin-top:3px; white-space:nowrap; }
    details.lessons { margin-top:5px; }
    details.lessons > summary { cursor:pointer; list-style:none; padding:3px 0; }
    details.lessons > summary::-webkit-details-marker { display:none; }
    details.lessons > summary .sumhint { color:var(--text-3); font-size:var(--fs-micro); }
    details.lessons > summary .sumhint::after { content:' ▸'; opacity:.6; }
    details.lessons[open] > summary .sumhint::after { content:' ▾'; }
    /* desktop: give the chart column room */
    table.rowcards th[data-col='price'], table.rowcards td.c-chart { width:360px; min-width:340px; }
    td.c-chart .cchart { margin-top:0; }
    td.c-narr { min-width:130px; }

    @media (max-width: 768px) {
      table.rowcards { min-width:0; border-collapse:separate; }
      table.rowcards thead { display:none; }
      table.rowcards tr { display:grid; grid-template-columns:repeat(6,1fr); gap:4px 8px;
        background:var(--surface); border:1px solid var(--line); border-radius:10px;
        padding:10px 10px 12px; margin:0 0 12px;
        content-visibility:auto; contain-intrinsic-size:auto 560px; }
      table.rowcards td { display:block; border-bottom:none; padding:0; font-size:var(--fs-table);
        grid-column:1/-1; }
      table.rowcards td.col-hidden { display:block !important; }   /* cards always show everything */
      table.rowcards th:first-child, table.rowcards td:first-child
        { position:static; background:transparent; }               /* no sticky col in cards */
      table.rowcards td.ticker, table.rowcards td.ep-ticker { grid-column:1/-1; order:0;
        display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; }
      #search { top:54px; z-index:8; }
      table.rowcards td.c-chart { grid-column:1/-1; order:1; width:auto; min-width:0; }
      table.rowcards td.c-chart .cchart { max-width:none; }
      table.rowcards td.c-plan { grid-column:1/-1; order:2; }
      table.rowcards td.c-narr { grid-column:1/-1; order:3; min-width:0; }
      table.rowcards td.c-stat { grid-column:span 2; order:4; text-align:left;
        background:var(--raised); border:1px solid var(--line); border-radius:var(--r-card); padding:4px 6px; }
      table.rowcards td[data-label]::before { content:attr(data-label); display:block;
        font-size:10px; color:var(--text-3); text-transform:uppercase; letter-spacing:.06em; }
      table.rowcards td.c-status { grid-column:1/-1; order:5; }
      table.rowcards td[colspan] { grid-column:1/-1; }
      .rowcards-container { max-height:none; overflow:visible; }
      .rowcards-container .colbar { display:none; }                /* column tools are a desktop feature */
      .tabs { flex-wrap:nowrap; overflow-x:auto; -webkit-overflow-scrolling:touch;
              position:sticky; top:0; background:var(--bg); z-index:9; }
      .tab-btn { flex:0 0 auto; }
      div.sortsel-bar { display:flex; }
    }
    /* mobile sort bridge (hidden on desktop; the header row is the desktop UI) */
    .sortsel-bar { display:none; gap:6px; margin:6px 0 10px; }
    .sortsel { flex:1; background:var(--surface); color:var(--text); border:1px solid var(--line);
               border-radius:var(--r-card); padding:9px 10px; font-size:15px; }
    .sortdir { background:var(--raised); color:var(--text-2); border:1px solid var(--line);
               border-radius:var(--r-card); padding:9px 13px; font-size:15px; cursor:pointer; }
    /* ---- IBD-style page-1 sections (2026-07-08) ---- */
    .bigpic-wrap { display:grid; grid-template-columns:1.6fr 1fr; gap:15px; }
    .bigpic-text { font-size:var(--fs-body); color:var(--text-2); line-height:1.65; }
    .bigpic-text p { margin:0 0 10px; }
    .bigpic-text b { color:var(--text); }
    .pulse-box h3 { margin:0 0 8px; }
    .pulse-state { display:inline-block; font-weight:700; font-size:var(--fs-body); letter-spacing:.04em;
                   border:1px solid; border-radius:6px; padding:3px 10px; margin-bottom:8px; }
    .pulse-line { font-size:var(--fs-table); color:var(--text-2); margin:7px 0; line-height:1.9; }
    .pulse-chip { display:inline-block; background:var(--raised); border:1px solid var(--line);
                  border-radius:5px; padding:1px 7px; margin:2px 3px 2px 0; white-space:nowrap; }
    .pulse-chip a { color:var(--accent); text-decoration:none; font-weight:600; }
    .t10-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px;
                background:var(--surface); border:1px solid var(--line); border-top:none;
                border-radius:0 0 8px 8px; padding:14px; margin-bottom:24px; }
    .t10-card { background:var(--raised); border:1px solid var(--line); border-radius:8px; padding:11px 13px; }
    .t10-head { display:flex; gap:9px; align-items:baseline; }
    .t10-num { flex:0 0 auto; font-weight:700; color:var(--accent); font-size:var(--fs-title); }
    .t10-title { font-size:var(--fs-body); font-weight:600; line-height:1.35; }
    .t10-title a { color:var(--text); text-decoration:none; }
    .t10-title a:hover { color:var(--accent); }
    .t10-orig { font-size:var(--fs-caption); color:var(--text-3); margin-top:3px; line-height:1.3; }
    .t10-meta { font-size:var(--fs-caption); color:var(--text-3); margin:4px 0 6px; }
    .t10-brief { font-size:var(--fs-table); color:var(--text-2); line-height:1.55; }
    @media (max-width:768px) {
      .bigpic-wrap { grid-template-columns:1fr; }
      .t10-grid { grid-template-columns:1fr; }
    }

    /* ================= v9 — chart-first cards + Foldable Desk ================= */
    body.v9 #search { position:static; }   /* the secnav is the sticky bar in v9 */
    .vchip { display:inline-flex; align-items:center; gap:3px; font:600 10px/1 var(--mono); letter-spacing:.06em;
             text-transform:uppercase; border:1px solid var(--flag-bd); color:var(--flag-ink);
             border-radius:4px; padding:3px 6px; white-space:nowrap; }
    .tier-chip { border-color:var(--tier); color:var(--tier); font-weight:700; }
    .secnav { position:sticky; top:0; z-index:12; background:var(--bg); margin:0 0 14px; padding:8px 0 6px;
              border-bottom:1px solid var(--line-2); }
    .secnav-in { display:flex; gap:6px; overflow-x:auto; -webkit-overflow-scrolling:touch; }
    .navchip { flex:0 0 auto; display:inline-flex; align-items:center; gap:5px; min-height:44px;
               border:1px solid var(--line-2); border-radius:999px; padding:0 13px;
               font:500 12px/1 var(--mono); color:var(--text-2); text-decoration:none;
               background:var(--surface); cursor:pointer; -webkit-tap-highlight-color:transparent; }
    .navchip b { color:var(--text); font-weight:600; }
    .navchip:hover { border-color:var(--act-bd); color:var(--act); }
    .navchip:focus-visible { border-color:var(--act-bd); color:var(--act); outline:2px solid var(--act-bd); outline-offset:2px; }
    /* REV 10c: chart control bar — outside #deck so desk mode can't hide it */
    .chartctl { position:sticky; top:0; z-index:13; background:var(--bg);
                padding:8px 0 8px; margin:0 0 10px; border-bottom:1px solid var(--line-2); }
    .ctl-row { display:flex; align-items:center; gap:8px; overflow-x:auto;
               -webkit-overflow-scrolling:touch; scrollbar-width:none; }
    .ctl-row::-webkit-scrollbar { display:none; }
    .ctl-row + .ctl-row { margin-top:7px; }
    .ctl-lbl { flex:0 0 auto; font:600 10px/1 var(--mono); letter-spacing:.08em;
               color:var(--text-3); text-transform:uppercase; }
    .fchip { flex:0 0 auto; min-height:34px; cursor:pointer; border:1px solid var(--line-2);
             border-radius:999px; padding:0 12px; background:var(--surface);
             font:600 12px/1 var(--mono); color:var(--text-2);
             display:inline-flex; align-items:center; gap:5px;
             -webkit-tap-highlight-color:transparent; }
    .fchip b { color:var(--text-3); font-weight:600; font-size:10px; }
    .fchip:hover { color:var(--text); border-color:var(--act-bd); }
    .fchip.on { background:var(--tint-act); border-color:var(--act-bd); color:var(--act); }
    .fchip.on b { color:var(--act); }
    .ctlsel { flex:0 0 auto; min-height:34px; background:var(--surface); color:var(--text);
              border:1px solid var(--line-2); border-radius:9px; padding:0 8px;
              font:600 12px/1 var(--mono); cursor:pointer; }
    /* filter states: hide non-matching cards, their now-empty sections, desk rows */
    #deck article.card.grp-off, #deck .secv9.grp-empty,
    #desklist .dl-row.grp-off, #desklist .dl-h.grp-off { display:none !important; }
    /* REV 10b: top-level Charts / Screener tabs */
    .v9tabs { display:flex; gap:4px; margin:0 0 10px; border-bottom:1px solid var(--line-2); }
    .v9tab { appearance:none; border:0; background:none; cursor:pointer; min-height:44px;
             padding:0 18px; font:600 13px/1 var(--mono); color:var(--text-3);
             border-bottom:2px solid transparent; margin-bottom:-1px;
             -webkit-tap-highlight-color:transparent; }
    .v9tab:hover { color:var(--text-2); }
    .v9tab.on { color:var(--act); border-bottom-color:var(--act); }
    .v9tab:focus-visible { outline:2px solid var(--act-bd); outline-offset:-2px; }
    .v9pane[hidden] { display:none; }
    /* screener table */
    .scr-head { color:var(--text-3); font-size:var(--fs-caption); margin:0 0 8px; }
    .scr-head b { color:var(--text); }
    .scr-wrap { max-height:none; }
    table.screener { min-width:760px; width:100%; }
    table.screener th { position:sticky; top:0; background:var(--raised); z-index:2;
                        white-space:nowrap; font-size:var(--fs-micro); }
    table.screener td { padding:7px 9px; white-space:nowrap; font-size:var(--fs-caption); }
    table.screener td.tl, table.screener th.tl { text-align:left; }
    table.screener tbody tr:hover td { background:var(--hover); }
    .scr-tk { font-weight:700; color:var(--text); text-decoration:none; font-family:var(--mono); }
    .scr-tk:hover { color:var(--act); }
    .scr-sec { color:var(--text-3); font-size:var(--fs-micro); }
    /* REV 10 global chart controls: timeframe + grid density (secnav row 2) */
    .secnav-ctl { margin-top:6px; gap:10px; }
    .ctlgrp { flex:0 0 auto; display:inline-flex; border:1px solid var(--line-2);
              border-radius:9px; overflow:hidden; background:var(--surface); }
    .ctlbtn { min-height:34px; min-width:38px; border:0; background:transparent; cursor:pointer;
              font:600 12px/1 var(--mono); color:var(--text-3); padding:0 10px;
              -webkit-tap-highlight-color:transparent; }
    .ctlbtn + .ctlbtn { border-left:1px solid var(--line-2); }
    .ctlbtn:hover { color:var(--text); }
    .ctlbtn.on { background:var(--tint-act); color:var(--act); }
    .ctlbtn:focus-visible { outline:2px solid var(--act-bd); outline-offset:-2px; }
    /* grid density — cards tile; dense modes shed everything but the chart so a
       wall of charts stays readable (USER: "3x3 or 4x4 charts per screen") */
    .cardlist.gcols { display:grid; gap:10px; }
    .cardlist.gcols .card { margin:0; }
    .cardlist.g2 { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .cardlist.g3 { grid-template-columns:repeat(3,minmax(0,1fr)); }
    .cardlist.g4 { grid-template-columns:repeat(4,minmax(0,1fr)); }
    .cardlist.gcols .fold, .cardlist.gcols .cardlines, .cardlist.gcols .card-flags { display:none; }
    .cardlist.g3 .card-head .lp-sub, .cardlist.g4 .card-head .lp-sub { display:none; }
    @media (max-width:640px) {
      .cardlist.g3, .cardlist.g4 { grid-template-columns:repeat(2,minmax(0,1fr)); }
    }
    .secv9 { scroll-margin-top:64px; }
    .sec-n { font-family:var(--mono); color:var(--text-3); font-weight:600; margin-left:8px; font-size:var(--fs-caption); }
    .cardlist { max-height:none !important; overflow:visible !important; background:transparent; border:none; padding:2px 0 6px; }
    .card { background:var(--surface); border:1px solid var(--line-2); border-radius:14px;
            padding:12px 12px 12px; margin:0 0 14px; content-visibility:auto; contain-intrinsic-size:auto 520px; }
    .card-empty { color:var(--text-3); font-size:var(--fs-table); padding:16px; contain-intrinsic-size:auto 60px; }
    .card-head { display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-bottom:8px; }
    .card-head .ticker { position:static; }
    .card-head .ticker a { font-family:var(--mono); font-size:18px; font-weight:800; letter-spacing:.02em;
                           color:var(--text); text-decoration:none; }
    .card-head .ticker a:hover { color:var(--act); }
    .card-head .ticker a:focus-visible { color:var(--act); outline:2px solid var(--act-bd); outline-offset:2px; }
    .card-head .lp { font-size:var(--fs-table); border:1px solid var(--line-2); border-radius:5px; padding:2px 7px; }
    .card-flags { margin-left:auto; display:flex; gap:5px; flex-wrap:wrap; justify-content:flex-end; }
    .card .cchart { max-width:none; width:calc(100% + 24px); margin:4px -12px 0; }
    .card .cchart:empty { height:262px; }
    .cardlines { margin-top:8px; }
    .fold { margin-top:10px; }
    .fold > summary { list-style:none; cursor:pointer; display:flex; align-items:center; gap:8px;
                      border:1px solid var(--line); border-radius:9px; background:var(--raised);
                      padding:0 12px; min-height:44px; font:600 10px/1 var(--mono); letter-spacing:.1em;
                      text-transform:uppercase; color:var(--text-3);
                      -webkit-tap-highlight-color:transparent; user-select:none; -webkit-user-select:none; }
    .fold > summary::-webkit-details-marker { display:none; }
    .fold .chev { margin-left:auto; transition:transform .15s ease; font-size:10px; }
    .fold[open] > summary .chev { transform:rotate(90deg); }
    @media (prefers-reduced-motion: reduce) { .fold .chev { transition:none; } }
    .foldin { margin-top:10px; display:flex; flex-direction:column; gap:10px; }
    .vplan { margin:0; }
    .stat6 { display:grid; grid-template-columns:repeat(3,1fr); gap:1px; background:var(--line-2);
             border:1px solid var(--line-2); border-radius:9px; overflow:hidden; }
    .stat6 .sc { background:var(--raised); padding:7px 9px; }
    .stat6 .sl { display:block; font:600 9px/1 var(--mono); letter-spacing:.06em; text-transform:uppercase;
                 color:var(--text-3); margin-bottom:4px; }
    .stat6 .sv { font-family:var(--mono); font-size:12.5px; color:var(--text); font-variant-numeric:tabular-nums; }
    .sv.up { color:var(--green); } .sv.dn { color:var(--red); }
    .ctxline { font-family:var(--mono); font-size:11px; line-height:1.7; color:var(--text-3); }
    .ctxline .up { color:var(--green); } .ctxline .dn { color:var(--red); } .ctxline .warn { color:var(--yellow); }
    .chiprow { display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
    .fundrow .sumhint { color:var(--text-3); font-size:var(--fs-micro); }
    .vnote { border:1px solid var(--bd-accent); background:var(--tint-accent);
             border-radius:0 8px 8px 0; padding:10px 14px; font-size:var(--fs-table); }
    .vnote-green { border-color:var(--bd-up); background:var(--tint-up); }
    .vnote-t { font-weight:700; color:var(--accent-2); margin-bottom:4px; letter-spacing:.04em; }
    .vnote-green .vnote-t { color:var(--green); }
    .vnote-b { color:var(--text-2); line-height:1.6; }
    @media (min-width:520px) { .stat6 { grid-template-columns:repeat(6,1fr); } }

    /* ---- the Foldable Desk (v9 wide mode; body.desk set by matchMedia JS:
       list right, ONE big card left — the cards never move in the DOM) ---- */
    body.desk #deskwrap { display:grid; grid-template-columns:minmax(0,1fr) 312px; gap:14px; align-items:start; }
    body.desk #desklist { display:block; position:sticky; top:8px; max-height:calc(100vh - 16px);
                          max-height:calc(100dvh - 16px); overflow-y:auto; background:var(--surface);
                          border:1px solid var(--line-2); border-radius:12px; }
    body.desk #deck > * { display:none; }
    body.desk #deck .secv9.has-active, body.desk #deck details.has-active { display:block; }
    body.desk #deck .secv9.has-active:not(.desk-full) .card { display:none; }
    body.desk #deck .secv9.has-active .card.desk-active { display:block; }
    body.desk #deck details.has-active .card { display:none; }
    body.desk #deck details.has-active .card.desk-active { display:block; }
    body.desk .card.desk-active { border-color:var(--act-bd); contain-intrinsic-size:auto 900px; }
    .dl-pos { position:sticky; top:0; z-index:2; background:var(--raised); padding:9px 12px;
              font:700 10px/1 var(--mono); letter-spacing:.1em; color:var(--act);
              border-bottom:1px solid var(--line-2); }
    .dl-h { padding:9px 10px 6px; font:700 10px/1 var(--mono); letter-spacing:.09em; text-transform:uppercase;
            color:var(--text-3); background:var(--bg); border-bottom:1px solid var(--line); }
    .dl-row { display:flex; gap:8px; align-items:center; min-height:44px; padding:6px 10px; text-decoration:none;
              color:var(--text); border-bottom:1px solid var(--line);
              font-family:var(--mono); font-size:12px; box-sizing:border-box; }
    .dl-row b { font-size:13.5px; font-weight:800; letter-spacing:.02em; }
    .dl-row .dlr { color:var(--text-3); font-size:10.5px; }
    .dl-row .dls { margin-left:auto; color:var(--text-2); font-variant-numeric:tabular-nums; }
    .dl-row:hover { background:var(--hover); }
    .dl-row:focus-visible { outline:2px solid var(--act-bd); outline-offset:-2px; }
    .dl-row.active { background:var(--tint-act); }
    .dl-row.active b { color:var(--act); }
    .dl-sec b { color:var(--accent-2); font-size:12px; letter-spacing:.06em; text-transform:uppercase; }

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
    var res = await fetch(LIVE_PRICE_PROXY.replace(/\\/+$/, "") + "/?symbols=" + tickers.map(encodeURIComponent).join(","), { cache: 'no-store' });
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
      // Index symbols (^IXIC) are points, not dollars — match the server render.
      var unit = (e.getAttribute('data-tkr') || '').charAt(0) === '^' ? "" : "$";
      e.textContent = unit + px.toFixed(2);
      e.style.color = px > snap ? "#54b87f" : (px < snap ? "#e06c6a" : "");
      e.classList.remove('flag-up', 'flag-down');
      var entry = parseFloat(e.getAttribute('data-entry'));
      var stop = parseFloat(e.getAttribute('data-stop'));
      if (!isNaN(stop) && px <= stop) e.classList.add('flag-down');
      else if (!isNaN(entry) && px >= entry) e.classList.add('flag-up');
    });
    if (stamp) {
      stamp.textContent = "🟢 LIVE " + new Date().toLocaleTimeString() + " · " + updated + "/" + tickers.length;
      stamp.style.color = "#54b87f";
    }
  } catch (err) {
    if (stamp) {
      stamp.textContent = "⚠️ fetch failed — set a proxy (see notes)";
      stamp.style.color = "#e06c6a";
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
    // a searched ticker may live only inside a collapsed <details> table
    // (e.g. the 3/4 Lesson Radar) — open those so hits are visible
    if (q) {
      document.querySelectorAll('details.funnel').forEach(function (d) {
        if (d.querySelector('table.rowcards, .card')) d.open = true;
      });
    }
    // v9: the ticker units are .card articles; legacy: table rows. One pass.
    document.querySelectorAll('table tr, article.card').forEach(function (tr) {
      if (tr.tagName === 'TR' && tr.querySelector('th')) return;
      if (tr.closest('.fund-tbl')) return;   // never hide nested fundamentals mini-table rows
      var tk = tr.getAttribute('data-tk');
      var tickerCell = tk ? null : tr.querySelector('.ticker a, .ep-ticker a');
      var okQ = !q || (tk ? tk.toUpperCase().indexOf(q) !== -1
                          : (tickerCell && tickerCell.textContent.toUpperCase().indexOf(q) !== -1));
      var okT = !activeTheme || (tr.getAttribute('data-sector') === activeTheme);
      tr.style.display = (okQ && okT) ? '' : 'none';
    });
    // mirror onto the Desk list (rows + group headers with no visible rows)
    document.querySelectorAll('#desklist .dl-row').forEach(function (a) {
      var tk2 = (a.getAttribute('data-tk') || '').toUpperCase();
      var okQ2 = !q || tk2.indexOf(q) !== -1;
      var okT2 = !activeTheme || (a.getAttribute('data-sector') === activeTheme);
      a.style.display = (okQ2 && okT2) ? '' : 'none';
    });
    document.querySelectorAll('#desklist .dl-h').forEach(function (h) {
      var n = h.nextElementSibling, any = false;
      while (n && !n.classList.contains('dl-h')) {
        if (n.classList.contains('dl-row') && n.style.display !== 'none') { any = true; break; }
        n = n.nextElementSibling;
      }
      h.style.display = any ? '' : 'none';
    });
    // whole card sections whose every card is filtered out: hide the title bar
    // too so a search doesn't leave a column of empty section headers
    var filtering = !!q || !!activeTheme;
    document.querySelectorAll('.secv9').forEach(function (sec) {
      if (!filtering) { sec.style.display = ''; return; }
      var cardsIn = sec.querySelectorAll('article.card');
      // non-card sections (top picks / tracking study) stay; a section whose
      // only content is an empty-state card ("No HVE today") hides on filter.
      if (!cardsIn.length && !sec.querySelector('.card-empty')) { sec.style.display = ''; return; }
      var anyVis = Array.prototype.some.call(cardsIn, function (c) { return c.style.display !== 'none'; });
      sec.style.display = anyVis ? '' : 'none';
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
    // REV 10b: the Screener opts into descending-first via data-sort-desc.
    // On a value column "sort by RS" should lead with RS 99, not with the rows
    // that have no RS (they carry a very negative data-sort so they park last).
    var descFirst = table.hasAttribute('data-sort-desc');
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
          if (rank < 0) keys.push({ th: th, asc: !descFirst });
          else keys[rank].asc = !keys[rank].asc;
        } else {
          // Plain click: collapse to a single-column sort. Re-clicking the lone
          // active column toggles its direction (original single-sort behaviour).
          if (keys.length === 1 && rank === 0) keys[0].asc = !keys[0].asc;
          else keys = [{ th: th, asc: !descFirst }];
        }
        resort();
      });
    });

    // Default sort (2026-07-06 USER): Fwd YoY DESCENDING — fastest-growing names
    // first on every table that has the column. Missing values ship data-sort=-999
    // so they park at the bottom; a header click still replaces this freely.
    var fyTh = headRow.querySelector("th[data-col='fyoy']");
    if (fyTh) { keys = [{ th: fyTh, asc: false }]; resort(); }
  });

  // plan → IBKR-draft jump chips: keep only those whose draft card exists
  // (only the drafted top-3 TOP-PICKS cards carry an id).
  Array.prototype.forEach.call(document.querySelectorAll('a.plan-jump'), function (a) {
    var id = (a.getAttribute('href') || '').slice(1);
    if (!id || !document.getElementById(id)) a.remove();
  });

  // sector chips whose taxonomy matches NO row anywhere would blank every table
  // when tapped — hide them (they got prominent once the chips moved above the
  // tab bar; adversarial review 2026-07-07).
  (function () {
    var secs = {};
    Array.prototype.forEach.call(document.querySelectorAll('table tr[data-sector], article.card[data-sector]'), function (r) {
      secs[r.getAttribute('data-sector')] = 1;
    });
    Array.prototype.forEach.call(document.querySelectorAll('.theme-chip[data-sector]'), function (ch) {
      var s = ch.getAttribute('data-sector');
      if (s && !secs[s]) ch.style.display = 'none';
    });
  })();

  // Floating toggle so touch users can build multi-tier sorts without a Shift key.
  // OFF (default): a tap sorts by one column, exactly as before. ON: each header tap
  // ADDS a tier (primary, then secondary…); tapping an active tier flips its arrow.
  (function () {
    if (!document.querySelector('table.rowcards')) return;   // v9 card layout: no sortable stock tables
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
      btn.style.background = on ? 'var(--accent,#8cb4d6)' : 'var(--surface,#18181b)';
      btn.style.color = on ? 'var(--bg,#111113)' : 'var(--text,#ececea)';
      btn.style.borderColor = 'var(--accent,#8cb4d6)';
    }
    btn.style.cssText = 'position:fixed;z-index:9999;right:12px;bottom:12px;padding:9px 13px;'
      + 'border-radius:20px;border:1px solid var(--accent,#8cb4d6);font:600 13px system-ui,-apple-system,sans-serif;'
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
    // A column added to the schema AFTER the user saved an order is absent from
    // `order`; splice it in at its natural header position (relative to already-placed
    // neighbours) instead of appending it last.
    keys.forEach(function (k, hi) {
      if (desired.indexOf(k) >= 0) return;
      var pos = desired.length;
      for (var j = 0; j < desired.length; j++) {
        if (keys.indexOf(desired[j]) > hi) { pos = j; break; }
      }
      desired.splice(pos, 0, k);
    });
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
  var DEFAULT = ['tk', 'price', 'plan'];
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

(function () {
  // --- mobile sort bridge: a <select> that clicks the hidden header ---
  // The card layout hides <thead>, so column taps are gone on phones. The
  // select reuses the whole existing sort engine by programmatically
  // clicking the matching th; the direction button re-clicks it to flip.
  document.querySelectorAll('table.rowcards').forEach(function (t) {
    var ths = Array.prototype.slice.call(t.querySelectorAll('thead th'));
    if (!ths.length) return;
    var bar = document.createElement('div'); bar.className = 'sortsel-bar';
    var sel = document.createElement('select'); sel.className = 'sortsel';
    var o0 = document.createElement('option'); o0.value = ''; o0.textContent = 'Sort by\u2026';
    sel.appendChild(o0);
    ths.forEach(function (th, i) {
      var o = document.createElement('option'); o.value = String(i);
      o.textContent = th.textContent.replace(/[\u25b2\u25bc\u2039\u203a\u21c5]/g, '').trim();
      sel.appendChild(o);
    });
    var dir = document.createElement('button'); dir.type = 'button';
    dir.className = 'sortdir'; dir.textContent = '\u21c5';
    function cur() { return sel.value === '' ? null : ths[+sel.value]; }
    sel.addEventListener('change', function () { var th = cur(); if (th) th.click(); });
    dir.addEventListener('click', function () { var th = cur(); if (th) th.click(); });
    bar.appendChild(sel); bar.appendChild(dir);
    t.parentNode.insertBefore(bar, t.parentNode.firstChild);
  });
})();

(function () {
  // ---- v9: expand/collapse-all folds (default closed; no-teaser ruling) ----
  var fa = document.getElementById('foldAll');
  if (fa) {
    fa.addEventListener('click', function () {
      var open = fa.getAttribute('data-open') !== '1';
      document.querySelectorAll('#deck details.fold').forEach(function (d) { d.open = open; });
      fa.setAttribute('data-open', open ? '1' : '0');
      fa.textContent = open ? 'collapse all ▾' : 'expand all ▸';
    });
  }

  // ---- REV 10b: Charts / Screener tabs ----
  function showPane(name) {
    document.querySelectorAll('.v9tab').forEach(function (t) {
      t.classList.toggle('on', t.getAttribute('data-pane') === name);
    });
    document.querySelectorAll('.v9pane').forEach(function (p) {
      p.hidden = (p.id !== 'pane-' + name);
    });
    // charts in a pane that was hidden measured 0 wide — repaint on return
    if (name === 'charts' && window.__candle && window.__candle.setTF) {
      window.__candle.setTF(window.__candle.getTF());
    }
  }
  document.querySelectorAll('.v9tab').forEach(function (t) {
    t.addEventListener('click', function () { showPane(t.getAttribute('data-pane')); });
  });
  // a screener ticker jumps back to its chart card
  document.addEventListener('click', function (ev) {
    var a = ev.target.closest && ev.target.closest('.scr-tk');
    if (!a) return;
    ev.preventDefault();
    showPane('charts');
    var id = (a.getAttribute('href') || '').slice(1);
    setTimeout(function () {
      var card = document.getElementById(id);
      if (!card) return;
      var open = card.closest('details');
      if (open) open.open = true;
      card.scrollIntoView({ behavior: 'smooth', block: 'center' });
      card.style.outline = '2px solid var(--act)';
      setTimeout(function () { card.style.outline = ''; }, 1600);
    }, 60);
  });

  // ---- REV 10c: the chart control bar (filter · timeframe · layout · sort) ----
  // Lives outside #deck so desk mode can't hide it. Every control drives BOTH
  // the card deck and the Desk ticker list.
  function mark(btn) {
    btn.parentNode.querySelectorAll('.ctlbtn, .fchip').forEach(function (b) {
      b.classList.toggle('on', b === btn);
    });
  }
  function repaint() {
    if (window.__candle && window.__candle.setTF) window.__candle.setTF(window.__candle.getTF());
  }

  var curGrp = 'all', curView = 'list', curCols = 2;

  // --- section filter: cards, their sections, and the Desk list rows ---
  function applyGroup() {
    document.querySelectorAll('#deck article.card').forEach(function (c) {
      c.classList.toggle('grp-off', curGrp !== 'all' && c.getAttribute('data-grp') !== curGrp);
    });
    document.querySelectorAll('#deck .secv9').forEach(function (sec) {
      var any = sec.querySelector('article.card:not(.grp-off)');
      sec.classList.toggle('grp-empty', !any);
    });
    document.querySelectorAll('#desklist .dl-row[data-grp], #desklist .dl-h[data-grp]')
      .forEach(function (el) {
        var off = curGrp !== 'all' && el.getAttribute('data-grp') !== curGrp;
        el.classList.toggle('grp-off', off);
      });
    if (window.__desk && window.__desk.reselectVisible) window.__desk.reselectVisible();
    repaint();
  }
  document.querySelectorAll('#fchips .fchip').forEach(function (b) {
    b.addEventListener('click', function () {
      mark(b); curGrp = b.getAttribute('data-grp'); applyGroup();
    });
  });

  // --- sort: reorder cards inside each section, then rebuild the Desk list ---
  function applySort(key) {
    document.querySelectorAll('#deck .cardlist').forEach(function (cl) {
      var cards = Array.prototype.slice.call(cl.querySelectorAll(':scope > article.card'));
      if (cards.length < 2) return;
      if (key === 'doc') {
        cards.sort(function (a, b) {
          return (+a.getAttribute('data-ord') || 0) - (+b.getAttribute('data-ord') || 0);
        });
      } else {
        cards.sort(function (a, b) {
          var va = a.getAttribute('data-' + key), vb = b.getAttribute('data-' + key);
          if (va === null && vb === null) return 0;
          if (va === null) return 1;            // missing always parks last
          if (vb === null) return -1;
          return parseFloat(vb) - parseFloat(va);   // best first
        });
      }
      cards.forEach(function (c) { cl.appendChild(c); });
    });
    if (window.__desk && window.__desk.rebuild) window.__desk.rebuild();
    applyGroup();
  }
  // remember the original order so "section order" can restore it
  document.querySelectorAll('#deck .cardlist').forEach(function (cl) {
    Array.prototype.slice.call(cl.querySelectorAll(':scope > article.card'))
      .forEach(function (c, i) { c.setAttribute('data-ord', i); });
  });
  var sortSel = document.getElementById('cardsort');
  if (sortSel) sortSel.addEventListener('change', function () { applySort(sortSel.value); });

  // --- layout: Desk / List / Grid (overrides the automatic width switch) ---
  function applyView() {
    var dens = document.getElementById('densgrp');
    if (dens) dens.style.display = (curView === 'grid') ? 'inline-flex' : 'none';
    if (window.__desk && window.__desk.setDesk) window.__desk.setDesk(curView === 'desk');
    document.querySelectorAll('#deck .cardlist').forEach(function (cl) {
      cl.classList.remove('gcols', 'g2', 'g3', 'g4');
      if (curView === 'grid' && curCols > 1) cl.classList.add('gcols', 'g' + curCols);
    });
    repaint();
  }
  document.querySelectorAll('.ctlbtn[data-view]').forEach(function (b) {
    b.addEventListener('click', function () {
      mark(b); curView = b.getAttribute('data-view'); window.__deskManual = true; applyView();
    });
  });
  document.querySelectorAll('.ctlbtn[data-cols]').forEach(function (b) {
    b.addEventListener('click', function () {
      mark(b); curCols = parseInt(b.getAttribute('data-cols'), 10) || 2; applyView();
    });
  });
  document.querySelectorAll('.ctlbtn[data-tf]').forEach(function (b) {
    b.addEventListener('click', function () {
      mark(b);
      if (window.__candle && window.__candle.setTF) {
        window.__candle.setTF(parseInt(b.getAttribute('data-tf'), 10) || 0);
      }
    });
  });
  // Recount every filter chip from the DOM. The server builds them from the
  // nav entries, which also count non-card sections (Picks, Tracking) and are
  // taken before late drops — so "All" read 1106 against 666 real cards. The
  // rendered cards are the only truth; chips with none left hide themselves.
  (function () {
    var tot = 0;
    document.querySelectorAll('#fchips .fchip').forEach(function (b) {
      var g = b.getAttribute('data-grp');
      if (g === 'all') return;
      var n = document.querySelectorAll('#deck article.card[data-grp="' + g + '"]').length;
      if (!n) { b.style.display = 'none'; return; }
      tot += n;
      var s = b.querySelector('b'); if (s) s.textContent = n;
    });
    var all = document.querySelector('#fchips .fchip[data-grp="all"] b');
    if (all) all.textContent = tot;
  })();

  // reflect the layout the width picked on load, and hide density until needed
  (function () {
    var isDesk = document.body.classList.contains('desk');
    curView = isDesk ? 'desk' : 'list';
    document.querySelectorAll('.ctlbtn[data-view]').forEach(function (b) {
      b.classList.toggle('on', b.getAttribute('data-view') === curView);
    });
    var dens = document.getElementById('densgrp');
    if (dens) dens.style.display = 'none';
  })();
})();

(function () {
  // ---- v9 Foldable Desk: wide screens (unfolded 8-inch / desktop) become
  // master-detail — ticker list right, ONE big card left. The cards never
  // move; the Desk is a class-driven view over the #deck DOM. Narrow keeps
  // the stacked card deck untouched. ----
  var deck = document.getElementById('deck'), list = document.getElementById('desklist');
  if (!deck || !list) return;
  var mq = window.matchMedia('(min-width: 769px)');
  var GROUPS = [['aplus', 'A+'], ['a', 'A'], ['aminus', 'A−'], ['radar', 'Lesson Radar'],
                ['radar3', 'Radar 3/4'], ['min', 'Minervini'], ['tri', 'Trilogy'],
                ['hve', 'HVE'], ['ur', 'U&R'], ['short', 'Short'], ['s4', 'Stage-4'],
                ['nh', '52W New Highs'], ['pull', '52W Pullback'], ['wk', 'Weekly Review']];
  var SECTIONS = [['sec-top', 'Top Picks'], ['sec-study', 'Tracking Study']];
  var rows = [], built = false, activeId = null, pos = null;

  function buildList() {
    if (built) return;
    built = true;
    var frag = document.createDocumentFragment();
    pos = document.createElement('div');
    pos.className = 'dl-pos';
    frag.appendChild(pos);
    GROUPS.forEach(function (g) {
      var cards = deck.querySelectorAll('article.card[data-grp="' + g[0] + '"]');
      if (!cards.length) return;
      var h = document.createElement('div');
      h.className = 'dl-h';
      h.setAttribute('data-grp', g[0]);
      h.textContent = g[1] + ' · ' + cards.length;
      frag.appendChild(h);
      cards.forEach(function (c) {
        if (!c.id) return;
        var a = document.createElement('a');
        a.className = 'dl-row';
        a.href = '#' + c.id;
        a.setAttribute('data-tk', c.getAttribute('data-tk') || '');
        a.setAttribute('data-sector', c.getAttribute('data-sector') || '');
        a.setAttribute('data-grp', g[0]);
        var sc = c.getAttribute('data-score') || '', rk = c.getAttribute('data-risk');
        a.innerHTML = '<b>' + (c.getAttribute('data-tk') || '') + '</b>'
          + (rk ? '<span class="dlr">risk ' + rk + '%</span>' : '')
          + '<span class="dls">' + sc + '</span>';
        a.addEventListener('click', function (ev) { ev.preventDefault(); select(c.id, true); });
        frag.appendChild(a);
        rows.push(a);
      });
    });
    SECTIONS.forEach(function (sdef) {
      if (!document.getElementById(sdef[0])) return;
      var a = document.createElement('a');
      a.className = 'dl-row dl-sec';
      a.href = '#' + sdef[0];
      a.innerHTML = '<b>' + sdef[1] + '</b>';
      a.addEventListener('click', function (ev) { ev.preventDefault(); select(sdef[0], true); });
      frag.appendChild(a);
    });
    list.appendChild(frag);
    list.hidden = false;
  }

  function clearActive() {
    deck.querySelectorAll('.desk-active, .has-active, .desk-full').forEach(function (n) {
      n.classList.remove('desk-active');
      n.classList.remove('has-active');
      n.classList.remove('desk-full');
    });
  }

  function select(id, push) {
    var el = document.getElementById(id);
    if (!el) return;
    clearActive();
    if (el.tagName === 'ARTICLE') {
      el.classList.add('desk-active');
      var p = el.parentElement;
      while (p && p !== deck) {
        if (p.classList && p.classList.contains('secv9')) p.classList.add('has-active');
        if (p.tagName === 'DETAILS') { p.open = true; p.classList.add('has-active'); }
        p = p.parentElement;
      }
      // the Desk stage: repaint this card's chart big (keyed render — the
      // engine skips repeats at the same key)
      var chart = el.querySelector('.cchart[data-c]');
      if (chart && window.__candle) {
        var w = Math.max(360, Math.min((deck.clientWidth || 700) - 28, 980));
        window.__candle.renderInto(chart, { key: 'stage' + w, geom: window.__candle.geom(w, true) });
      }
    } else {
      el.classList.add('has-active');
      el.classList.add('desk-full');
    }
    activeId = id;
    var vis = 0, at = 0;
    rows.forEach(function (a) {
      var on = a.getAttribute('href') === '#' + id;
      a.classList.toggle('active', on);
      if (a.style.display !== 'none') { vis++; if (on) at = vis; }
    });
    if (pos) pos.textContent = at ? (at + ' / ' + vis) : (vis + ' charts');
    var act = list.querySelector('.dl-row.active');
    if (push && act && act.scrollIntoView) act.scrollIntoView({ block: 'nearest' });
    if (push && window.history && history.replaceState) history.replaceState(null, '', '#' + id);
    // Only scroll on an explicit user action (row click / j-k / deep link) —
    // NEVER on the initial matchMedia apply(), or opening the report on a wide
    // screen would yank the page past page-1 down to the deck.
    if (push) {
      var dw = document.getElementById('deskwrap');
      if (dw) window.scrollTo(0, Math.max(0, dw.offsetTop - 6));
    }
  }

  window.__desk = {
    rebuild: function () {                 // re-read card DOM order after a sort
      if (!built) return;
      built = false; rows = [];
      while (list.firstChild) list.removeChild(list.firstChild);
      buildList();
      if (activeId && document.getElementById(activeId)) select(activeId, false);
    },
    setDesk: function (on) { apply(on); },
    // After a filter the active card can be hidden — the Desk stage then goes
    // blank. Re-point it at the first row that survived the filter.
    reselectVisible: function () {
      if (!document.body.classList.contains('desk')) return;
      var cur = deck.querySelector('.card.desk-active');
      if (cur && !cur.classList.contains('grp-off')) return;
      var r = null, all = list.querySelectorAll('.dl-row[data-grp]');
      for (var i = 0; i < all.length; i++) {
        if (!all[i].classList.contains('grp-off')) { r = all[i]; break; }
      }
      if (r) select(r.getAttribute('href').slice(1), false);
    }
  };

  function firstVisibleRow() {
    for (var i = 0; i < rows.length; i++) { if (rows[i].style.display !== 'none') return rows[i]; }
    return null;
  }

  function apply(on) {
    document.body.classList.toggle('desk', on);
    if (on) {
      buildList();
      var id = null;
      if (location.hash.indexOf('#card-') === 0 && document.getElementById(location.hash.slice(1))) {
        id = location.hash.slice(1);
      } else if (activeId && document.getElementById(activeId)) {
        id = activeId;
      } else {
        var f = firstVisibleRow();
        if (f) id = f.getAttribute('href').slice(1);
      }
      if (id) select(id, false);
    } else {
      // fold back to the deck: repaint EVERY chart that was drawn at a stage
      // (Desk) size back to deck geometry — not just the last-active one, or
      // previously-viewed cards stay oversized in the narrow deck.
      if (window.__candle) {
        deck.querySelectorAll('.cchart[data-c]').forEach(function (ch) {
          if (ch.__cc && ch.__cc !== 'deck') window.__candle.renderInto(ch, { key: 'deck' });
        });
      }
      clearActive();
    }
  }
  // REV 10c: once the user picks a layout in the control bar, stop letting a
  // resize yank them back into (or out of) Desk.
  if (mq.addEventListener) mq.addEventListener('change', function (e) { if (!window.__deskManual) apply(e.matches); });
  else if (mq.addListener) mq.addListener(function (e) { if (!window.__deskManual) apply(e.matches); });
  apply(mq.matches);

  document.addEventListener('keydown', function (ev) {
    if (!document.body.classList.contains('desk')) return;
    if (/input|textarea|select/i.test(ev.target.tagName)) return;
    var d = (ev.key === 'j' || ev.key === 'ArrowDown') ? 1 : ((ev.key === 'k' || ev.key === 'ArrowUp') ? -1 : 0);
    if (!d) return;
    ev.preventDefault();
    var vis = rows.filter(function (a) { return a.style.display !== 'none'; });
    if (!vis.length) return;
    var i = -1;
    for (var k = 0; k < vis.length; k++) { if (vis[k].classList.contains('active')) { i = k; break; } }
    var nx = vis[Math.max(0, Math.min(vis.length - 1, i + d))];
    if (nx) select(nx.getAttribute('href').slice(1), true);
  });

  function gotoHash() {
    var id = location.hash.slice(1);
    if (!id || id.indexOf('card-') !== 0) return;
    var el = document.getElementById(id);
    if (!el) return;
    if (document.body.classList.contains('desk')) { select(id, true); return; }  // deep link → scroll
    var p = el.parentElement;
    while (p && p !== deck) { if (p.tagName === 'DETAILS') p.open = true; p = p.parentElement; }
    el.scrollIntoView({ block: 'start' });
  }
  window.addEventListener('hashchange', gotoHash);
  gotoHash();
})();

</script>
"""

# Shared client-side candlestick renderer. Each .cchart div carries its own
# compact OHLCV+overlay JSON (data-c, written by make_candle_chart); ONE
# IntersectionObserver renders charts lazily as they scroll into view, and a
# beforeprint hook renders everything for print/PDF. Colors come from the
# :root tokens so the palette stays single-sourced.
CANDLE_JS = """
<script>
(function () {
  'use strict';
  var css = getComputedStyle(document.documentElement);
  function tok(n, fb) { var v = css.getPropertyValue(n); return v ? v.trim() : fb; }
  var UP = tok('--candle-up', '#9aa7b3'), DN = tok('--candle-down', '#ff5c5c'),
      EN = tok('--up', '#54b87f'), ST = tok('--down', '#e06c6a'),
      GRID = tok('--chart-grid', '#232327'), AXIS = tok('--chart-axis', '#82827c'),
      VMA = tok('--vol-ma', '#9aa4ae'), ACC = tok('--accent', '#8cb4d6'),
      WRN = tok('--warn', '#d3a04d'), SURF = tok('--surface', '#161616'),
      INK = tok('--text', '#e6e6ea'),
      MASPEC = [[10, tok('--ma-fast', '#8cb4d6')], [20, tok('--ma-mid', '#d3a04d')], [50, tok('--ma-slow', '#6b6b74')]];
  var uid = 0;
  var MABBR = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

  // Parameterized geometry (v9): deck charts render at the element's own width
  // so text is 1:1 crisp; the Desk stage re-renders the same payload bigger via
  // renderInto(). Height is width-independent: PT 28 (two-line readout band) +
  // price pane + 8 gap + volume pane + 14 month axis.
  function geom(w, big) {
    var PT = 28, priceH = big ? 400 : 172, volH = big ? 70 : 40;
    var PB = PT + priceH, VT = PB + 8, VB = VT + volH;
    return { W: w, H: VB + 14, PT: PT, PB: PB, VT: VT, VB: VB,
             PL: 4, PR: w - (big ? 46 : 36), fs: big ? 10 : 8, big: !!big };
  }

  function sma(vals, p) {
    var out = [], s = 0, q = [];
    for (var i = 0; i < vals.length; i++) {
      var v = vals[i]; if (v == null) { out.push(null); s = 0; q = []; continue; }
      q.push(v); s += v; if (q.length > p) s -= q.shift();
      out.push(q.length === p ? s / p : null);
    }
    return out;
  }
  function fmt(v) { return v >= 1000 ? v.toFixed(0) : (v >= 100 ? v.toFixed(1) : (v >= 1 ? v.toFixed(2) : v.toFixed(4))); }
  function vfmt(v) { return v >= 1000 ? (v / 1000).toFixed(1) + 'M' : v + 'K'; }

  // ---- timeframe window (USER 2026-07-18: "6 months to 1y history i can
  // select"). Payloads ship ~1 trading year; TF slices the VISIBLE window and
  // hands the dropped bars to the MA warm-up tail (d.p/d.pv) so the 10/20/50
  // ladder stays valid at the left edge. ov levels are anchored at the last bar
  // with a per-BAR slope, so the S/R zone and trendlines re-project themselves
  // across whatever window is showing — no refit, still the engine's own read.
  var TF = 130;                        // default ~6 months; 0 = everything
  function sliceTF(d, tf) {
    var n = d.c.length;
    if (!tf || tf >= n) return d;
    var s = n - tf, o = {}, k;
    for (k in d) { if (Object.prototype.hasOwnProperty.call(d, k)) o[k] = d[k]; }
    o.p  = (d.p  || []).concat(d.c.slice(0, s)).slice(-49);
    o.pv = (d.pv || []).concat(d.v.slice(0, s)).slice(-49);
    o.o = d.o.slice(s); o.h = d.h.slice(s); o.l = d.l.slice(s);
    o.c = d.c.slice(s); o.v = d.v.slice(s);
    var add = 0; for (var i = 0; i <= s; i++) add += (d.dt[i] || 0);
    o.t0 = new Date(Date.parse(d.t0 + 'T00:00:00Z') + add * 86400000).toISOString().slice(0, 10);
    o.dt = d.dt.slice(s); o.dt[0] = 0;
    return o;
  }

  function render(el, opts) {
    opts = opts || {};
    var key = (opts.key || 'deck') + ':' + TF;   // TF is part of the paint key
    if (el.__cc === key) return;      // keyed, not boolean: the Desk repaints at other sizes
    el.__cc = key;
    var d; try { d = JSON.parse(el.getAttribute('data-c')); } catch (e) { return; }
    if (!d || !d.c || d.c.length < 2) return;
    d = sliceTF(d, TF);
    if (!d.c || d.c.length < 2) return;
    var G = opts.geom;
    if (!G) { var cw = el.clientWidth || 0; G = geom(cw >= 120 ? Math.min(Math.round(cw), 1000) : 340, false); }
    var W = G.W, PT = G.PT, PB = G.PB, VT = G.VT, VB = G.VB, PL = G.PL, PR = G.PR, FS = G.fs;
    var n = d.c.length, ov = d.ov || {}, i, k;
    var dates = [], tms = Date.parse(d.t0 + 'T00:00:00Z');
    for (i = 0; i < n; i++) { tms += d.dt[i] * 86400000; dates.push(new Date(tms).toISOString().slice(0, 10)); }
    // TRUE 4-state hollow candles: day COLOUR = close vs PREVIOUS close (bar 0
    // compares to the pre-window tail d.p); body HOLLOW/FILLED = close vs open.
    var dayUp = new Array(n);
    for (i = 0; i < n; i++) {
      var pc0 = i > 0 ? d.c[i - 1] : (d.p && d.p.length ? d.p[d.p.length - 1] : null);
      dayUp[i] = (pc0 != null && d.c[i] != null) ? d.c[i] >= pc0
               : (d.c[i] != null && d.o[i] != null ? d.c[i] >= d.o[i] : true);
    }
    var lo = Infinity, hi = -Infinity;
    for (i = 0; i < n; i++) {
      if (d.l[i] != null && d.l[i] < lo) lo = d.l[i];
      if (d.h[i] != null && d.h[i] > hi) hi = d.h[i];
    }
    if (!isFinite(lo) || !isFinite(hi) || hi <= lo) return;
    var R = hi - lo;
    var mas = MASPEC.map(function (sp) { return { p: sp[0], col: sp[1], v: sma((d.p || []).concat(d.c), sp[0]).slice(-n) }; });
    mas.forEach(function (m) { m.v.forEach(function (v) { if (v != null) { if (v < lo) lo = v; if (v > hi) hi = v; } }); });
    // horizontal overlay levels join the y-scale only when near the bar range
    // (v9: entry/stop lines are gone from the canvas, so 'e'/'s' left the join)
    ['srl', 'srh'].forEach(function (key2) {
      var v = ov[key2];
      if (v != null && v >= lo - 0.15 * R && v <= hi + 0.15 * R) { if (v < lo) lo = v; if (v > hi) hi = v; }
    });
    var pad = (hi - lo) * 0.03 || 0.5; lo -= pad; hi += pad;
    var step = (PR - PL) / n, bw = Math.max(step * 0.62, 1);
    function X(i2) { return PL + (i2 + 0.5) * step; }
    function Y(v) { return PT + (1 - (v - lo) / (hi - lo)) * (PB - PT); }
    function inP(y) { return y >= PT && y <= PB; }
    function px(v) { return Math.round(v) + 0.5; }      // half-pixel snap for crisp hairlines
    var id = 'cc' + (++uid), s = [];
    s.push('<svg viewBox="0 0 ' + W + ' ' + G.H + '" xmlns="http://www.w3.org/2000/svg">');
    s.push('<defs><clipPath id="' + id + '"><rect x="' + PL + '" y="' + PT + '" width="' + (PR - PL) + '" height="' + (PB - PT) + '"/></clipPath></defs>');

    // ---- faint grid + right-gutter price scale (minimal: 4 hairlines) ----
    var gut = [];      // gutter texts; the last-price marker wins collisions
    for (k = 0; k <= 3; k++) {
      var gv = lo + (hi - lo) * k / 3, gy = Y(gv);
      s.push('<line x1="' + PL + '" y1="' + px(gy) + '" x2="' + PR + '" y2="' + px(gy) + '" stroke="' + GRID + '" stroke-width="1" opacity="0.55" shape-rendering="crispEdges"/>');
      gut.push({ y: gy, t: fmt(gv), c: AXIS, w: 400 });
    }

    // ---- SR zone band ----
    if (ov.srl != null && ov.srh != null) {
      var zt = Math.max(PT, Math.min(PB, Y(ov.srh))), zb = Math.max(PT, Math.min(PB, Y(ov.srl)));
      if (zb - zt > 0.5) {
        // one quiet wash — no border lines (2026-07-06 round-2 minimalism)
        s.push('<rect x="' + PL + '" y="' + zt.toFixed(1) + '" width="' + (PR - PL) + '" height="' + (zb - zt).toFixed(1) + '" fill="' + ACC + '" opacity="0.08"/>');
      }
    }
    // ---- diagonals: trendlines (no text — the shapes read themselves) ----
    var f = d.w ? 5 : 1;
    function dline(now, slope, col, dash) {
      if (now == null || slope == null) return;
      var y1 = Y(now - slope * (n - 1) * f), y2 = Y(now);
      s.push('<line x1="' + X(0).toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + X(n - 1).toFixed(1) + '" y2="' + y2.toFixed(1) + '" stroke="' + col + '" stroke-width="1.2" opacity="0.7"' + (dash ? ' stroke-dasharray="' + dash + '"' : '') + ' clip-path="url(#' + id + ')"/>');
    }
    // structural trendlines are DASHED so they read as levels, not as another
    // MA line (they share the accent/warn hues with the MA ladder)
    dline(ov.tsn, ov.tsd, ACC, '6 3');
    dline(ov.trn, ov.trd, WRN, '6 3');
    // ---- moving averages: 10 / 20 / 50 — named by the fixed top legend ----
    mas.forEach(function (m) {
      var seg = [];
      for (i = 0; i < n; i++) {
        if (m.v[i] == null) { if (seg.length > 1) s.push('<polyline points="' + seg.join(' ') + '" fill="none" stroke="' + m.col + '" stroke-width="0.9" opacity="0.55"/>'); seg = []; continue; }
        seg.push(X(i).toFixed(1) + ',' + Y(m.v[i]).toFixed(1));
      }
      if (seg.length > 1) s.push('<polyline points="' + seg.join(' ') + '" fill="none" stroke="' + m.col + '" stroke-width="0.9" opacity="0.55"/>');
    });
    // ---- candles: TRUE HOLLOW, 4 states — colour from dayUp (close vs prev
    // close), hollow/filled from close vs open. Hollow bodies are BACKED with
    // the surface colour so MA lines never show through (rev-4 rule); a
    // near-zero body renders as a doji tick.
    for (i = 0; i < n; i++) {
      if (d.o[i] == null || d.h[i] == null || d.l[i] == null || d.c[i] == null) continue;
      var col = dayUp[i] ? UP : DN, hollow = d.c[i] >= d.o[i];
      var x = X(i), yo = Y(d.o[i]), yc = Y(d.c[i]);
      var bt = Math.min(yo, yc), bb = Math.max(yo, yc), yh = Y(d.h[i]), yl = Y(d.l[i]);
      var wx = px(x);
      if (yh < bt - 0.2) s.push('<line x1="' + wx + '" y1="' + yh.toFixed(1) + '" x2="' + wx + '" y2="' + bt.toFixed(1) + '" stroke="' + col + '" stroke-width="1" shape-rendering="crispEdges"/>');
      if (yl > bb + 0.2) s.push('<line x1="' + wx + '" y1="' + bb.toFixed(1) + '" x2="' + wx + '" y2="' + yl.toFixed(1) + '" stroke="' + col + '" stroke-width="1" shape-rendering="crispEdges"/>');
      if (bb - bt < 1.0) {
        s.push('<line x1="' + (x - bw / 2).toFixed(1) + '" y1="' + px(bt) + '" x2="' + (x + bw / 2).toFixed(1) + '" y2="' + px(bt) + '" stroke="' + col + '" stroke-width="1" shape-rendering="crispEdges"/>');
      } else {
        s.push('<rect x="' + px(x - bw / 2) + '" y="' + px(bt) + '" width="' + bw.toFixed(1) + '" height="' + (bb - bt).toFixed(1) + '" fill="' + (hollow ? SURF : col) + '" stroke="' + col + '" stroke-width="1" shape-rendering="crispEdges"/>');
      }
    }
    // ---- volume pane: quiet filled bars in the day colour + 50-day volume MA.
    // The MA tail comes from d.pv (49 pre-window volumes), so it is valid at the
    // left edge; vmax includes the MA so a quiet window under a loud past stays
    // in-pane instead of clipping the line into the price pane.
    var vma = sma((d.pv || []).concat(d.v), 50).slice(-n);
    var vmax = 0;
    for (i = 0; i < n; i++) if (d.v[i] > vmax) vmax = d.v[i];
    vma.forEach(function (mv) { if (mv != null && mv > vmax) vmax = mv; });
    if (vmax > 0) {
      for (i = 0; i < n; i++) {
        if (d.o[i] == null || d.c[i] == null) continue;   // no OHLC, no bar
        var vh = d.v[i] / vmax * (VB - VT);
        if (vh < 0.8) continue;
        var vcol = dayUp[i] ? UP : DN;
        s.push('<rect x="' + (X(i) - bw / 2).toFixed(1) + '" y="' + (VB - vh).toFixed(1) + '" width="' + bw.toFixed(1) + '" height="' + vh.toFixed(1) + '" fill="' + vcol + '" opacity="0.4"/>');
      }
      var vseg = [];
      for (i = 0; i < n; i++) {
        if (vma[i] == null) continue;
        vseg.push(X(i).toFixed(1) + ',' + (VB - vma[i] / vmax * (VB - VT)).toFixed(1));
      }
      if (vseg.length > 1) s.push('<polyline points="' + vseg.join(' ') + '" fill="none" stroke="' + VMA + '" stroke-width="0.9" opacity="0.8"/>');
    }
    // ---- month axis strip under the volume pane ----
    var lastLx = -99;
    for (i = 1; i < n; i++) {
      if (dates[i].slice(5, 7) === dates[i - 1].slice(5, 7)) continue;
      var mx = X(i);
      s.push('<line x1="' + px(mx) + '" y1="' + (VB + 1) + '" x2="' + px(mx) + '" y2="' + (VB + 4) + '" stroke="' + AXIS + '" stroke-width="1" opacity="0.6" shape-rendering="crispEdges"/>');
      if (mx - lastLx < 26 || mx > PR - 20) continue;
      var mm = +dates[i].slice(5, 7) - 1;
      var mlbl = mm === 0 ? "'" + dates[i].slice(2, 4) : MABBR[mm];
      s.push('<text x="' + (mx + 2).toFixed(1) + '" y="' + (VB + 11.5) + '" font-size="' + (G.big ? 9 : 7.5) + '" font-family="ui-monospace,monospace" fill="' + AXIS + '" opacity="0.9">' + mlbl + '</text>');
      lastLx = mx;
    }
    // ---- last price: dotted hairline + the only bright number in the gutter ----
    var lp = d.c[n - 1];
    if (lp != null) {
      var lpy = Y(lp);
      if (inP(lpy)) {
        s.push('<line x1="' + PL + '" y1="' + px(lpy) + '" x2="' + PR + '" y2="' + px(lpy) + '" stroke="' + AXIS + '" stroke-width="1" stroke-dasharray="1 3" opacity="0.8" shape-rendering="crispEdges"/>');
        gut.push({ y: lpy, t: fmt(lp), c: INK, w: 600, last: 1 });
      }
    }
    // ---- right-gutter price texts: the last-price marker wins collisions.
    // Clamp FIRST, then measure the gap: the top tick's render position is
    // shifted down by the clamp, so comparing raw y would pass ticks that
    // still overprint a last-price sitting just under the window high — the
    // modal breakout shape (found by adversarial review 2026-07-06).
    var lpe = null;
    gut.forEach(function (g) { g.ry = Math.min(Math.max(g.y, PT + 4), PB); if (g.last) lpe = g; });
    gut.filter(function (g) { return g.last || !lpe || Math.abs(g.ry - lpe.ry) > 8; })
      .forEach(function (g) {
        s.push('<text x="' + (PR + 3) + '" y="' + (g.ry + 2.5).toFixed(1) + '" font-size="' + FS + '" font-weight="' + g.w + '" font-family="ui-monospace,monospace" fill="' + g.c + '">' + g.t + '</text>');
      });
    // ---- readout band line 1: MA legend + last close/volume — reserved band
    // y in [0, PT), cannot overlap candles by construction ----
    var lx = PL + 2;
    MASPEC.forEach(function (sp) {
      var lt = 'MA' + sp[0];
      s.push('<text x="' + lx.toFixed(1) + '" y="11" font-size="' + FS + '" font-weight="700" font-family="ui-monospace,monospace" fill="' + sp[1] + '">' + lt + '</text>');
      lx += lt.length * (FS * 0.62) + 9;
    });
    if (lp != null) {
      s.push('<text x="' + (lx + 4) + '" y="11" font-size="' + FS + '" font-family="ui-monospace,monospace" fill="' + AXIS + '">C ' + fmt(lp) + (d.v[n - 1] != null ? ' \\u00b7 V ' + vfmt(d.v[n - 1]) : '') + '</text>');
    }
    // ---- readout band line 2 (dynamic) + crosshair hairline: the floating
    // tooltip is gone — the readout lives in the reserved band, updated via
    // cached tspans (no re-render). Tap pins it; tap again unpins. ----
    s.push('<line id="' + id + 'x" x1="-9" y1="' + PT + '" x2="-9" y2="' + VB + '" stroke="' + AXIS + '" stroke-width="0.7" stroke-dasharray="2 3" opacity="0" pointer-events="none"/>');
    s.push('<text x="' + (PL + 2) + '" y="23" font-size="' + FS + '" font-family="ui-monospace,monospace" fill="' + INK + '"><tspan id="' + id + 'a"></tspan><tspan id="' + id + 'b" dx="6"></tspan><tspan id="' + id + 'c" dx="6" fill="' + AXIS + '"></tspan></text>');
    s.push('</svg>');
    el.innerHTML = s.join('');
    var svg = el.firstChild;
    var tA = document.getElementById(id + 'a'), tB = document.getElementById(id + 'b'),
        tC = document.getElementById(id + 'c'), xh = document.getElementById(id + 'x');
    var pinned = -1;
    function mmd(ds) { return MABBR[+ds.slice(5, 7) - 1] + ' ' + (+ds.slice(8, 10)); }
    function setReadout(xi, cross) {
      if (!tA || d.o[xi] == null || d.h[xi] == null || d.l[xi] == null || d.c[xi] == null) return;
      // keep chg NUMERIC for the sign test — comparing the toFixed() string
      // coerces "-0.0" >= 0 to true and prints "+-0.0%" in green
      var pc2 = xi > 0 ? d.c[xi - 1] : (d.p && d.p.length ? d.p[d.p.length - 1] : null);
      var chg = pc2 ? (d.c[xi] / pc2 - 1) * 100 : null;
      tA.textContent = mmd(dates[xi]) + (d.w ? ' wk' : '') + ' \\u00b7 O ' + fmt(d.o[xi]) + ' H ' + fmt(d.h[xi]) + ' L ' + fmt(d.l[xi]) + ' C ' + fmt(d.c[xi]);
      tB.textContent = chg == null ? '' : (chg >= 0 ? '+' : '') + chg.toFixed(1) + '%';
      tB.setAttribute('fill', chg != null && chg < 0 ? ST : EN);
      tC.textContent = d.v[xi] != null ? 'V ' + vfmt(d.v[xi]) : '';
      if (cross) { var cx = X(xi).toFixed(1); xh.setAttribute('x1', cx); xh.setAttribute('x2', cx); xh.setAttribute('opacity', pinned >= 0 ? '0.9' : '0.55'); }
      else { xh.setAttribute('opacity', '0'); }
    }
    setReadout(n - 1, false);
    function barAt(ev) {
      var r = svg.getBoundingClientRect();
      return Math.max(0, Math.min(n - 1, Math.floor(((ev.clientX - r.left) / r.width * W - PL) / step)));
    }
    svg.addEventListener('pointermove', function (ev) { if (pinned < 0 && ev.pointerType === 'mouse') setReadout(barAt(ev), true); });
    svg.addEventListener('pointerleave', function () { if (pinned < 0) setReadout(n - 1, false); });
    svg.addEventListener('pointercancel', function () { if (pinned < 0) setReadout(n - 1, false); });
    svg.addEventListener('click', function (ev) {
      var xi = barAt(ev);
      if (pinned === xi) { pinned = -1; setReadout(n - 1, false); return; }
      pinned = xi; setReadout(xi, true);
    });
  }
  function renderInto(el, opts) { el.__cc = null; render(el, opts); }
  window.__candle = { render: render, renderInto: renderInto, geom: geom,
                      setTF: setTF, getTF: function () { return TF; } };

  var els = document.querySelectorAll('.cchart[data-c]');

  // Switch every chart to a new window: drop the paint key, repaint what's on
  // screen now, and let the existing debounced scroll pass repaint the rest as
  // they come back into view. The Desk stage re-renders through renderInto().
  function setTF(tf) {
    TF = tf | 0;
    for (var i = 0; i < els.length; i++) { els[i].__cc = null; }
    if (typeof paintNear === 'function') paintNear();
    var stage = document.querySelector('.desk-active .cchart[data-c]');
    if (stage && window.__candle) window.__candle.renderInto(stage, { key: 'stage', geom: geom(stage.clientWidth || 900, true) });
  }
  // Paint any chart already within a screenful now, so the first view is never
  // blank if the IntersectionObserver is slow to fire (or suspended, as it is
  // in some embedded/hidden documents). IO + a debounced scroll pass then keep
  // the rest lazy.
  function paintNear() {
    var vh = window.innerHeight || 800;
    for (var i = 0; i < els.length; i++) {
      var el = els[i];
      if (el.__cc) continue;
      var r = el.getBoundingClientRect();
      if (r.bottom > -600 && r.top < vh + 600) render(el);
    }
  }
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { render(en.target); io.unobserve(en.target); } });
    }, { rootMargin: '500px' });
    els.forEach(function (el) { io.observe(el); });
    var st = null;
    window.addEventListener('scroll', function () {
      if (st) return; st = setTimeout(function () { st = null; paintNear(); }, 120);
    }, { passive: true });
    paintNear();
  } else {
    els.forEach(function (el) { render(el); });
  }
  window.addEventListener('beforeprint', function () { els.forEach(function (el) { render(el); }); });
})();
</script>
"""


def _ind_badge(m: dict) -> str:
    """Industry-group RS chip (Fred6725 rs_industries) for a pick; '' if untagged."""
    pct = m.get("ind_rs")
    if not isinstance(pct, int):
        return ""
    col = "#54b87f" if pct >= IND_RS_STRONG else ("#d3a04d" if pct >= 70 else "#82827c")
    return (f"<br><span class='tag' style='border-color:{col};color:{col};' "
            f"title='Industry-group RS percentile (Fred6725)'>🏭 {esc(m.get('ind_name') or '')} {pct}</span>")


def _meta_details_block(meta_details: List[str]) -> str:
    if not meta_details:
        return ""
    items = "".join(f"<li>{esc(x)}</li>" for x in meta_details)
    return f"<details class='meta'><summary>▸ M.E.T.A. breakdown</summary><ul>{items}</ul></details>"


_SR_FLAG_LABELS = {
    "flip": "flip", "shakeout": "shakeout", "wk_confl": "wkly",
    "lowvol_pullback": "lo-vol", "blue_sky": "blue-sky",
    "barrier_worn": "lid-worn", "prot_worn": "⚠️ worn",
    "extended": "⚠️ extended", "no_headroom": "⚠️ lid",
    "wide_zone_stop": "⚠️ wide", "no_protection": "⚠️ no-zone",
}


def _sr_tp_token(s: dict) -> str:
    """Compact S/R grade chip for the Top Picks card. Never raises."""
    try:
        g = s.get("sr_grade")
        if not g:
            return ""
        col = {"A": "#54b87f", "B": "#d3a04d"}.get(g, "#82827c")
        rr = s.get("sr_rr_wk") if s.get("sr_rr_wk") is not None else s.get("sr_rr")
        rr_txt = f" R:R {rr}" if rr is not None else ""
        return (f" <span class=\"tp-meta\" style=\"color:{col};\" title=\"S/R entry quality "
                f"(informational): zone structure grade per the S&R playbook\">📐 SR {esc(str(g))}{esc(rr_txt)}</span>")
    except Exception:  # noqa: BLE001
        return ""


def _sr_line(m: dict) -> str:
    """One-line S/R entry-quality annotation under the trade plan (shadow mode:
    informational only, filters nothing). Copies the never-raises cell pattern —
    any problem returns '' so a bad pick can't take the report down."""
    try:
        g = m.get("sr_grade")
        if not g:
            return ""
        col = {"A": "#54b87f", "B": "#d3a04d"}.get(g, "#82827c")
        toks = []
        for f in (m.get("sr_flags") or []):
            f = str(f)
            if f.startswith("ma_confl:"):
                toks.append(esc(f.split(":", 1)[1]))
            elif f in _SR_FLAG_LABELS:
                lbl = _SR_FLAG_LABELS[f]
                toks.append(f"<span class='warn-flag'>{esc(_strip_lead_emoji(lbl))}</span>"
                            if lbl.startswith("⚠️") else esc(lbl))
        bits = []
        lo, hi = m.get("sr_prot_lo"), m.get("sr_prot_hi")
        if lo is not None and hi is not None:
            bits.append(esc(f"zone ${lo}–${hi}"))
        if toks:
            bits.append("+".join(toks))
        if m.get("sr_stop_suggest") is not None:
            bits.append(esc(f"zone-stop ${m['sr_stop_suggest']}"))
        rr, rrw = m.get("sr_rr"), m.get("sr_rr_wk")
        if rr is not None:
            bits.append(esc(f"R:R {rr}" + (f" (wk {rrw})" if rrw is not None else "")))
        tip = ("S/R entry quality (informational — filters nothing, printed stop unchanged): "
               "grade from zone structure per the S&R playbook: flip (broken level changed sides), "
               "shakeout (overshoot & reclaim), weekly-zone confluence, MA confluence, low-volume "
               "pullback, headroom to the next opposing zone. zone-stop = suggested stop just "
               "OUTSIDE the protecting zone; R:R measured to the first daily barrier (wk = to the "
               "first weekly barrier).")
        return (f"<div class='edge-line' title='{esc(tip)}'><span class='lbl'>SR</span> "
                f"<span style='color:{col};font-weight:600;'>entry {esc(str(g))}</span> · "
                + " · ".join(bits) + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _support_line(m: dict) -> str:
    """Which verified entry engines back this pick (the Stage-4 gate). Never
    raises - '' when the key is absent (HVE/UR/HTF rows, older snapshots)."""
    try:
        eng = m.get("edge_support")
        if not eng:
            return ""
        names = {"SR": "S/R zone", "PB": "pullback-recovery", "TL": "trendline"}
        label = " + ".join(esc(str(e)) for e in eng)
        tip = ("Backed by: " + ", ".join(names.get(str(e), str(e)) for e in eng)
               + " — the Stage-4 entry-engine gate (each engine verified on its "
                 "tutorial's own trades; details in the lines below)")
        return (f"<div class='edge-line' title='{esc(tip)}'><span class='lbl'>GATE</span> "
                f"<span style='color:#54b87f;font-weight:600;'>backed by {label}</span></div>")
    except Exception:  # noqa: BLE001
        return ""


def _tl_line(m: dict, short: bool = False) -> str:
    """One-line trendline-v2 annotation (shadow mode, filters nothing).
    Never raises - returns '' on any problem. short=True relabels the overhead
    line as the RISK side (a '+7%' there is adverse move, not headroom)."""
    try:
        sup_k, res_k = m.get("tl_sup_kind"), m.get("tl_res_kind")
        if not sup_k and not res_k:
            return ""
        flags = [str(f) for f in (m.get("tl_flags") or [])]
        bits = []
        if sup_k:
            tf = "" if m.get("tl_sup_tf") == "D" else " (wk)"
            tail = "below (cover target)" if short else "below"
            bits.append(esc(f"{sup_k}{tf} ${m.get('tl_sup_at')} "
                            f"{m.get('tl_sup_dist_atr')}ATR {tail}"))
        if res_k:
            tf = "" if m.get("tl_res_tf") == "D" else " (wk)"
            lbl = (f"risk-to-line +{m.get('tl_res_headroom_pct')}%" if short
                   else f"+{m.get('tl_res_headroom_pct')}%")
            bits.append(esc(f"{res_k}{tf} ${m.get('tl_res_at')} {lbl}"))
        show = [f for f in flags if f.startswith(("at_", "fresh_break", "sup_steep",
                                                  "sup_worn", "res_worn", "sup_shakeout"))]
        if show:
            bits.append(esc("+".join(f.replace("sup_steep", "⚠steep-line")
                                     .replace("sup_worn", "⚠line-worn") for f in show)))
        anch = []
        if m.get("tl_sup_anchors"):
            anch.append(f"{m.get('tl_sup_kind')} drawn through "
                        + " & ".join(str(a) for a in m["tl_sup_anchors"]))
        if m.get("tl_res_anchors"):
            anch.append(f"{m.get('tl_res_kind')} drawn through "
                        + " & ".join(str(a) for a in m["tl_res_anchors"]))
        tip = ("Trendline read (informational - filters nothing): nearest governing "
               "diagonal support (UTL/TSL) and overhead line (TRL/DTL = the diagonal "
               "profit-target/lid), two-point swing construction with zone semantics; "
               "steep lines are weak; (wk) = weekly timeframe dominates daily. "
               + ("; ".join(anch) + ". " if anch else "")
               + "Flags: " + (", ".join(flags) if flags else "none"))
        return (f"<div class='edge-line' title='{esc(tip)}'><span class='lbl'>TL</span> "
                + " · ".join(bits) + "</div>")
    except Exception:  # noqa: BLE001
        return ""


# Label valence follows the EVIDENCE, not the tutorial's swing-trading frame:
# ch_study (n=12,871, era-consistent) shows near_top longs OUTPERFORM in this
# breakout universe (+0.096 vs +0.046) - an entry trigger at the lid is the
# lid breaking, not a swing at the rail - so no warning styling there. Only
# fake_break_watch keeps the warning (explicit trap, no contrary evidence).
_CH_FLAG_LABELS = {
    "near_top": "at-channel-top",
    "near_bottom": "at-channel-bottom",
    "fresh_projection": "fresh-rail",
    "proj_worn": "rail-worn",
    "steep": "steep-channel",
    "counter_trend": "counter-trend",
    "fake_break_watch": "⚠️fake-break-watch",
    "top_overshoot_recent": "top-shakeout",
    "bot_overshoot_recent": "bottom-shakeout",
    "top_sr_confluence": "top=SR-zone",
    "higher_tf": "",          # already shown as (wk)
}


def _ch_line(m: dict, short: bool = False) -> str:
    """One-line parallel-channel annotation (tutorial #4, shadow mode, filters
    nothing, NOT in the Stage-4 gate). Never raises - '' on any problem.
    short=True relabels the rails: the bottom is the cover map, the top is
    the risk side."""
    try:
        d = m.get("ch_dir")
        flags = [str(f) for f in (m.get("ch_flags") or [])]
        if not d:
            # a fresh channel break can outlive the channel itself (no alive
            # governing structure left) - the break IS the event, show it
            ev = [f for f in flags
                  if f.startswith("fresh_ch_break") or f == "fake_break_watch"]
            if not ev:
                return ""
            toks = []
            for f in ev:
                lbl = _CH_FLAG_LABELS.get(f, f)
                toks.append(f"<span class='warn-flag'>{esc(_strip_lead_emoji(lbl))}</span>"
                            if lbl.startswith("⚠️") else esc(lbl))
            tip = ("Parallel channel break event (informational - filters nothing): "
                   "a rail was decisively broken within the last 3 bars and no alive "
                   "channel contains the entry. An upside break OUT of a down channel "
                   "is squeeze bait until proven (fake-break-watch). Flags: "
                   + ", ".join(flags))
            return (f"<div class='edge-line' title='{esc(tip)}'><span class='lbl'>CH</span> "
                    f"<span style='color:#d3a04d;font-weight:600;'>break</span> · "
                    + "+".join(toks) + "</div>")
        tf = "" if m.get("ch_tf") == "D" else ",wk"
        col = "#54b87f" if d == "up" else "#e06c6a"
        bits = [f"<span class='lbl'>CH</span> <span style='color:{col};font-weight:600;'>{esc(str(d))}{tf}</span>"]
        if m.get("ch_pos_pct") is not None:
            bits.append(esc(f"pos {m['ch_pos_pct']}%"))
        if m.get("ch_top_at") is not None:
            lbl = (f"risk-to-rail +{m.get('ch_top_headroom_pct')}%" if short
                   else f"+{m.get('ch_top_headroom_pct')}%")
            bits.append(esc(f"top ${m['ch_top_at']} {lbl}"))
        if m.get("ch_bot_at") is not None:
            tail = "cover rail" if short else "floor"
            bits.append(esc(f"{tail} ${m['ch_bot_at']} {m.get('ch_bot_dist_atr')}ATR"))
        toks = []
        for f in flags:
            lbl = _CH_FLAG_LABELS.get(f, f if f.startswith("fresh_ch_break") else "")
            if not lbl:
                continue
            toks.append(f"<span class='warn-flag'>{esc(_strip_lead_emoji(lbl))}</span>"
                        if lbl.startswith("⚠️") else esc(lbl))
        if toks:
            bits.append("+".join(toks))
        drawn = ""
        if m.get("ch_anchors"):
            drawn = (f"Drawn: {esc(str(m.get('ch_base_kind') or 'base'))} through "
                     + " & ".join(str(a) for a in m["ch_anchors"])
                     + (f", copied parallel through {m.get('ch_proj_anchor')}"
                        if m.get("ch_proj_anchor") else "") + ". ")
        tip = ("Parallel channel read (informational - filters nothing, not part of "
               "the Stage-4 gate): 2+1 construction (a two-anchor trendline copied "
               "through one opposite swing pivot); rails are zones; the projected "
               "rail is most reliable on its first touches and wears out from the "
               "3rd on. " + drawn
               + "The channel top is the swing take-profit rail - but in this "
               "breakout universe at-the-lid entries have historically OUTPERFORMED "
               "(era-consistent, ch_study n=12.9k): the trigger at the lid is the "
               "lid breaking. Flags: "
               + (", ".join(flags) if flags else "none"))
        return (f"<div class='edge-line' title='{esc(tip)}'>"
                + " · ".join(bits) + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def _pb2_line(m: dict) -> str:
    """One-line pullback-recovery annotation (shadow mode, filters nothing).
    Never raises - returns '' on any problem."""
    try:
        st = m.get("pb2_state")
        if not st:
            return ""
        col = "#54b87f" if st == "recovery" else "#aecfe8"
        flags = [str(f) for f in (m.get("pb2_flags") or [])]
        tip = ("Pullback-recovery read (informational - filters nothing, printed plan "
               "unchanged): Stage-2 name in a short natural pullback whose mini downtrend "
               "line is breaking (recovery) or pending (setup - the trigger is tomorrow's "
               "buy-stop level). Stop goes under the prior low; risk capped at 10%. "
               "Flags: " + (", ".join(flags) if flags else "none"))
        parts = [f"<span class='lbl'>PB</span> <span style='color:{col};font-weight:600;'>{esc(str(st))}</span>"]
        if m.get("pb2_trigger") is not None:
            parts.append(esc(f"trig ${m['pb2_trigger']}"))
        if m.get("pb2_stop") is not None:
            parts.append(esc(f"stop ${m['pb2_stop']}"))
        if m.get("pb2_risk_pct") is not None:
            parts.append(esc(f"risk {m['pb2_risk_pct']}%"))
        if flags:
            parts.append(esc("+".join(flags)))
        return f"<div class='edge-line' title='{esc(tip)}'>" + " · ".join(parts) + "</div>"
    except Exception:  # noqa: BLE001
        return ""


def _trendline_block(m: dict) -> str:
    """Setup-column trendline badges — REPLACED (USER-RATIFIED 2026-07-04) to
    read the verified v2 engine (tl_* keys) instead of the legacy
    calculate_trendline_analysis lines, which carried no validity/break
    checking (a line price had crashed through still displayed). The legacy
    trendline_data stays computed+persisted for snapshot continuity only.
    Never raises - '' on anything unexpected."""
    try:
        parts = []
        flags = [str(f) for f in (m.get("tl_flags") or [])]
        d_sup = m.get("tl_sup_dist_atr")
        sup_k = m.get("tl_sup_kind")
        if sup_k and isinstance(d_sup, (int, float)):
            if any(f.startswith(("at_UTL", "at_TSL")) for f in flags):
                parts.append(f"<span style='color:#54b87f;'>🎯 {esc(str(sup_k))}: {d_sup:.1f} ATR</span>")
            elif d_sup <= 3:
                parts.append(f"<span style='color:#d3a04d;'>📐 {esc(str(sup_k))}: {d_sup:.1f} ATR</span>")
        if any(f.startswith("fresh_break_up") for f in flags):
            parts.append("<span style='color:#e06c6a;font-weight:bold;'>🔥 Line Break ↑</span>")
        hr = m.get("tl_res_headroom_pct")
        res_k = m.get("tl_res_kind")
        if res_k and isinstance(hr, (int, float)) and 0 < hr <= 50:
            worn = " (worn)" if "res_worn" in flags else ""
            parts.append(f"<span style='color:#aecfe8;'>🎯 {esc(str(res_k))}: +{hr:.1f}%{esc(worn)}</span>")
        if not parts:
            return ""
        return ("<div style='font-size:var(--fs-caption);margin-bottom:6px;font-weight:500;'>"
                + " | ".join(parts) + "</div>")
    except Exception:  # noqa: BLE001
        return ""


def build_filter_funnel(fn: dict, n_aplus: int, n_a: int, n_aminus: int) -> str:
    """How the ~10k US stock + ETF universe narrows to the coil tiers. The survivor
    count after each stage comes from scan_coil's per-stage instrumentation (this run's
    real numbers — see the `funnel` dict it returns). Plain divs so the table JS ignores
    it. universe_total/stage1_fetched are the STOCK leg; the ETF leg (2026-07-15) rides
    in the additive etf_* keys and is summed here for display."""
    fn = fn or {}

    def _n(v):
        return f"{v:,}" if isinstance(v, int) else "—"

    def _sum2(a, b):
        # stock + ETF legs; either may be missing — sum what exists, else None
        vals = [v for v in (a, b) if isinstance(v, int)]
        return sum(vals) if vals else None

    s1_total = _sum2(fn.get("universe_total"), fn.get("etf_universe_total"))
    s1_fetched = _sum2(fn.get("stage1_fetched"), fn.get("etf_stage1_fetched"))
    s1 = _n(s1_total if isinstance(s1_total, int) else s1_fetched)
    n_etf = fn.get("etf_universe_total")
    etf_note = f" (incl. {n_etf} ETFs)" if isinstance(n_etf, int) and n_etf > 0 else ""
    cap_note = ""
    if isinstance(s1_total, int) and isinstance(s1_fetched, int) and s1_total > s1_fetched:
        cap_note = f" <span class='fn-sub'>(top {s1_fetched:,} by ADR fetched)</span>"
    s2 = _n(fn.get("stage2_final") if fn.get("stage2_final") is not None
            else fn.get("stage2_candidates"))
    # 2026-07-06 USER: "too complicated, just simple present, and make it
    # collapsed" — one collapsed <details>, four one-line steps, counts only.
    dropped = _n(fn.get("drop_unsupported")) if fn.get("drop_unsupported") is not None else "–"
    return (
        "<details class='funnel'><summary class='fn-cap'>📋 How this list was built "
        f"<span class='fn-sub'>10,000+ stocks &amp; ETFs → {s1} liquid leaders → {s2} RS leaders near highs → "
        f"A+ {n_aplus} · A {n_a} · A− {n_aminus}</span></summary>"
        "<div class='fn-stage'><div class='fn-body'><div class='fn-title'>1 · Liquid leaders</div>"
        "<div class='fn-crit'>$10+ · ADR ≥ 1.5% · 500k+ volume · above 200MA · $2B+ cap "
        "(ETFs: $2B+ AUM)</div>"
        f"</div><div class='fn-count'>{s1}{esc(etf_note)}{cap_note}</div></div>"
        "<div class='fn-stage'><div class='fn-body'><div class='fn-title'>2 · RS leaders near highs</div>"
        "<div class='fn-crit'>within 20% of the 52-week high · stock or industry-group RS 80+ · "
        "RS line rising swing-over-swing</div>"
        f"</div><div class='fn-count'>{s2}</div></div>"
        "<div class='fn-stage'><div class='fn-body'><div class='fn-title'>3 · Coil tiers</div>"
        "<div class='fn-crit'>tight flag + volume dry-up, graded A+ / A / A−</div>"
        f"</div><div class='fn-count'><span style='color:var(--green);'>A+ {n_aplus}</span> <span class='fn-dot'>·</span> "
        f"<span style='color:var(--yellow);'>A {n_a}</span> <span class='fn-dot'>·</span> "
        f"<span style='color:var(--red);'>A− {n_aminus}</span></div></div>"
        "<div class='fn-stage'><div class='fn-body'><div class='fn-title'>4 · Entry check</div>"
        "<div class='fn-crit'>each pick must pass ≥1 tutorial entry engine (SR zones / pullback-recovery / trendlines)</div>"
        f"</div><div class='fn-count'>{dropped}<span class='fn-sub'> dropped</span></div></div>"
        "</details>"
    )


def build_lesson_radar(radar: List[dict]) -> str:
    """Stage-2 survivors with >=3/4 lesson confluence that made NO coil tier —
    what the tightness/vol filters reject despite textbook lesson stacking.
    DISPLAY-ONLY: never tiered, never tracked, never drafted (IBKR plan and
    latest_setups untouched). 4/4 renders open, 3/4 collapsed."""
    if not radar:
        return ""
    for m in radar:
        missed = m.get("radar_missed_tier", "none")
        why = str(m.get("radar_reason", "")) or "tightness / vol dry-up"
        tag = ("🛰 Radar: no tier (" + why + ")" if missed in (None, "none")
               else "🛰 Radar: no tier (missed " + str(missed) + " — " + why + ")")
        if tag not in m.get("status_labels", []):
            m.setdefault("status_labels", []).append(tag)
    four = [m for m in radar if len(m.get("lesson_confluence") or []) >= 4]
    three = [m for m in radar if len(m.get("lesson_confluence") or []) == 3]
    out = generate_coil_table(four, "Lesson Radar — 4/4 lessons, no tier", "bg-a",
                              subtitle="all four tutorial lessons agree, but the Stage-3 "
                                       "tightness/vol dry-up filters rejected it · informational only")
    if three:
        out += ("<details class='funnel'><summary class='fn-cap'>🎓 Lesson Radar — 3/4 lessons, no tier "
                + f"<span class='fn-sub'>{len(three)} names</span></summary>"
                + generate_coil_table(three, "3/4 lessons — filtered", "bg-aminus",
                                      subtitle="one lesson short of full confluence · informational only")
                + "</details>")
    return out


def generate_coil_table(matches: List[dict], title: str, bg_class: str,
                        subtitle: str = "") -> str:
    if not matches:
        return ""
    sub_html = (f"<span class='section-sub'>{esc(subtitle)}</span>"
                if subtitle else "")
    out = [
        f'<div class="section-title {bg_class}"><span class="tdot"></span>{esc(title)}{sub_html}</div>',
        '<div class="table-container rowcards-container"><table data-schema="coil2" class="rowcards">',
        "<thead><tr><th data-col='tk'>Ticker</th><th data-col='price'>Chart</th><th data-col='plan'>Trade Plan</th>"
        "<th data-col='narr'>Narrative</th>"
        "<th class='num' data-col='adr' title='Average Daily Range — 20-day avg of (High/Low−1), % · how much it typically moves per day (TradingView ADRP, or an equivalent 20-day calc on the external/HTF tabs)'>ADR</th>"
        "<th data-col='rs'>RS</th><th data-col='meta'>M.E.T.A.</th><th class='num' data-col='ants'>ANTS</th>"
        + _MA_YOY_HEADERS +
        "<th data-col='status'>Status (Vol &amp; MA)</th></tr></thead>",
    ]
    for m in matches:
        vol_color = "good" if m["vol_pct"] <= 55 else ("warn" if m["vol_pct"] <= 75 else "bad")
        dist_color = "good" if m["dist_pct"] <= 4.0 else ("warn" if m["dist_pct"] <= 8.0 else "bad")
        dist52_color = "good" if m["dist_52w"] <= 25 else "bad"

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
        trendline_html = _trendline_block(m)
        rs_val = m.get("rs_rating", "N/A")
        rs_asof = m.get("rs_asof")
        rs_mark = (f"<span title='carried-forward from {esc(rs_asof)} — not in today&#39;s RS source' "
                   f"style='color:var(--text-3);font-size:var(--fs-micro);vertical-align:super;'>*</span>"
                   if rs_asof else "")
        spark = m.get("spark", "")
        # ANTS / M.E.T.A. cells + leadership badges come from the shared
        # helpers (also used by Minervini/Trilogy) — one implementation.
        leader_html = _ext_leader_badges(m)

        # Martin pullback plan (additive second trade plan)
        pb_html = ""
        pb_risk = m.get("pb_risk")
        if pb_risk is not None:
            pb_html = (
                "<div class='entry-box' style='border-color:var(--bd-accent);background:var(--tint-accent);margin-top:6px;'>"
                "<span style='color:var(--text-3);font-weight:500;font-size:var(--fs-micro);'>PULLBACK</span><br>"
                f"<span class='entry-text' style='color:var(--accent-2);'>Buy ≈ ${m.get('pb_entry', m['entry'])}</span> <span class='stop-reason'>(to {esc(m['hugging'])})</span><br>"
                f"<span class='stop-text'>Stop: ${m['pb_stop']} <span class='stop-reason'>(~3% under 9EMA)</span></span><br>"
                f"<span class='{_risk_cls(pb_risk)}'>Risk: {pb_risk}%</span></div>"
            )

        geo_line = _geo_line(m)
        out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
            {_tk_cell(m, entry=m['entry'], stop=m['stop'])}
            {_chart_cell(spark, m['close'])}
            <td class="c-plan" data-sort="{m['risk_pct']}">
                <div class="entry-box">
                    {_plan_kicker(m)}
                    <span class="entry-text">Buy: ${m['entry']}</span><br>
                    <span class="stop-text">Stop: ${m['stop']} <span class="stop-reason">({esc(m['stop_reason'])})</span></span><br>
                    <span class="{_risk_cls(m['risk_pct'])}">Risk: {m['risk_pct']}%</span>{geo_line}
                    {_plan_jump(m['ticker'])}
                </div>
                {pb_html}
                {_edge_details(m, [_lessons_line(m), _mtf_line(m), _lbw_line(m), _support_line(m),
                                   _sr_line(m), _pb2_line(m), _tl_line(m), _ch_line(m),
                                   _stage_line(m), _mans_line(m), _oh_line(m),
                                   _pba_line(m), _tc_line(m), _group_line(m)])}
            </td>
            {_narr_cell(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span><br><span class="tag">{esc(m['sector'])}</span>{_ind_badge(m)}''')}
            <td class="num c-stat" data-label="ADR" data-sort="{m['adr']}">{m['adr']}%</td>
            <td class="c-stat" data-label="RS" data-sort="{rs_val if isinstance(rs_val, int) else 0}"><span class="score">{esc(rs_val)}</span>{rs_mark}<br><span class="sub">1M: +{m['perf_1m']}%</span></td>
            {_ext_meta_cell(m)}
            {_ext_ants_cell(m)}
            {_ma_cells(m.get('_ma_dist'))}{_fwd_yoy_cell(m['ticker'])}{_eps_accel_cell(m['ticker'])}
            <td class="c-status" style="text-align:left;" data-sort="{m['vol_pct']}">
                {leader_html}{status_html}{fp_html}{trendline_html}
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
# Trilogy feed: the Downloads original is TCC-protected (launchd runs get
# PermissionError(1) — the documented com.madrry.scanner failure signature), so the
# Trilogy nightly (08:10, runs WITH Downloads access) mirrors the final file to
# feeds/ (step 7d, feed_exports in its pipeline_config). Prefer whichever readable
# copy is FRESHER; the Downloads path still works for manual Terminal runs.
TRILOGY_RTB_PATHS = [
    os.path.expanduser("~/.openclaw/workspace/feeds/trilogy_ready_to_buy.json"),
    os.path.expanduser("~/Downloads/Chart learning project claude/webapp/ready_to_buy.json"),
]
TRILOGY_RTB = TRILOGY_RTB_PATHS[0]  # kept for log messages / back-compat


def _read_trilogy_rtb():
    """Read the freshest readable Trilogy ready_to_buy copy; raise if none is."""
    best, best_mtime, last_exc = None, -1.0, None
    for _p in TRILOGY_RTB_PATHS:
        try:
            _m = os.path.getmtime(_p)
        except OSError as exc:
            last_exc = exc
            continue
        if _m > best_mtime:
            best, best_mtime = _p, _m
    if best is None:
        raise last_exc if last_exc is not None else FileNotFoundError(TRILOGY_RTB_PATHS[0])
    try:
        return _read_json_retry(best), best
    except (OSError, ValueError) as exc:
        # freshest copy unreadable (e.g. Downloads under launchd/TCC) -> try the other(s)
        for _p in TRILOGY_RTB_PATHS:
            if _p == best:
                continue
            try:
                return _read_json_retry(_p), _p
            except (OSError, ValueError) as exc2:
                exc = exc2
        raise exc


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
        r["_adr20"] = round(adr, 2)
        if adr < 1.5:                             # same dead-stock floor as the coil universe (USER 2026-07-06)
            r["_drop_adr"] = True
        _hi, _lo = float(df["High"].iloc[-1]), float(df["Low"].iloc[-1])
        day_range_pct = ((_hi / _lo) - 1.0) * 100.0 if _lo else 0.0   # High/Low basis, same as ADR
        hi52 = float(df["High"].iloc[-252:].max())
        dist_52w = (hi52 - close) / hi52 * 100.0 if hi52 > 0 else 0.0
        v50 = float(df["Volume"].iloc[-50:].mean())
        vol_pct = float(df["Volume"].iloc[-20:].mean() / v50 * 100.0) if v50 > 0 else 100.0
        # v9 cards: surface the two stat-strip values the external tables never
        # printed (Off-52W + dry-up Vol%) — additive keys, legacy tables ignore
        r.setdefault("dist_52w", round(dist_52w, 1))
        r.setdefault("vol_pct", round(vol_pct))
        # REV 10b: the Screener needs these as sortable numbers on the ROW.
        # _perf()/adr were only ever fed into meta_input, so external-engine
        # rows (Minervini/Trilogy) showed "—" for 1M/6M/ADR. setdefault keeps
        # any value the source feed already supplied.
        r.setdefault("perf_1m", round(_perf(21), 1))
        r.setdefault("perf_6m", round(_perf(126), 1))
        r.setdefault("adr", round(adr, 2))
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
            # Candlestick chart (weekly bars for Trilogy). Plan overlay = the
            # engine's own pivot/stop; Trilogy has no explicit stop, so mirror
            # its -8%-of-pivot convention used for _risk_pct above.
            entry = r.get("pivot") if isinstance(r.get("pivot"), (int, float)) \
                else (r.get("ideal_buy") if isinstance(r.get("ideal_buy"), (int, float)) else None)
            stop = r.get("stop") if isinstance(r.get("stop"), (int, float)) else None
            if stop is None and weekly_spark and entry is not None:
                stop = round(float(entry) * 0.92, 2)
            plan = {}
            if entry is not None:
                plan["entry"] = entry
            if stop is not None:
                plan["stop"] = stop
            r["_spark"] = make_candle_chart(df, _chart_plan(plan, df),
                                            max(spark_n, CHART_WINDOW) if not weekly_spark else spark_n,
                                            weekly=weekly_spark)
        except Exception:  # noqa: BLE001
            pass
    # 52wk-high persistence + ANTS + RS-line leadership (self-fetch 2y + ^GSPC).
    try:
        attach_persistence(rows)
    except Exception:  # noqa: BLE001
        pass
    try:
        attach_ants(rows)
    except Exception:  # noqa: BLE001
        pass


def _ext_meta_cell(m: dict) -> str:
    """M.E.T.A. column cell (score badge + expandable details) — shared by the
    coil tiers (meta_score/meta_details keys) and the external engines
    (_meta_score/_meta_details keys)."""
    score = m.get("_meta_score", m.get("meta_score", 0))
    cls = "meta-hi" if score >= 70 else ("meta-md" if score >= 50 else "meta-lo")
    details = m.get("_meta_details", m.get("meta_details", []))
    return (f'<td class="c-stat" data-label="META" data-sort="{score}">'
            f'<span class="meta-pill {cls}">{score}</span>'
            f'{_meta_details_block(details)}</td>')


def _ext_ants_cell(m: dict) -> str:
    """ANTS column cell (today chip + 3M peak), coil-identical."""
    a_lvl = m.get("ants_level", 0)
    a_chain = m.get("ants_chain", 0)
    a_3m_peak = m.get("ants_3m_peak", 0)
    a_3m_days = m.get("ants_3m_days", 0)
    if m.get("ants_ok") and a_lvl > 0:
        _col = {1: "#aecfe8", 2: "#d3a04d", 3: "#d3a04d", 4: "#54b87f", 5: "#2f6b4b"}.get(a_lvl, "#82827c")
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
        _p3col = "#54b87f" if a_3m_peak >= 4 else ("#d3a04d" if a_3m_peak >= 2 else "#82827c")
        p3 = ("<div style='font-size:var(--fs-micro);color:%s;' title='Peak ANTS level in the last ~3 months "
              "over %d active days'>3M %s·%dd</div>" % (_p3col, a_3m_days, esc(_p3lbl), a_3m_days))
    if m.get("ants_ok") and (a_lvl > 0 or a_3m_peak > 0):
        srt = a_lvl * 100000 + a_3m_peak * 1000 + min(a_chain, 999)
    else:
        srt = -1
    return f'<td class="num c-stat" data-label="ANTS" data-sort="{srt}">{today}{p3}</td>'


def _ext_leader_badges(m: dict) -> str:
    """RS▲ Leader + ★ Persistent/Relentless Leader + 🆕 At 52W High, coil-identical."""
    rs_badge = ""
    if m.get("rs_ok"):
        if m.get("rs_nh_before_price"):
            rs_badge = ("<div class='fp-badge' style='border-color:#aecfe8;color:#aecfe8;font-weight:bold;' "
                        "title='RS line near its 1-year high while price has NOT broken out — stealth relative-strength leader'>"
                        "RS▲ ‹ Px</div>")
        elif m.get("rs_new_high"):
            rs_badge = ("<div class='fp-badge' style='border-color:#aecfe8;color:#aecfe8;' "
                        "title='RS line (close/SPX) at or near its 1-year high — relative-strength leader vs the market'>RS▲ Leader</div>")
    nh_html = ""
    if m.get("at_high"):
        nh_html += "<div class='fp-badge' style='border-color:#54b87f;color:#54b87f;'>52W HIGH</div>"
    if m.get("persist_tier"):
        _star = "★★" if m["persist_tier"] == "R" else "★"
        nh_html += (f"<div class='fp-badge' style='border-color:#d3a04d;color:#d3a04d;font-weight:bold;'>"
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


# =============================================================================
# v9 CHART-FIRST CARDS (LAYOUT_V9) — one unified <article class='card'> per
# ticker, used by EVERY section (coil tiers, Minervini, Trilogy, HVE, U&R,
# shorts, 52W highs, pullback monitor, Weekly Review). Anatomy per the
# user-ratified 62-field pick: header (ticker · live price · N/4 badge · tier
# + flag chips) → candle chart (clean: no entry/stop drawn) → TF + gold
# LESSONS lines → one collapsed "Details" fold (plan · stat strip · fundamentals
# · META · context/extras · edges · trendline block). Every block self-omits on
# missing data, so sparse sections (HVE/U&R/shorts/monitor) degrade gracefully.
# Legacy table generators stay untouched for LAYOUT_V9=False rollback.
# =============================================================================

def _v9_chip(label: str, *, color: str = "", border: str = "", title: str = "",
             strong: bool = False) -> str:
    """Bordered uppercase text chip (v9 chip system — no emoji)."""
    st = ""
    if color:
        st += f"color:{color};"
    if border or color:
        st += f"border-color:{border or color};"
    t = f" title='{esc(title)}'" if title else ""
    w = "font-weight:700;" if strong else ""
    return f"<span class='vchip' style='{st}{w}'{t}>{esc(label)}</span>"


def _card_flags_v9(m: dict, tier: str) -> str:
    """Header chip row: tier + the kept identity flags (HTF / LINE / 52W★ /
    persist / RS▲). Warn + footprint tokens live in the fold, not here."""
    chips = []
    if tier:
        chips.append(f"<span class='vchip tier-chip'>{esc(tier)}</span>")
    for lbl in m.get("status_labels", []) or []:
        if lbl.startswith("🚩"):
            chips.append(_v9_chip("HTF", color="var(--red)",
                                  title=_strip_lead_emoji(lbl), strong=True))
    if m.get("lbw_state") == "break":
        chips.append(_v9_chip("LINE", color="var(--yellow)",
                              title="closed above the resistance line (line-break)"))
    if m.get("at_high"):
        chips.append(_v9_chip("52W HIGH", color="var(--green)"))
    if m.get("persist_tier"):
        star = "★★" if m.get("persist_tier") == "R" else "★"
        chips.append(_v9_chip(f"{star} {m.get('persist_label', 'PERSIST')}",
                              color="var(--yellow)",
                              title=f"{m.get('nh_3m', 0)} NH/3M ({m.get('weeks_3m', 0)}w)"))
    if m.get("rs_ok") and m.get("rs_nh_before_price"):
        chips.append(_v9_chip("RS▲ ‹ PX", color="var(--accent-2)",
                              title="RS line near a 1-year high while price has not broken out"))
    elif m.get("rs_ok") and m.get("rs_new_high"):
        chips.append(_v9_chip("RS▲", color="var(--accent-2)",
                              title="RS line at/near its 1-year high"))
    return f"<span class='card-flags'>{''.join(chips)}</span>" if chips else ""


def _fwd_yoy_chip(ticker: str) -> str:
    """Fwd-YoY as a fold chip (reads the same fundamentals cache as the cell)."""
    y, lbl = None, ""
    if _fund is not None and ticker not in _ETF_TICKERS:
        try:
            rec = _fund.get(ticker)
            if rec:
                for r in rec.get("rev", []):
                    if r.get("est"):
                        y, lbl = r.get("yoy"), r.get("lbl", "")
                        break
        except Exception:  # noqa: BLE001
            y = None
    if y is None:
        return ""
    pct = y * 100.0
    col = "var(--green)" if pct > 0.5 else ("var(--red)" if pct < -0.5 else "var(--text-3)")
    return _v9_chip(f"FWD {'+' if pct >= 0 else ''}{pct:.0f}%", color=col,
                    title=f"next forward quarter ({lbl}) consensus revenue YoY")


def _eps_accel_chip(ticker: str) -> str:
    """EPS-acceleration verdict as a fold chip (same cache as the cell)."""
    a = None
    if _fund is not None and ticker not in _ETF_TICKERS:
        try:
            rec = _fund.get(ticker)
            a = rec.get("eps_accel") if rec else None
        except Exception:  # noqa: BLE001
            a = None
    score, verdict = (a.get("accel_score"), a.get("verdict")) if a else (None, None)
    if score is None or verdict is None:
        return ""
    if verdict == "accel":
        arrow, col = ("▲▲" if score >= 20 else "▲"), "var(--green)"
    elif verdict == "decel":
        arrow, col = ("▼▼" if score <= -20 else "▼"), "var(--red)"
    else:
        arrow, col = "→", "var(--text-3)"
    qs = [q for q in (a.get("quarters") or []) if q.get("yoy") is not None]
    latest = f" {'+' if qs[-1]['yoy'] >= 0 else ''}{qs[-1]['yoy'] * 100:.0f}%" if qs else ""
    ttm = " ✦TTM" if a.get("ttm_new_high") else ""
    return _v9_chip(f"EPS {arrow}{latest}{ttm}", color=col,
                    title="O'Neil earnings acceleration: trend in quarterly EPS YoY growth"
                          + (" · TTM EPS at a new high" if a.get("ttm_new_high") else ""))


def _stat_strip_v9(m: dict) -> str:
    """6-cell stat strip: ADR · RS · ANTS 3M · Vol% · Off 52W · Risk%.
    (User ruling: NO vs-MA cells, NO dist-to-EMA, NO RS-1M, NO ANTS-today.)
    Each cell self-omits; '' when nothing is available."""
    cells = []

    def cell(lab, val, title="", cls=""):
        t = (" title='" + esc(title) + "'") if title else ""
        sv = f"sv {cls}" if cls else "sv"
        cells.append(f"<div class='sc'><span class='sl'{t}>{lab}</span>"
                     f"<span class='{sv}'>{val}</span></div>")

    adr = m.get("adr", m.get("_adr20"))
    if adr is not None:
        cell("ADR", f"{adr}%", "Average Daily Range — 20-day avg of (High/Low−1)")
    rs_val = m.get("rs_rating", m.get("rs"))
    if rs_val not in (None, "N/A", ""):
        mark = "*" if m.get("rs_asof") else ""
        cell("RS", f"{esc(rs_val)}{mark}",
             "Fred6725 relative-strength percentile" + (f" · carried from {m.get('rs_asof')}" if m.get("rs_asof") else ""))
    if m.get("ants_ok") and (m.get("ants_3m_peak") or 0) >= 1:
        cell("ANTS 3M", f"{esc(_ANTS_LABELS.get(m.get('ants_3m_peak', 0), ''))}·{m.get('ants_3m_days', 0)}d",
             "peak David-Ryan accumulation level in the last ~3 months")
    if m.get("vol_pct") is not None:
        v = m["vol_pct"]
        cell("VOL", f"{v}%", "20-day avg volume as % of the 50-day (dry-up read)",
             "up" if v <= 55 else ("" if v <= 75 else "dn"))
    d52 = m.get("dist_52w")
    if d52 is not None:
        cell("OFF 52W", f"−{d52}%", "% below the 52-week high",
             "up" if d52 <= 25 else "dn")
    risk = m.get("risk_pct", m.get("_risk_pct"))
    if risk is not None:
        cell("RISK", f"{risk}%", "entry → stop distance",
             {"risk-lo": "up", "risk-hi": "dn"}.get(_risk_cls(risk), ""))
    if not cells:
        return ""
    return f"<div class='stat6'>{''.join(cells)}</div>"


def _fund_block_v9(tk: str) -> str:
    """Fundamentals: Fwd-YoY + EPS-accel chips as the tap-target, expanding to
    the full fundamentals panel (forward Rev/EPS table, acceleration table,
    margin line, revisions) via the existing _fund.details_html wrapper."""
    chips = _fwd_yoy_chip(tk) + _eps_accel_chip(tk)
    if not chips:
        return ""
    inner = f"<div class='chiprow'>{chips}<span class='sumhint'>earnings ▸</span></div>"
    return f"<div class='fundrow'>{_narrative(tk, inner)}</div>"


def _meta_block_v9(m: dict) -> str:
    """META score chip + momentum badge + expandable component breakdown.
    Keys off key-PRESENCE (never the 0-default) so sparse rows show nothing."""
    if "_meta_score" not in m and "meta_score" not in m:
        return ""
    score = m.get("_meta_score", m.get("meta_score"))
    if not score:
        return ""
    cls = "meta-hi" if score >= 70 else ("meta-md" if score >= 50 else "meta-lo")
    parts = [f"<span class='meta-pill {cls}'>META {score}</span>"]
    if score >= 80:
        parts.append(_v9_chip("SUPER MOMENTUM", color="var(--green)"))
    elif score >= 60:
        parts.append(_v9_chip("STRONG MOMENTUM", color="var(--yellow)"))
    details = m.get("_meta_details", m.get("meta_details", []))
    return (f"<div class='chiprow'>{''.join(parts)}</div>"
            + _meta_details_block(details))


# ---- per-section PLAN renderers (all return '' when the plan is absent) ----
def _plan_wrap(kicker_html: str, body: str, *, accent: bool = False) -> str:
    cls = "entry-box vplan" + (" vplan-acc" if accent else "")
    return f"<div class='{cls}'>{kicker_html}{body}</div>"


def _plan_coil_v9(m: dict) -> str:
    if m.get("entry") is None:
        return ""
    geo = _geo_line(m)
    body = (f"<span class='entry-text'>Buy: ${m['entry']}</span><br>"
            f"<span class='stop-text'>Stop: ${m['stop']} <span class='stop-reason'>({esc(m.get('stop_reason', ''))})</span></span><br>"
            f"<span class='{_risk_cls(m.get('risk_pct'))}'>Risk: {m.get('risk_pct')}%</span>{geo}"
            f"{_plan_jump(m['ticker'])}")
    out = _plan_wrap(_plan_kicker(m), body)
    pb_risk = m.get("pb_risk")
    if pb_risk is not None:
        out += ("<div class='entry-box vplan' style='border-color:var(--bd-accent);background:var(--tint-accent);margin-top:6px;'>"
                "<span class='kicker'>PULLBACK</span>"
                f"<span class='entry-text' style='color:var(--accent-2);'>Buy ≈ ${m.get('pb_entry', m['entry'])}</span> <span class='stop-reason'>(to {esc(m.get('hugging', ''))})</span><br>"
                f"<span class='stop-text'>Stop: ${m.get('pb_stop')} <span class='stop-reason'>(~3% under 9EMA)</span></span><br>"
                f"<span class='{_risk_cls(pb_risk)}'>Risk: {pb_risk}%</span></div>")
    return out


def _plan_vcp_v9(m: dict) -> str:
    pivot, stop = m.get("pivot"), m.get("stop")
    if pivot is None:
        return ""
    risk = round((m.get("stop_frac") or 0.0) * 100, 1)
    body = (f"<span class='entry-text'>Buy: ${pivot}</span><br>"
            f"<span class='stop-text'>Stop: ${stop}</span><br>"
            f"<span class='{_risk_cls(risk, 4.0, 8.0)}'>Risk: {risk}%</span>")
    return _plan_wrap("<span class='kicker'>VCP PIVOT</span>", body)


def _plan_buystop_v9(c: dict) -> str:
    ideal, pivot, top = c.get("ideal_buy"), c.get("pivot"), c.get("buy_range_top")
    if ideal is None:
        return ""
    stop = round(pivot * 0.92, 2) if isinstance(pivot, (int, float)) else None
    risk = (round((ideal - stop) / ideal * 100, 1)
            if isinstance(ideal, (int, float)) and isinstance(stop, (int, float)) and ideal else 0.0)
    rng = f" <span class='stop-reason'>(top ${top})</span>" if top is not None else ""
    body = (f"<span class='entry-text'>Buy: ${ideal}</span>{rng}<br>"
            f"<span class='stop-text'>Stop: ${stop} <span class='stop-reason'>(pivot −8%)</span></span><br>"
            f"<span class='{_risk_cls(risk, 8.0, 12.0)}'>Risk: {risk}%</span>")
    return _plan_wrap("<span class='kicker'>BUY-STOP</span>", body)


def _plan_hve_v9(m: dict) -> str:
    if m.get("entry") is None:
        return ""
    rc = "var(--green)" if m.get("risk_pct", 9) <= 4.0 else ("var(--yellow)" if m.get("risk_pct", 9) <= 6.0 else "var(--red)")
    body = (f"<span class='sub'>✓ close range {m.get('close_range', '—')}%</span><br>"
            f"<span class='entry-text'>Buy-Stop: ${m['entry']}</span><br>"
            f"<span class='stop-text'>Stop: ${m.get('stop')} <span class='stop-reason'>({esc(m.get('stop_reason', ''))})</span></span><br>"
            f"<span style='color:{rc};'>Risk: {m.get('risk_pct')}%</span>")
    return _plan_wrap("<span class='kicker'>QM EPISODIC PIVOT</span>", body)


def _plan_ur_v9(m: dict) -> str:
    if m.get("entry") is None:
        return ""
    rc = "var(--green)" if m.get("risk_pct", 9) <= 3.5 else ("var(--yellow)" if m.get("risk_pct", 9) <= 5.0 else "var(--red)")
    body = (f"<span class='sub'>U&amp;R: undercut D{max((m.get('days_since_hve') or 1) - 1, 1)} low, then reclaim</span><br>"
            f"<span class='entry-text' style='color:var(--accent-2);'>Buy: ${m['entry']}</span><br>"
            f"<span class='stop-text'>Stop: ${m.get('stop')} <span class='stop-reason'>({esc(m.get('stop_reason', ''))})</span></span><br>"
            f"<span style='color:{rc};'>Risk: {m.get('risk_pct')}%</span>")
    return _plan_wrap("<span class='kicker'>POST-HVE U&amp;R</span>", body, accent=True)


def _plan_short_para_v9(m: dict) -> str:
    if m.get("entry") is None:
        return ""
    risk = m.get("risk_pct")
    tt = m.get("to_target")
    body = (f"<span class='stop-text'>Trigger: break of ORL / AVWAP retest</span><br>"
            f"<span class='stop-reason'>daily proxy entry ${m['entry']} · stop &gt; day-high ${m.get('stop')}"
            f"{f' ({risk}%)' if risk is not None else ''}</span><br>"
            f"<span style='color:var(--green);'>Cover → 21EMA ${m.get('target')}"
            f"{f' (+{tt}% away)' if tt is not None else ''}</span>"
            "<div class='sub' style='margin-top:4px;'>intraday stop is far tighter (above ORH, ~0.4–2%). "
            "Best on a gap UP (exhaustion); stand aside if it reclaims AVWAP.</div>")
    return _plan_wrap("<span class='kicker' style='color:var(--red);'>SHORT SETUP</span>", body)


def _plan_short_s4_v9(m: dict) -> str:
    if m.get("entry") is None:
        return ""
    tgt = m.get("target_swing")
    tgt_txt = (f"Cover ½ near ${tgt} <span class='stop-reason'>(swing rule, −{m.get('to_target_pct')}%)</span>"
               if tgt else "<span class='stop-reason'>measured move exceeds price — deep-target flag</span>")
    body = (f"<span class='entry-text' style='color:var(--red);'>Short &lt; ${m['entry']}</span><br>"
            f"<span class='stop-text'>Buy-stop ${m.get('stop')} <span class='stop-reason'>"
            f"({esc(m.get('buy_stop_basis') or '')}, {m.get('risk_pct')}%)</span></span><br>"
            f"<span style='color:var(--green);'>{tgt_txt}</span>")
    return _plan_wrap("<span class='kicker' style='color:var(--red);'>STAGE-4 BREAKDOWN</span>", body)


def _plan_nh_v9(m: dict) -> str:
    if m.get("entry") is None:
        return ""
    risk = m.get("risk_pct")
    rc = "var(--green)" if (risk or 9) <= 4 else ("var(--yellow)" if (risk or 9) <= 6 else "var(--red)")
    kick = _plan_kicker(m) if m.get("plan_src") else "<span class='kicker'>CONTINUATION</span>"
    body = (f"<span class='entry-text'>Buy &gt; ${m['entry']}</span><br>"
            f"<span class='stop-text'>Stop: ${m.get('stop')} <span class='stop-reason'>"
            f"({esc(m.get('stop_reason') or '21EMA / −1.5×ADR')})</span></span><br>"
            f"<span style='color:{rc};'>Risk: {f'{risk}%' if risk is not None else 'n/a'}</span>")
    return _plan_wrap(kick, body)


# ---- per-section EXTRAS (context chips/lines unique to the engine) ----
def _x_coil_v9(m: dict) -> str:
    toks = []
    for lbl in m.get("status_labels", []) or []:
        if lbl.startswith("🚩"):
            continue                       # HTF already a header chip
        if lbl.startswith("⚠️"):
            toks.append(f"<span class='warn-flag'>{esc(_strip_lead_emoji(lbl))}</span>")
        else:
            toks.append(esc(_strip_lead_emoji(lbl)))
    for b in m.get("footprint", {}).get("badges", []):
        if b.startswith("⚠️"):
            toks.append(f"<span class='warn-flag'>{esc(_strip_lead_emoji(b))}</span>")
        else:
            toks.append(esc(_strip_lead_emoji(b)))
    out = f"<div class='edge-line'>{' · '.join(toks)}</div>" if toks else ""
    dist = m.get("dist_pct")
    if dist is not None and m.get("hugging"):
        c = "up" if dist <= 4.0 else ("" if dist <= 8.0 else "dn")
        out += f"<div class='sub ctxline'><span class='{c}'>dist to {esc(m['hugging'])}: {dist}%</span></div>"
    return out


def _x_min_v9(m: dict) -> str:
    status = (m.get("status") or "").replace("_", " ")
    trig = "TRIGGER" in status
    vcp = m.get("vcp_score", 0) or 0
    vc = "var(--red)" if vcp >= 85 else ("var(--yellow)" if vcp >= 75 else "var(--text-3)")
    bits = [_v9_chip(status or "—", color="var(--green)" if trig else "var(--yellow)"),
            _v9_chip(f"VCP {vcp}", color=vc, title="Minervini VCP score")]
    ptp = m.get("pct_to_pivot")
    line = ""
    if ptp is not None:
        line = ("<span class='up'>▲ triggered / through pivot</span>" if ptp <= 0
                else f"<span class='warn'>{ptp:.1f}% to pivot</span>")
    offhi = m.get("pct_from_high")
    if isinstance(offhi, (int, float)):
        line += f" · off high {offhi:.1f}%"
    perf6 = m.get("perf6m")
    if isinstance(perf6, (int, float)):
        line += f" · 6M {perf6:+.0f}%"
    return f"<div class='chiprow'>{''.join(bits)}</div>" + (f"<div class='sub ctxline'>{line}</div>" if line else "") + _ext_fp_badges(m)


def _x_tri_v9(c: dict) -> str:
    gcol = {"A": "var(--green)", "B": "var(--accent-2)", "C": "var(--yellow)",
            "D": "var(--red)", "F": "var(--red)"}.get(c.get("grade", ""), "var(--text-3)")
    bits = [_v9_chip(f"GRADE {c.get('grade') or '—'}", color=gcol, strong=True)]
    win20 = c.get("win20_rate")
    if isinstance(win20, (int, float)):
        bits.append(_v9_chip(f"WIN20 {win20 * 100:.0f}%", title="20-day win rate of the reference class"))
    if c.get("monster_tail"):
        bits.append(_v9_chip(f"MONSTER-TAIL d{c.get('monster_tail_decile')}", color="var(--yellow)"))
    status = c.get("status", "")
    if status:
        stc = "var(--red)" if c.get("gated") else ("var(--green)" if "BUYING RANGE" in status else "var(--text-3)")
        bits.append(_v9_chip(status, color=stc))
    line = " · ".join(x for x in [
        (c.get("pattern") or "").replace("_", " "), c.get("family", ""),
        (f"Stage {c.get('stage')}" if c.get("stage") is not None else ""),
        (c.get("rs_line") or "").replace("_", " "),
        (f"likeness Q{c.get('likeness_q')}" if c.get("likeness_q") is not None else ""),
        esc(c.get("checklist") or "")] if x)
    return (f"<div class='chiprow'>{''.join(bits)}</div>"
            + (f"<div class='sub ctxline'>{line}</div>" if line else "") + _ext_fp_badges(c))


def _x_hve_v9(m: dict) -> str:
    bits = [_v9_chip(f"{m.get('rel_vol')}× AVG VOL", color="var(--red)", strong=True)]
    flt = f"{m.get('float_shares')}M" if m.get("float_shares") else "n/a"
    line = f"+{m.get('change')}% today · gap {m.get('gap')}% · float {flt}"
    return f"<div class='chiprow'>{''.join(bits)}</div><div class='sub ctxline'>{line}</div>"


def _x_ur_v9(m: dict) -> str:
    vc = m.get("vol_contraction")
    vcol = "up" if (vc or 99) <= 40 else ("" if (vc or 99) <= 60 else "dn")
    hold = m.get("holding_above_low")
    return ("<div class='sub ctxline'>"
            f"day {m.get('days_since_hve')} since HVE · <span class='{vcol}'>vol {vc:.0f}% of day 1</span> · "
            f"<span class='{'up' if hold else 'dn'}'>above D1 low: {'yes' if hold else 'NO'}</span> · "
            f"D1 high ${m.get('day1_high')}</div>")


def _x_short_para_v9(m: dict) -> str:
    return ("<div class='sub ctxline'>"
            f"<span class='dn'>+{m.get('dist9')}% vs 9EMA</span> · +{m.get('dist21')}% vs 21EMA · "
            f"vol {m.get('vol_ratio')}× · {m.get('gap_ups')} gap-ups · accel +{m.get('accel')}% · "
            f"1M +{m.get('perf_1m')}%</div>")


def _x_short_s4_v9(m: dict) -> str:
    rs_bits = []
    if m.get("rs_pct") is not None:
        rs_bits.append(f"RS {m['rs_pct']}")
    if m.get("rs_line_down"):
        rs_bits.append("RS line ↘")
    return ("<div class='sub ctxline'>"
            f"<span class='dn'>{esc(m.get('wk_stage') or 'S4')}</span> · 30wk MA {(m.get('wk_ma_slope') or 0.0):+.1f}%/5wk · "
            f"{esc(' · '.join(rs_bits))} · {esc(m.get('ind_name') or '')} ind-RS {m.get('ind_rs')}<br>"
            f"shelf ${m.get('support_low')} · peak ${m.get('peak')} · "
            f"1M {m.get('perf_1m')}% · 6M {m.get('perf_6m')}%</div>")


def _x_nh_v9(m: dict) -> str:
    _tag_col = {"GRN": "var(--green)", "YEL": "var(--yellow)", "RED": "var(--red)"}
    bits = [_v9_chip(m.get("label", ""), color=_tag_col.get(m.get("tag"), "var(--green)"))]
    for b in m.get("fp_badges", []) or []:
        bits.append(_v9_chip(b))
    base = (f"{m['base_weeks']:.0f}w / {m['base_depth']:.0f}% deep"
            if m.get("base_depth") is not None else f"{m.get('base_weeks', 0):.0f}w base")
    ext9 = f"{m['ext9']:.0f}%" if m.get("ext9") is not None else "–"
    ext50 = f"{m['ext50']:.0f}%" if m.get("ext50") is not None else "–"
    e50c = "up" if (m.get("ext50") or 0) <= 15 else ("" if (m.get("ext50") or 0) <= 25 else "dn")
    return (f"<div class='chiprow'>{''.join(bits)}</div>"
            f"<div class='sub ctxline'>{esc(base)} · {m.get('higher_lows', 0)} HL · "
            f"<span class='up'>+{ext9} vs 9EMA</span> · <span class='{e50c}'>+{ext50} vs 50EMA</span></div>")


def _x_pull_v9(m: dict) -> str:
    _tag_col = {"GRN": "var(--green)", "RED": "var(--red)", "HOLD": "var(--text-3)"}
    vs50, vsprev, vr = m.get("vs_50"), m.get("vs_prev"), m.get("vol_ratio")
    line = " · ".join(x for x in [
        (f"vs 50MA {vs50:+.1f}%" if vs50 is not None else ""),
        (f"vs prev {vsprev:+.1f}%" if vsprev is not None else ""),
        (f"vol {vr:.2f}× 30d avg" if vr is not None else ""),
        f"{m.get('days_since_high')}d since high",
        f"{m.get('high_count')}× NH"] if x)
    return (f"<div class='chiprow'>{_v9_chip(m.get('status', ''), color=_tag_col.get(m.get('tag'), 'var(--text-3)'))}</div>"
            f"<div class='sub ctxline'>{line}</div>" + _pb2_line(m))


def _x_weekly_v9(m: dict) -> str:
    wk = m.get("perf_w", 0.0)
    note = f"up {wk:+.1f}% this week"
    if m.get("wk_vol_x"):
        note += f" on {m['wk_vol_x']:.1f}× avg weekly volume"
    note += f" · {m.get('off_high', 0):.0f}% off the 52-wk high"
    if m.get("perf_1m") is not None:
        note += f" · 1M {m.get('perf_1m'):+.1f}%"
    return f"<div class='chiprow'>{_v9_chip(f'WEEK {wk:+.1f}%', color='var(--green)')}</div><div class='sub ctxline'>{esc(note)}</div>"


# ---- section specs: what differs in KIND per section; availability is data ----
def _edges_coil_v9(m):
    return [_lbw_line(m), _support_line(m), _sr_line(m), _pb2_line(m), _tl_line(m),
            _ch_line(m), _stage_line(m), _mans_line(m), _oh_line(m),
            _pba_line(m), _tc_line(m), _group_line(m)]


def _edges_nh_v9(m):
    # TC + GRP added 2026-07-18: NH rows DO carry these keys (attach_weinstein
    # runs inside scan_new_highs; attach_industry_rs covers nh green rows), so
    # omitting them was a silent gap. _pba_line is deliberately NOT here —
    # pba_pct is computed in scan_coil only, so it would be dead code.
    return [_sr_line(m), _pb2_line(m), _tl_line(m), _ch_line(m),
            _stage_line(m), _mans_line(m), _oh_line(m),
            _tc_line(m), _group_line(m)]


def _edges_short_v9(m):
    # _group_line added 2026-07-18: attach_industry_rs stamps grp_* on stage4
    # rows (all 10 in the last run) but there was no render site — dead compute.
    # It self-suppresses on parabolic rows, which carry no ind_name.
    return [_dtc_line(m), _group_line(m), _sr_line(m),
            _tl_line(m, short=True), _ch_line(m, short=True)]


def _edges_ctx_v9(m):
    """Context-only edges for sections that receive industry/group data but had
    no edges entry (HVE, U&R). Added 2026-07-18: attach_industry_rs stamps
    grp_* on these rows, so the group read was computed and discarded."""
    return [_group_line(m)]


SECTION_SPECS_V9 = {
    "coil":   dict(plan=_plan_coil_v9, extras=_x_coil_v9, edges=_edges_coil_v9,
                   prices=lambda m: (m.get("close"), m.get("entry"), m.get("stop")),
                   score=lambda m: m.get("meta_score"), risk=lambda m: m.get("risk_pct")),
    "min":    dict(plan=_plan_vcp_v9, extras=_x_min_v9, edges=None,
                   prices=lambda m: (m.get("last_close"), m.get("pivot"), m.get("stop")),
                   score=lambda m: m.get("_meta_score"),
                   risk=lambda m: round((m.get("stop_frac") or 0.0) * 100, 1)),
    "tri":    dict(plan=_plan_buystop_v9, extras=_x_tri_v9, edges=None,
                   prices=lambda m: (m.get("last_close"), m.get("ideal_buy"),
                                     round(m["pivot"] * 0.92, 2) if isinstance(m.get("pivot"), (int, float)) else None),
                   score=lambda m: m.get("grade"), risk=lambda m: m.get("_risk_pct")),
    "hve":    dict(plan=_plan_hve_v9, extras=_x_hve_v9, edges=_edges_ctx_v9,
                   prices=lambda m: (m.get("close"), m.get("entry"), m.get("stop")),
                   score=lambda m: f"{m.get('rel_vol')}×", risk=lambda m: m.get("risk_pct")),
    "ur":     dict(plan=_plan_ur_v9, extras=_x_ur_v9, edges=_edges_ctx_v9,
                   prices=lambda m: (m.get("close"), m.get("entry"), m.get("stop")),
                   score=lambda m: f"D{m.get('days_since_hve')}", risk=lambda m: m.get("risk_pct")),
    "short":  dict(plan=_plan_short_para_v9, extras=_x_short_para_v9, edges=_edges_short_v9,
                   prices=lambda m: (m.get("close"), None, None),
                   score=lambda m: f"+{m.get('dist9')}%", risk=lambda m: m.get("risk_pct")),
    "s4":     dict(plan=_plan_short_s4_v9, extras=_x_short_s4_v9, edges=_edges_short_v9,
                   prices=lambda m: (m.get("close"), None, None),
                   score=lambda m: m.get("wk_stage") or "S4", risk=lambda m: m.get("risk_pct")),
    "nh":     dict(plan=_plan_nh_v9, extras=_x_nh_v9, edges=_edges_nh_v9,
                   prices=lambda m: (m.get("close"), m.get("entry"), m.get("stop")),
                   score=lambda m: m.get("meta_score", m.get("_meta_score")),
                   risk=lambda m: m.get("risk_pct")),
    "pull":   dict(plan=None, extras=_x_pull_v9, edges=None,
                   prices=lambda m: (m.get("close"), None, None),
                   score=lambda m: m.get("rs_rating"), risk=lambda m: None),
    "wk":     dict(plan=None, extras=_x_weekly_v9, edges=None,
                   prices=lambda m: (m.get("close"), None, None),
                   score=lambda m: m.get("rs"), risk=lambda m: None),
}


def _card_v9(m: dict, spec: dict, *, tier: str = "", grp: str = "", seen=None) -> str:
    """One chart-first card. Never raises; every block self-omits on missing data."""
    tk = str(m.get("ticker", "") or "")
    if not tk:
        return ""
    cid = f"card-{grp}-{tk}"
    if seen is not None:
        if cid in seen:
            k = 2
            while f"{cid}-{k}" in seen:
                k += 1
            cid = f"{cid}-{k}"
        seen.add(cid)
    close, entry, stop = spec["prices"](m)
    lp = _lp(tk, close, entry=entry, stop=stop) if close is not None else ""
    score = spec["score"](m)
    risk = spec["risk"](m)
    head = (f"<div class='card-head'>"
            f"<span class='ticker'><a href='https://www.tradingview.com/chart/?symbol={esc(tk)}' target='_blank'>{esc(tk)}</a></span>"
            + lp + _lesson_badge(m) + _card_flags_v9(m, tier) + "</div>")
    chart = m.get("spark") or m.get("_spark") or ""
    above = _mtf_line(m) + _lessons_line(m)
    if above:
        above = f"<div class='cardlines'>{above}</div>"
    theme_bits = " · ".join(x for x in [esc(m.get("theme") or ""), esc(m.get("sector") or "")] if x)
    ctx_head = ""
    if theme_bits or _ind_badge(m):
        ctx_head = f"<div class='sub ctxline'>{theme_bits}{_ind_badge(m)}</div>"
    # Each fold block is independently guarded (2026-07-18 review): the
    # docstring promises "every block self-omits on missing data", but an
    # unguarded None reaching a format string in ANY spec function used to
    # abort the whole report. Now one bad block drops out; the card survives.
    def _blk(fn, *a):
        try:
            return fn(*a) or ""
        except Exception:  # noqa: BLE001
            log.warning("v9 card block %s failed for %s", getattr(fn, "__name__", fn), tk)
            return ""

    fold_parts = [
        # REV 10 (USER 2026-07-18: "i dont need this boxes 'BREAKOUT · SR STOP /
        # Buy / Stop / Risk / PULLBACK …'"): the per-section plan box is no
        # longer rendered on ANY card. The numbers survive where they are read
        # from the chart instead: entry/stop ride the .lp span (data-entry/
        # data-stop) in the head and risk% stays a cell in the stat strip. The
        # _plan_*_v9 builders + SECTION_SPECS_V9['plan'] are kept for rollback.
        _blk(_stat_strip_v9, m),
        _blk(_fund_block_v9, tk),
        _blk(_meta_block_v9, m),
        ctx_head,
        _blk(spec.get("extras") or (lambda _m: ""), m),
        _blk(lambda _m: _edge_details(_m, spec["edges"](_m)) if spec.get("edges") else "", m),
        _blk(_trendline_block, m),
    ]
    inner = "".join(x for x in fold_parts if x)
    fold = (f"<details class='fold'><summary><span>Details</span><span class='chev'>▸</span></summary>"
            f"<div class='foldin'>{inner}</div></details>") if inner else ""
    # REV 10c: numeric keys so the control bar can sort cards (and the Desk
    # list) by META/RS/ADR/1M/6M. Same normaliser the Screener uses, because
    # sections name the same idea differently (meta_score vs _meta_score …).
    _srt = {"meta": _scr_num(m, "meta_score", "_meta_score"),
            "rs": _scr_num(m, "rs_rating", "rs"),
            "adr": _scr_num(m, "adr", "_adr20"),
            "p1m": _scr_num(m, "perf_1m"),
            "p6m": _scr_num(m, "perf_6m")}
    attrs = (f" data-tk='{esc(tk)}' data-grp='{esc(grp)}' data-sector='{esc(m.get('sector', '') or '')}'"
             + "".join(f" data-{k}='{v:.4f}'" for k, v in _srt.items() if v is not None)
             + (f" data-score='{esc(score)}'" if score not in (None, "") else "")
             + (f" data-risk='{risk}'" if risk not in (None, "") else ""))
    return (f"<article class='card' id='{cid}'{attrs}>"
            + head + chart + above + fold + "</article>")


# Short per-section labels for the Screener's "Sec" column + the row register
# that _section_v9 fills as it renders (REV 10b).
_SCR_LABELS = {
    "aplus": "A+", "a": "A", "aminus": "A−", "radar": "Radar", "radar3": "Radar",
    "min": "Minervini", "tri": "Trilogy", "hve": "HVE", "ur": "U&R",
    "short": "Short", "s4": "Stage-4", "nh": "52W High", "pull": "Pullback",
    "wk": "Weekly",
}
_V9_SECTION_ROWS: List[Tuple[str, str, list]] = []


def _section_v9(key: str, title: str, rows, spec: dict, *, subtitle: str = "",
                bg: str = "", tier: str = "", grp: str = "", lead: str = "",
                tail: str = "", empty: str = "") -> str:
    """A stacked v9 section: title bar + card list. Keeps the .section-title +
    .table-container adjacency so the existing collapse JS/CSS still work."""
    grp = grp or key
    # REV 10b: every card section funnels through here, so this is the one place
    # that sees ALL rows (incl. Minervini/Trilogy, whose generators only hand
    # back HTML). The Screener tab is built from this register.
    if rows:
        _V9_SECTION_ROWS.append((grp, _SCR_LABELS.get(key, key.upper()), list(rows)))
    seen: set = set()
    cards = "".join(_card_v9(m, spec, tier=tier, grp=grp, seen=seen) for m in rows)
    if not cards and not empty and not lead:
        return ""
    if not cards and empty:
        cards = f"<div class='card card-empty'>{esc(empty)}</div>"
    sub_html = f"<span class='section-sub'>{esc(subtitle)}</span>" if subtitle else ""
    return (f"<section class='secv9' id='sec-{key}'>"
            f"<div class='section-title {bg}'><span class='tdot'></span>{esc(title)}"
            f"<span class='sec-n'>{len(rows)}</span>{sub_html}</div>"
            f"<div class='table-container cardlist'>{lead}{cards}{tail}</div></section>")


def _eps_yoy_num(ticker: str) -> Optional[float]:
    """Latest reported-quarter EPS YoY % as a sortable NUMBER for the screener
    (_eps_accel_cell returns a whole <td>, no good here).

    STRICTLY CACHE-ONLY. madrry_fundamentals.get() does a synchronous per-ticker
    TradingView POST + Yahoo scrape + cache flush on a miss (see the note by the
    import); the screener asks for ~700 tickers, and routing that through get()
    added ~190s to the run. Read the prefetch-warmed cache directly instead and
    show '—' for anything the prefetch budget didn't reach."""
    if _fund is None or ticker in _ETF_TICKERS:
        return None
    try:
        rec = _fund._load_cache().get(str(ticker).strip().upper())
        if not rec or not rec.get("ok"):
            return None
        qs = [q for q in ((rec.get("eps_accel") or {}).get("quarters") or [])
              if q.get("yoy") is not None]
        return qs[-1]["yoy"] * 100.0 if qs else None
    except Exception:  # noqa: BLE001
        return None


# REV 10b (USER 2026-07-18: "your list of stock is so hard to see, a list of 600
# stocks … at least let me sort by rs/ meta score/eps etc." + "i cant believe
# that you make a whole 6xx row of scrolling … at least make different tabs").
# ONE flat, sortable table over every ticker in the report. Rendered server-side
# as a plain <table> so the existing click-to-sort engine binds to it for free
# (it walks every table, reads td[data-sort], and supports shift/multi-sort);
# data-tk + data-sector make the existing global search filter it too.
_SCR_COLS = [
    ("Ticker", "tl"), ("Sec", "tl"), ("Last", "num"), ("1M %", "num"),
    ("6M %", "num"), ("RS", "num"), ("META", "num"), ("EPS YoY", "num"),
    ("ADR", "num"), ("Vol %", "num"), ("Off 52W", "num"), ("Sector", "tl"),
]


def _scr_num(m: dict, *keys):
    """First numeric value among `keys` (sections name the same idea differently:
    close/last_close, meta_score/_meta_score, rs_rating/rs, dist_52w/off_high)."""
    for k in keys:
        v = m.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _scr_cell(v, *, pct=False, dp=1, colour=False) -> str:
    if v is None:
        return "<td class='num' data-sort='-99999'>—</td>"
    cls = "num"
    if colour:
        cls += " val-green" if v > 0 else (" val-red" if v < 0 else "")
    txt = f"{v:+.{dp}f}%" if (pct and colour) else (f"{v:.{dp}f}%" if pct else f"{v:,.{dp}f}")
    return f"<td class='{cls}' data-sort='{v:.4f}'>{txt}</td>"


def build_screener_v9(groups=None) -> str:
    """Flat sortable table over every card row rendered this run. `groups`
    defaults to the register `_section_v9` filled while building the sections."""
    body, n = [], 0
    seen_tk: set = set()
    # Flatten first so the table can OPEN on something useful (META desc, blanks
    # last) instead of section order — a header click still replaces this freely.
    flat = [(grp, label, m)
            for grp, label, rows in (groups if groups is not None else _V9_SECTION_ROWS)
            for m in (rows or [])]
    def _meta_of(t):
        return _scr_num(t[2], "meta_score", "_meta_score")
    flat.sort(key=lambda t: (_meta_of(t) is None, -(_meta_of(t) or 0.0)))
    for grp, label, m in flat:
        tk = str(m.get("ticker", "") or "")
        if not tk or (grp, tk) in seen_tk:
            continue
        seen_tk.add((grp, tk))
        n += 1
        sec = esc(m.get("sector") or "")
        body.append(
            f"<tr data-tk='{esc(tk)}' data-sector='{sec}'>"
            f"<td class='tl'><a href='#card-{esc(grp)}-{esc(tk)}' class='scr-tk'>{esc(tk)}</a></td>"
            f"<td class='tl'><span class='scr-sec'>{esc(label)}</span></td>"
            + _scr_cell(_scr_num(m, "close", "last_close"), dp=2)
            + _scr_cell(_scr_num(m, "perf_1m"), pct=True, colour=True)
            + _scr_cell(_scr_num(m, "perf_6m"), pct=True, colour=True, dp=0)
            + _scr_cell(_scr_num(m, "rs_rating", "rs"), dp=0)
            + _scr_cell(_meta_of((grp, label, m)), dp=0)
            + _scr_cell(_eps_yoy_num(tk), pct=True, colour=True, dp=0)
            + _scr_cell(_scr_num(m, "adr", "_adr20"), pct=True)
            + _scr_cell(_scr_num(m, "vol_pct", "rel_vol"), pct=True, colour=True, dp=0)
            + _scr_cell(_scr_num(m, "dist_52w", "off_high"), pct=True, dp=1)
            + f"<td class='tl'>{sec or '—'}</td></tr>")
    if not body:
        return "<div class='sub'>No rows to screen.</div>"
    head = "".join(f"<th class='{c}'>{esc(t)}</th>" for t, c in _SCR_COLS)
    return (f"<div class='scr-head'><b>{n}</b> tickers · tap any column to sort "
            f"(shift-click or the Multi-sort toggle adds a tier) · tap a ticker to jump to its chart</div>"
            f"<div class='table-container scr-wrap'><table class='screener' data-sort-desc>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


def _secnav_v9(entries) -> str:
    """Sticky anchor-chip nav (the one nav contract — replaces the 8-tab bar)
    + the expand/collapse-all-folds control."""
    chips = []
    for key, label, count in entries:
        n = f"<b>{count}</b>" if count not in (None, 0) else ""
        chips.append(f"<a class='navchip' href='#sec-{key}'>{esc(label)}{n}</a>")
    # REV 10 (USER 2026-07-18: "make the chart has 6 months to 1y history 'i can
    # select'" + "make another grid … have a button for me to choose as well").
    # Both controls are global: they retint every chart / the whole deck at once.
    return ("<div class='secnav' id='secnav'><div class='secnav-in'>"
            + "".join(chips)
            + "<button class='navchip' id='foldAll' type='button' data-open='0'>expand all ▸</button>"
            + "</div></div>")


def _chartctl_v9(entries) -> str:
    """REV 10c: the chart control bar. Lives OUTSIDE #deck, because
    `body.desk #deck > * { display:none }` hid the previous in-secnav version
    entirely on desktop/foldable — the user saw no timeframe, grid, filter or
    sort at all there. Section filter + sort drive the cards AND the Desk list."""
    total = sum(int(c or 0) for _k, _l, c in entries)
    chips = [f"<button class='fchip on' data-grp='all' type='button'>All"
             f"<b>{total}</b></button>"]
    for key, label, count in entries:
        if not count:
            continue
        chips.append(f"<button class='fchip' data-grp='{esc(key)}' type='button'>"
                     f"{esc(label)}<b>{count}</b></button>")
    tf = ("<span class='ctlgrp' role='group' aria-label='Chart timeframe'>"
          "<button class='ctlbtn' data-tf='63' type='button'>3M</button>"
          "<button class='ctlbtn on' data-tf='130' type='button'>6M</button>"
          "<button class='ctlbtn' data-tf='0' type='button'>1Y</button></span>")
    view = ("<span class='ctlgrp' role='group' aria-label='Layout'>"
            "<button class='ctlbtn' data-view='desk' type='button' title='Desk — list + one big chart'>&#9707;</button>"
            "<button class='ctlbtn on' data-view='list' type='button' title='One chart per row'>&#9776;</button>"
            "<button class='ctlbtn' data-view='grid' type='button' title='Grid of charts'>&#9638;</button></span>")
    gr = ("<span class='ctlgrp' id='densgrp' role='group' aria-label='Grid density'>"
          "<button class='ctlbtn on' data-cols='2' type='button'>2</button>"
          "<button class='ctlbtn' data-cols='3' type='button'>3</button>"
          "<button class='ctlbtn' data-cols='4' type='button'>4</button></span>")
    srt = ("<select id='cardsort' class='ctlsel' aria-label='Sort charts'>"
           "<option value='meta'>Sort: META</option>"
           "<option value='rs'>Sort: RS</option>"
           "<option value='adr'>Sort: ADR</option>"
           "<option value='p1m'>Sort: 1M %</option>"
           "<option value='p6m'>Sort: 6M %</option>"
           "<option value='doc'>Sort: section order</option></select>")
    return ("<div class='chartctl' id='chartctl'>"
            f"<div class='ctl-row fchips' id='fchips'>{''.join(chips)}</div>"
            f"<div class='ctl-row'><span class='ctl-lbl'>TF</span>{tf}{view}{gr}{srt}</div>"
            "</div>")


# ---- v9 section generators (card versions of the tabbed tables). Each does
# its OWN data prep (copied from the legacy generator) because exactly one
# path runs per report — legacy generators stay untouched for rollback. ----

_UR_NOTE_V9 = (
    "<div class='vnote'>"
    "<div class='vnote-t'>POST-HVE U&amp;R STRATEGY (like MXL Apr 28)</div>"
    "<div class='vnote-b'><b>Setup:</b> day 2–5 after a massive HVE breakout; volume dries to &lt;50% of day 1 "
    "(institutional holding). <b>Trigger:</b> price undercuts the previous day low, flushes stops, then violently "
    "reclaims. <b>Entry:</b> on the reclaim with volume expansion. <b>Stop:</b> 1% below the undercut-day wick low "
    "(typically 2–4% risk).</div></div>")

_S4_SUB_V9 = ("below a flat/declining 30-week MA · RS ≤25 or falling RS line · industry RS ≤25 · "
              "at/just breaking the shelf · DTC ≥10× excluded · informational — never drafted")


def generate_minervini_cards(market_modifier: float = 1.0) -> Tuple[str, int]:
    """v9 card version of the Minervini buy list (own data prep; runs INSTEAD
    of generate_minervini_table when LAYOUT_V9)."""
    try:
        files = [f for f in os.listdir(MINERVINI_DIR)
                 if f.startswith("buy_list_") and f.endswith(".json")]
    except OSError:
        files = []
    if not files:
        return _section_v9("min", "MINERVINI — VCP / SEPA BUY LIST", [],
                           SECTION_SPECS_V9["min"],
                           empty="Minervini engine: no buy_list file found."), 0
    latest = sorted(files)[-1]
    asof = latest[len("buy_list_"):-len(".json")]
    try:
        rows = _read_json_retry(os.path.join(MINERVINI_DIR, latest))
    except (OSError, ValueError) as exc:
        log.error("Minervini feed unreadable (%s): %r", latest, exc)
        return _section_v9("min", "MINERVINI — VCP / SEPA BUY LIST", [],
                           SECTION_SPECS_V9["min"],
                           empty=f"Minervini engine: could not read {latest}."), 0
    if not rows:
        return _section_v9("min", "MINERVINI — VCP / SEPA BUY LIST", [],
                           SECTION_SPECS_V9["min"],
                           empty=f"Minervini engine: 0 picks on {asof}."), 0
    for m in rows:
        m["_risk_pct"] = round((m.get("stop_frac") or 0.0) * 100, 1)
    _enrich_external_rows(rows, weekly_spark=False, spark_field="Close", spark_n=60,
                          market_modifier=market_modifier, spark_ma_spec=_MA_SPEC_DARK)
    rows = [m for m in rows if not m.get("_drop_adr")]
    _prefetch_fundamentals([m.get("ticker") for m in rows], budget_s=30.0)
    return _section_v9("min", "MINERVINI — VCP / SEPA BUY LIST", rows,
                       SECTION_SPECS_V9["min"],
                       subtitle=f"as of {asof} · trade plan from minervini_engine · "
                                f"META/ANTS/chart computed by MADRRY"), len(rows)


def generate_trilogy_cards(limit=None, market_modifier: float = 1.0) -> Tuple[str, int]:
    """v9 card version of the Trilogy buy-stop list (weekly charts)."""
    try:
        data, _rtb_src = _read_trilogy_rtb()
        log.info("Trilogy feed source: %s", _rtb_src)
    except (OSError, ValueError) as exc:
        log.error("Trilogy feed unreadable (tried %s): %r", TRILOGY_RTB_PATHS, exc)
        return _section_v9("tri", "TRILOGY — O'NEIL BUY-STOP LIST", [],
                           SECTION_SPECS_V9["tri"],
                           empty="Trilogy: ready_to_buy.json not found / unreadable."), 0
    cands = data.get("candidates", []) or []
    asof = data.get("asof", "")
    total = len(cands)
    if not cands:
        return _section_v9("tri", "TRILOGY — O'NEIL BUY-STOP LIST", [],
                           SECTION_SPECS_V9["tri"],
                           empty=f"Trilogy: 0 candidates on {asof}."), 0
    shown = cands if limit is None else cands[:limit]
    for c in shown:
        _piv = c.get("pivot")
        _stp = round(_piv * 0.92, 2) if isinstance(_piv, (int, float)) else None
        _ib = c.get("ideal_buy")
        c["_risk_pct"] = (round((_ib - _stp) / _ib * 100, 1)
                          if isinstance(_ib, (int, float)) and isinstance(_stp, (int, float)) and _ib else 10.0)
    _enrich_external_rows(shown, weekly_spark=True, market_modifier=market_modifier,
                          spark_ma_spec=_MA_SPEC_10W)
    shown = [c for c in shown if not c.get("_drop_adr")]
    _prefetch_fundamentals([c.get("ticker") for c in shown], budget_s=30.0)
    return _section_v9("tri", "TRILOGY — O'NEIL BUY-STOP LIST", shown,
                       SECTION_SPECS_V9["tri"],
                       subtitle=f"as of {asof} · weekly charts · trade plan from trilogy webapp · "
                                f"META/ANTS computed by MADRRY"), total


def build_lesson_radar_v9(radar: List[dict]) -> str:
    """v9 card version of the Lesson Radar (4/4 open, 3/4 collapsed).
    Keeps the legacy status-label mutation (the radar tag is data)."""
    if not radar:
        return ""
    for m in radar:
        missed = m.get("radar_missed_tier", "none")
        why = str(m.get("radar_reason", "")) or "tightness / vol dry-up"
        tag = ("🛰 Radar: no tier (" + why + ")" if missed in (None, "none")
               else "🛰 Radar: no tier (missed " + str(missed) + " — " + why + ")")
        if tag not in m.get("status_labels", []):
            m.setdefault("status_labels", []).append(tag)
    four = [m for m in radar if len(m.get("lesson_confluence") or []) >= 4]
    three = [m for m in radar if len(m.get("lesson_confluence") or []) == 3]
    out = _section_v9("radar", "LESSON RADAR — 4/4 LESSONS, NO TIER", four,
                      SECTION_SPECS_V9["coil"], bg="bg-a", grp="radar",
                      subtitle="all four tutorial lessons agree, but the tightness/vol filters "
                               "rejected it · informational only")
    if three:
        seen: set = set()
        cards3 = "".join(_card_v9(m, SECTION_SPECS_V9["coil"], grp="radar3", seen=seen)
                         for m in three)
        out += ("<details class='funnel'><summary class='fn-cap'>Lesson Radar — 3/4 lessons, no tier "
                + f"<span class='fn-sub'>{len(three)} names</span></summary>"
                + f"<div class='table-container cardlist'>{cards3}</div></details>")
    return out


def generate_new_highs_cards(nh: dict) -> str:
    """v9 card version of the 52-week-high leadership section."""
    green = nh.get("green", [])
    clusters = nh.get("clusters", [])
    total = nh.get("total", 0)
    if total == 0:
        return ""
    n_persist = sum(1 for m in green if m.get("persist_tier"))
    lead = ""
    if clusters:
        chips = "".join(
            f"<span class='theme-chip' data-sector='{esc(s)}' style='border-color:var(--green);'>"
            f"{esc(s)} <b style='color:var(--green);'>×{n}</b></span>" for s, n in clusters)
        lead = (f"<div class='vnote vnote-green'><div class='vnote-t'>Sectors making COLLECTIVE new highs "
                f"({total} total · ≥3 = cluster)</div>"
                f"<div class='hot-themes' style='margin:6px 0 0;'>{chips}</div></div>")
    elif total:
        lead = (f"<div class='sub' style='margin:0 0 10px;'>{total} new 52-wk highs today · "
                "no single sector reached a ≥3 cluster.</div>")
    return _section_v9("nh", f"NEW 52-WEEK HIGHS — LEADERSHIP", green,
                       SECTION_SPECS_V9["nh"], bg="bg-aplus", lead=lead,
                       subtitle=f"{total} new highs · {n_persist}★ persistent · "
                                f"{len(clusters)} sector clusters · constructive + persistent names carded",
                       empty="No constructive or persistent names today — the rest of the new highs "
                             "are extended or still developing.")


def generate_nh52_monitor_cards(pullbacks: List[dict], monitored: List[dict]) -> str:
    """v9 card version of the 52wk-high pullback monitor."""
    n_pull = len(pullbacks)
    n_break = sum(1 for m in monitored if m.get("tag") == "RED")
    lead = ""
    if pullbacks:
        lead = ("<div class='vnote vnote-green'><div class='vnote-t'>Low-volume pullback</div>"
                "<div class='vnote-b'>price slipped below its 50-day MA or the prior close while volume "
                "dried up below its 30-day average — supply exhausting, a constructive continuation watch."
                "</div></div>")
    return _section_v9("pull", "52-WEEK-HIGH PULLBACK MONITOR", monitored,
                       SECTION_SPECS_V9["pull"], lead=lead,
                       subtitle=f"new highs from the last {NH52_WATCH_DAYS} sessions, re-checked each run · "
                                f"{n_pull} low-vol pullbacks · {n_break} high-vol breakdowns",
                       empty="No names on the 52wk-high monitor yet — they accumulate as the daily scan "
                             "prints fresh new highs.")


def generate_weekly_review_cards(data: dict) -> str:
    """v9 card version of the IBD-style Weekly Review."""
    rows = (data or {}).get("rows", [])
    ok = (data or {}).get("ok", True)
    empty = ("No leader gained 5%+ this week — a quiet tape for the strongest names." if ok
             else "Weekly Review unavailable — screener or RS data did not load this run.")
    return _section_v9("wk", "YOUR WEEKLY REVIEW", rows, SECTION_SPECS_V9["wk"],
                       subtitle="leaders up ≥5% this week · RS≥80 · above 200-day · "
                                "within 25% of 52-wk high", empty=empty)


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
    except (OSError, ValueError) as exc:
        log.error("Minervini feed unreadable (%s): %r", latest, exc)
        return _ext_empty(f"Minervini engine: could not read {latest}."), 0
    if not rows:
        return _ext_empty(f"Minervini engine: 0 picks on {asof}."), 0

    for m in rows:
        m["_risk_pct"] = round((m.get("stop_frac") or 0.0) * 100, 1)
    _enrich_external_rows(rows, weekly_spark=False, spark_field="Close", spark_n=60,
                          market_modifier=market_modifier, spark_ma_spec=_MA_SPEC_DARK)
    rows = [m for m in rows if not m.get("_drop_adr")]   # dead-stock floor (ADR<1.5%)
    _prefetch_fundamentals([m.get("ticker") for m in rows], budget_s=30.0)

    out = [
        f"<div class='ext-asof'>Minervini engine · daily VCP/SEPA buy list · as of {esc(asof)} · "
        f"{len(rows)} names · trade plan from <code>minervini_engine</code> · "
        f"M.E.T.A./ANTS/candlestick chart (daily · 60 bars + 10·20·50 MA + volume)/leader badges computed by MADRRY</div>",
        "<div class='table-container rowcards-container'><table data-schema='minervini2' class='rowcards'>",
        "<thead><tr><th data-col='tk'>Ticker</th><th data-col='price'>Chart</th><th data-col='plan'>Trade Plan</th>"
        "<th data-col='narr'>Narrative</th>"
        "<th class='num' data-col='adr' title='Average Daily Range — 20-day avg of (High/Low−1), % · how much it typically moves per day (TradingView ADRP, or an equivalent 20-day calc on the external/HTF tabs)'>ADR</th>"
        "<th data-col='rs'>RS</th><th data-col='meta'>M.E.T.A.</th><th class='num' data-col='ants'>ANTS</th>"
        + _MA_YOY_HEADERS +
        "<th data-col='status'>Status (VCP &amp; Vol)</th></tr></thead>",
    ]
    for m in rows:
        tk = m.get("ticker", "")
        pivot, stop, close = m.get("pivot"), m.get("stop"), m.get("last_close")
        risk = round((m.get("stop_frac") or 0.0) * 100, 1)
        status = (m.get("status") or "").replace("_", " ")
        triggered = "TRIGGER" in status
        st_color = "#54b87f" if triggered else "#d3a04d"
        st_bg = "var(--tint-green)" if triggered else "var(--tint-yellow)"
        vcp = m.get("vcp_score", 0) or 0
        vc_color = "#e06c6a" if vcp >= 85 else ("#d3a04d" if vcp >= 75 else "#82827c")
        rs = m.get("rs", "N/A")
        adr = m.get("adr", 0)
        sector = m.get("sector", "")
        foot = m.get("footprint", "")
        rev = m.get("rev_yoy")
        perf6 = m.get("perf6m")
        ptp = m.get("pct_to_pivot")
        offhi = m.get("pct_from_high")
        slope = m.get("vol50_slope")
        rev_line = (f"<br><span class='sub'>Rev YoY {rev:+.0f}%</span>"
                    if isinstance(rev, (int, float)) else "")
        perf6_line = (f"<br><span class='sub'>6M: {perf6:+.0f}%</span>"
                      if isinstance(perf6, (int, float)) else "")
        if ptp is None:
            ptp_line = ""
        elif ptp <= 0:
            ptp_line = "<span class='good'>▲ triggered / through pivot</span>"
        else:
            ptp_line = f"<span class='warn'>{ptp:.1f}% to pivot</span>"
        offhi_line = (f"<br><span class='{'good' if (offhi or 0) >= -8 else 'bad'}'>Off high: {offhi:.1f}%</span>"
                      if isinstance(offhi, (int, float)) else "")
        slope_line = (f"<br><span class='sub'>Vol50 slope: {slope:+.1f}%</span>"
                      if isinstance(slope, (int, float)) else "")
        leader_html = _ext_leader_badges(m)
        fp_html = _ext_fp_badges(m)
        out.append(f"""<tr data-sector="{esc(sector)}">
            {_tk_cell({"ticker": tk, "close": close}, entry=pivot, stop=stop)}
            {_chart_cell(m.get('_spark', ''), close if close is not None else 0)}
            <td class="c-plan" data-sort="{risk}">
                <div class="entry-box">
                    <span class="kicker">VCP PIVOT</span>
                    <span class="entry-text">Buy: ${pivot}</span><br>
                    <span class="stop-text">Stop: ${stop}</span><br>
                    <span class="{_risk_cls(risk, 4.0, 8.0)}">Risk: {risk}%</span>
                </div>
            </td>
            {_narr_cell(tk, f'''<span class="tag">{esc(foot)}</span>{rev_line}<br>
                <span class="tag">{esc(sector)}</span>''')}
            <td class="num c-stat" data-label="ADR" data-sort="{adr}">{adr}%</td>
            <td class="c-stat" data-label="RS" data-sort="{rs if isinstance(rs, int) else 0}"><span class="score">{esc(rs)}</span>{perf6_line}</td>
            {_ext_meta_cell(m)}
            {_ext_ants_cell(m)}
            {_ma_cells(m.get('_ma_dist'))}{_fwd_yoy_cell(tk)}{_eps_accel_cell(tk)}
            <td class="c-status" style="text-align:left;" data-sort="{ptp if ptp is not None else 999}">
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
        data, _rtb_src = _read_trilogy_rtb()
        log.info("Trilogy feed source: %s", _rtb_src)
    except (OSError, ValueError) as exc:
        # Surface the real errno — a background launchd run failing here on a file that
        # a manual run reads fine is the TCC/Full-Disk-Access signature (PermissionError).
        # With the feeds/ mirror in place this should only fire if the Trilogy nightly
        # itself never ran (no mirror) AND Downloads is TCC-blocked.
        log.error("Trilogy feed unreadable (tried %s): %r", TRILOGY_RTB_PATHS, exc)
        return _ext_empty("Trilogy: ready_to_buy.json not found / unreadable "
                          "(no readable copy among mirrors)."), 0
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
    shown = [c for c in shown if not c.get("_drop_adr")]   # dead-stock floor (ADR<1.5%)
    _prefetch_fundamentals([c.get("ticker") for c in shown], budget_s=30.0)

    out = [
        f"<div class='ext-asof'>Trilogy nightly · O'Neil reference-class buy-stop list · as of {esc(asof)} · "
        f"{total} candidates{extra} · trade plan from <code>trilogy webapp</code> · "
        f"M.E.T.A./ANTS/weekly candlestick chart (+10·20·50-week MA)/leader badges computed by MADRRY</div>",
        "<div class='table-container rowcards-container'><table data-schema='trilogy2' class='rowcards'>",
        "<thead><tr><th data-col='tk'>Ticker</th><th data-col='price'>Chart</th><th data-col='plan'>Trade Plan</th>"
        "<th data-col='narr'>Narrative</th><th data-col='grade'>Grade</th>"
        "<th class='num' data-col='win20'>Win20</th><th data-col='meta'>M.E.T.A.</th><th class='num' data-col='ants'>ANTS</th>"
        + _MA_YOY_HEADERS +
        "<th data-col='status'>Status</th></tr></thead>",
    ]
    grade_col = {"A": "#54b87f", "B": "#aecfe8", "C": "#d3a04d", "D": "#e06c6a", "F": "#e06c6a"}
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
        grade = c.get("grade", "")
        gcol = grade_col.get(grade, "#82827c")
        likeness = c.get("likeness_q")
        likeness_line = (f"<br><span class='sub'>likeness Q{likeness}</span>"
                         if likeness is not None else "")
        pattern = (c.get("pattern") or "").replace("_", " ")
        family = c.get("family", "")
        stage = c.get("stage")
        stage_txt = f"Stage {stage}" if stage is not None else ""
        sector = c.get("sector", "")
        rs_line = (c.get("rs_line") or "").replace("_", " ")
        rs_rank = c.get("sector_rs_rank")
        rs_rank_line = (f"<br><span class='sub'>sector RS #{rs_rank}</span>"
                        if rs_rank else "")
        win20 = c.get("win20_rate")
        win_txt = f"{win20 * 100:.0f}%" if isinstance(win20, (int, float)) else "—"
        status = c.get("status", "")
        gated = c.get("gated")
        checklist = c.get("checklist", "")
        mtail = c.get("monster_tail")
        mdec = c.get("monster_tail_decile")
        detail = c.get("detail", "")
        st_color = "#e06c6a" if gated else ("#54b87f" if "BUYING RANGE" in status else "#82827c")
        mtail_line = (f"<br><span class='fp-badge fp-warn'>~MONSTER-TAIL d{mdec}</span>" if mtail else "")
        range_txt = f" <span class='stop-reason'>(top ${top})</span>" if top is not None else ""
        leader_html = _ext_leader_badges(c)
        fp_html = _ext_fp_badges(c)
        rs_line_html = ((f"<span style='color:var(--accent-2);font-size:var(--fs-micro);'>RS line: {esc(rs_line)}</span>"
                         + (f" <span class='sub'>· sector RS #{rs_rank}</span>" if rs_rank else "")
                         + "<br>") if rs_line else "")
        out.append(f"""<tr data-sector="{esc(sector)}">
            {_tk_cell({"ticker": tk, "close": close}, entry=ideal, stop=stop)}
            {_chart_cell(c.get('_spark', ''), close if close is not None else 0)}
            <td class="c-plan" data-sort="{risk}">
                <div class="entry-box">
                    <span class="kicker">BUY-STOP</span>
                    <span class="entry-text">Buy: ${ideal}</span>{range_txt}<br>
                    <span class="stop-text">Stop: ${stop} <span class="stop-reason">(pivot −8%)</span></span><br>
                    <span class="{_risk_cls(risk, 8.0, 12.0)}">Risk: {risk}%</span>
                </div>
            </td>
            {_narr_cell(tk, f'''<span class="theme-tag">{esc(pattern)}</span><br>
                <span class="tag">{esc(family)}</span> <span class="tag">{esc(stage_txt)}</span> <span class="tag">{esc(sector)}</span>''')}
            <td class="c-stat" data-label="Grade" data-sort="{esc(grade)}"><span class="grade-badge" style="color:{gcol};border:1px solid {gcol};background:rgba(0,0,0,0.18);">{esc(grade) or '—'}</span>{likeness_line}</td>
            <td class="num c-stat" data-label="Win20" data-sort="{win20 if isinstance(win20, (int, float)) else 0}">{win_txt}</td>
            {_ext_meta_cell(c)}
            {_ext_ants_cell(c)}
            {_ma_cells(c.get('_ma_dist'))}{_fwd_yoy_cell(tk)}{_eps_accel_cell(tk)}
            <td class="c-status" style="text-align:left;" data-sort="{esc(status)}" title="{esc(detail)}">
                {leader_html}{fp_html}{rs_line_html}
                <span class="fp-badge" style="border-color:{st_color};color:{st_color};">{esc(status)}</span>
                <span class="warn-flag" style="font-size:var(--fs-micro);"> {esc(checklist)}</span>{mtail_line}
            </td>
        </tr>""")
    out.append("</table></div>")
    return "".join(out), total


def generate_hve_table(ep_matches: List[dict]) -> str:
    out = ['<div class="section-title bg-hve"><span class="tdot"></span>HVE (EPISODIC PIVOTS) — LOW FLOAT ≤200M</div>',
           '<div class="table-container rowcards-container"><table class="rowcards">',
           "<thead><tr><th>Ticker</th><th>Price &amp; Gap</th><th>Narrative &amp; Conviction</th><th>QM Trade Plan</th></tr></thead>"]
    if not ep_matches:
        out.append("<tr><td colspan='4' style='color:#82827c;'>No HVE events detected today.</td></tr>")
    else:
        for m in ep_matches:
            risk_color = "#54b87f" if m["risk_pct"] <= 4.0 else ("#d3a04d" if m["risk_pct"] <= 6.0 else "#e06c6a")
            float_txt = f"{m['float_shares']}M" if m["float_shares"] else "N/A"
            out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
                <td class="ep-ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a></td>
                <td data-sort="{m['close']}">{_lp(m['ticker'], m['close'], entry=m['entry'], stop=m['stop'])} <span class="good">(+{m['change']}%)</span><br><span style="font-size:var(--fs-caption);color:#82827c;">Gap: {m['gap']}%</span></td>
                <td data-sort="{m['rel_vol']}">{_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span>''')}<br><br><span class="hve-badge">{m['rel_vol']}x Avg!</span><br><span style="font-size:var(--fs-caption);color:#82827c;margin-top:4px;display:inline-block;">Float: {float_txt}</span></td>
                <td data-sort="{m['risk_pct']}">
                    <div style="font-size:var(--fs-caption);color:#aecfe8;text-align:left;margin-bottom:4px;">✓ Close Range {m['close_range']}%</div>
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
        return (f"<div class='table-container' style='padding:16px;color:#82827c;'>"
                f"Tier-A tracking study not available yet "
                f"({esc(str(exc))}). It is generated after the next scan by "
                f"<code>madrry_tier_a_tracker.py</code>.</div>", 0)

    def wr_color(wr):
        if wr is None:
            return "#82827c"
        if wr >= 60:
            return "#54b87f"
        if wr >= 50:
            return "#d29922"
        if wr >= 40:
            return "#db6d28"
        return "#f85149"

    def bucket_table(title, rows, label="bucket"):
        h = [f"<div class='section-title' style='background-color:var(--surface);"
             f"color:#aecfe8;border-bottom:2px solid #26262b;'>{title}</div>",
             "<div class='table-container'><table>",
             f"<tr><th>{label}</th><th>Win</th><th>Loss</th><th>N</th><th>Win%</th></tr>"]
        if not rows:
            h.append("<tr><td colspan='5' style='color:#82827c;'>No resolved names yet.</td></tr>")
        for r in rows:
            wr = r.get("wr")
            h.append(
                f"<tr><td style='text-align:left;font-weight:600;'>{esc(r.get('k', r.get('bucket','')))}</td>"
                f"<td style='color:#54b87f;'>{r['w']}</td>"
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
        "color:#aecfe8;border-bottom:3px solid #aecfe8;'>TIER-A FORWARD WIN/LOSS STUDY</div>",
        "<div class='table-container' style='padding:14px 16px;line-height:1.6;'>",
        f"<div style='font-size:var(--fs-body);'>"
        f"<b style='color:#ececea;'>{ov['total']}</b> Tier-A names tracked from first appearance · "
        f"<b style='color:#54b87f;'>{ov['w']} win</b> / <b style='color:#f85149;'>{ov['l']} loss</b> resolved · "
        f"overall win-rate <b style='color:{wr_color(ov['wr'])};'>{ov['wr']}%</b> · "
        f"<span style='color:#82827c;'>{ov['open']} still open · data as-of {esc(s.get('asof',''))}</span></div>",
        "<div style='font-size:var(--fs-caption);color:#82827c;margin-top:6px;'>"
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
          "color:#aecfe8;border-bottom:2px solid #26262b;'>"
          "M.E.T.A. WEIGHTS — active (live) vs next-fit preview</div>",
          "<div class='table-container' style='padding:8px 12px;font-size:var(--fs-caption);"
          "color:#82827c;'>Active weights are LIVE in scoring now. ‘Preview’ = the "
          "weekend re-fit candidate; it is applied only if it separates winners "
          "strictly better than the active set (else held). Edge &gt;0 = component "
          "scored higher on winners.</div>",
          "<div class='table-container'><table>",
          "<tr><th>Component</th><th>Active</th><th>Edge</th><th>Preview</th><th>Δ</th></tr>"]
    for w in recal["weights"]:
        d = w["new"] - w["cur"]
        dcol = "#54b87f" if d > 0 else ("#f85149" if d < 0 else "#82827c")
        ecol = "#54b87f" if w["edge"] > 0 else ("#f85149" if w["edge"] < 0 else "#82827c")
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
          "color:#aecfe8;border-bottom:2px solid #26262b;'>"
          "META COMPONENT — winner vs loser average points</div>",
          "<div class='table-container'><table>",
          "<tr><th>Component</th><th>Win avg</th><th>Loss avg</th><th>Edge</th></tr>"]
    for c in s["components"]:
        e = c["edge"]
        ecol = "#54b87f" if (e or 0) > 0 else ("#f85149" if (e or 0) < 0 else "#82827c")
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
    out = ['<div class="section-title"><span class="tdot" style="background:var(--accent);"></span>POST-HVE U&amp;R (PULLBACK &amp; UNDERCUT)</div>',
           '<div class="table-container rowcards-container"><table class="rowcards">',
           "<thead><tr><th>Ticker</th><th>Price &amp; 1W Perf</th><th>Narrative &amp; Status</th><th>U&amp;R Trade Plan</th></tr></thead>"]
    if not ur_matches:
        out.append("<tr><td colspan='4' style='color:#82827c;'>No Post-HVE U&amp;R candidates. Waiting for HVE stocks to consolidate...</td></tr>")
    else:
        for m in ur_matches:
            risk_color = "#54b87f" if m["risk_pct"] <= 3.5 else ("#d3a04d" if m["risk_pct"] <= 5.0 else "#e06c6a")
            vol_color = "good" if m["vol_contraction"] <= 40 else ("warn" if m["vol_contraction"] <= 60 else "bad")
            holding_color = "good" if m["holding_above_low"] else "bad"
            out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
                <td class="ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank" style="color:#aecfe8;">{esc(m['ticker'])}</a></td>
                <td data-sort="{m['close']}">{_lp(m['ticker'], m['close'], entry=m['entry'], stop=m['stop'])} <span style="color:#82827c;">({m['change']:+}%)</span><br><span style="font-size:var(--fs-caption);color:#aecfe8;background:var(--tint-accent);padding:2px 6px;border-radius:4px;">Day {m['days_since_hve']} since HVE</span></td>
                <td data-sort="{m['vol_contraction']}">{_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span>''')}<br><br>
                    <span class="squat-badge {vol_color}">Vol: {m['vol_contraction']:.0f}% of Day 1</span><br>
                    <span class="squat-badge {holding_color}">Above D1 Low: {'Yes ✓' if m['holding_above_low'] else 'No ✗'}</span><br>
                    <span class="sub">D1 High: ${m['day1_high']}</span>
                </td>
                <td data-sort="{m['risk_pct']}">
                    <div style="font-size:var(--fs-caption);color:#aecfe8;text-align:left;margin-bottom:4px;">⚡ U&amp;R: Undercut D{m['days_since_hve']-1}L then reclaim</div>
                    <div class="entry-box" style="border-color:#aecfe8;background-color:rgba(210,168,255,0.1);">
                        <span class="entry-text" style="color:#aecfe8;">Buy: ${m['entry']}</span><br>
                        <span class="stop-text">Stop: ${m['stop']} <span class="stop-reason">({esc(m['stop_reason'])})</span></span><br>
                        <span style="color:{risk_color};font-size:var(--fs-body);">Risk: {m['risk_pct']}%</span>
                    </div>
                </td>
            </tr>""")
    out.append("</table></div>")
    out.append("""
    <div style="background-color:var(--tint-accent);padding:15px;margin:20px 0;border-radius:8px;">
        <div style="color:#aecfe8;font-weight:bold;margin-bottom:8px;">📖 Post-HVE U&amp;R Strategy (Like MXL Apr 28)</div>
        <div style="font-size:var(--fs-table);color:#ececea;line-height:1.6;">
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


# ---- forward base rates: "if this state, what has ^GSPC/^NDX/IWM done next?" ----
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
        return "<span style='color:#4a4a52;'>—</span>"
    med, win = cell["median"], cell["win"]
    col = "val-green" if med > 0 else ("val-red" if med < 0 else "#ececea")
    return f"<span class='{col}'>{med:+.1f}%·{win:.0f}%</span>"


def _forward_block(md: dict) -> str:
    """Compact 'if this state → forward 1w/2w/3w/4w' card line for one index
    (2026-07-06 USER: near horizons — was 1w/4w/8w)."""
    br = forward_baserate(md.get("ticker", ""), md.get("dist_days", 0), md.get("above_200"))
    if not br:
        return ""
    s = br["stats"]
    n = (s.get("f4w") or {}).get("n", 0)
    # Small-n caution (display only; the min_n=30 lookup gate is unchanged). Rare
    # states cluster in a handful of episodes — e.g. ^IXIC bear·9+ dist (n=41) is
    # dominated by the 1987/2001/2020 capitulation lows and reads strongly bullish
    # while ^NDX's same cell (2008/2022 mid-bear grind) reads bearish — so a thin
    # cell is composition-sensitive, not a stable base rate (audit 2026-07-02).
    small_n = " <span style='color:var(--yellow,#d3a04d);' title=\"Thin sample — this state is rare, so the stats lean on a few historical episodes; treat as context, not a stable base rate.\">⚠️ small n</span>" if n < 100 else ""
    rows = (f"<div style='margin-top:6px;border-top:1px solid #1f1f23;padding-top:5px;"
            f"font-size:var(--fs-table);color:#82827c;'>"
            f"<span title=\"Historical forward price return after days in the SAME state "
            f"(median · win-rate). Conditioner: 200-day-MA regime × O'Neil distribution-day "
            f"bucket — the most predictive combination in out-of-sample testing. "
            f"Built from full history since inception.\">"
            f"📊 If this state → forward <span style='color:#4a4a52;'>({br['label']}, n={n})</span></span>{small_n}<br>"
            f"&nbsp;&nbsp;1w {_fwd_num(s.get('f1w'))} · "
            f"2w {_fwd_num(s.get('f2w'))} · "
            f"3w {_fwd_num(s.get('f3w'))} · "
            f"4w {_fwd_num(s.get('f4w'))}")
    ext = forward_ext_baserate(md.get("ticker", ""), md.get("ext_10", 0.0),
                               md.get("ext_20", 0.0), md.get("ext_50", 0.0), md.get("above_200"))
    if ext:
        es = ext["stats"]
        rows += (f"<br><span style='color:#4a4a52;'>&nbsp;&nbsp;{ext['label']}:</span> "
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


# ---- Weinstein ch.8 market internals (2026-07-17): A-D line, weekly common-
# stock NH-NL, Momentum Index (200d MA of net advances). Display-only card;
# history backfilled from the survivor-only local dump (shape/sign reads, not
# levels). Any future use as a FILTER needs its own §2.7 era-rule backtest.
_INTERNALS_BASE_FILTER = [
    {"left": "type", "operation": "in_range", "right": ["stock"]},
    {"left": "typespecs", "operation": "has", "right": ["common"]},
    {"left": "exchange", "operation": "in_range", "right": ["NYSE", "NASDAQ", "AMEX"]},
    {"left": "close", "operation": "egreater", "right": 2},
]


def fetch_market_internals(diag: Optional[Diagnostics] = None) -> Optional[dict]:
    """Five cheap TradingView COUNT queries (range [0,1] -> totalCount only):
    universe / advancers / decliners / new 52w highs / new 52w lows over
    NYSE+NASDAQ+AMEX common stocks >= $2 (probed working 2026-07-17). Never
    fatal: any failure -> diag.warn -> None (the card just doesn't update)."""
    try:
        def _count(extra, label):
            payload = {"filter": _INTERNALS_BASE_FILTER + extra,
                       "columns": ["name"], "range": [0, 1]}
            # NO diag= here (2026-07-18 review, BLOCKER): _request_json records
            # diag.error() BEFORE raising, so passing diag would put an entry in
            # diag.errors and trip the errors=0 publish gate — even though this
            # card is optional and our own except-handler downgrades to warn.
            d = tv_post(payload, label=f"internals:{label}")
            n = d.get("totalCount")
            if n is None:
                raise ValueError(f"{label}: no totalCount")
            return int(n)
        issues = _count([], "universe")
        adv = _count([{"left": "change", "operation": "greater", "right": 0}], "adv")
        dec = _count([{"left": "change", "operation": "less", "right": 0}], "dec")
        nh = _count([{"left": "high", "operation": "egreater", "right": "price_52_week_high"}], "nh")
        nl = _count([{"left": "low", "operation": "eless", "right": "price_52_week_low"}], "nl")
        if not issues or adv + dec > issues:
            raise ValueError(f"implausible counts: issues={issues} adv={adv} dec={dec}")
        return {"issues": issues, "adv": adv, "dec": dec, "nh": nh, "nl": nl}
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"Market internals fetch skipped: {exc}")
        return None


def _load_market_internals() -> Optional[dict]:
    """Read market_internals.json ({} -> None). Never raises."""
    try:
        with open(MARKET_INTERNALS_PATH) as fh:
            d = json.load(fh)
        return d if d.get("daily") else None
    except Exception:  # noqa: BLE001
        return None


def _persist_market_internals(counts: Optional[dict], data_date: Optional[str]) -> None:
    """Append today's counts keyed by data_date (same-day re-run overwrites),
    derive net_adv / ad_line / mi / mi_pct from the stored series, cap at 600
    rows. Non-fatal, atomic."""
    try:
        if not counts or not data_date:
            return
        # Never overwrite an accumulated history we failed to READ (2026-07-18
        # review): a transient unreadable/corrupt file would otherwise be
        # replaced by a one-row file, destroying the series. Absent file =
        # legitimate fresh start; present-but-unreadable = bail out.
        d = _load_market_internals()
        if d is None and os.path.exists(MARKET_INTERNALS_PATH):
            log.warning("market_internals.json unreadable — skipping append to "
                        "avoid destroying the accumulated series")
            return
        d = d or {"_meta": {"spec": "live TV counts"}, "daily": {}}
        daily = d.setdefault("daily", {})
        rows = sorted(k for k in daily if k < data_date)
        prev_ad = daily[rows[-1]].get("ad_line", 0) if rows else 0
        net = counts["adv"] - counts["dec"]
        rec = dict(counts)
        rec.update({"net_adv": net, "ad_line": prev_ad + net, "src": "live"})
        daily[data_date] = rec
        keep = sorted(daily)[-600:]
        d["daily"] = {k: daily[k] for k in keep}
        # MI over the last <=200 stored rows (tolerates gaps; sign is the read)
        tail = [d["daily"][k] for k in sorted(d["daily"])[-200:]]
        nets = [r.get("net_adv", 0) for r in tail]
        isss = [max(1, r.get("issues", 1)) for r in tail]
        rec["mi"] = round(sum(nets) / len(nets), 1)
        rec["mi_pct"] = round(sum(n / i for n, i in zip(nets, isss)) / len(nets) * 100.0, 2)
        _atomic_write(MARKET_INTERNALS_PATH, json.dumps(d))
    except Exception:  # noqa: BLE001
        pass


def _internals_card(internals: Optional[dict], market_data: Optional[List[dict]]) -> str:
    """One compact Market Overview card: A-D + divergence state, weekly NH-NL
    + sign streak, Momentum Index + confirmed/unconfirmed zero-cross. ''
    when data is missing or >5 sessions stale. Display-only. Never raises."""
    try:
        if not internals:
            return ""
        daily = internals.get("daily") or {}
        keys = sorted(daily)
        if len(keys) < 30:
            return ""
        last = daily[keys[-1]]
        # stale guard: newest row must be recent (>5 stored sessions gap = off)
        try:
            from datetime import date as _date
            newest = datetime.strptime(keys[-1], "%Y-%m-%d").date()
            if (_date.today() - newest).days > 9:
                return ""
        except Exception:  # noqa: BLE001
            pass
        rows = [daily[k] for k in keys]
        # --- A-D line state / divergence vs SPX ---
        ads = [r.get("ad_line", 0) for r in rows]
        tail252 = ads[-252:]
        ad_stale = (len(tail252) - 1) - max(range(len(tail252)), key=lambda i: tail252[i])
        spx = next((m for m in (market_data or []) if m.get("ticker") == "^GSPC"), None)
        spx_off = (spx or {}).get("pct_off_52wk")
        if spx_off is not None and spx_off >= -0.5 and ad_stale >= 20:
            l1 = ("<span class='val-warn'>⚠️ SPX near its 52w high UNCONFIRMED — "
                  "A-D line stale %d sess (~%.1f mo) · backfill is survivor-biased, read "
                  "shape not level</span>" % (ad_stale, ad_stale / 21.0))
        else:
            state = ("at its 252d high" if ad_stale == 0
                     else "%d sess off its 252d high" % ad_stale)
            l1 = ("%s▲ / %s▼ · net %+d · line %s"
                  % (last.get("adv", "?"), last.get("dec", "?"), last.get("net_adv", 0), state))
        # --- weekly NH-NL + streak (W-FRI buckets) ---
        import collections
        wk = collections.OrderedDict()
        for k in keys[-400:]:
            dt = datetime.strptime(k, "%Y-%m-%d").date()
            fri = dt + timedelta(days=(4 - dt.weekday()) % 7)
            wk.setdefault(fri.isoformat(), 0)
            wk[fri.isoformat()] += (daily[k].get("nh", 0) - daily[k].get("nl", 0))
        wvals = list(wk.values())
        streak = 0
        for v in reversed(wvals):
            if (v > 0) == (wvals[-1] > 0) and (v != 0 or wvals[-1] == 0):
                streak += 1
            else:
                break
        l2 = ("weekly NH-NL %+d · %d wk %s"
              % (wvals[-1], streak, "positive" if wvals[-1] > 0 else "negative"))
        # --- Momentum Index + zero-cross confirmation (>=10 sessions held) ---
        mis = [(r.get("net_adv", 0) / max(1, r.get("issues", 1))) for r in rows]
        mi_series = []
        for i in range(len(mis)):
            w = mis[max(0, i - 199):i + 1]
            mi_series.append(sum(w) / len(w))
        cur = mi_series[-1] * 100.0
        sign = cur > 0
        held = 0
        for v in reversed(mi_series):
            if (v > 0) == sign:
                held += 1
            else:
                break
        cross_date = keys[-held] if held < len(keys) else keys[0]
        conf = ("positive" if sign else "negative")
        conf += " since %s (%s)" % (cross_date,
                                    "confirmed" if held >= 10 else "unconfirmed flip %d/10" % held)
        l3 = "MI %+.2f%% · %s" % (cur, conf)
        tip = ("Weinstein ch.8 internals. Common stocks >=$2, NYSE+NASDAQ+AMEX (~4.4k live / "
               "5.9k backfill). History backfilled from a survivor-only dump - decliner/new-low "
               "counts understated historically; read shape and sign, not levels. MI = 200d MA "
               "of daily net advances as % of issues; a zero-cross is 'confirmed' after the sign "
               "holds 10 sessions. A-D divergence rule: index near its 52w high while the A-D "
               "line is >=20 sessions past its own high. Display-only - no gate.")
        return (f"""<div class="market-card" title="{esc(tip)}">
                <h3>Internals (A-D · NH-NL · MI)</h3>
                <div style="font-size:var(--fs-table);color:#82827c;">
                    {l1}<br>
                    {l2}<br>
                    {l3}<br>
                    <span style="font-size:var(--fs-caption);">Weinstein ch.8 · common stocks · display-only</span>
                </div>
            </div>""")
    except Exception:  # noqa: BLE001
        return ""


def _persist_headline_meter(data_date: Optional[str], market_data: Optional[List[dict]] = None) -> None:
    """Append today's headline-meter reading to headline_meter_history.json
    (this history is what makes a future §2.7 era-gated study possible — the
    meter itself stays display-only until then). Non-fatal; 400-row cap."""
    try:
        if not data_date or not HEADLINE_METER:
            return
        rec = dict(HEADLINE_METER)
        try:
            spx = next((m for m in (market_data or []) if m.get("ticker") == "^GSPC"), None)
            rec["spx_off_hi"] = (spx or {}).get("pct_off_52wk")
        except Exception:  # noqa: BLE001
            rec["spx_off_hi"] = None
        hist = {}
        if os.path.exists(HEADLINE_METER_HISTORY_PATH):
            try:
                with open(HEADLINE_METER_HISTORY_PATH) as fh:
                    hist = json.load(fh)
                if not isinstance(hist, dict):
                    raise ValueError("not a dict")
            except Exception:  # noqa: BLE001
                # Present but unreadable -> preserve on disk, skip this append
                # (2026-07-18 review: the old fallback wrote a 1-row file over
                # the accumulated series).
                log.warning("headline_meter_history.json unreadable — skipping append")
                return
        hist[data_date] = rec
        keep = sorted(hist)[-400:]
        _atomic_write(HEADLINE_METER_HISTORY_PATH, json.dumps({k: hist[k] for k in keep}))
    except Exception:  # noqa: BLE001
        pass


def _persist_breadth_history(breadth: dict, data_date: Optional[str]) -> None:
    """Append the session's S&P breadth to breadth_history.json keyed by the
    US-session DATA DATE (never wall-clock). This is the durable regime feed for
    downstream studies. breadth_day_over_day() above used to be the writer but was
    orphaned when the day-over-day chip switched to Barchart's native priceChange
    (~2026-06-08), so the file stalled. Re-wired here. Non-fatal, additive, 120-day
    cap. Same-day re-runs overwrite (final run of a data date wins)."""
    if not breadth or not breadth.get("ok") or not data_date:
        return
    if breadth.get("stale"):
        return   # never re-persist a carried reading as today's fresh data
    try:
        hist = {}
        if os.path.exists(BREADTH_HISTORY_PATH):
            with open(BREADTH_HISTORY_PATH) as fh:
                hist = json.load(fh)
        entry = {"br50": round(float(breadth.get("above50", 0.0)), 1),
                 "br200": round(float(breadth.get("above200", 0.0)), 1)}
        a20 = breadth.get("above20")
        if a20 is not None:
            entry["br20"] = round(float(a20), 1)
        hist[data_date] = entry                       # same-day re-run overwrites
        hist = {k: hist[k] for k in sorted(hist)[-120:]}
        _atomic_write(BREADTH_HISTORY_PATH, json.dumps(hist))
    except Exception:  # noqa: BLE001
        pass


def _ext_line(label: str, ext: float, pct: Optional[float]) -> str:
    """One extension row: '+X.X% · P##' colored by historical percentile."""
    base = f"{ext:+.1f}%"
    if pct is None:
        return f"{label}: <span style='color:#ececea;'>{base}</span>"
    pcol = "val-red" if pct >= 90 else ("val-warn" if pct >= 75 else "val-green")
    flag = " ⚠️ stretched" if pct >= 90 else ("" if pct < 75 else " hot")
    return f"{label}: <span class='{pcol}'>{base} · P{pct:.0f}{flag}</span>"


def _bd_chip(d: Optional[float]) -> str:
    if d is None:
        return ""
    arr = "▲" if d > 0 else ("▼" if d < 0 else "▬")
    col = "val-green" if d > 0 else ("val-red" if d < 0 else "")
    return f" <span class='{col}' style='font-size:var(--fs-table);'>{arr} {d:+.1f}pp</span>"


_INDEX_LABELS = {"^IXIC": "IXIC", "^NDX": "NDX", "^GSPC": "SPX"}


def _index_display(tk: str) -> str:
    """Card/label name for an index. The internal key stays the Yahoo symbol
    (^IXIC/^NDX/^GSPC) for data lookups + the live-refresh fetch; the caret is
    dropped only for display so the header reads cleanly as 'IXIC/NDX/SPX'
    alongside IWM (2026-07-06 USER: ^NDX=Nasdaq-100, ^GSPC=S&P-500/SPX)."""
    return _INDEX_LABELS.get(tk, tk)


def build_market_section(market_data: List[dict], breadth: dict,
                         regime: str = "GREEN", allow_breakouts: bool = True,
                         internals: Optional[dict] = None,
                         sector_waves: Optional[List[dict]] = None) -> Tuple[str, str]:
    """Returns (html, overall_trend)."""
    # Weinstein index-stage banner (2026-07-16, ch.8): presentation-level only —
    # never gates scans, tiers or drafts. Red styling + the ch.8 Stage-4 protocol
    # text when any tracked index reads Stage 4; amber on Stage-3 risk.
    stage_banner = ""
    try:
        stages = [(md["ticker"], md["wk_stage"]) for md in market_data if md.get("wk_stage")]
        if stages:
            any_s4 = any(s.startswith("S4") for _, s in stages)
            any_s3 = any(s.startswith("S3") for _, s in stages)
            toks = " · ".join(f"{esc(_index_display(t))} <b>{esc(s)}</b>" for t, s in stages)
            note = ""
            if any_s4:
                note = (" — <b>Stage-4 tape (Weinstein ch.8): no new buys, sell weak-RS "
                        "holdings, tighten stops, favor the SHORT section</b>")
            elif any_s3:
                note = " — Stage-3 caution: 30wk MA flattening, tighten stops"
            col = "var(--red)" if any_s4 else ("var(--yellow)" if any_s3 else "var(--text-2)")
            tip = ("Weinstein stage read per index (30-week ~ 150d MA): slope over ~5wk "
                   "(rising > +0.5%, falling < -0.5%), price vs MA, churn = >=3 MA crossings "
                   "in 50 sessions. Informational banner only - no gate.")
            stage_banner = ("<div style='font-size:var(--fs-caption);margin:2px 0 10px;"
                            f"color:{col};' title='{esc(tip)}'>WEINSTEIN STAGE · {toks}{note}</div>")
    except Exception:  # noqa: BLE001
        stage_banner = ""
    # Sector-wave banner (Weinstein ch.3 group-ignition tally; informational).
    waves_banner = ""
    try:
        if sector_waves:
            toks = " · ".join(
                ("%s%s %d/%s" % ("<b>NEW</b> " if w.get("is_new") else "",
                                 esc(w["industry"]), w["n"], w.get("size") or "?"))
                for w in sector_waves[:5])
            wtip = ("Distinct 5-day breakout WINNERS per industry group (breakout_log, "
                    "non-ETF). Fires at >=3 winners covering >=20% of the group, or >=10 "
                    "absolute. Weinstein ch.3: clustered group ignition marked the casino "
                    "'78 / motor-home '82 / oil '86 waves. Thresholds cadence-calibrated on 28 days "
                    "of live log - no forward-return claim tested. Informational only - no gate.")
            waves_banner = ("<div style='font-size:var(--fs-caption);margin:2px 0 8px;"
                            "color:var(--text-2);' title='" + esc(wtip) + "'>"
                            "SECTOR WAVES · " + toks + "</div>")
    except Exception:  # noqa: BLE001
        waves_banner = ""
    out = ['<div class="market-panel"><div class="market-title">MARKET OVERVIEW</div>'
           + stage_banner + waves_banner + '<div class="market-grid">']
    overall_trend = "GREEN"
    for md in market_data:
        tk = md["ticker"]
        trend_col = "val-green" if md["trend"] == "GREEN" else "val-red"
        dist_col = "val-red" if md["dist_days"] >= 6 else ("val-warn" if md["dist_days"] >= 4 else "val-green")  # align to regime thresholds (audit)
        if tk == "^IXIC" and md["trend"] == "RED":
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
                <h3>{esc(_index_display(tk))} {_lp(tk, md['close'], style="float:right", fmt="{:.2f}")}</h3>
                <div style="font-size:var(--fs-caption);margin-bottom:4px;"><span class="{chg_col}">{chg:+.2f}% ({chg_pt:+.2f})</span> <span style="color:#82827c;">vs prev close</span></div>
                <div class="idx-spark">{spark}</div>
                <div style="font-size:var(--fs-table);color:#82827c;margin:5px 0;">10SMA/21SMA: <span class="{trend_col}">{md['trend']}</span></div>
                <div style="font-size:var(--fs-table);color:#82827c;">
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
        foot = (f"% of S&amp;P 500 members · carried from {esc(asof)} (live feed unavailable)"
                if breadth.get("stale")
                else f"% of S&amp;P 500 members · vs prev day · Barchart {esc(asof)}")
        breadth_html = (
            f"""<div class="market-card">
                <h3>S&amp;P 500 Breadth</h3>
                <div style="font-size:var(--fs-table);color:#82827c;">
                    {brow("20MA (S5TW)", breadth['above20'], breadth.get('chg20'))}<br>
                    {brow("50MA (S5FI)", breadth['above50'], breadth.get('chg50'))}<br>
                    {brow("200MA (S5TH)", breadth['above200'], breadth.get('chg200'))}<br>
                    <span style="font-size:var(--fs-caption);">{foot}</span>
                </div>
            </div>""")
    else:
        breadth_html = ('<div class="market-card"><h3>S&amp;P 500 Breadth</h3>'
                        '<div style="font-size:var(--fs-table);color:#82827c;">Unavailable (Barchart fetch failed).</div></div>')

    out.append(f"""
            {breadth_html}
            {_internals_card(internals, market_data)}
        </div>
    </div>""")
    return "".join(out), overall_trend


# ----------------------------------------------------------------------------
# IBD-STYLE PAGE-1 SECTIONS (2026-07-08): MADRRY TOP 10 news briefs, THE BIG
# PICTURE + Market Pulse, YOUR WEEKLY REVIEW. All fetchers follow the house
# fail-safe pattern: catch -> diag.warn -> sentinel; builders are pure
# formatters that render "" / an Unavailable note on empty input, so a feed
# outage can never take the report down.
# ----------------------------------------------------------------------------

def _fetch_url_bytes(url: str, timeout: int = 15, wall_s: float = 20.0,
                     max_bytes: int = 4_000_000) -> bytes:
    """GET raw bytes with a browser UA (some feeds 403 the default urllib UA).
    urlopen(timeout=) is a PER-SOCKET-OP deadline: a CDN that drips >=1 byte
    every <timeout seconds keeps resp.read() alive forever. Read in chunks so a
    total wall-clock cap (and a size cap) can actually fire — this is the
    difference between a bounded fetch and a silent overnight hang (audit
    2026-07-08)."""
    req = urllib.request.Request(url, headers={"User-Agent": HEADERS.get("User-Agent", "Mozilla/5.0")})
    t0 = time.time()
    chunks: List[bytes] = []
    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            if time.time() - t0 > wall_s:
                raise TimeoutError(f"read exceeded {wall_s:.0f}s wall budget")
            # read1(), NOT read(n): read(n) blocks INSIDE the call until it has n
            # bytes, so a drip feed never returns control to the wall check above.
            # read1() returns after one socket read (bounded by `timeout`), so a
            # slow-drip CDN actually trips the wall budget (audit 2026-07-08).
            chunk = resp.read1(65536)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"feed exceeded {max_bytes} bytes")
    return b"".join(chunks)


_NEWS_TAG_RE = re.compile(r"<[^>]+>")
_NEWS_WS_RE = re.compile(r"\s+")


def _news_clean(text: str) -> str:
    """Strip tags/entities/whitespace from feed text."""
    t = html_lib.unescape(_NEWS_TAG_RE.sub(" ", text or ""))
    return _NEWS_WS_RE.sub(" ", t).strip()


def _rss_items(url: str, feed: str, weight: float, max_age_h: float = 26.0) -> List[dict]:
    """Parse one RSS feed with stdlib ElementTree (feedparser is not installed
    on this Mac — probed 2026-07-08). Freshness is asserted per item: several
    big-name feeds (WSJ dj.com) return HTTP 200 with months-old content."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    root = ET.fromstring(_fetch_url_bytes(url))
    now = datetime.now(timezone.utc)
    out: List[dict] = []
    for it in root.findall(".//item"):
        try:
            pub = parsedate_to_datetime(it.findtext("pubDate") or "")
            if pub.tzinfo is None:
                pub = pub.replace(tzinfo=timezone.utc)
            age_h = (now - pub).total_seconds() / 3600
        except Exception:  # noqa: BLE001 - unparseable date -> not trustworthy
            continue
        if age_h > max_age_h or age_h < -1:
            continue
        title = _news_clean(it.findtext("title") or "")
        if not title:
            continue
        summary = _news_clean(it.findtext("description") or "")
        # Google News quirks: titles end " - Source"; descriptions are a list of
        # related-coverage links (an importance tell, not a summary).
        n_related = 0
        if feed == "gnews":
            n_related = (it.findtext("description") or "").count("<a ")
            src = it.findtext("source")
            if src and title.endswith(" - " + src.strip()):
                title = title[: -(len(src.strip()) + 3)].rstrip()
            summary = ""
        if summary == title:
            summary = ""
        out.append({
            "title": title, "summary": summary, "source": feed,
            "provider": (it.findtext("source") or "").strip() if feed == "gnews" else feed,
            "url": (it.findtext("link") or "").strip(),
            "age_h": age_h, "weight": weight + min(n_related, 5) * 0.25,
        })
    return out


_NEWS_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have in into is it its of on or "
    "that the their this to was were will with after amid over more than says "
    "say said new report reports today just how what why who when your you"
    .split())

# Macro/market-wide words that mark a story as broadly important (vs one-stock
# color). Counted over the representative title, capped, in the cluster score.
_NEWS_MACRO_WORDS = frozenset(
    "fed fomc powell rates rate inflation cpi ppi tariff tariffs treasury yield "
    "yields jobs payrolls unemployment gdp recession opec oil crude stimulus "
    "congress senate house shutdown china trade deal earnings nasdaq dow "
    "stocks stock market markets selloff sell-off rally record futures economy "
    "dollar bitcoin bonds".split())

# Providers inside the yfinance pool: curated outlets score up, listicle mills
# score down (they dominate item counts but are never the day's top stories).
_NEWS_GOOD_PROVIDERS = ("reuters", "bloomberg", "barron", "investor's business daily",
                        "cnbc", "yahoo finance", "associated press", "ap finance",
                        "wall street journal", "marketwatch", "financial times", "fortune")
_NEWS_JUNK_PROVIDERS = ("motley fool", "24/7 wall st", "zacks", "simply wall st",
                        "insider monkey", "gurufocus", "benzinga")


def _news_tokens(title: str) -> frozenset:
    toks = re.findall(r"[a-z0-9&$%.']+", (title or "").lower())
    return frozenset(t.strip(".'") for t in toks
                     if len(t) >= 3 and t not in _NEWS_STOPWORDS)


# ---- Weinstein ch.8 contrary-opinion headline meter (2026-07-17) ------------
# Deterministic lexicon score over the day's MARKET-SCOPED headlines. INFO-ONLY
# soft line on the Big Picture card — no gate, no score input, no timing claim
# (no historical headline archive exists, so no backtest is possible until
# headline_meter_history.json accrues >=2 eras). Probe-earned rules: scope
# words are SUBJECT-only (never sentiment words) and foreign-market stories are
# excluded — see the F5 probe (2026-07-17).
_HM_SCOPE_PHRASES = ("wall street", "s&p 500", "s&p500", "sp 500", "stock market",
                     "bull market", "bear market", "risk assets")
_HM_SCOPE_WORDS = frozenset(
    "stocks market markets nasdaq dow s&p equities investors traders bulls "
    "bears futures indexes indices fed".split())
_HM_FOREIGN_WORDS = frozenset(
    "india indian korea korean china chinese japan japanese europe european "
    "nikkei kospi sensex nifty dax ftse cac hang seng shanghai shenzhen "
    "taiwan taiex asx".split())
_HM_EUPHORIA_PHRASES = (
    "record high", "record highs", "all-time high", "all-time highs",
    "record close", "record closes", "new high", "new highs", "fresh high",
    "fresh highs", "fresh record", "notches record", "best day", "best week",
    "best month", "best quarter", "best year", "winning streak", "bull run",
    "bull market", "melt-up", "melt up", "risk-on", "buying frenzy",
    "no stopping", "cant lose", "can't lose", "to the moon", "goldilocks",
    "soft landing", "fear of missing out")
_HM_EUPHORIA_WORDS = frozenset(
    "soar soars soared soaring surge surges surged surging rally rallies "
    "rallied rallying boom booms booming skyrocket skyrockets skyrocketed "
    "rocket rockets rocketed euphoria euphoric mania frenzy fomo unstoppable "
    "bullish upbeat buoyant optimism optimistic cheer cheers cheered roaring "
    "milestone blockbuster".split())
_HM_DOOM_PHRASES = (
    "bear market", "hard landing", "worst day", "worst week", "worst month",
    "worst quarter", "worst year", "losing streak", "free fall", "freefall",
    "wiped out", "wipe out", "black monday", "flash crash", "credit crunch",
    "death cross", "on edge", "brace for", "braces for", "no way out",
    "recession fears", "recession risk", "recession warning")
_HM_DOOM_WORDS = frozenset(
    "crash crashes crashed crashing plunge plunges plunged plunging plummet "
    "plummets plummeted tumble tumbles tumbled slump slumps slumped sink "
    "sinks sank rout routed meltdown collapse collapses collapsed panic "
    "fear fears fearful turmoil crisis recession bearish gloom doom "
    "bloodbath carnage contagion default defaults bankruptcy bankruptcies "
    "layoffs stagflation selloff sell-off capitulation jitters dread tank "
    "tanks tanked crater craters cratered dive dives nosedive nosedived "
    "spiral spirals warns warning warnings".split())
_HM_TOKEN_RE = re.compile(r"[a-z0-9&$%.'\-]+")

HEADLINE_METER: dict = {}   # filled by fetch_top10_news; read by build_big_picture


def _hm_side_hits(t: str, phrases, words) -> int:
    hits, masked = 0, t
    for p in phrases:
        if p in masked:
            hits += 1
            masked = masked.replace(p, " ")   # mask: "record high" != "record"+"high"
    return hits + sum(1 for w in masked.split() if w in words)


def _headline_meter(pool: List[dict]) -> dict:
    """Score the day's unique market-scoped headlines. Returns {} when the
    pool is empty. Shrunk index (add-2 pseudo-count per side) kills small-n
    false precision; word buckets, never decimals, in the display."""
    seen, n_e, n_d, n_neu = set(), 0, 0, 0
    for it in pool:
        t = " ".join(_HM_TOKEN_RE.findall((it.get("title") or "").lower()))
        if not t or t in seen:
            continue
        seen.add(t)
        toks = set(t.split())
        if toks & _HM_FOREIGN_WORDS:
            continue
        if not (any(p in t for p in _HM_SCOPE_PHRASES) or (toks & _HM_SCOPE_WORDS)):
            continue
        e = _hm_side_hits(t, _HM_EUPHORIA_PHRASES, _HM_EUPHORIA_WORDS)
        d = _hm_side_hits(t, _HM_DOOM_PHRASES, _HM_DOOM_WORDS)
        if e > d:
            n_e += 1
        elif d > e:
            n_d += 1
        else:
            n_neu += 1
    n_cls = n_e + n_d
    idx_raw = (n_e - n_d) / n_cls if n_cls else None
    idx = (n_e - n_d) / (n_cls + 4) if n_cls else None
    bucket = None
    if idx is not None:
        bucket = ("one-sided cheer" if idx >= 0.45 else
                  "leaning upbeat" if idx >= 0.20 else
                  "uniform gloom" if idx <= -0.45 else
                  "leaning fearful" if idx <= -0.20 else "mixed")
    return {"n_scoped": n_e + n_d + n_neu, "n_euphoria": n_e, "n_doom": n_d,
            "idx_raw": (round(idx_raw, 3) if idx_raw is not None else None),
            "idx": (round(idx, 3) if idx is not None else None),
            "bucket": bucket, "printable": n_cls >= 8}


def fetch_top10_news(diag: Optional[Diagnostics] = None, deadline_s: float = 50.0) -> List[dict]:
    """The day's ten most important market/business stories, ranked WITHOUT an
    LLM: substantive summaries come from the yfinance index/ETF news pool (the
    only free source with multi-sentence text — probed 2026-07-08); importance
    comes from cross-outlet corroboration (CNBC Top / MarketWatch Top / Google
    News Business clusters / NYT Business) + macro-keyword and freshness boosts.
    Returns [] on any failure — never raises."""
    t_start = time.time()
    left = lambda: deadline_s - (time.time() - t_start)  # noqa: E731
    pool: List[dict] = []
    try:
        # -- content pool: yfinance news on the majors (parallel, budgeted) --
        def _yf_news(sym):
            items = []
            for x in (yf.Ticker(sym).news or []):
                c = x.get("content") or {}
                title = _news_clean(c.get("title") or "")
                pub = c.get("pubDate") or ""
                if not title or not pub:
                    continue
                try:
                    dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
                    age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                except Exception:  # noqa: BLE001
                    continue
                if age_h > 26:
                    continue
                prov = ((c.get("provider") or {}).get("displayName") or "").strip()
                pl = prov.lower()
                w = 1.2
                if any(g in pl for g in _NEWS_GOOD_PROVIDERS):
                    w += 0.8
                if any(j in pl for j in _NEWS_JUNK_PROVIDERS):
                    w -= 1.2
                items.append({
                    "id": x.get("id"), "title": title,
                    "summary": _news_clean(c.get("summary") or ""),
                    "source": "yf", "provider": prov,
                    "url": ((c.get("canonicalUrl") or {}).get("url")
                            or (c.get("clickThroughUrl") or {}).get("url") or ""),
                    "age_h": age_h, "weight": w,
                })
            return items

        seen_ids = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(_yf_news, s) for s in ("^GSPC", "^DJI", "^IXIC", "SPY", "QQQ")]
            for f in futs:
                try:
                    for it in f.result(timeout=max(5.0, left())):
                        if it["id"] and it["id"] in seen_ids:
                            continue
                        seen_ids.add(it["id"])
                        pool.append(it)
                except Exception as exc:  # noqa: BLE001
                    log.warning("top10 yf pool: %s", exc)

        # -- importance pool: curated RSS (each guarded; freshness asserted) --
        feeds = [
            ("https://www.cnbc.com/id/100003114/device/rss/rss.html", "cnbc", 3.0),
            ("https://feeds.content.dowjones.io/public/rss/mw_topstories", "marketwatch", 2.5),
            ("https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-US&gl=US&ceid=US:en", "gnews", 2.0),
            ("https://rss.nytimes.com/services/xml/rss/nyt/Business.xml", "nyt", 2.5),
        ]
        for url, feed, w in feeds:
            if left() < 8:
                log.warning("top10: deadline reached, skipping remaining feeds")
                break
            try:
                pool.extend(_rss_items(url, feed, w))
            except Exception as exc:  # noqa: BLE001
                log.warning("top10 feed %s: %s", feed, exc)

        if not pool:
            if diag:
                diag.warn("Top 10 news: every source came back empty")
            return []

        # Weinstein headline meter — side product of the pool, never fatal.
        try:
            HEADLINE_METER.update(_headline_meter(pool))
        except Exception as exc:  # noqa: BLE001
            log.warning("headline meter skipped: %s", exc)

        # -- cluster near-duplicate stories across outlets by title tokens.
        # Match against the SEED item's fixed token set (never a growing union):
        # a union that accretes tokens as items join lets story A→B→C chain
        # through a shared bridge word even when A and C are unrelated (audit
        # 2026-07-08). Anchoring to the seed keeps every member close to it. --
        clusters: List[dict] = []
        for it in pool:
            toks = _news_tokens(it["title"])
            if len(toks) < 2:
                continue
            best = None
            for cl in clusters:
                shared = len(toks & cl["tokens"])
                ratio = shared / max(1, min(len(toks), len(cl["tokens"])))
                if shared >= 4 or (shared >= 3 and ratio >= 0.5):
                    best = cl
                    break
            if best is None:
                clusters.append({"tokens": frozenset(toks), "items": [it]})
            else:
                best["items"].append(it)      # tokens stay pinned to the seed

        # -- score: corroboration + source weight + macro words + freshness --
        scored = []
        for cl in clusters:
            items = cl["items"]
            rep = max(items, key=lambda x: x["weight"])
            body_item = max(items, key=lambda x: len(x.get("summary") or ""))
            body = body_item.get("summary") or ""
            n_feeds = len({x["source"] for x in items})
            fresh = min(x["age_h"] for x in items)
            score = sum(x["weight"] for x in items)
            score += (n_feeds - 1) * 0.8
            score += min(sum(1 for t in _news_tokens(rep["title"]) if t in _NEWS_MACRO_WORDS), 3) * 0.6
            score += 1.0 if fresh <= 6 else (0.5 if fresh <= 14 else 0.0)
            score += 0.5 if len(body) >= 150 else 0.0
            if len(body) > 420:                    # sentence-boundary truncate
                cut = body[:420]
                dot = cut.rfind(". ")
                body = (cut[:dot + 1] if dot > 150 else cut.rstrip() + "…")
            url = rep.get("url") or body_item.get("url") or ""
            if rep["source"] == "gnews" and body_item.get("url"):
                url = body_item["url"]             # prefer a direct link over the redirect
            scored.append({
                "title": rep["title"], "brief": body,
                "provider": rep.get("provider") or rep["source"],
                "url": url, "age_h": fresh, "n_src": n_feeds,
                "score": score,
            })
        scored.sort(key=lambda s: s["score"], reverse=True)
        top = scored[:10]
        for i, s in enumerate(top, 1):
            s["rank"] = i
        _attach_zh_translations(top, diag)   # additive title_zh/brief_zh; fail-safe
        log.info("top10 news: %d pool items -> %d clusters -> %d briefs in %.1fs",
                 len(pool), len(clusters), len(top), time.time() - t_start)
        return top
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"Top 10 news failed: {exc}")
        return []


_NEWS_FEED_DISPLAY = {"cnbc": "CNBC", "marketwatch": "MarketWatch", "nyt": "NY Times",
                      "gnews": "Google News", "yf": "Yahoo Finance"}

NEWS_ZH_CACHE_PATH = os.path.join(WORKSPACE, "news_zh_cache.json")


def _translate_zh_tw(text: str, timeout: float = 6.0) -> Optional[str]:
    """One short EN -> zh-TW translation via the keyless Google endpoint.
    Returns None on ANY failure — callers keep the English original."""
    try:
        if not text or not text.strip():
            return None
        import urllib.parse as _up
        url = ("https://translate.googleapis.com/translate_a/single"
               "?client=gtx&sl=en&tl=zh-TW&dt=t&q=" + _up.quote(text[:900]))
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        out = "".join(seg[0] for seg in (data[0] or []) if seg and seg[0])
        return out.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _attach_zh_translations(stories: List[dict], diag: Optional[Diagnostics] = None,
                            budget_s: float = 25.0) -> None:
    """Traditional-Chinese titles/briefs for the Top-10 briefs (2026-07-10 USER).
    ADDITIVE keys title_zh / brief_zh; disk-cached by English text so re-runs and
    unchanged stories cost nothing; time-boxed; NEVER raises — any miss simply
    renders in English."""
    try:
        cache = {}
        if os.path.exists(NEWS_ZH_CACHE_PATH):
            try:
                cache = json.load(open(NEWS_ZH_CACHE_PATH))
            except Exception:  # noqa: BLE001
                cache = {}
        t0 = time.time()
        dirty = False
        misses = 0
        for s in stories:
            for key_en, key_zh in (("title", "title_zh"), ("brief", "brief_zh")):
                en = (s.get(key_en) or "").strip()
                if not en:
                    continue
                if en in cache:
                    s[key_zh] = cache[en]
                    continue
                if time.time() - t0 > budget_s:
                    misses += 1
                    continue
                zh = _translate_zh_tw(en)
                if zh:
                    s[key_zh] = zh
                    cache[en] = zh
                    dirty = True
                else:
                    misses += 1
        if dirty:
            try:
                # keep the cache bounded (~400 newest entries)
                if len(cache) > 400:
                    cache = dict(list(cache.items())[-400:])
                _atomic_write(NEWS_ZH_CACHE_PATH, json.dumps(cache, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                pass
        if misses and diag:
            diag.warn(f"Top 10 news: {misses} translation(s) missed — shown in English")
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"Top 10 news translation skipped: {exc}")


def build_top10_news(stories: List[dict]) -> str:
    """Numbered IBD-style briefs. Collapsible, open by default; absent when the
    fetch produced nothing (the diag panel carries the warning)."""
    if not stories:
        return ""
    cards = []
    for s in stories:
        n = s.get("rank", 0)
        title_en = esc(s.get("title", ""))
        title_zh = esc(s.get("title_zh") or "")
        title = title_zh or title_en          # 2026-07-10 USER: 繁體中文 headline
        url = s.get("url") or ""
        head = (f"<a href='{esc(url)}' target='_blank' rel='noopener'>{title}</a>"
                if url.startswith("http") else title)
        # keep the English original visible (links go to English articles)
        orig = (f"<div class='t10-orig'>{title_en}</div>" if title_zh else "")
        prov = s.get("provider") or ""
        meta_bits = [esc(_NEWS_FEED_DISPLAY.get(prov, prov))]
        age = s.get("age_h")
        if age is not None:
            meta_bits.append(f"{age:.0f}h ago" if age >= 1 else "just in")
        if (s.get("n_src") or 1) > 1:
            meta_bits.append(f"{s['n_src']} outlets")
        brief = esc(s.get("brief_zh") or s.get("brief") or "")
        cards.append(
            f"<div class='t10-card'><div class='t10-head'><span class='t10-num'>{n}</span>"
            f"<span class='t10-title'>{head}</span></div>"
            + orig
            + f"<div class='t10-meta'>{' · '.join(b for b in meta_bits if b)}</div>"
            + (f"<div class='t10-brief'>{brief}</div>" if brief else "")
            + "</div>")
    return (
        "<details class='collapsis' open><summary class='section-title' "
        "style='background-color:var(--surface);color:var(--accent);border-bottom:3px solid var(--accent);'>"
        "MADRRY TOP 10<span class='section-sub'>當日十大要聞 · ranked by "
        "cross-outlet corroboration, no LLM</span></summary>"
        f"<div class='t10-grid'>{''.join(cards)}</div></details>")


def fetch_market_pulse_movers(rs_map: Dict[str, Any], diag: Optional[Diagnostics] = None) -> dict:
    """IBD Market Pulse lists: leaders (RS>=87) up / down on >=1.3x 10-day
    relative volume. Two cheap TradingView POSTs with the house liquidity
    floors; RS is intersected client-side (it lives in the Fred6725 CSV, not in
    TradingView). Sentinel {'ok': False} on any failure."""
    def _one(direction: str) -> List[dict]:
        op = "egreater" if direction == "up" else "eless"
        payload = {
            "filter": [
                {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
                {"left": "close", "operation": "egreater", "right": 10},
                {"left": "average_volume_30d_calc", "operation": "egreater", "right": 500000},
                {"left": "market_cap_basic", "operation": "egreater", "right": 2000000000},
                {"left": "change", "operation": op, "right": 1.5 if direction == "up" else -1.5},
                {"left": "relative_volume_10d_calc", "operation": "egreater", "right": 1.3},
            ],
            "columns": ["name", "close", "change", "relative_volume_10d_calc", "volume"],
            "sort": {"sortBy": "relative_volume_10d_calc", "sortOrder": "desc"},
            "range": [0, 300],
        }
        rows = []
        # diag=None on purpose: tv_post -> _request_json records diag.error() on
        # final failure, which would trip the morning script's errors=0 publish
        # gate. This section is non-fatal — the except below diag.warns instead.
        data = tv_post(payload, label=f"pulse_{direction}", diag=None)
        for r in data.get("data", []):
            d = r.get("d")
            if not d or d[1] is None or d[2] is None:
                continue
            sym = str(d[0]).upper().replace(".", "-")
            rs = rs_map.get(sym)
            if not isinstance(rs, (int, float)) or rs < 87:
                continue
            if sym in EXCLUDED_TICKERS:
                continue
            rows.append({"ticker": sym, "close": float(d[1]), "change": float(d[2]),
                         "relvol": float(d[3] or 0), "rs": int(rs)})
        rows.sort(key=lambda x: x["relvol"], reverse=True)
        return rows[:8]

    try:
        time.sleep(1)
        up = _one("up")
        down = _one("down")
        return {"ok": True, "up": up, "down": down}
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"Market Pulse movers failed: {exc}")
        return {"ok": False, "up": [], "down": []}


_PULSE_STATE = {
    "GREEN": ("CONFIRMED UPTREND", "var(--up)"),
    "YELLOW": ("UPTREND UNDER PRESSURE", "var(--warn)"),
    "RED": ("MARKET IN CORRECTION", "var(--down)"),
}


def _pulse_chip(m: dict, up: bool) -> str:
    col = "val-green" if up else "val-red"
    return (f"<span class='pulse-chip'><a href='https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}'"
            f" target='_blank' rel='noopener'>{esc(m['ticker'])}</a> "
            f"<span class='{col}'>{m['change']:+.1f}%</span>"
            f" <span class='sub'>{m['relvol']:.1f}×</span></span>")


def _vol_phrase(v: Optional[float]) -> str:
    if v is None:
        return ""
    if v >= 8:
        return f"volume up {v:.0f}%"
    if v <= -8:
        return f"volume down {abs(v):.0f}%"
    return "volume about even"


def _hm_soft_line(meter: Optional[dict], spx: Optional[dict]) -> str:
    """Weinstein ch.8 headline-meter soft line for the Big Picture card. ''
    unless printable (>=8 classified headlines). Complacency flag: index down
    >=8% from its 52w high while headlines stay cheerful = historically
    bearish; mirrored gloom-at-lows note. Display-only. Never raises."""
    try:
        m = meter or {}
        if not m.get("printable"):
            return ""
        off = (spx or {}).get("pct_off_52wk")
        idx = m.get("idx") or 0.0
        pre = ""
        if off is not None and off <= -8.0 and idx >= 0.30:
            pre = ("⚠ Complacency: S&amp;P %.0f%% off its high yet headlines stay "
                   "cheerful — historically bearish (Weinstein ch.8). " % abs(off))
        elif off is not None and off <= -15.0 and idx <= -0.5:
            pre = ("Uniform gloom deep in a decline — crowd capitulation is how "
                   "bottoms form (Weinstein ch.8). ")
        return ("<p class='sub'>%sHeadline meter: <b>%s</b> — %d cheer / %d fear of %d "
                "market headlines. Psychological potential only — act only when the "
                "technical indicators agree.</p>"
                % (pre, esc(m.get("bucket") or "mixed"), m.get("n_euphoria", 0),
                   m.get("n_doom", 0), m.get("n_scoped", 0)))
    except Exception:  # noqa: BLE001
        return ""


def build_big_picture(market_data: List[dict], breadth: dict, t2108: dict,
                      regime: str, allow_breakouts: bool,
                      leader_stats: dict, movers: dict,
                      headline_meter: Optional[dict] = None) -> str:
    """IBD 'The Big Picture' column: a short data-composed narrative of the
    day's action + the Market Pulse box (outlook state, IBD dist-day counts,
    leaders up/down in volume). Pure formatter — no network I/O."""
    try:
        by_tk = {m["ticker"]: m for m in market_data if m}
        ixic, spx = by_tk.get("^IXIC"), by_tk.get("^GSPC")
        iwm, ndx = by_tk.get("IWM"), by_tk.get("^NDX")
        # Degrade gracefully: both cards come from the same Yahoo endpoint, so a
        # single miss is rare — but if one fails, still render from the survivor
        # rather than dropping the whole section (audit 2026-07-08).
        if not ixic and not spx:
            return ""
        ref = spx or ixic          # reference index for cross-index comparisons

        def _act(md):
            chg = md.get("change_pct", 0.0)
            if chg > 0.05:
                return f"rose {abs(chg):.1f}%"
            if chg < -0.05:
                return f"fell {abs(chg):.1f}%"
            return "closed about flat"

        def _dd(md):
            return str(md.get("dist_days_ibd", 0)) if md else "n/a"

        # P1 — the tape: index moves with the volume comparison O'Neil reads.
        if ixic and spx:
            vol_bits = [p for p in (_vol_phrase(ixic.get("vol_vs_prev")),) if p]
            p1 = (f"The Nasdaq Composite {_act(ixic)} and the S&amp;P 500 {_act(spx)}"
                  + (f", with Nasdaq {vol_bits[0]} versus the prior session" if vol_bits else "") + ".")
        else:
            one = ixic or spx
            name = "Nasdaq Composite" if ixic else "S&amp;P 500"
            vb = _vol_phrase(one.get("vol_vs_prev"))
            p1 = (f"The {name} {_act(one)}"
                  + (f", with {vb} versus the prior session" if vb else "") + ".")
        if iwm and abs(iwm.get("change_pct", 0) - ref.get("change_pct", 0)) >= 0.75:
            lead = "outperformed" if iwm["change_pct"] > ref["change_pct"] else "lagged"
            p1 += f" Small caps {lead}: the Russell 2000 ETF {_act(iwm)}."
        elif ndx and abs(ndx.get("change_pct", 0) - ref.get("change_pct", 0)) >= 0.75:
            lead = "led" if ndx["change_pct"] > ref["change_pct"] else "lagged"
            p1 += f" Big-cap tech {lead}: the Nasdaq-100 {_act(ndx)}."

        # P2 — character: distribution, breadth, leaders. "IBD-style" not "IBD
        # count": we count close-down >=0.2% on higher volume over 25 sessions
        # with a close-based 5% expiry, but have no intraday data for stalling
        # days, so the number runs a touch lighter than IBD's published one.
        p2 = (f"Distribution days (25 sessions, IBD-style; excl. stalling days): "
              f"Nasdaq {_dd(ixic)}, S&amp;P 500 {_dd(spx)}.")
        if breadth.get("ok"):
            p2 += (f" {breadth['above50']:.0f}% of S&amp;P 500 members hold their 50-day line"
                   f" ({breadth['above200']:.0f}% the 200-day).")
        if leader_stats.get("n"):
            p2 += (f" Among the {leader_stats['n']} scanned leaders, "
                   f"{leader_stats['below50']:.0f}% sit below their own 50-day.")
        if t2108 and t2108.get("ok") and t2108.get("t2108") is not None:
            p2 += f" T2108 stands at {t2108['t2108']:.0f}%."

        # P3 — the verdict, in regime terms the rest of the report already uses.
        state, scol = _PULSE_STATE.get(regime, _PULSE_STATE["YELLOW"])
        bo = "breakout buys are ON" if allow_breakouts else "breakout buys are OFF"
        p3 = (f"The 10-tell regime verdict is <b style='color:{scol};'>{regime}</b> — "
              f"in IBD terms, <b style='color:{scol};'>{state.title()}</b> — and {bo}.")
        if regime == "RED":
            # A rally count only means something off a FRESH low — past ~25
            # sessions the 60-bar argmin is just an old uptrend low, and "day 60"
            # would read as noise (caught in the standalone exercise 2026-07-08).
            rd = max((ixic or {}).get("rally_day", 0), (spx or {}).get("rally_day", 0))
            fresh = 0 < rd <= 25
            ftd_ix, ftd_sp = (ixic or {}).get("ftd_today"), (spx or {}).get("ftd_today")
            if fresh and (ftd_ix or ftd_sp):
                which = "Nasdaq" if ftd_ix else "S&amp;P 500"
                p3 += (f" <b>Today qualified as a possible follow-through day on the {which}</b>"
                       " (day-4+ rally of 1.2%+ on rising volume) — watch for confirmation.")
            elif fresh:
                p3 += (f" Rally attempt: day {rd}. A follow-through day (a 1.2%+ gain on rising"
                       " volume, day 4 or later) would signal a new uptrend attempt.")
            else:
                p3 += (" Watch for a fresh rally attempt — the first close above a new low"
                       " starts the follow-through-day count.")

        # Market Pulse box.
        pulse_rows = [f"<div class='pulse-state' style='color:{scol};border-color:{scol};'>{state}</div>"]
        pulse_rows.append(
            f"<div class='pulse-line sub'>Dist days: IXIC {_dd(ixic)} · "
            f"SPX {_dd(spx)} <span class='sub'>(25-session, close-based 5% expiry)</span></div>")
        if movers.get("ok"):
            up_h = "".join(_pulse_chip(m, True) for m in movers.get("up", [])) or "<span class='sub'>none today</span>"
            dn_h = "".join(_pulse_chip(m, False) for m in movers.get("down", [])) or "<span class='sub'>none today</span>"
            pulse_rows.append(f"<div class='pulse-line'><b>Leaders up in volume</b><br>{up_h}</div>")
            pulse_rows.append(f"<div class='pulse-line'><b>Leaders down in volume</b><br>{dn_h}</div>")
            pulse_rows.append("<div class='pulse-line sub'>leaders = RS≥87 · move ≥1.5% · ≥1.3× 10-day rel-vol</div>")
        else:
            pulse_rows.append("<div class='pulse-line sub'>Leaders up/down in volume unavailable (screener fetch failed).</div>")

        # 2026-07-10 USER: too complicated -> collapsed by default with the ONLY
        # must-see facts on the summary line (verdict · breakouts · dist days ·
        # breadth); the narrative + Market Pulse open on demand.
        summary_bits = [f"<b style='color:{scol};'>{regime}</b>",
                        esc(state.title()),
                        ("Breakouts ON" if allow_breakouts else "Breakouts OFF"),
                        f"Dist days IXIC {_dd(ixic)} · SPX {_dd(spx)}"]
        if breadth.get("ok"):
            summary_bits.append(f"50MA breadth {breadth['above50']:.0f}%")
        return (
            "<details class='collapsis'><summary class='section-title' "
            "style='background-color:var(--surface);color:var(--accent);border-bottom:3px solid var(--accent);'>"
            "THE BIG PICTURE<span class='section-sub'>"
            + " · ".join(summary_bits)
            + " — tap for the story</span></summary>"
            "<div class='market-panel bigpic'>"
            "<div class='bigpic-wrap'>"
            f"<div class='bigpic-text'><p>{p3}</p><p>{p1}</p><p>{p2}</p>{_hm_soft_line(headline_meter, spx)}</div>"
            f"<div class='market-card pulse-box'><h3>Market Pulse</h3>{''.join(pulse_rows)}</div>"
            "</div></div></details>")
    except Exception as exc:  # noqa: BLE001
        log.warning("build_big_picture failed: %s", exc)
        return ""


WEEKLY_REVIEW_N = 15


def fetch_weekly_review(rs_map: Dict[str, Any], diag: Optional[Diagnostics] = None) -> List[dict]:
    """YOUR WEEKLY REVIEW rows: quality leaders up >=5% over the trailing week
    (TradingView Perf.W), RS>=80, above the 200-day, within 25% of the 52-week
    high, house liquidity floors. One screener POST + one batched yfinance
    download for the charts. Returns {'ok': bool, 'rows': [...]} — 'ok' is False
    on a fetch failure or an empty RS map, so the renderer can tell a genuinely
    quiet week from a data outage (never raises)."""
    if not rs_map:                     # RS-gating an empty map guarantees 0 rows
        if diag:
            diag.warn("Weekly Review: RS map empty — section unavailable")
        return {"ok": False, "rows": []}
    try:
        time.sleep(1)
        payload = {
            "filter": [
                {"left": "type", "operation": "in_range", "right": ["stock", "dr"]},
                {"left": "close", "operation": "egreater", "right": 10},
                {"left": "average_volume_30d_calc", "operation": "egreater", "right": 500000},
                {"left": "market_cap_basic", "operation": "egreater", "right": 2000000000},
                {"left": "close", "operation": "egreater", "right": "SMA200"},
                {"left": "Perf.W", "operation": "egreater", "right": 5},
            ],
            "columns": ["name", "close", "Perf.W", "Perf.1M", "price_52_week_high",
                        "sector", "industry", "relative_volume_10d_calc", "change"],
            "sort": {"sortBy": "Perf.W", "sortOrder": "desc"},
            "range": [0, 400],
        }
        # diag=None: non-fatal section — keep tv_post's failure out of diag.errors
        # (else the errors=0 publish gate trips). The except below diag.warns.
        data = tv_post(payload, label="weekly_review", diag=None)
        cands: List[dict] = []
        for r in data.get("data", []):
            d = r.get("d")
            if not d or d[1] is None or d[2] is None:
                continue
            sym = str(d[0]).upper().replace(".", "-")
            if sym in EXCLUDED_TICKERS:
                continue
            rs = rs_map.get(sym)
            if not isinstance(rs, (int, float)) or rs < 80:
                continue
            hi52 = d[4]
            off_high = ((hi52 - d[1]) / hi52 * 100) if hi52 else None
            if off_high is None or off_high > 25:
                continue
            cands.append({
                "ticker": sym, "close": float(d[1]), "perf_w": float(d[2]),
                "perf_1m": float(d[3]) if d[3] is not None else None,
                "off_high": off_high, "sector": d[5] or "", "industry": d[6] or "",
                "relvol": float(d[7]) if d[7] is not None else None,
                "rs": int(rs),
            })
            if len(cands) >= WEEKLY_REVIEW_N:
                break
        if not cands:
            return {"ok": True, "rows": []}       # genuine quiet week
        hmap = fetch_histories_batch([c["ticker"] for c in cands], period="1y", min_rows=60)
        for c in cands:
            hist = hmap.get(c["ticker"])
            c["spark"] = make_candle_chart(hist, _chart_plan(c, hist), CHART_WINDOW)
            c["wk_vol_x"] = None
            try:
                if hist is not None and len(hist) >= 55:
                    v = hist["Volume"].astype(float)
                    # Drop zero-volume bars (a TV-patched EOD-lag bar carries vol=0,
                    # see :1106) and compare per-day AVERAGES so a dropped bar can't
                    # itself deflate the week side (audit 2026-07-08).
                    recent = [x for x in v.iloc[-5:] if x > 0]
                    basev = [x for x in v.iloc[-55:-5] if x > 0]
                    if len(recent) >= 3 and len(basev) >= 20:
                        c["wk_vol_x"] = (sum(recent) / len(recent)) / (sum(basev) / len(basev))
            except Exception:  # noqa: BLE001
                pass
        return {"ok": True, "rows": cands}
    except Exception as exc:  # noqa: BLE001
        if diag:
            diag.warn(f"Weekly Review fetch failed: {exc}")
        return {"ok": False, "rows": []}


def generate_weekly_review_table(data: dict) -> str:
    """IBD 'Your Weekly Review' analog: the week's strongest quality leaders,
    each with the house candlestick chart. Rolling 5-session window so the
    daily report always carries it (Saturday's run covers the calendar week).
    `data` is fetch_weekly_review's {'ok', 'rows'} sentinel."""
    rows = (data or {}).get("rows", [])
    ok = (data or {}).get("ok", True)
    out = ["<div class='section-title' style='background-color:var(--surface);color:var(--accent);"
           "border-bottom:3px solid var(--accent);'><span class='tdot' style='background:var(--accent);'></span>"
           "YOUR WEEKLY REVIEW<span class='section-sub'>leaders up ≥5% this week · RS≥80 · above 200-day · "
           "within 25% of 52-wk high</span></div>",
           "<div class='table-container rowcards-container'><table class='rowcards'>",
           "<thead><tr><th data-col='tk'>Ticker</th><th data-col='chart'>Chart</th>"
           "<th data-col='wk' class='num'>Week %</th><th data-col='rs' class='num'>RS</th>"
           "<th data-col='off' class='num'>Off High</th><th data-col='m1' class='num'>1M %</th>"
           "<th data-col='note'>Note</th></tr></thead>"]
    if not rows:
        msg = ("No leader gained 5%+ this week — a quiet tape for the strongest names."
               if ok else "Weekly Review unavailable — screener or RS data did not load this run.")
        out.append(f"<tr><td colspan='7' style='color:#82827c;'>{msg}</td></tr>")
    for m in rows:
        wk = m.get("perf_w", 0.0)
        note_bits = [f"Up {wk:+.1f}% for the week"]
        if m.get("wk_vol_x"):
            note_bits.append(f"on {m['wk_vol_x']:.1f}× average weekly volume")
        note = " ".join(note_bits) + f"; {m.get('off_high', 0):.0f}% off the 52-week high."
        sec = esc(m.get("sector", ""))
        ind = esc(m.get("industry", ""))
        m1 = m.get("perf_1m")
        out.append(
            f"<tr data-sector='{sec}'>"
            + _tk_cell(m)
            + _chart_cell(m.get("spark", ""), m.get("close", 0))
            + f"<td class='c-stat num' data-label='Week %' data-sort='{wk:.2f}'><span class='val-green'>{wk:+.1f}%</span></td>"
            + f"<td class='c-stat num' data-label='RS' data-sort='{m.get('rs', 0)}'>{m.get('rs', 0)}</td>"
            + f"<td class='c-stat num' data-label='Off High' data-sort='{m.get('off_high', 0):.1f}'>{m.get('off_high', 0):.1f}%</td>"
            + (f"<td class='c-stat num' data-label='1M %' data-sort='{m1:.1f}'>{m1:+.1f}%</td>" if m1 is not None
               else "<td class='c-stat num' data-label='1M %' data-sort='0'>—</td>")
            + _narr_cell(m.get("ticker", ""), f"<span class='theme-tag'>{sec}</span> <span class='sub'>{ind}</span><br>"
                                              f"<span class='sub'>{esc(note)}</span>")
            + "</tr>")
    out.append("</table></div>")
    return "".join(out)


def build_runbar(counts: Dict[str, int], market_modifier: float, runtime: float,
                 regime: str, allow_breakouts: bool) -> str:
    cls = {"GREEN": "green", "YELLOW": "warn", "RED": "red"}.get(regime, "")
    # No emoji here: a red 🚫 inside a YELLOW chip reads as mixed signals.
    bo = ("<span style='color:var(--green);'>Breakouts ON</span>" if allow_breakouts
          else "<span style='color:var(--red);'>Breakouts OFF</span>")
    # GLOBAL chips only (2026-07-10 USER: per-section counts were mixed into the
    # global bar — tier counts now render inside their own tabs via build_tab_counts).
    # REV 10 (USER 2026-07-18: "the [runbar] at the top is very not useful"):
    # collapsed to ONE compact chip. Dropped: Mkt Mod, Run time, the 🔄 Refresh
    # Prices button and the "prices frozen at scan" stamp. refreshPrices() is
    # only ever invoked by that button and null-guards #liveStamp, so removing
    # both is safe; the live-quote helper stays in PAGE_JS for future use.
    return f"""
    <div class="runbar">
        <span class="chip {cls}"><b>{regime}</b> · {bo} · {market_modifier}×</span>
    </div>"""


def build_tab_counts(pairs: List[Tuple[str, int, str]]) -> str:
    """Per-tab count pills (label, count, extra chip class). Lives at the TOP of
    the tab whose sections it counts — A+/A/A− in the MADRRY tab, HVE/U&R in
    Pivots, Short in Short (2026-07-10 USER placement fix)."""
    chips = "".join(
        f"<span class='chip {xcls}'>{esc(lab)} <b>{n}</b></span>"
        for lab, n, xcls in pairs)
    return f"<div class='runbar' style='justify-content:flex-start;margin:2px 0 14px;'>{chips}</div>"


def build_diag_panel(diag: Diagnostics) -> str:
    if not diag.errors and not diag.warnings:
        return ""
    items = "".join(f"<li>✗ {esc(e)}</li>" for e in diag.errors)
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
    # trendline edge now reads the VERIFIED v2 engine (USER-RATIFIED 2026-07-04:
    # the legacy block's lines carry no validity/break checking); legacy
    # trendline_data remains computed+persisted for snapshot continuity only.
    _tf = [str(f) for f in (m.get("tl_flags") or [])]
    if any(f.startswith(("at_UTL", "at_TSL", "fresh_break_up")) for f in _tf):
        e += 1                                            # 📐 trendline v2 (verified)
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


# Plain-language meaning for each regime tell (2026-07-06 USER: click-to-expand).
# Matched by label PREFIX (labels carry live numbers/entities); html-escaped
# prefixes match the escaped labels. First hit wins; fallback is generic.
_REGIME_EXPLAIN: List[Tuple[str, str]] = [
    ("Trend", "Are IXIC and SPX both above their rising 10&gt;21-day MAs? Both up = green tape, "
              "both down = red. Setups fight the tide when this is red — 環境優先."),
    ("Breadth", "Percent of S&amp;P 500 stocks above their own 50-day MA. &gt;50% = broad participation; "
                "&lt;40% = a narrow rally carried by few names — fragile."),
    ("&gt;200MA", "Percent of S&amp;P 500 stocks above their 200-day MA — the market's long-term health. "
                 "Below 50% means most stocks are in downtrends even if the index looks fine."),
    ("Distribution", "O'Neil distribution days in the last ~20 sessions: the index fell &gt;0.2% on HIGHER "
                     "volume than the prior day = institutions selling into strength. 4-5 = caution, 6+ = danger."),
    ("Climax", "How stretched the leading index is above its 50-day MA, ranked against its ENTIRE history "
               "(P90 = more extended than 90% of all days). P90+ often precedes a blow-off / mean reversion."),
    ("Leaders", "Percent of the market's leading stocks that broke below their 50-day MA / stopped making new "
                "highs. Leaders roll over BEFORE the index does — this is the early-warning line."),
    ("Topping", "The index closed BELOW the floor of its recent ~20-session range — a range breakdown after "
                "an advance is the classic topping footprint."),
    ("Range", "Where the index sits inside its recent ~20-session range. Holding the upper half = healthy; "
              "sliding toward the floor = watch for a breakdown."),
    ("T2108", "Worden T2108: percent of NYSE stocks above their 40-day MA. &lt;40% = weak internals. "
              "DIVERGE = the index makes highs while T2108 falls — a hard red flag."),
    ("Breakout win: building", "The win-rate of this scanner's own recent breakout signals — "
                               "still building a sample (needs 8+ graded trades to score)."),
    ("Breakout win", "The win-rate of THIS scanner's own recent breakout signals. When fresh breakouts keep "
                     "failing (&lt;35%), the tape is hostile no matter how good the charts look — hard override."),
    ("Leader momentum", "Percent of the market's leading stocks that broke below their 50-day MA / stopped "
                        "making new highs — unavailable this run."),
    ("Sectors", "How many of the leading sectors are losing relative strength vs SPX. Leadership rotating "
                "off (2+ leaders weak) is an early distribution signal."),
    ("Sector RS", "Per-sector relative strength vs SPX — unavailable this run."),
    ("VIX", "The options-market fear gauge. Calm (&lt;20) is informational only; &gt;20 or a +15% one-day "
            "spike = expect wider swings and failed follow-through."),
]


def _regime_explain(lbl: str) -> str:
    for prefix, txt in _REGIME_EXPLAIN:
        if lbl.startswith(prefix):
            return txt
    return "One of the ten market-top early-warning tells scored into the regime verdict."


def build_regime(market_data: List[dict], breadth: dict,
                 t2108: Optional[dict] = None, vix: Optional[dict] = None,
                 sector_rs: Optional[dict] = None, leader_stats: Optional[dict] = None,
                 winrate: Optional[dict] = None) -> Tuple[str, str, bool]:
    """Market-top early-warning grid: 10 scored tells (+ VIX info) rolled into a
    GREEN/YELLOW/RED verdict with hard 🔴 overrides. Returns (html, regime, allow_breakouts)."""
    md = {m["ticker"]: m for m in market_data}
    ixic, spx = md.get("^IXIC"), md.get("^GSPC")   # S&P 500 index (was SPY ETF)
    br50 = breadth.get("above50", 50.0)
    br200 = breadth.get("above200", 50.0)
    dist_max = max((m.get("dist_days", 0) for m in market_data), default=0)
    ext_max = max((m.get("ext_50", 0) for m in market_data), default=0.0)

    sigs = []   # (state, label)  state ∈ g/y/r/i ; i = info (not scored)

    # 1) Trend
    if ixic and spx and ixic["trend"] == "GREEN" and spx["trend"] == "GREEN":
        sigs.append(("g", "Trend ✓ (IXIC/SPX 10&gt;21)"))
    elif ixic and spx and ixic["trend"] == "RED" and spx["trend"] == "RED":
        sigs.append(("r", "Trend ✗ (IXIC &amp; SPX 10&lt;21)"))
    else:
        sigs.append(("y", "Trend mixed (IXIC/SPX)"))
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
    #    (same P## as the IXIC/SPX cards; take the more-stretched of the two).
    climax = []
    for tk in ("^IXIC", "^GSPC"):
        m = md.get(tk)
        if m is not None:
            p = ext_percentile(tk, "SMA50", m.get("ext_50", 0.0))
            if p is not None:
                climax.append((p, tk, m.get("ext_50", 0.0)))
    if climax:
        cp, ctk, cext = max(climax)
        cst = "r" if cp >= 90 else ("y" if cp >= 75 else "g")
        cflag = " ⚠️stretched" if cp >= 90 else (" hot" if cp >= 75 else "")
        sigs.append((cst, f"Climax {_index_display(ctk)} +{cext:.0f}% · P{cp:.0f}{cflag}"))
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
    # 7) Topping-range breakdown (IXIC/SPX)
    idx_below = [m for m in (ixic, spx) if m and m.get("close_below_range")]
    idx_pos = [m.get("range_pos", 1.0) for m in (ixic, spx) if m]
    if idx_below:
        sigs.append(("r", f"Topping: {'/'.join(_index_display(m['ticker']) for m in idx_below)} broke range"))
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

    color = {"GREEN": "#54b87f", "YELLOW": "#d3a04d", "RED": "#e06c6a"}[regime]
    bg = {"GREEN": "rgba(84,184,127,.08)", "YELLOW": "rgba(211,160,77,.08)",
          "RED": "rgba(224,108,106,.10)"}[regime]
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
    cmap = {"g": "#54b87f", "y": "#d3a04d", "r": "#e06c6a", "i": "#aecfe8"}
    # 2026-07-06 USER: each tell is click-to-expand — a plain-language meaning
    # opens under the chip (native <details>, no JS). Matched by label prefix
    # because the labels carry live numbers.
    grid = "".join(
        f"<details class='reg-sig-w'><summary class='reg-sig' style='border-color:{cmap[s]};color:{cmap[s]};'>{dot[s]}{lbl}</summary>"
        f"<div class='reg-exp'>{_regime_explain(lbl)}</div></details>"
        for s, lbl in sigs
    )
    html = (
        f"<div class='regime' style='border-color:{color};background:{bg};'>"
        f"<div class='reg-head'><span style='color:{color};'>REGIME: {regime}</span>"
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
        # DASHBOARD-ONLY (same pattern as the ANTS bonus): lesson-confluence
        # picks (>=3 of the 4 tutorial lessons' entry criteria at once) float
        # up per the user's 2026-07-05 direction. The order plan calls with
        # the default, so the IBKR draft ranking is byte-identical.
        def _lesson_bonus(s):
            # graded: 4/4 lessons +2, 3/4 +1 (7 vs 42 names on the first
            # badged run - the full-house is the scarce signal)
            return max(0, len(s.get("lesson_confluence") or []) - 2)
        pool.sort(key=lambda r: (not r[1]["_high_conviction"], r[0],
                                 -(r[1]["_edges"] + _ants_edge_bonus(r[1])
                                   + _lesson_bonus(r[1])),
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
    each card shows its N/4 legs + whether it was drafted or ⤷ skipped."""
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
            _ds = (f'<span style="color:var(--up);" title="drafted to IBKR — top-3 by high-conviction '
                   f'then tier/edges/M.E.T.A., one per sector">DRAFT #{draft_pos[s["ticker"]]}</span>')
        elif _sec in drafted_sectors:
            _ds = '<span title="one-per-sector rule: this sector is already taken by a higher-ranked pick">sector taken</span>'
        else:
            _ds = '<span title="shown for context — outside the top-3 drafted">watch</span>'
        _legcol = "var(--accent-2)" if _legs >= 3 else "var(--text-3)"
        _score = (f'<span class="tp-legs" style="color:{_legcol};" title="high-conviction legs met (of 4): '
                  f'coiled · RS≥90 · within 10% of 52wk high · risk≤3.5%. 4/4 = the validated SPY-beating edge; '
                  f'3/4 = one leg away; ≤2/4 = no measured edge.">{_legs}/4 legs</span>')
        tcol = {"A+": "var(--up)", "A": "var(--warn)", "A-": "var(--down)"}.get(tr, "var(--text-3)")
        _tcls = {"A+": "t-aplus", "A": "t-a", "A-": "t-aminus"}.get(tr, "")
        _atp = ""
        if s.get("ants_ok") and s.get("ants_level", 0) >= 1:
            _atp_s = ("·%db" % s.get("ants_chain", 0)) if s.get("ants_chain") else ""
            _atp = '<span class="tp-meta" title="ANTS accumulation (David Ryan)">ANTS %s%s</span>' % (esc(s.get("ants_label", "")), _atp_s)
        elif s.get("ants_ok") and s.get("ants_3m_peak", 0) >= 4:
            _atp = '<span class="tp-meta">ANTS 3M %s</span>' % esc(_ANTS_LABELS.get(s.get("ants_3m_peak", 0), ""))
        _rsl = ""
        if s.get("rs_ok") and s.get("rs_nh_before_price"):
            _rsl = '<span class="tp-meta" style="color:var(--accent-2);">RS▲ ‹ Px</span>'
        elif s.get("rs_ok") and s.get("rs_new_high"):
            _rsl = '<span class="tp-meta" style="color:var(--accent-2);">RS▲ Leader</span>'
        _lc = s.get("lesson_confluence") or []
        _lcb = (f'<span class="tp-tier" style="color:var(--warn);border-color:var(--bd-warn);" '
                f'title="{len(_lc)}/4 tutorial lessons meet their quality entry criteria at once: '
                f'{esc(" + ".join(str(x) for x in _lc))}">{len(_lc)}/4 LESSONS</span>') if len(_lc) >= 3 else ""
        _hc = ('<span class="tp-tier" style="color:var(--accent-2);border-color:var(--accent);" '
               'title="coiled · RS≥90 · within 10% of 52wk high · risk≤3.5% — the validated SPY-beating overlay">HI-CONV</span>'
               ) if s.get("_high_conviction") else ""
        # drafted cards carry an id so a plan cell's "→ IBKR draft" chip can jump here
        _tpid = f" id=\"tp-{esc(s['ticker'])}\"" if s["ticker"] in draft_pos else ""
        cards.append(f"""
        <div class="tp-card {_tcls}"{_tpid}>
            <div class="tp-top"><a href="https://www.tradingview.com/chart/?symbol={esc(s['ticker'])}" target="_blank">{esc(s['ticker'])}</a>
                <span class="tp-tier" style="color:{tcol};border-color:{tcol};">{tr}</span>
                <span class="tp-edges" title="independent verified edges stacked on this entry">↯{s.get('_edges',0)}</span>{_hc}{_lcb}
                <span class="tp-draft">{_ds}</span></div>
            <div class="tp-px">{_lp(s['ticker'], s['close'], style='', entry=s.get('entry'), stop=s.get('stop'))} {_score}</div>
            <div class="tp-signals"><span class="tp-meta">M.E.T.A. {s.get('meta_score',0)}</span>{_atp}{_rsl}</div>
            <div class="tp-theme">{esc(s['theme'])}</div>
        </div>""")
    return (f"<div class='section-title' style='background-color:var(--surface);'>"
            f"<span class='tdot' style='background:var(--up);'></span>"
            f"TOP PICKS — TODAY'S BEST MULTIPLE-EDGE SETUPS</div>"
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
        col = "#54b87f" if r["pct"] >= IND_RS_STRONG else ("#d3a04d" if r["pct"] >= 70 else "#82827c")
        n = held.get(r["industry"], 0)
        mark = f" <b style='color:#56d364;'>🎯{n}</b>" if n else ""
        chips.append(
            f"<span class='theme-chip' data-sector='{esc(r['sector'])}' style='border-color:{col};'>"
            f"🏭 {esc(r['industry'])} <b style='color:{col};'>{r['pct']}</b>"
            f"<span style='color:#82827c;'> · {esc(r['sector'])}</span>{mark}</span>")
    n_strong = sum(1 for r in rows if r["pct"] >= IND_RS_STRONG)
    return (
        "<details class='collapsis'><summary class='section-title' style='background-color:var(--surface);"
        "color:#56d364;border-bottom:3px solid #56d364;'>"
        f"HOT INDUSTRY GROUPS — Fred6725 RS · {n_strong} groups ≥{IND_RS_STRONG} "
        f"(top 12 of {len(rows)}; ● = your picks' group)</summary>"
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
        col = "#54b87f" if ratio >= 0.66 else ("#d3a04d" if ratio >= 0.4 else "#82827c")
        nh = f" · {a['near']} NH" if a["near"] >= 2 else ""
        chips.append(
            f"<span class='theme-chip' data-sector='{esc(th)}' style='border-color:{col};'>"
            f"{esc(th)} <b style='color:{col};'>{a['n']}</b> · avg {avg:.0f}{nh}</span>"
        )
    chips.append("<span class='theme-chip' id='themeClear' style='border-color:#82827c;display:none;'>✕ Show all</span>")
    return (f"<div class='section-title' style='background-color:var(--surface);color:#d3a04d;border-bottom:3px solid #d3a04d;'>"
            f"HOT SECTORS — 今日強勢板塊 · tap to filter</div>"
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
    summary = (f'<summary class="section-title" style="background-color:var(--tint-green);color:#54b87f;border-bottom:3px solid #54b87f;">'
               f'NEW 52-WEEK HIGHS — LEADERSHIP · {total} new highs · '
               f'{n_persist}★ persistent · {len(clusters)} sector clusters</summary>')
    out = ['<details class="collapsis nh">', summary]

    # --- sector-cluster breadth strip ---
    if clusters:
        chips = []
        for s, n in clusters:
            chips.append(f"<span class='theme-chip' data-sector='{esc(s)}' style='border-color:#54b87f;'>"
                         f"{esc(s)} <b style='color:#54b87f;'>×{n}</b></span>")
        out.append(
            "<div style='background:var(--tint-green);padding:10px 14px;margin:0 0 14px;border-radius:8px;'>"
            "<div style='color:#54b87f;font-weight:bold;font-size:var(--fs-body);margin-bottom:6px;'>"
            f"Sectors making COLLECTIVE new highs today ({total} new highs total · ≥3 = cluster)</div>"
            f"<div class='hot-themes' style='margin:0;'>{''.join(chips)}</div></div>")
    else:
        out.append(f"<div style='color:#82827c;font-size:var(--fs-table);margin-bottom:12px;'>"
                   f"{total} new 52-wk highs today · no single sector reached a ≥3 cluster.</div>")

    # --- leaders table: constructive (🟢) + persistent (⭐) names ---
    if not green:
        out.append("<div style='color:#82827c;font-size:var(--fs-body);padding:6px 0;'>"
                   "No constructive or persistent names today — "
                   "the rest of the new highs are extended or still developing.</div></details>")
        return "".join(out)

    out.append('<div class="table-container rowcards-container"><table data-schema="newhighs2" class="rowcards">')
    out.append("<thead><tr><th data-col='tk'>Ticker</th><th data-col='price'>Chart</th><th data-col='plan'>Continuation Plan</th>"
               "<th data-col='narr'>Narrative</th>"
               "<th data-col='adr' title='Average Daily Range — 20-day avg of (High/Low−1), % · how much it typically moves per day (TradingView ADRP, or an equivalent 20-day calc on the external/HTF tabs)'>ADR</th><th data-col='rs'>RS</th>"
               "<th data-col='pattern'>3-Month Pattern &amp; Persistence</th><th data-col='meta'>M.E.T.A.</th>"
               + _MA_YOY_HEADERS + "</tr></thead>")
    _tag_style = {"GRN": ("var(--tint-green)", "#54b87f"), "YEL": ("var(--tint-yellow)", "#d3a04d"), "RED": ("var(--tint-red)", "#e06c6a")}
    for m in green:
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
        rc = "#54b87f" if (risk or 9) <= 4 else ("#d3a04d" if (risk or 9) <= 6 else "#e06c6a")
        risk_txt = f"{risk}%" if risk is not None else "n/a"
        # pattern badge colored by grade
        dot = {"GRN": "●", "YEL": "●", "RED": "●"}.get(m.get("tag"), "●")
        pbg, pcol = _tag_style.get(m.get("tag"), ("var(--tint-green)", "#54b87f"))
        pattern_badge = f"<div class='squat-badge' style='background:{pbg};color:{pcol};border-color:{pcol};'>{dot} {esc(m['label'])}</div>"
        # persistence badge (⭐ / ⭐⭐) from recurring new highs
        persist_badge = ""
        if m.get("persist_tier"):
            star = "★★" if m["persist_tier"] == "R" else "★"
            pc = "#d3a04d" if m["persist_tier"] == "R" else "#d3a04d"
            persist_badge = (f"<div class='squat-badge' style='background:#221d08;color:{pc};border-color:{pc};font-weight:bold;'>"
                             f"{star} {esc(m['persist_label'])} · {m['nh_3m']} NH-days/3M ({m['weeks_3m']} wks) · {m['nh_1m']}/1M</div>")
        out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
            {_tk_cell(m, entry=m['entry'], stop=m['stop'])}
            {_chart_cell(m.get('spark', ''), m['close'])}
            <td class="c-plan" data-sort="{risk if risk is not None else 999}">
                <div class="entry-box">
                    {_plan_kicker(m) if m.get('plan_src') else "<span class='kicker'>CONTINUATION</span>"}
                    <span class="entry-text">Buy &gt; ${m['entry']}</span><br>
                    <span class="stop-text">Stop: ${m['stop']} <span class="stop-reason">({esc(m.get('stop_reason') or '21EMA / −1.5×ADR')})</span></span><br>
                    <span style="color:{rc};font-size:var(--fs-body);">Risk: {risk_txt}</span>
                </div>
                {_edge_details(m, [_lessons_line(m), _sr_line(m), _pb2_line(m),
                                   _tl_line(m), _ch_line(m),
                                   _stage_line(m), _mans_line(m), _oh_line(m),
                                   _tc_line(m), _group_line(m)])}
            </td>
            {_narr_cell(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span><br><span class="tag">{esc(m['sector'])}</span>{_ind_badge(m)}''')}
            <td class="num c-stat" data-label="ADR" data-sort="{m['adr']}">{m['adr']}%</td>
            <td class="c-stat" data-label="RS" data-sort="{rs_val if isinstance(rs_val,int) else 0}"><span class="score">{esc(rs_val)}</span><br><span class="sub">1M:+{m['perf_1m']}% · 3M:+{m['perf_3m']}%</span></td>
            <td class="c-status" style="font-size:var(--fs-table);text-align:left;">
                {persist_badge}{pattern_badge}
                {fp_html}
                <div style="margin-top:4px;color:#82827c;">{esc(base)} · {m['higher_lows']} HL</div>
                <div><span class="good">+{ext9} vs 9EMA</span> · <span class="{ext50_col}">+{ext50} vs 50EMA</span></div>
            </td>
            {_ext_meta_cell(m)}
            {_ma_cells(m.get('_ma_dist'))}{_fwd_yoy_cell(m['ticker'])}{_eps_accel_cell(m['ticker'])}
        </tr>""")
    out.append("</table></div></details>")
    return "".join(out)


def generate_nh52_monitor_section(pullbacks: List[dict], monitored: List[dict]) -> str:
    """Dedicated tab: every name that printed a 52wk high in the last
    NH52_WATCH_DAYS trading days, re-checked daily. Low-volume pullbacks (the
    awareness signal) are sorted to the top and highlighted."""
    n_pull = len(pullbacks)
    n_break = sum(1 for m in monitored if m.get("tag") == "RED")
    head = (f"<h2 style='margin:4px 0 2px;'>52-Week-High Pullback Monitor</h2>"
            f"<p class='header-sub' style='margin:0 0 14px;'>Names that printed a new "
            f"52wk high in the last {NH52_WATCH_DAYS} trading days, re-checked each run · "
            f"<b style='color:#54b87f;'>{n_pull}</b> low-vol pullback"
            f"{'' if n_pull == 1 else 's'} · "
            f"<b style='color:#e06c6a;'>{n_break}</b> high-vol breakdown"
            f"{'' if n_break == 1 else 's'} · {len(monitored)} watched</p>")
    if not monitored:
        return (head + "<div style='color:#82827c;font-size:var(--fs-body);padding:8px 0;'>"
                "No names on the 52wk-high monitor yet — they accumulate as the daily "
                "scan prints fresh new highs, then stay here for "
                f"{NH52_WATCH_DAYS} trading days.</div>")

    out = [head]
    if pullbacks:
        out.append("<div style='background:var(--tint-green);"
                   "padding:10px 14px;margin:0 0 14px;border-radius:0 8px 8px 0;color:#54b87f;"
                   "font-size:var(--fs-table);'><b>Low-volume pullback</b> = price slipped below "
                   "its 50-day MA or the prior close while volume dried up below its 30-day average "
                   "— supply exhausting, a constructive continuation watch.</div>")
    out.append('<div class="table-container rowcards-container"><table class="rowcards">')
    out.append("<thead><tr><th>Ticker</th><th>Status</th><th>Price</th>"
               "<th>vs 50-MA</th><th>vs Prev Close</th><th>Volume vs 30d Avg</th>"
               "<th>RS</th><th>Watch</th></tr></thead>")
    _tag_col = {"GRN": "#54b87f", "RED": "#e06c6a", "HOLD": "#82827c"}
    for m in monitored:
        col = _tag_col.get(m["tag"], "#82827c")
        spark_html = f"<div class='spark'>{m['spark']}</div>" if m.get("spark") else ""
        rs_val = m.get("rs_rating", "N/A")
        vs50 = m.get("vs_50"); vsprev = m.get("vs_prev"); vr = m.get("vol_ratio")
        vs50_col = "#e06c6a" if (vs50 is not None and vs50 < 0) else "#54b87f"
        vsprev_col = "#e06c6a" if (vsprev is not None and vsprev < 0) else "#54b87f"
        vr_col = "#54b87f" if (vr is not None and vr < 1) else "#e06c6a"
        vs50_txt = f"{vs50:+.1f}%" if vs50 is not None else "–"
        vsprev_txt = f"{vsprev:+.1f}%" if vsprev is not None else "–"
        vr_txt = f"{vr:.2f}×" if vr is not None else "–"
        out.append(f"""<tr>
            <td class="ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a>{spark_html}</td>
            <td><span class="squat-badge" style="background:rgba(0,0,0,0.12);color:{col};border-color:{col};font-weight:bold;">{esc(m['status'])}</span></td>
            <td data-sort="{m['close']}">${m['close']}</td>
            <td data-sort="{vs50 if vs50 is not None else 0}"><span style="color:{vs50_col};">{vs50_txt}</span><br><span class="sub">50MA ${m['sma50']}</span></td>
            <td data-sort="{vsprev if vsprev is not None else 0}"><span style="color:{vsprev_col};">{vsprev_txt}</span></td>
            <td data-sort="{vr if vr is not None else 9}"><span style="color:{vr_col};font-weight:bold;">{vr_txt}</span><br><span class="sub">{'below' if (vr is not None and vr < 1) else 'above'} avg</span></td>
            <td data-sort="{rs_val if isinstance(rs_val,int) else 0}"><span class="score">{esc(rs_val)}</span></td>
            <td data-sort="{m['days_since_high']}"><span style="font-size:var(--fs-table);">{m['days_since_high']}d since high</span><br><span class="sub">{m['high_count']}× NH · last {esc(m.get('last_high') or '–')}</span>{_pb2_line(m)}</td>
        </tr>""")
    out.append("</table></div>")
    return "".join(out)


def generate_short_table(shorts: List[dict]) -> str:
    out = [
        '<div class="section-title bg-short"><span class="tdot"></span>PARABOLIC SHORT — CLIMAX / EXHAUSTION '
        '(乖離過大 · 拋物線見頂)</div>',
        '<div class="table-container rowcards-container"><table class="rowcards">',
        "<thead><tr><th>Ticker</th><th>Price &amp; Extension</th><th>Climax Stats</th>"
        "<th>Short Plan (intraday)</th></tr></thead>",
    ]
    if not shorts:
        out.append("<tr><td colspan='4' style='color:#82827c;'>No parabolic-short "
                   "candidates — nothing is climactically extended right now.</td></tr>")
    else:
        for m in shorts:
            risk = m.get("risk_pct")
            risk_txt = f"{risk}%" if risk is not None else "n/a"
            tt = m.get("to_target")
            tt_txt = f"+{tt}% to 21EMA" if tt is not None else ""
            out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
                <td class="ep-ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a></td>
                <td data-sort="{m['dist9']}">{_lp(m['ticker'], m['close'])}<br><span class="bad">+{m['dist9']}% above 9EMA</span><br><span style="font-size:var(--fs-caption);color:#82827c;">+{m['dist21']}% above 21EMA</span><br>{_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m['theme'])}</span>''')}</td>
                <td data-sort="{m['vol_ratio']}" style="font-size:var(--fs-table);text-align:left;">
                    <span class="bad">Vol {m['vol_ratio']}x</span><br>
                    <span class="warn">{m['gap_ups']} recent gap-up{'s' if m['gap_ups'] != 1 else ''}</span><br>
                    <span style="color:#82827c;">⚡ accel +{m['accel']}%</span><br>
                    <span style="color:#82827c;">1M: +{m['perf_1m']}%</span>
                </td>
                <td data-sort="{risk if risk is not None else 999}">
                    <div class="entry-box" style="border-color:#8a4341;background:rgba(224,108,106,.08);">
                        <span style="color:#e06c6a;font-weight:bold;font-size:var(--fs-table);">SHORT SETUP</span><br>
                        <span class="stop-text">Trigger: break of ORL / AVWAP retest</span><br>
                        <span class="stop-reason">Daily proxy entry ${m['entry']} · stop &gt; day-high ${m['stop']} ({risk_txt})</span><br>
                        <span style="color:#54b87f;">Cover → 21EMA ${m['target']} <span class="stop-reason">({tt_txt})</span></span>
                    </div>
                    <div style="font-size:var(--fs-micro);color:#82827c;margin-top:4px;">⚠️ Intraday stop is far tighter (above ORH, ~0.4–2%). Best when it gaps UP (exhaustion). Stand aside if it reclaims AVWAP.</div>
                    {_sr_line(m)}
                    {_tl_line(m, short=True)}
                    {_ch_line(m, short=True)}
                </td>
            </tr>""")
    out.append("</table></div>")
    return "".join(out)


def generate_stage4_short_table(shorts: List[dict]) -> str:
    """Weinstein ch.7 Stage-4 breakdown table (second table in the Short tab)."""
    out = [
        '<div class="section-title bg-short"><span class="tdot"></span>STAGE-4 BREAKDOWN — WEINSTEIN CH.7 '
        '(跌破支撐 · 空頭第四階段)</div>',
        "<div style='font-size:var(--fs-caption);color:#82827c;margin:0 0 8px;'>"
        "Below a flat/declining 30-week MA · RS ≤25 or falling RS line · industry RS ≤25 · "
        "at/just breaking the shelf · DTC ≥10x excluded. Volume NOT required on breakdowns "
        "(Weinstein ch.2/7 asymmetry). Informational leg — never drafted, learning loops skip shorts.</div>",
        '<div class="table-container rowcards-container"><table class="rowcards">',
        "<thead><tr><th>Ticker</th><th>Stage &amp; RS</th><th>Structure</th>"
        "<th>Short Plan (Weinstein)</th></tr></thead>",
    ]
    if not shorts:
        out.append("<tr><td colspan='4' style='color:#82827c;'>No Stage-4 breakdown candidates "
                   "— no weak-group name is at its shelf right now.</td></tr>")
    else:
        for m in shorts:
            tgt = m.get("target_swing")
            tgt_txt = (f"Cover ½ near ${tgt} <span class='stop-reason'>(swing rule, −{m.get('to_target_pct')}%)</span>"
                       if tgt else "<span class='stop-reason'>measured move exceeds price — deep-target flag</span>")
            rs_bits = []
            if m.get("rs_pct") is not None:
                rs_bits.append(f"RS {m['rs_pct']}")
            if m.get("rs_line_down"):
                rs_bits.append("RS line ↘ swing-over-swing")
            out.append(f"""<tr data-sector="{esc(m.get('sector',''))}">
                <td class="ep-ticker" data-sort="{esc(m['ticker'])}"><a href="https://www.tradingview.com/chart/?symbol={esc(m['ticker'])}" target="_blank">{esc(m['ticker'])}</a></td>
                <td data-sort="{m.get('ind_rs') or 99}">{_lp(m['ticker'], m['close'])}<br>
                    <span class="bad">{esc(m.get('wk_stage') or 'S4')}</span> <span style="font-size:var(--fs-caption);color:#82827c;">30wk MA {m.get('wk_ma_slope', 0):+.1f}%/5wk</span><br>
                    <span style="font-size:var(--fs-caption);color:#82827c;">{esc(' · '.join(rs_bits))}</span><br>
                    <span style="font-size:var(--fs-caption);color:#e06c6a;">{esc(m.get('ind_name') or '')} · industry RS {m.get('ind_rs')}</span><br>
                    {_narrative(m['ticker'], f'''<span class="theme-tag">{esc(m.get('theme') or '')}</span>''')}</td>
                <td style="font-size:var(--fs-table);text-align:left;">
                    shelf ${m.get('support_low')}<br>
                    <span style="color:#82827c;">top peak ${m.get('peak')}</span><br>
                    <span style="color:#82827c;">1M {m.get('perf_1m')}% · 6M {m.get('perf_6m')}%</span>
                </td>
                <td data-sort="{m.get('risk_pct') or 999}">
                    <div class="entry-box" style="border-color:#8a4341;background:rgba(224,108,106,.08);">
                        <span style="color:#e06c6a;font-weight:bold;font-size:var(--fs-table);">SHORT &lt; ${m['entry']}</span><br>
                        <span class="stop-text">Buy-stop ${m['stop']} <span class="stop-reason">({esc(m.get('buy_stop_basis') or '')}, {m.get('risk_pct')}%)</span></span><br>
                        <span style="color:#54b87f;">{tgt_txt}</span>
                    </div>
                    {_dtc_line(m)}
                    {_sr_line(m)}
                    {_tl_line(m, short=True)}
                    {_ch_line(m, short=True)}
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
    """LIVE intraday execution plan (ORB + VWAP + volume), amended from
    madrry_trade_plan.py and rendered into the report. Operates on the current
    scan's high-conviction names (A+ VCP power + HVE rel-vol)."""
    targets = [s for s in setups_pool
               if (s.get("power_score") or 0) > 100 or (s.get("rel_vol") or 0) > 3.0]
    targets.sort(key=lambda x: (x.get("power_score") or (x.get("rel_vol", 0) * 100)),
                 reverse=True)
    targets = targets[:6]

    # Fetch each target's 5m snapshot concurrently (per-stock VWAP/ORB gating).
    need = [t["ticker"] for t in targets]
    intra: Dict[str, Optional[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futs = {pool.submit(fetch_intraday_5m, tk): tk for tk in dict.fromkeys(need)}
        for fut in concurrent.futures.as_completed(futs):
            try:
                intra[futs[fut]] = fut.result()
            except Exception:  # noqa: BLE001
                intra[futs[fut]] = None

    rows = []
    for s in targets:
        tk = s["ticker"]
        is_hve = "rel_vol" in s
        kind = "HVE" if is_hve else "A+ VCP"
        kind_color = "#e06c6a" if is_hve else "#54b87f"
        it = intra.get(tk)
        if not it or it["bars"] < 5:
            rows.append(f"""<tr>
                <td class="ticker" data-sort="{esc(tk)}"><a href="https://www.tradingview.com/chart/?symbol={esc(tk)}" target="_blank">{esc(tk)}</a><br><span style="font-size:var(--fs-micro);font-weight:bold;color:{kind_color};">{kind}</span></td>
                <td colspan="3" style="color:#82827c;text-align:left;">⏳ Waiting for intraday data to populate (market closed or pre-open).</td>
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
        vwap_txt = "✓ Above VWAP (buyers)" if above_vwap else "✗ Below VWAP (CANCEL)"
        vol_cls = "good" if vol_ok else "warn"
        risk_color = "#54b87f" if risk_pct <= 5.0 else "#e06c6a"

        live = (f'<span style="font-weight:bold;">${cp:.2f}</span> '
                f'<span class="stop-reason">VWAP ${vwap:.2f}</span><br>'
                f'<span class="stop-reason">ORB ${orb_l:.2f} – ${orb_h:.2f}</span><br>'
                f'<span class="{vwap_cls}">{vwap_txt}</span>')

        if not above_vwap:
            protocol = ('<span class="bad">✗ STAND DOWN — price under VWAP, sellers in control.</span>'
                        '<br><span class="stop-reason">Re-evaluate only on a reclaim of VWAP.</span>')
        else:
            steps = [
                f'1️⃣ <b>不要盲掛 buy-stop.</b> 等 5m/30m K棒<b>收盤站上</b> ${trigger:.2f}.',
                f'2️⃣ 突破K棒量必須 &gt; 5根均量 '
                f'<span class="{vol_cls}">(now {cur_v:,.0f} vs {avg_v:,.0f})</span>. 無量=陷阱.',
            ]
            if is_hve:
                steps.append(f'3️⃣ <b>較佳:</b> 等回測 ${trigger:.2f} 守住再進 (避免追高).')
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
    <div class="section-title" style="background-color:#10243a;color:#aecfe8;border-bottom:3px solid #8cb4d6;">⚔️ INTRADAY EXECUTION PLAN — 盤中執行 (ORB + VWAP · Live · Risk ${risk_dollar:.0f})</div>
    <div class="table-container"><table>
        <thead><tr><th>Ticker</th><th>Live (Price / VWAP / ORB)</th><th>Execution Protocol</th><th>Trigger / Stop / Size</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
    </table></div>
    <div style="background:#10243a;padding:12px 15px;margin:0 0 20px;border-radius:8px;font-size:var(--fs-table);color:#ececea;line-height:1.6;">
        <b style="color:#aecfe8;">紀律：</b> 開盤前 15–30 分鐘別急；等 5m/30m <b>收盤</b>確認、量過均，再進。價在 VWAP 下一律不做多。<br>
        賣在強勢、沿 9/21 EMA 移動停損；風險每筆固定 ${risk_dollar:.0f}（部位是緊停損的副產品）。
    </div>
    """


def _minervini_snapshot_rows(data_date: Optional[str] = None) -> List[dict]:
    """Lightweight Minervini buy-list rows for the ledger snapshot (IMPROVEMENT_PLAN
    Phase 1a). Reads the external minervini_engine buy_list JSON (prefers the scan's
    data_date, else the newest file). Additive & best-effort — any failure returns []
    so it can never break the report. pivot is the buy-stop entry (long, stop-style)."""
    try:
        files = [f for f in os.listdir(MINERVINI_DIR)
                 if f.startswith("buy_list_") and f.endswith(".json")]
    except OSError:
        return []
    if not files:
        return []
    want = f"buy_list_{data_date}.json" if data_date else None
    fname = want if (want and want in files) else sorted(files)[-1]
    try:
        rows = json.load(open(os.path.join(MINERVINI_DIR, fname)))
    except (OSError, ValueError):
        return []
    out: List[dict] = []
    for m in rows or []:
        tk = m.get("ticker")
        pivot, stop = m.get("pivot"), m.get("stop")
        if not tk or pivot is None or stop is None:
            continue
        out.append({
            "ticker": tk, "tier": "MINERVINI", "section": "minervini",
            "close": m.get("last_close"), "entry": pivot, "stop": stop,
            "risk_pct": round((m.get("stop_frac") or 0.0) * 100, 1),
            "status": m.get("status"), "vcp_score": m.get("vcp_score"),
            "rs_rating": m.get("rs"), "adr": m.get("adr"), "sector": m.get("sector"),
            "pattern": m.get("pattern"), "pct_to_pivot": m.get("pct_to_pivot"),
            "buy_list_asof": fname[len("buy_list_"):-len(".json")],
        })
    return out


# ----------------------------------------------------------------------------
# ORCHESTRATION
# ----------------------------------------------------------------------------
def run_scanners_and_generate_html() -> str:
    diag = Diagnostics()
    t0 = time.time()

    # The lesson engines must never degrade SILENTLY: a failed import would
    # otherwise produce a report with no zones/pullback/trendline/channel
    # reads (and an all-None SR/PB/TL would starve the Stage-4 gate). Surface
    # it in the DONE line's warning count so the morning gate/log shows it.
    for _name, _mod in (("madrry_sr_zones", _srz), ("madrry_pullback_buy", _pbb),
                        ("madrry_trendlines", _tlv2), ("madrry_channels", _chv)):
        if _mod is None:
            diag.warn(f"lesson engine FAILED TO IMPORT: {_name} — report runs "
                      f"without its reads (Stage-4 gate affected if SR/PB/TL)")

    with timed(diag, "rs_scores"):
        rs_map = fetch_and_load_rs_scores(diag)
    with timed(diag, "industry_rs"):
        industry_rs = fetch_and_load_industry_rs(diag)

    with timed(diag, "market_health"):
        market_data, breadth = fetch_market_health(diag)

    # Top-10 news runs in its OWN executor (not the regime pool): its result()
    # gets a hard timeout, and on timeout we shutdown(wait=False) so a straggler
    # feed thread can never block the report — the shared pool's implicit
    # shutdown(wait=True) would otherwise join it (audit 2026-07-08).
    _news_ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    f_news = _news_ex.submit(fetch_top10_news, diag)
    with timed(diag, "regime_data"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as rpool:
            f_vix = rpool.submit(fetch_vix, diag)
            f_t2108 = rpool.submit(fetch_t2108, diag)
            f_sect = rpool.submit(fetch_sector_rs, diag)
            vix = f_vix.result()
            t2108 = f_t2108.result()
            sector_rs = f_sect.result()
    try:
        top10_stories = f_news.result(timeout=120)
        _news_ex.shutdown(wait=False)
    except Exception as exc:  # noqa: BLE001 - incl. TimeoutError
        diag.warn(f"Top 10 news timed out/failed: {exc}")
        top10_stories = []
        _news_ex.shutdown(wait=False)

    ixic_trend = next((m["trend"] for m in market_data if m["ticker"] == "^IXIC"), "GREEN")
    spx_trend = next((m["trend"] for m in market_data if m["ticker"] == "^GSPC"), "GREEN")
    above_50_pct = breadth.get("above50", 50.0)
    if ixic_trend == "GREEN" and spx_trend == "GREEN":
        market_modifier = 1.2
    elif ixic_trend == "RED" and spx_trend == "RED":
        market_modifier = 0.4 if above_50_pct < 40.0 else 0.7
    else:
        market_modifier = 1.0
    log.info("IXIC=%s SPX=%s Above50=%.1f%% MarketMod=%s",
             ixic_trend, spx_trend, above_50_pct, market_modifier)

    # 10 calendar days > the 5 TRADING-day U&R window, so weekend-spanning day-4/5
    # setups aren't purged before scan_ur can see them (audit H4).
    hve_history = cleanup_old_hve(load_hve_history(), days=10)

    # Data date = trading date of the last daily bar, taken as the FRESHEST asof
    # across the index cards. Computed BEFORE the scans so they can reject stale
    # history feeds (Yahoo's bulk endpoint can lag the chart endpoint by a session
    # during EOD consolidation). Why max and not just ^IXIC (the primary index):
    # TradingView's scan API has no caret indices, so the stale-bar TV-append
    # inside _fetch_one_index can only patch the ETF card (IWM) — under Yahoo's
    # EOD lag IWM carries the real current session while the ^IXIC/^NDX/^GSPC
    # indices would sit a day behind, and keying on ^IXIC alone would mark the
    # whole report provisional. (2026-07-06: S&P/Nasdaq-100 cards are now indices
    # too, so IWM is the sole ETF freshness anchor — still sufficient.)
    _asofs = [m["asof"] for m in market_data if m.get("asof")]  # ISO strings → max() is latest
    data_date = max(_asofs) if _asofs else None
    if not next((m for m in market_data if m.get("ticker") == "^IXIC" and m.get("asof")), None):
        diag.warn("^IXIC asof unavailable — data_date taken from the other index cards "
                  + (f"({data_date})" if data_date else "(none — today())"))
    data_date = data_date or date.today().isoformat()
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

    # Persist S&P breadth to the durable regime feed (keyed by data_date). The
    # writer had been orphaned since ~2026-06-08; re-wired here. Non-fatal.
    _persist_breadth_history(breadth, data_date)
    _persist_headline_meter(data_date, market_data)
    _persist_market_internals(fetch_market_internals(diag), data_date)

    with timed(diag, "scan_coil"):
        tier_a_plus, tier_a, tier_a_minus, tier_a_minus_full, lesson_radar, coil_funnel = scan_coil(
            rs_map, market_modifier, diag, industry_rs.get("by_ticker") or {})
        # Geometry ratification: switch coil A+/A/A- to the 1.5×ADR stop + 5-day validity for
        # sessions >= STOP_REGIME_SWITCH_SESSION. Runs BEFORE write_order_plan + HTML render so
        # the printed stop, IBKR sizing, and the ledger snapshot all follow. Date-gated: on a
        # pre-switch session (e.g. a holiday re-run of older data) it only stamps stop_version +
        # the stable stop_tight/stop_atr fields and changes nothing printed. Idempotent.
        _apply_stop_regime(tier_a_plus + tier_a + tier_a_minus + tier_a_minus_full + lesson_radar, data_date)
    with timed(diag, "scan_htf"):
        htf_matches = scan_htf(rs_map, market_modifier, diag, data_date)
        # Merge HTF fires INTO Tier A+ (user's chosen placement). Dedup against
        # any name already surfaced by the coil scan; re-sort A+ by M.E.T.A.
        _coil_names = {s["ticker"] for s in tier_a_plus + tier_a + tier_a_minus_full}
        new_htf = [h for h in htf_matches if h["ticker"] not in _coil_names]
        if new_htf:
            tier_a_plus = sorted(tier_a_plus + new_htf,
                                 key=lambda x: x["meta_score"], reverse=True)
        # a name can be tierless in the coil scan yet enter A+ via HTF — never show twice
        _apn = {s["ticker"] for s in tier_a_plus}
        lesson_radar = [s for s in lesson_radar if s["ticker"] not in _apn]
    with timed(diag, "scan_hve"):
        ep_matches = scan_hve(hve_history, diag)
    with timed(diag, "scan_ur"):
        ur_matches = scan_ur(hve_history, ep_matches, diag)
    with timed(diag, "scan_short"):
        short_matches = scan_parabolic_short(diag)
        for _s in short_matches:
            _s["short_style"] = "parabolic"   # new-key tag, no rename (2026-07-17)
        stage4_matches = scan_stage4_short(diag, rs_map, industry_rs.get("by_ticker") or {})
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
        lesson_radar = _drop(lesson_radar)
        ep_matches = _drop(ep_matches)
        ur_matches = _drop(ur_matches)
        short_matches = _drop(short_matches)
        stage4_matches = _drop(stage4_matches)
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
    # SHADOW weekly/1h lesson read on the displayed names (radar prioritized
    # ahead of the deep A- pool so the 80-ticker cap favors it).
    with timed(diag, "mtf_lessons"):
        _enrich_mtf_lessons(tier_a_plus + tier_a + lesson_radar + tier_a_minus_full, diag)
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
        # Lesson Radar rows render through the same coil table but skip
        # attach_ants — give them the Weinstein chips too (2026-07-17 review).
        attach_weinstein(lesson_radar, diag)
    # Industry-group RS (Fred6725 rs_industries) — display-only leadership tag.
    sector_waves = compute_sector_waves(industry_rs.get("by_ticker") or {})
    attach_industry_rs(tier_a_plus + tier_a + tier_a_minus_full + lesson_radar
                       + nh_data.get("green", []) + ep_matches + ur_matches
                       + stage4_matches, industry_rs["by_ticker"])

    # IBD-style extras: Weekly Review rows (own screener POST + chart batch) and
    # the Market Pulse leader movers. Both fetchers are fail-safe (sentinel on
    # error) — neither can block the report.
    with timed(diag, "weekly_review"):
        weekly_data = fetch_weekly_review(rs_map, diag)
        weekly_rows = weekly_data.get("rows", [])
    with timed(diag, "market_pulse"):
        pulse_movers = fetch_market_pulse_movers(rs_map, diag)

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
    market_html, overall_trend = build_market_section(
        market_data, breadth, regime, allow_breakouts,
        internals=_load_market_internals(), sector_waves=sector_waves)
    # IBD-style page-1 sections: The Big Picture (narrative + Market Pulse) and
    # the Top-10 briefs. Builders are pure formatters that return "" on empty /
    # broken input, so they can never take the report down.
    big_picture_html = build_big_picture(market_data, breadth, t2108, regime,
                                         allow_breakouts, leader_stats, pulse_movers,
                                         headline_meter=HEADLINE_METER)
    top10_html = build_top10_news(top10_stories)
    weekly_review_html = (generate_weekly_review_cards(weekly_data) if LAYOUT_V9
                          else generate_weekly_review_table(weekly_data))
    # Deterministic IBKR draft-order INTENT (top_picks_orders.json) FIRST, so the
    # dashboard cards can show which picks were actually drafted. No orders, no
    # account data — the staging step alone touches IBKR (drafts only).
    _plan = write_order_plan(tier_a_plus, tier_a, tier_a_minus, regime, allow_breakouts, data_date)
    _drafted = [p.get("ticker") for p in (_plan.get("picks") or [])]
    top_picks_html = build_top_picks(tier_a_plus, tier_a, tier_a_minus, drafted=_drafted)
    # ETF rows are excluded from HOT SECTORS: sector="ETF" is a plumbing label,
    # not a sector-leadership read, and the chip doubles as a row filter.
    hot_themes_html = build_hot_themes(
        [s for s in (tier_a_plus + tier_a + tier_a_minus + ep_matches)
         if not s.get("is_etf")])
    hot_industries_html = build_hot_industries(industry_rs, tier_a_plus + tier_a + tier_a_minus)

    # Warm the fundamentals cache for EVERY narrative-bearing MADRRY ticker BEFORE the
    # first table renders. Must precede generate_new_highs_section (below): its 🟢 green
    # rows also tap fundamentals and were previously omitted from the batch, forcing one
    # synchronous TradingView POST + cache flush per new-high name during HTML assembly.
    # Batched + disk-cached + time-boxed; never fatal (Minervini/Trilogy warm their own).
    _prefetch_fundamentals(
        [s.get("ticker") for s in (tier_a_plus + tier_a + tier_a_minus_full
                                   + lesson_radar + ep_matches + ur_matches + short_matches)
         if not s.get("is_etf")]     # funds have no fundamentals — skip, don't cache junk
        + [m.get("ticker") for m in nh_data.get("green", [])]
        + [m.get("ticker") for m in weekly_rows])   # Weekly Review narrative cells tap fundamentals too
    # Tier 3 — estimate-revision counts (per-ticker yfinance) for the TOP PICKS only.
    _prefetch_revisions([s.get("ticker") for _, s in
                         _rank_top_picks(tier_a_plus, tier_a, tier_a_minus_full)[:REVISIONS_TOP_N]
                         if not s.get("is_etf")])

    if LAYOUT_V9:
        new_highs_html = generate_new_highs_cards(nh_data)
        nh52_monitor_html = generate_nh52_monitor_cards(nh52_pullbacks, nh52_monitored)
    else:
        new_highs_html = generate_new_highs_section(nh_data)
        nh52_monitor_html = generate_nh52_monitor_section(nh52_pullbacks, nh52_monitored)
    counts = {
        "a_plus": len(tier_a_plus), "a": len(tier_a), "a_minus": len(tier_a_minus_full),
        "hve": len(ep_matches), "ur": len(ur_matches), "short": len(short_matches),
    }
    runtime = time.time() - t0

    # External engines (own trade plans). Exactly ONE path runs per report —
    # the prep is side-effectful (feed read + enrichment fetch + prefetch).
    if LAYOUT_V9:
        minervini_html, minervini_n = generate_minervini_cards(market_modifier)
        trilogy_html, trilogy_n = generate_trilogy_cards(market_modifier=market_modifier)
    else:
        minervini_html, minervini_n = generate_minervini_table(market_modifier)
        trilogy_html, trilogy_n = generate_trilogy_table(market_modifier=market_modifier)
    tier_a_study_html, tier_a_study_n = generate_tier_a_study_tab()
    tabs_bar = (
        "<div class='tabs' role='tablist'>"
        "<button class='tab-btn active' data-tab='madrry'>MADRRY Watchlist</button>"
        f"<button class='tab-btn' data-tab='minervini'>Minervini<span class='tab-count'>{minervini_n}</span></button>"
        f"<button class='tab-btn' data-tab='trilogy'>Trilogy<span class='tab-count'>{trilogy_n}</span></button>"
        f"<button class='tab-btn' data-tab='pivots'>Pivots &amp; U&amp;R<span class='tab-count'>{len(ep_matches) + len(ur_matches)}</span></button>"
        f"<button class='tab-btn' data-tab='short'>Short<span class='tab-count'>{len(short_matches) + len(stage4_matches)}</span></button>"
        f"<button class='tab-btn' data-tab='hi52'>52-Week High<span class='tab-count'>{nh_data.get('total', 0)}</span></button>"
        f"<button class='tab-btn' data-tab='weekly'>Weekly Review<span class='tab-count'>{len(weekly_rows)}</span></button>"
        f"<button class='tab-btn' data-tab='tracking'>Tracking<span class='tab-count'>{tier_a_study_n}</span></button>"
        "</div>"
    )

    if LAYOUT_V9:
        # ---- v9: stacked chart-first sections + anchor-chip nav; the Desk
        # (wide screens) is a client-side view over the same #deck DOM ----
        _nav_entries = [("top", "Picks", None)]
        if lesson_radar:
            _nav_entries.append(("radar", "Radar", len(lesson_radar)))
        _nav_entries += [
            ("aplus", "A+", counts["a_plus"]), ("a", "A", counts["a"]),
            ("aminus", "A−", counts["a_minus"]),
            ("min", "Minervini", minervini_n), ("tri", "Trilogy", trilogy_n),
            ("hve", "HVE", counts["hve"]), ("ur", "U&R", counts["ur"]),
            ("short", "Short", len(short_matches)), ("s4", "Stage-4", len(stage4_matches)),
        ]
        if nh_data.get("total", 0):
            _nav_entries.append(("nh", "52W High", nh_data.get("total", 0)))
        _nav_entries += [("pull", "Pullback", len(nh52_pullbacks)),
                         ("wk", "Weekly", len(weekly_rows)),
                         ("study", "Tracking", tier_a_study_n)]
        section_parts = [
            # REV 10b: two top-level panes. Charts = the card deck (unchanged
            # structure, so the Foldable Desk keeps working on #deskwrap/#deck/
            # #desklist); Screener = one flat sortable table over every ticker,
            # replacing the ~600-card scroll as the way to compare names.
            "<div class='v9tabs' role='tablist'>"
            "<button class='v9tab on' data-pane='charts' role='tab'>Charts</button>"
            "<button class='v9tab' data-pane='screener' role='tab'>Screener</button>"
            "</div>",
            "<div class='v9pane' id='pane-charts'>",
            _chartctl_v9(_nav_entries),
            "<div id='deskwrap'><div id='deck'>",
            _secnav_v9(_nav_entries),
            f"<div id='sec-top' class='secv9'>{top_picks_html}</div>",
            tracking_html,
            build_filter_funnel(coil_funnel, len(tier_a_plus), len(tier_a), len(tier_a_minus_full)),
            build_lesson_radar_v9(lesson_radar),
            _section_v9("aplus", "TIER A+ — TRIGGER READY", tier_a_plus, SECTION_SPECS_V9["coil"],
                        bg="bg-aplus", tier="A+", grp="aplus",
                        subtitle="strict 3-day flag · ≤1% from EMA · 3-day vol ≤50% of prev-day or 50-day avg · incl. HTF",
                        empty="No A+ trigger-ready coils today."),
            _section_v9("a", "TIER A — DEVELOPING", tier_a, SECTION_SPECS_V9["coil"],
                        bg="bg-a", tier="A", grp="a",
                        subtitle="2-day tight candle · ≤1% from EMA · 2-day vol ≤55% of prev-day or 50-day avg",
                        empty="No developing A coils today."),
            _section_v9("aminus", "TIER A− — EXTENDED / MESSY", tier_a_minus_full, SECTION_SPECS_V9["coil"],
                        bg="bg-aminus", tier="A−", grp="aminus",
                        subtitle="1-day tight candle · ≤2% from EMA · 1-day vol ≤ prev-day or 50-day avg",
                        empty="No A− coils today."),
            minervini_html,
            trilogy_html,
            _section_v9("hve", "EPISODIC PIVOTS (HVE) — LOW FLOAT ≤200M", ep_matches,
                        SECTION_SPECS_V9["hve"], bg="bg-hve",
                        empty="No HVE events detected today."),
            _section_v9("ur", "POST-HVE U&R (PULLBACK & UNDERCUT)", ur_matches,
                        SECTION_SPECS_V9["ur"], tail=_UR_NOTE_V9,
                        empty="No Post-HVE U&R candidates. Waiting for HVE stocks to consolidate…"),
            _section_v9("short", "PARABOLIC SHORT — 乖離過大 · 拋物線見頂", short_matches,
                        SECTION_SPECS_V9["short"], bg="bg-short",
                        empty="No parabolic-short candidates — nothing is climactically extended."),
            _section_v9("s4", "STAGE-4 BREAKDOWN — WEINSTEIN CH.7", stage4_matches,
                        SECTION_SPECS_V9["s4"], bg="bg-short", subtitle=_S4_SUB_V9,
                        empty="No Stage-4 breakdown candidates — no weak-group name is at its shelf."),
            new_highs_html,
            nh52_monitor_html,
            weekly_review_html,
            "<section class='secv9' id='sec-study'>"
            "<div class='section-title'><span class='tdot'></span>TRACKING — TIER-A FORWARD WIN/LOSS STUDY</div>"
            f"<div class='table-container'>{tier_a_study_html}</div></section>",
            "</div><nav id='desklist' hidden></nav></div>",
            "</div>",                                   # /#pane-charts
            # built LAST so _section_v9 has registered every section's rows
            "<div class='v9pane' id='pane-screener' hidden>",
            build_screener_v9(),
            "</div>",
        ]
    else:
        section_parts = [
            tabs_bar,
            "<div class='tab-panel active' id='tab-madrry'>",
            build_tab_counts([("A+", counts["a_plus"], ""), ("A", counts["a"], ""),
                              ("A−", counts["a_minus"], "")]),
            top_picks_html,
            tracking_html,
            build_filter_funnel(coil_funnel, len(tier_a_plus), len(tier_a), len(tier_a_minus_full)),
            build_lesson_radar(lesson_radar),
            generate_coil_table(tier_a_plus, "Tier A+ — trigger ready", "bg-aplus",
                                subtitle="strict 3-day flag · ≤1% from EMA · 3-day vol ≤50% of prev-day or 50-day avg · incl. HTF"),
            generate_coil_table(tier_a, "Tier A — developing", "bg-a",
                                subtitle="2-day tight candle · ≤1% from EMA · 2-day vol ≤55% of prev-day or 50-day avg"),
            generate_coil_table(tier_a_minus_full, "Tier A− — extended / messy", "bg-aminus",
                                subtitle="1-day tight candle · ≤2% from EMA · 1-day vol ≤ prev-day or 50-day avg"),
            "</div>",  # /tab-madrry
            f"<div class='tab-panel' id='tab-minervini'>{minervini_html}</div>",
            f"<div class='tab-panel' id='tab-trilogy'>{trilogy_html}</div>",
            # ---- Episodic Pivots (HVE) + Post-HVE U&R — own tab ----
            "<div class='tab-panel' id='tab-pivots'>",
            build_tab_counts([("HVE", counts["hve"], ""), ("U&R", counts["ur"], "")]),
            generate_hve_table(ep_matches),
            generate_ur_table(ur_matches),
            "</div>",
            # ---- Parabolic Short — own tab ----
            f"<div class='tab-panel' id='tab-short'>"
            f"{build_tab_counts([('Short', counts.get('short', 0), 'red')])}"
            f"{generate_short_table(short_matches)}"
            f"{generate_stage4_short_table(stage4_matches)}</div>",
            # ---- 52-Week High — New Highs + Pullback as two sub-tabs ----
            "<div class='tab-panel' id='tab-hi52'>",
            "<div class='subtabs' role='tablist'>",
            f"<button class='subtab-btn active' data-subtab='nh'>New 52wk Highs<span class='tab-count'>{nh_data.get('total', 0)}</span></button>",
            f"<button class='subtab-btn' data-subtab='pull'>52wk Pullback<span class='tab-count'>{len(nh52_pullbacks)}</span></button>",
            "</div>",
            f"<div class='subtab-panel active' id='subtab-nh'>{new_highs_html}</div>",
            f"<div class='subtab-panel' id='subtab-pull'>{nh52_monitor_html}</div>",
            "</div>",
            # ---- Your Weekly Review (IBD-style weekly leaders) — own tab ----
            f"<div class='tab-panel' id='tab-weekly'>{weekly_review_html}</div>",
            f"<div class='tab-panel' id='tab-tracking'>{tier_a_study_html}</div>",
        ]

    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "<title>MADRRY Ultimate Scanner Report</title>",
        f"<style>{PAGE_CSS}</style></head><body class='{'v9' if LAYOUT_V9 else ''}'>" ,
        "<h1>MADRRY Watchlist</h1>",
        f"<p class='header-sub'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        build_runbar(counts, market_modifier, runtime, regime, allow_breakouts),
        stale_banner,
        # 2026-07-06 USER layout: market overview FIRST, regime second, then the
        # hot sector/industry chips + the global search ABOVE the sections (they
        # act on every section, so they live outside the deck).
        market_html,
        regime_html,
        # IBD-style page 1 (2026-07-08): The Big Picture narrative + Market Pulse
        # right after the regime it interprets, then the Top-10 news briefs.
        big_picture_html,
        top10_html,
        hot_themes_html,
        hot_industries_html,
        "<input id='search' type='search' placeholder='🔎 Search the whole report — ticker (e.g. NVDA)…' autocomplete='off'>",
        *section_parts,
        build_mindset_panel(),
        build_diag_panel(diag),
        PAGE_JS.replace("__LIVE_PRICE_PROXY__", LIVE_PRICE_PROXY),
        CANDLE_JS,
        "</body></html>",
    ]
    html = "".join(parts)
    # strip f-string indentation runs — pure whitespace minify (~3% of the file);
    # a lone \n renders identically, and JS line comments stay intact.
    html = re.sub(r"\n[ \t]+", "\n", html)

    # v9 contract guard: the new HVE/U&R/Short/Stage-4 chart payloads are for
    # the HTML only — pop them so latest_setups.json stays byte-compatible
    # with every downstream consumer (coil rows keep spark exactly as before).
    for s in ep_matches + ur_matches + short_matches + (stage4_matches or []):
        s.pop("spark", None)

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
    _base_rows = tier_a_plus + tier_a + tier_a_minus_full + ep_matches + ur_matches
    # ---- Ledger snapshot enrichment (IMPROVEMENT_PLAN Phase 1a) — ADDITIVE & fully
    # wrapped so a failure here can NEVER break report persistence (falls back to the
    # legacy coil+HVE+U&R payload). Adds: a `section` tag on every row; the Short,
    # 52wk-high and Minervini sections (previously outcome-UNTRACKED — the core gap);
    # and a `top_pick` flag on the 5 dashboard picks. Existing consumers
    # (tier_a_tracker, v4_tracker, track_previous_setups, premarket_execution_engine)
    # all filter to A+/A/A- and ignore the new tier/section values. ----
    _setup_rows = _base_rows
    try:
        for s in ep_matches:
            s["section"] = "hve"
        for s in ur_matches:
            s["section"] = "ur"
        for s in tier_a_plus + tier_a + tier_a_minus_full:
            s.setdefault("section", "coil")
        _short_rows = []
        for s in (short_matches or []):
            r = dict(s); r["tier"] = "SHORT"; r["section"] = "short"; r["direction"] = "short"
            _short_rows.append(r)
        for s in (stage4_matches or []):
            r = dict(s); r["tier"] = "SHORT"; r["section"] = "short"; r["direction"] = "short"
            _short_rows.append(r)
        _nh_rows = []
        for s in (nh_data.get("green", []) or []):
            r = dict(s); r["tier"] = "NH52"; r.setdefault("section", "nh52")
            _nh_rows.append(r)
        _min_rows = _minervini_snapshot_rows(data_date)
        _top_tickers = {st.get("ticker") for _, st in
                        _rank_top_picks(tier_a_plus, tier_a, tier_a_minus, ants_boost=True)[:5]}
        _setup_rows = _base_rows + _short_rows + _nh_rows + _min_rows
        for s in _setup_rows:
            if s.get("ticker") in _top_tickers:
                s["top_pick"] = True
        # Phase-4: print the 1.5×ADR alternate stop ALONGSIDE the tight stop (additive, does
        # NOT change the printed `stop` — that switch is a user decision downstream). Lets the
        # ledger label BOTH stop geometries on live signals from now on, starting the OOS
        # validation clock for the stop-width finding. Direction-aware; needs entry + adr.
        _ATR_STOP_MULT = 1.5
        for s in _setup_rows:
            # Every snapshot row carries a stop_version regime marker (coil A+/A/A- get atr_5day
            # from _apply_stop_regime; all others stay tight_3day — they did not switch, §2.1).
            s.setdefault("stop_version", "tight_3day")
            _e, _a = s.get("entry"), s.get("adr")
            if _e is None or _a is None:
                continue
            try:
                _e = float(_e); _a = float(_a)
            except (TypeError, ValueError):
                continue
            if _a <= 0:
                continue
            if s.get("direction") == "short" or s.get("section") == "short":
                s["stop_atr"] = round(_e * (1 + _ATR_STOP_MULT * _a / 100.0), 2)
            else:
                s["stop_atr"] = round(_e * (1 - _ATR_STOP_MULT * _a / 100.0), 2)
    except Exception as exc:  # noqa: BLE001 — snapshot enrichment must never break persistence
        diag.warn(f"ledger snapshot enrichment skipped: {exc}")
        _setup_rows = _base_rows
    _setups_payload = json.dumps(_setup_rows)
    _atomic_write(LATEST_SETUPS_PATH, _setups_payload)   # legacy/stable alias
    _save_dated_setups(_setups_payload, data_date)
    _atomic_write(HTML_REPORT_PATH, html)
    # Stable "latest" alias so the preview / bookmarks always show the newest run
    _atomic_write(os.path.join(WORKSPACE, "madrry_report.html"), html)

    # ---- also generate Markdown report for cron delivery ----
    md_path = MD_REPORT_PATH
    md_content = build_markdown_report(
        tier_a_plus, tier_a, tier_a_minus, ep_matches, ur_matches, short_matches,
        stage4_matches, market_data, breadth, overall_trend, market_modifier, runtime, diag
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
    stage4_matches, market_data, breadth, overall_trend, market_modifier, runtime, diag
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
    lines.append(f"- **Short (Stage-4 breakdown):** {len(stage4_matches)}")
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
                rsl = "RS▲‹Px"
            elif s.get("rs_ok") and s.get("rs_new_high"):
                rsl = "RS▲"
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
    # The count line above advertises the Stage-4 leg; emit its body too
    # (2026-07-18 review: markdown/cron delivery listed the count but no names).
    lines.extend(_tier_section("Short — Stage-4 Breakdown", stage4_matches))

    if diag.errors:
        lines.extend(["", "## Errors", ""])
        for e in diag.errors[:10]:
            lines.append(f"- ⚠️ {e}")

    lines.extend(["", "---", "*Generated by MADRRY Scanner v2*", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    run_scanners_and_generate_html()
