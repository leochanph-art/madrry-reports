"""
MADRRY FUNDAMENTALS — past + MANY-forward-quarter sales/revenue/EPS for the report.
=================================================================================
Tap the "theme / sector / 🏭 industry" narrative cell to reveal a small table:

      Qtr        Rev       YoY      EPS
      Apr'26     81.61B    +69%     0.81     <- 1 most-recent REPORTED quarter (actual)
      Jul'26e    91.71B    +50%     2.08     <- forward consensus estimates, as many
      Oct'26e   103.37B    +53%     2.45        quarters out as the source provides
      ...        ...       ...      ...         (TradingView gives ~8-13 quarters)

PRIMARY SOURCE: TradingView's public scanner (scanner.tradingview.com) — the same
data feed the scanner already screens from. Its `revenues_fq_h` / `earnings_fq_h`
arrays carry the full consensus series: every reported quarter (Actual) AND ~8-13
FORWARD quarters (Estimate, IsReported=False) of revenue and EPS, for essentially
every US-listed ticker incl. small/micro caps, ADRs and recent IPOs. Values are
normalised to USD. One POST returns a whole batch of tickers (the `name in_range`
filter also resolves each ticker's exchange), so ~380 names = a handful of requests.

FALLBACK: Yahoo Finance (yfinance) — 1 reported + 2 estimate quarters — used only for
any ticker TradingView does not return, so a single-source hiccup never blanks a row.

Robustness (runs inside a time-boxed morning cron):
  * Disk cache (fundamentals_cache.json) keyed by ticker + fetch date, refreshed once
    per calendar day (REFRESH_DAYS). Within a day, re-reads from cache.
  * Every network path is wrapped — failure -> the plain narrative, never a scan abort.
  * Both endpoints are UNOFFICIAL (same risk class as the existing yfinance dependency);
    for personal/internal use only, wrapped defensively.

Public API:
  prefetch(tickers, ...)              -> warm the cache (batched TV + Yahoo fallback)
  get(ticker)                         -> cached dict | None
  details_html(ticker, inner_html)    -> <details> wrapping inner_html, or unchanged
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime

try:
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    import yfinance as yf
    import pandas as pd
except Exception:  # pragma: no cover
    yf = None
    pd = None

_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_DIR, "fundamentals_cache.json")

REFRESH_DAYS = 1      # re-fetch a ticker's fundamentals once per calendar day
NEG_DAYS = 1          # re-try a "no data" ticker next day too (self-heals hiccups)
CACHE_SCHEMA = 3      # record-shape version; bump when fields change (O'Neil layer = v2,
                      # +ticker tag = v3) so pre-existing caches are treated as stale & re-fetched

TV_URL = "https://scanner.tradingview.com/america/scan"
# Columns 0-4 feed the forward Rev/EPS narrative; 5-8 are the O'Neil layer fetched in the
# SAME POST (zero extra round-trips): per-quarter net income + revenue (after-tax MARGIN
# series → "latest quarter margin at a new high"), current TTM net margin, and industry
# (best-in-industry margin percentile, Tier 2). earnings_fq_h's reported rows also carry
# the pre-report consensus Estimate → earnings-surprise/"beat" %, computed for free.
TV_COLUMNS = ["name", "exchange", "fundamental_currency_code", "revenues_fq_h", "earnings_fq_h",
              "net_income_fq_h", "total_revenue_fq_h", "net_margin", "industry"]
TV_CHUNK = 80         # tickers per scanner POST
TV_TIMEOUT = 15
EPS_ACCEL_DISPLAY_Q = 8    # trailing reported quarters shown in the acceleration panel
INDUSTRY_MARGIN_CACHE = os.path.join(_DIR, "industry_margins_cache.json")
REVISIONS_CACHE = os.path.join(_DIR, "revisions_cache.json")

_LOCK = threading.Lock()
_CACHE: dict | None = None
_DIRTY = False


# ----------------------------------------------------------------------------- cache
def _load_cache() -> dict:
    global _CACHE
    if _CACHE is None:
        try:
            with open(CACHE_PATH) as fh:
                _CACHE = json.load(fh)
        except Exception:
            _CACHE = {}
    return _CACHE


def _flush() -> None:
    global _DIRTY
    if not _DIRTY:
        return
    with _LOCK:
        try:
            tmp = CACHE_PATH + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(_CACHE, fh)
            os.replace(tmp, CACHE_PATH)
            _DIRTY = False
        except Exception:
            pass


def _fresh(rec: dict) -> bool:
    if rec.get("sv") != CACHE_SCHEMA:        # old-shape record -> force a re-fetch
        return False
    try:
        f = datetime.strptime(rec.get("fetched", ""), "%Y-%m-%d").date()
    except Exception:
        return False
    ttl = REFRESH_DAYS if rec.get("ok") else NEG_DAYS
    return (date.today() - f).days < ttl


def _store(ticker: str, rec: dict) -> None:
    global _DIRTY
    with _LOCK:
        _load_cache()[ticker] = rec
        _DIRTY = True


# ----------------------------------------------------------------------------- labels
_FP_RE = re.compile(r"^(\d{4})-Q([1-4])$")


def _year_ago_fp(fp: str) -> str:
    m = _FP_RE.match(fp or "")
    return f"{int(m.group(1)) - 1}-Q{m.group(2)}" if m else ""


def _fp_label(fp: str, est: bool) -> str:
    """'2026-Q2' -> "Q2'26" (+'e' for an estimate)."""
    m = _FP_RE.match(fp or "")
    base = f"Q{m.group(2)}'{m.group(1)[2:]}" if m else (fp or "?")
    return base + ("e" if est else "")


def _fp_next(fp: str) -> str:
    """'2025-Q3' -> '2025-Q4'; '2025-Q4' -> '2026-Q1'.  '' if unparseable."""
    m = _FP_RE.match(fp or "")
    if not m:
        return ""
    y, q = int(m.group(1)), int(m.group(2))
    return f"{y + 1}-Q1" if q == 4 else f"{y}-Q{q + 1}"


# ----------------------------------------------------------------------------- O'Neil layer
def _eps_accel_from_arr(arr) -> dict | None:
    """O'Neil earnings-acceleration read from a TradingView earnings_fq_h array.

    Returns the trailing reported quarters (EPS, YoY%, and "beat" = earnings-surprise %
    from Actual vs the pre-report consensus Estimate), an acceleration verdict (the TREND
    in the YoY growth RATE — accel/steady/decel, the CANSLIM 'C' refined), a sortable
    accel_score (change in YoY rate, percentage points), and a trailing-12-month EPS
    new-high flag (rolling 4-quarter sum at its peak — the "best companies" tell).
    None if too few reported quarters to say anything.

    Robust to: missing/negative year-ago bases (YoY suppressed, never a sign-flip %),
    non-positive estimates (beat suppressed), non-contiguous quarters (TTM only sums 4
    CONSECUTIVE fiscal periods)."""
    if not arr:
        return None
    rep = sorted(
        [e for e in arr if isinstance(e, dict) and e.get("IsReported")
         and e.get("Actual") is not None],
        key=lambda e: e.get("FiscalPeriod") or "")
    if len(rep) < 2:
        return None
    actual = {e["FiscalPeriod"]: e["Actual"] for e in rep if e.get("FiscalPeriod")}

    # trailing quarters for display (oldest -> newest). YoY only when a POSITIVE year-ago
    # base exists (a growth % across a sign change is meaningless for EPS). beat only when
    # a positive consensus estimate is on file.
    quarters = []
    for e in rep[-EPS_ACCEL_DISPLAY_Q:]:
        fp, v = e["FiscalPeriod"], e["Actual"]
        ya = actual.get(_year_ago_fp(fp))
        yoy = (v / ya - 1.0) if (ya and ya > 0) else None
        est = e.get("Estimate")
        beat = (v / est - 1.0) if (est and est > 0) else None
        quarters.append({"lbl": _fp_label(fp, False), "eps": v, "yoy": yoy, "beat": beat})

    # verdict + score from the YoY GROWTH-RATE trend over the last 3 reported quarters.
    # "Acceleration" = the YoY rate rising two quarters running (BOTH consecutive steps up).
    # This is seasonality-safe (YoY, not QoQ) and rejects a single rebound off a depressed
    # base — e.g. a cyclical's one-quarter snap-back (Ford -8%→-67%→+371%) — which a plain
    # recent-vs-prior-mean delta would mislabel "accelerating" and float to the top of the
    # sort. score (pp, for the column sort) = the WEAKER of the two up-steps when
    # accelerating / the milder down-step when decelerating / 0 when mixed.
    # Evaluate on the 3 most-recent ADJACENT reported quarters, computed from the raw
    # entries — NOT by compacting None-YoY quarters out of `quarters`, which would splice a
    # calendar gap into the "consecutive" triple (e.g. a turnaround with one loss quarter
    # mid-series) and could mislabel the trend. Require all three to be calendar-consecutive
    # AND have a positive year-ago base; otherwise the verdict is undecidable (None).
    verdict, score = None, None
    tail = rep[-3:]
    if len(tail) == 3 and all(_fp_next(tail[k]["FiscalPeriod"]) == tail[k + 1]["FiscalPeriod"]
                              for k in range(2)):
        yv, ok = [], True
        for e in tail:
            ya = actual.get(_year_ago_fp(e["FiscalPeriod"]))
            if not (ya and ya > 0):
                ok = False
                break
            yv.append(e["Actual"] / ya - 1.0)
        if ok:
            s1, s2 = yv[1] - yv[0], yv[2] - yv[1]
            tol = 0.005                          # 0.5pp dead-band vs float/rounding noise
            if s1 > tol and s2 > tol:
                verdict, score = "accel", min(s1, s2) * 100.0
            elif s1 < -tol and s2 < -tol:
                verdict, score = "decel", max(s1, s2) * 100.0
            else:
                verdict, score = "steady", 0.0

    # trailing-12-month EPS: rolling sum of 4 CONSECUTIVE reported quarters
    ttm_vals, ttm_latest = [], None
    for i in range(3, len(rep)):
        fps = [rep[j].get("FiscalPeriod") or "" for j in range(i - 3, i + 1)]
        if all(_fp_next(fps[k]) == fps[k + 1] for k in range(3)):
            s = sum(rep[j]["Actual"] for j in range(i - 3, i + 1))
            ttm_vals.append(s)
            if i == len(rep) - 1:                # window ends at the most-recent quarter
                ttm_latest = s
    ttm_new_high = bool(ttm_latest is not None and ttm_latest > 0
                        and ttm_latest >= max(ttm_vals) - 1e-9)

    n_beats = sum(1 for q in quarters[-4:] if q["beat"] is not None and q["beat"] > 0)
    n_beat_obs = sum(1 for q in quarters[-4:] if q["beat"] is not None)
    return {"quarters": quarters, "verdict": verdict, "accel_score": score,
            "ttm_new_high": ttm_new_high, "beat_streak": n_beats, "beat_obs": n_beat_obs}


def _margin_from_arrays(ni, rev, ttm_margin) -> dict | None:
    """Per-quarter after-tax (net) profit margin from the parallel net_income_fq_h /
    total_revenue_fq_h arrays. IMPORTANT: those arrays are NEWEST-FIRST (index 0 = most
    recent quarter) — the OPPOSITE orientation to the FiscalPeriod-sorted *_fq_h dict
    arrays — verified: the first-4-quarter margin equals TradingView's net_margin_ttm.

    Returns {latest, new_high, ttm, series} where latest = most-recent single-quarter
    margin %, new_high = latest is at/above every prior quarter's margin, ttm = current
    trailing-12-month net margin (TradingView's net_margin field, already after-tax).
    None if the arrays are unusable."""
    if not isinstance(ni, list) or not isinstance(rev, list) or not ni or not rev:
        return None
    n = min(len(ni), len(rev))
    series = []                                  # newest -> oldest, % (None where rev<=0)
    for i in range(n):
        r, x = rev[i], ni[i]
        if isinstance(r, (int, float)) and isinstance(x, (int, float)) and r > 0:
            series.append(x / r * 100.0)
        else:
            series.append(None)
    vals = [m for m in series if m is not None]
    if not vals:
        return None
    latest = series[0]
    # only star a profitable margin at a new high (a loss-maker's "least-negative" quarter
    # is technically a margin peak but not O'Neil's "best companies" tell)
    new_high = bool(latest is not None and latest > 0
                    and latest >= max(vals) - 1e-9 and len(vals) >= 4)
    ttm = float(ttm_margin) if isinstance(ttm_margin, (int, float)) else None
    return {"latest": latest, "new_high": new_high, "ttm": ttm,
            "series": series[:EPS_ACCEL_DISPLAY_Q]}


# ----------------------------------------------------------------------------- TradingView
def _rows_from_tv(arr) -> list:
    """1 most-recent reported quarter + ALL forward (estimate) quarters from a
    TradingView *_fq_h array of {Actual, Estimate, FiscalPeriod, IsReported}."""
    if not arr:
        return []
    # Defensive chronological sort (FiscalPeriod 'YYYY-Qn' sorts lexically) — the
    # unofficial TV feed is *observed* to be chronological but not contracted to be,
    # so don't trust raw order for "most recent reported" / forward sequencing.
    arr = sorted([e for e in arr if isinstance(e, dict)],
                 key=lambda e: e.get("FiscalPeriod") or "")
    # value map (actual if reported else estimate) for YoY lookups across the series
    valmap = {}
    for e in arr:
        fp = e.get("FiscalPeriod")
        v = e.get("Actual") if e.get("IsReported") else e.get("Estimate")
        if fp and v is not None:
            valmap[fp] = v
    reported = [e for e in arr if e.get("IsReported") and e.get("Actual") is not None]
    forward = [e for e in arr if not e.get("IsReported") and e.get("Estimate") is not None]
    chosen = reported[-1:] + forward          # 1 reported + every forward quarter
    rows = []
    for e in chosen:
        fp = e.get("FiscalPeriod")
        est = not e.get("IsReported")
        v = e.get("Estimate") if est else e.get("Actual")
        yoy = None
        ya = valmap.get(_year_ago_fp(fp))
        # Require a POSITIVE year-ago base: a growth % across a sign change (common for
        # EPS of loss-making names, e.g. -0.12 -> -0.07 is an IMPROVEMENT but v/ya-1
        # prints a misleading red "-42%") is meaningless. Revenue is always >0 so this
        # never suppresses a real revenue YoY.
        if ya and ya > 0 and v is not None:
            try:
                yoy = v / ya - 1.0
            except Exception:
                yoy = None
        rows.append({"lbl": _fp_label(fp, est), "v": v, "yoy": yoy, "est": est})
    return rows


def _parse_tv_row(d: list) -> tuple[str, dict] | None:
    """One scanner row -> (TICKER, record)."""
    try:
        def col(i):                              # defensive: TV may return a short row
            return d[i] if isinstance(d, list) and len(d) > i else None
        name = (col(0) or "").upper()
        cur = col(2) or "USD"
        rev = _rows_from_tv(col(3))
        eps = _rows_from_tv(col(4))
        if not name or not (rev or eps):
            return None
        return name, {"fetched": date.today().isoformat(), "ok": True, "cur": cur,
                      "src": "TradingView", "sv": CACHE_SCHEMA, "tk": name, "rev": rev, "eps": eps,
                      "eps_accel": _eps_accel_from_arr(col(4)),
                      "margin": _margin_from_arrays(col(5), col(6), col(7)),
                      "industry": col(8) or None}
    except Exception:
        return None


def _tv_fetch(tickers: list) -> dict:
    """Batch scanner POST -> {TICKER: record}. `name in_range` also resolves exchange.
    Never raises."""
    out: dict = {}
    if not tickers:
        return out
    body = json.dumps({
        "filter": [{"left": "name", "operation": "in_range",
                    "right": [t.upper() for t in tickers]}],
        "columns": TV_COLUMNS,
        "range": [0, len(tickers) * 2 + 20],
    }).encode()
    try:
        req = urllib.request.Request(
            TV_URL, data=body,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TV_TIMEOUT) as r:
            data = json.loads(r.read())
        # Parsing stays INSIDE the try: the public endpoint can return a non-dict error
        # envelope / rate-limit page / changed schema (a JSON list or string) on which
        # data.get / row.get would raise — that must degrade to {} (-> Yahoo fallback in
        # get()), honouring the "Never raises" contract, not crash the batch.
        rows = data.get("data") if isinstance(data, dict) else None
        for row in (rows or []):
            if not isinstance(row, dict):
                continue
            parsed = _parse_tv_row(row.get("d") or [])
            if parsed and parsed[0] not in out:        # first row per ticker wins
                out[parsed[0]] = parsed[1]
    except Exception:
        return out
    return out


# ----------------------------------------------------------------------------- Yahoo fallback
def _extract_yahoo(ticker: str) -> dict:
    """Fallback: 1 reported + 2 estimate quarters from yfinance. Never raises."""
    rec = {"fetched": date.today().isoformat(), "ok": False, "cur": "USD",
           "src": "Yahoo", "sv": CACHE_SCHEMA, "tk": (ticker or "").strip().upper(),
           "rev": [], "eps": []}
    if yf is None:
        return rec
    try:
        t = yf.Ticker(ticker)
        rev_hist, eps_hist = {}, {}
        q = t.quarterly_income_stmt
        if q is not None and not q.empty:
            if "Total Revenue" in q.index:
                for c, v in q.loc["Total Revenue"].items():
                    if pd.notna(v):
                        rev_hist[pd.Timestamp(c).date().isoformat()] = float(v)
            er = "Diluted EPS" if "Diluted EPS" in q.index else (
                "Basic EPS" if "Basic EPS" in q.index else None)
            if er:
                for c, v in q.loc[er].items():
                    if pd.notna(v):
                        eps_hist[pd.Timestamp(c).date().isoformat()] = float(v)
        rev_dates, eps_dates = sorted(rev_hist), sorted(eps_hist)
        last = datetime.strptime(rev_dates[-1], "%Y-%m-%d").date() if rev_dates else None

        def lbl(d_iso, est):
            d = datetime.strptime(d_iso, "%Y-%m-%d").date()
            return d.strftime("%b'%y") + ("e" if est else "")

        def yoy(hist, d_iso, val):
            try:
                d = datetime.strptime(d_iso, "%Y-%m-%d").date().replace(
                    year=datetime.strptime(d_iso, "%Y-%m-%d").year - 1)
            except Exception:
                return None
            best, gap = None, 26
            for k, v in hist.items():
                kd = datetime.strptime(k, "%Y-%m-%d").date()
                g = abs((kd - d).days)
                if g < gap and v:
                    best, gap = v, g
            return (val / best - 1.0) if best else None

        rev_rows, eps_rows = [], []
        for d_iso in rev_dates[-1:]:
            rev_rows.append({"lbl": lbl(d_iso, False), "v": rev_hist[d_iso],
                             "yoy": yoy(rev_hist, d_iso, rev_hist[d_iso]), "est": False})
        for d_iso in eps_dates[-1:]:
            eps_rows.append({"lbl": lbl(d_iso, False), "v": eps_hist[d_iso],
                             "yoy": yoy(eps_hist, d_iso, eps_hist[d_iso]), "est": False})

        def two_ends(base):
            """The two forward quarter-ends after `base` (label month/year only)."""
            if not base:
                return []
            m1 = base.month + 3
            e1 = date(base.year + (m1 - 1) // 12, (m1 - 1) % 12 + 1, 1)
            m2 = e1.month + 3
            return [e1, date(e1.year + (m2 - 1) // 12, (m2 - 1) % 12 + 1, 1)]

        last_eps = datetime.strptime(eps_dates[-1], "%Y-%m-%d").date() if eps_dates else None
        # Forward-quarter labels derived SEPARATELY for revenue vs EPS — yfinance can
        # report the two on different date grids, so reusing revenue's ends for EPS
        # would mis-date the EPS estimate rows.
        ends_rev, ends_eps = two_ends(last), two_ends(last_eps)
        re_, ee = t.revenue_estimate, t.earnings_estimate
        for i, p in enumerate(["0q", "+1q"]):
            for src, rows, ends in ((re_, rev_rows, ends_rev), (ee, eps_rows, ends_eps)):
                if src is not None and not src.empty and p in src.index:
                    avg = src.loc[p].get("avg")
                    g = src.loc[p].get("growth")
                    if pd.notna(avg):
                        L = (ends[i].strftime("%b'%y") + "e") if i < len(ends) else p + "e"
                        rows.append({"lbl": L, "v": float(avg),
                                     "yoy": (float(g) if pd.notna(g) else None), "est": True})
        cur = "USD"
        try:
            info = t.get_info() or {}
            cur = info.get("financialCurrency") or info.get("currency") or "USD"
        except Exception:
            pass
        rec.update({"cur": cur, "rev": rev_rows, "eps": eps_rows,
                    "ok": bool(rev_rows or eps_rows)})
    except Exception:
        pass
    return rec


# ----------------------------------------------------------------------------- API
def get(ticker: str) -> dict | None:
    if not ticker:
        return None
    tk = ticker.strip().upper()
    rec = _load_cache().get(tk)
    if rec is not None and _fresh(rec):
        return rec if rec.get("ok") else None
    found = _tv_fetch([tk])
    rec = found.get(tk) or _extract_yahoo(tk)
    _store(tk, rec)
    _flush()
    return rec if rec.get("ok") else None


def prefetch(tickers, workers: int = 8, budget_s: float = 90.0) -> None:
    """Warm the cache: batched TradingView first, then Yahoo for any ticker TV missed."""
    cache = _load_cache()
    todo, seen = [], set()
    for t in tickers:
        if not t:
            continue
        tk = str(t).strip().upper()
        if tk in seen:
            continue
        seen.add(tk)
        rec = cache.get(tk)
        if rec is None or not _fresh(rec):
            todo.append(tk)
    if not todo:
        return
    t0 = time.time()

    # 1) TradingView in batches (fast, covers the vast majority)
    missing = []
    for i in range(0, len(todo), TV_CHUNK):
        if time.time() - t0 > budget_s:
            missing.extend(todo[i:])
            break
        chunk = todo[i:i + TV_CHUNK]
        found = _tv_fetch(chunk)
        for tk in chunk:
            if tk in found:
                _store(tk, found[tk])
            else:
                missing.append(tk)
    _flush()

    # 2) Yahoo fallback for stragglers (concurrent, still under the budget)
    if missing and yf is not None:
        try:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_extract_yahoo, tk): tk for tk in missing}
                for fut in as_completed(futs):
                    tk = futs[fut]
                    try:
                        _store(tk, fut.result())
                    except Exception:
                        pass
                    if time.time() - t0 > budget_s:
                        break
        except Exception:
            pass
        _flush()


# ------------------------------------------------------------------- Tier 2: best-in-industry margin
# One batched scan per UNIQUE industry present in the report -> the industry's full TTM
# net-margin distribution, cached per calendar day. industry_percentile() is then a pure
# lookup, so "among the very best in its industry" (O'Neil) costs ~a dozen scans/day total.
_IND_DIST: dict | None = None


def _load_ind_dist() -> dict:
    global _IND_DIST
    if _IND_DIST is None:
        try:
            with open(INDUSTRY_MARGIN_CACHE) as fh:
                blob = json.load(fh)
            _IND_DIST = blob.get("dist") or {} if blob.get("date") == date.today().isoformat() else {}
        except Exception:
            _IND_DIST = {}
    return _IND_DIST


def _save_ind_dist() -> None:
    try:
        tmp = INDUSTRY_MARGIN_CACHE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"date": date.today().isoformat(), "dist": _IND_DIST}, fh)
        os.replace(tmp, INDUSTRY_MARGIN_CACHE)
    except Exception:
        pass


def _scan_industry_margins(industry: str) -> list:
    """Every constituent's TTM net margin for one industry. Never raises -> []."""
    body = json.dumps({
        "filter": [{"left": "industry", "operation": "equal", "right": industry}],
        "columns": ["net_margin"],
        "range": [0, 2000],
    }).encode()
    try:
        req = urllib.request.Request(
            TV_URL, data=body,
            headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TV_TIMEOUT) as r:
            data = json.loads(r.read())
        rows = data.get("data") if isinstance(data, dict) else None
        out = []
        for row in (rows or []):
            if isinstance(row, dict):
                d = row.get("d") or []
                if d and isinstance(d[0], (int, float)):
                    out.append(float(d[0]))
        return out
    except Exception:
        return []


def prefetch_industry_margins(tickers, budget_s: float = 30.0) -> None:
    """Warm the per-industry net-margin distribution for every industry present among the
    cached recs of `tickers`. Cached per day, time-boxed; a failure just leaves an industry
    without a percentile (the panel omits that clause). Run AFTER prefetch()."""
    dist = _load_ind_dist()
    cache = _load_cache()
    industries, seen = [], set()
    for t in tickers:
        rec = cache.get(str(t).strip().upper()) if t else None
        ind = rec.get("industry") if rec else None
        if ind and ind not in seen and ind not in dist:
            seen.add(ind)
            industries.append(ind)
    if not industries:
        return
    t0, changed = time.time(), False
    for ind in industries:
        if time.time() - t0 > budget_s:
            break
        vals = _scan_industry_margins(ind)
        if vals:
            dist[ind] = sorted(vals)
            changed = True
    if changed:
        _save_ind_dist()


def industry_percentile(industry, margin_ttm) -> float | None:
    """Percentile rank (0-100) of `margin_ttm` within its industry's TTM-net-margin
    distribution — 95 => only ~5% of the industry is more profitable. None if unknown or
    the industry sample is too thin to rank."""
    if not industry or not isinstance(margin_ttm, (int, float)):
        return None
    vals = _load_ind_dist().get(industry)
    if not vals or len(vals) < 5:
        return None
    below = sum(1 for v in vals if v <= margin_ttm)
    return round(below / len(vals) * 100.0)


# ------------------------------------------------------------------- Tier 3: estimate revisions
# How many analysts RAISED vs CUT estimates recently + how far consensus drifted (O'Neil:
# "how many times analysts have raised their estimates"). yfinance only — a PER-TICKER call,
# so warmed for the TOP PICKS only (not the full universe). Cached per calendar day.
_REV: dict | None = None


def _load_rev() -> dict:
    global _REV
    if _REV is None:
        try:
            blob = json.load(open(REVISIONS_CACHE))
            _REV = blob.get("data") or {} if blob.get("date") == date.today().isoformat() else {}
        except Exception:
            _REV = {}
    return _REV


def _save_rev() -> None:
    try:
        tmp = REVISIONS_CACHE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"date": date.today().isoformat(), "data": _REV}, fh)
        os.replace(tmp, REVISIONS_CACHE)
    except Exception:
        pass


def _extract_revisions(ticker: str) -> dict | None:
    """{up30,down30,up7,down7,drift90,period} from yfinance eps_revisions + eps_trend for
    the nearest estimate quarter. Never raises -> None."""
    if yf is None:
        return None
    try:
        t = yf.Ticker(ticker)
        rev = getattr(t, "eps_revisions", None)
        if rev is None or not hasattr(rev, "loc"):
            return None
        period = next((p for p in ("0q", "+1q") if p in getattr(rev, "index", [])), None)
        if period is None:
            return None
        row = rev.loc[period]

        def gi(col):
            try:
                v = row.get(col)
                return int(v) if (v is not None and v == v) else None
            except Exception:
                return None

        up30, down30, up7, down7 = gi("upLast30days"), gi("downLast30days"), gi("upLast7days"), gi("downLast7Days")
        if up30 is None and down30 is None:
            return None
        drift = None
        trend = getattr(t, "eps_trend", None)
        if trend is not None and hasattr(trend, "loc") and period in getattr(trend, "index", []):
            tr = trend.loc[period]
            try:
                cur, old = tr.get("current"), tr.get("90daysAgo")
                if cur is not None and cur == cur and old not in (None, 0) and old == old:
                    drift = float(cur) / float(old) - 1.0
            except Exception:
                drift = None
        return {"up30": up30, "down30": down30, "up7": up7, "down7": down7,
                "drift90": drift, "period": period}
    except Exception:
        return None


def prefetch_revisions(tickers, budget_s: float = 25.0, workers: int = 6) -> None:
    """Warm estimate revisions for a SMALL set (top picks) — concurrent, time-boxed, cached
    per day. Per-ticker yfinance, so deliberately NOT run across the full universe."""
    if yf is None:
        return
    cache = _load_rev()
    todo, seen = [], set()
    for t in tickers:
        if not t:
            continue
        tk = str(t).strip().upper()
        if tk in seen or tk in cache:
            continue
        seen.add(tk)
        todo.append(tk)
    if not todo:
        return
    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_extract_revisions, tk): tk for tk in todo}
            for fut in as_completed(futs):
                tk = futs[fut]
                try:
                    cache[tk] = fut.result()        # may be None (cached as "checked, no data")
                except Exception:
                    cache[tk] = None
                if time.time() - t0 > budget_s:
                    break
    except Exception:
        pass
    _save_rev()


def revisions(ticker: str) -> dict | None:
    return _load_rev().get(ticker.strip().upper()) if ticker else None


# ----------------------------------------------------------------------------- render
def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_money(v: float) -> str:
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.0f}M"
    if a >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


def _fmt_eps(v: float) -> str:
    return f"{v:.2f}"


def _yoy_html(y) -> str:
    if y is None:
        return "<span class='fund-na'>—</span>"
    pct = y * 100.0
    if pct > 0.5:
        return f"<span class='fund-up'>+{pct:.0f}%</span>"
    if pct < -0.5:
        return f"<span class='fund-dn'>{pct:.0f}%</span>"
    return "<span class='fund-flat'>0%</span>"   # flat band: avoid a '+-0%' rounding artifact


_ACCEL_CHIP = {
    "accel":  ("&#9650;&#9650; Accelerating", "#3fb950"),
    "steady": ("&#8594; Steady", "#8b949e"),
    "decel":  ("&#9660; Decelerating", "#ff7b72"),
}


def _beat_html(b) -> str:
    """Earnings-surprise % (Actual vs pre-report consensus). Green beat / red miss."""
    if b is None:
        return "<span class='fund-na'>—</span>"
    pct = b * 100.0
    if pct >= 0.5:
        return f"<span class='fund-up'>+{pct:.0f}%</span>"
    if pct <= -0.5:
        return f"<span class='fund-dn'>{pct:.0f}%</span>"
    return "<span class='fund-flat'>0%</span>"   # in-line: avoid a '+-0%' rounding artifact


def _margin_line(rec: dict) -> str:
    """'Net margin 71.5% (qtr) ⭐ new high · 63.0% TTM · top 5% in <industry>'."""
    m = rec.get("margin") or {}
    latest, ttm = m.get("latest"), m.get("ttm")
    if latest is None and ttm is None:
        return ""
    bits = []
    if latest is not None:
        star = (" <span style='color:#3fb950;font-weight:600;'>&#11088; new high</span>"
                if m.get("new_high") else "")
        bits.append(f"<b>Net margin</b> {latest:.1f}% <span style='color:#8b949e;'>(qtr)</span>{star}")
    if ttm is not None:
        bits.append(f"{ttm:.1f}% TTM")
    pct = industry_percentile(rec.get("industry"), ttm)
    if pct is not None:
        ind = _esc(rec.get("industry") or "industry")
        bits.append(f"<span style='color:#3fb950;'>top {max(1, 100 - int(pct))}% in {ind}</span>"
                    if pct >= 90 else f"{int(pct)}th pct in {ind}")
    return "<div class='fund-src' style='margin-top:4px;'>" + " &middot; ".join(bits) + "</div>"


def _revisions_line(ticker: str) -> str:
    """'Est. revisions (30d) ▲35 / ▼2 · consensus raised +7% vs 90d ago' (top picks only)."""
    r = revisions(ticker) if ticker else None
    if not r:
        return ""
    up, dn = r.get("up30"), r.get("down30")
    if up is None and dn is None:
        return ""
    cells = []
    if up is not None:
        cells.append(f"<span style='color:#3fb950;'>&#9650;{up}</span>")
    if dn is not None:
        cells.append(f"<span style='color:#ff7b72;'>&#9660;{dn}</span>")
    bits = ["<b>Est. revisions</b> <span style='color:#8b949e;'>(30d)</span> " + " / ".join(cells)]
    drift = r.get("drift90")
    if drift is not None and abs(drift) >= 0.005:
        col = "#3fb950" if drift > 0 else "#ff7b72"
        verb = "raised" if drift > 0 else "cut"
        bits.append(f"consensus {verb} <span style='color:{col};'>"
                    f"{'+' if drift >= 0 else ''}{drift * 100:.0f}%</span> vs 90d ago")
    return "<div class='fund-src' style='margin-top:2px;'>" + " &middot; ".join(bits) + "</div>"


def render_accel(rec: dict) -> str:
    """O'Neil block beneath the forward Rev/EPS table: trailing reported quarters (EPS,
    YoY%, earnings-beat %), an acceleration verdict, a TTM-EPS new-high badge, and the
    net-margin (new-high / TTM / best-in-industry) line. '' when there's nothing to show."""
    a = rec.get("eps_accel") or {}
    qs = a.get("quarters") or []
    margin = rec.get("margin") or {}
    if len(qs) < 2 and not margin:
        return ""
    chips = []
    v = a.get("verdict")
    if v in _ACCEL_CHIP:
        txt, col = _ACCEL_CHIP[v]
        chips.append(f"<span style='color:{col};font-weight:600;'>{txt}</span>")
    if a.get("ttm_new_high"):
        chips.append("<span style='color:#3fb950;font-weight:600;'>&#11088; TTM EPS new high</span>")
    head = " &nbsp;&middot;&nbsp; ".join(chips)
    out = ["<div class='fund-wrap' style='margin-top:8px;'>",
           "<div class='fund-src' style='margin:2px 0 4px;letter-spacing:.04em;'>EARNINGS ACCELERATION"
           f"{(' &mdash; ' + head) if head else ''}</div>"]
    if len(qs) >= 2:
        out.append("<table class='fund-tbl'>"
                   "<tr class='fund-head'><td>Qtr</td><td>EPS</td><td>YoY</td><td>Beat</td></tr>")
        for q in qs:
            eps_txt = (_fmt_eps(q["eps"]) if q.get("eps") is not None
                       else "<span class='fund-na'>—</span>")
            out.append(f"<tr class='fund-act'><td>{_esc(q['lbl'])}</td><td>{eps_txt}</td>"
                       f"<td>{_yoy_html(q.get('yoy'))}</td><td>{_beat_html(q.get('beat'))}</td></tr>")
        out.append("</table>")
    out.append(_margin_line(rec))
    out.append(_revisions_line(rec.get("tk")))   # Tier 3 — populated for top picks only
    out.append("</div>")
    return "".join(out)


def render_body(rec: dict) -> str:
    if not rec or not rec.get("ok"):
        return ""
    cur = rec.get("cur", "USD")
    rev = rec.get("rev", [])
    eps = rec.get("eps", [])
    # Label-driven: look up Rev and EPS for each period SEPARATELY (keyed by label,
    # which already encodes the est flag). This avoids the rev-empty bug where the
    # EPS value would otherwise be formatted into the Rev column.
    rev_by = {r["lbl"]: r for r in rev}
    eps_by = {e["lbl"]: e for e in eps}
    order = rev if rev else eps          # period ordering + est flag from the primary
    if not order:
        return ""
    na = "<span class='fund-na'>—</span>"
    out = ["<div class='fund-wrap'><table class='fund-tbl'>",
           "<tr class='fund-head'><td>Qtr</td><td>Rev</td><td>YoY</td><td>EPS</td></tr>"]
    for o in order:
        lbl = o["lbl"]
        rr, ee = rev_by.get(lbl), eps_by.get(lbl)
        cls = "fund-est" if o.get("est") else "fund-act"
        rev_txt = _esc(_fmt_money(rr["v"])) if (rr and rr.get("v") is not None) else na
        eps_txt = _fmt_eps(ee["v"]) if (ee and ee.get("v") is not None) else na
        yoy = rr.get("yoy") if rr else o.get("yoy")    # YoY column = revenue YoY
        out.append(
            f"<tr class='{cls}'><td>{_esc(lbl)}</td><td>{rev_txt}</td>"
            f"<td>{_yoy_html(yoy)}</td><td>{eps_txt}</td></tr>")
    out.append("</table>")
    n_est = sum(1 for o in order if o.get("est"))
    n_act = len(order) - n_est
    cur_note = cur if cur and cur != "USD" else "USD"
    src = rec.get("src", "Yahoo")
    out.append(f"<div class='fund-src'>{_esc(cur_note)} · {n_act} reported + {n_est} est · "
               f"{_esc(src)}</div></div>")
    out.append(render_accel(rec))            # O'Neil acceleration + beat + margin block
    return "".join(out)


def details_html(ticker: str, inner_html: str) -> str:
    rec = get(ticker)
    body = render_body(rec) if rec else ""
    if not body:
        return inner_html
    return f"<details class='fund'><summary>{inner_html}</summary>{body}</details>"


if __name__ == "__main__":
    import sys
    for tk in (sys.argv[1:] or ["NVDA", "MGM", "RKLB", "ASML", "QUBT"]):
        r = get(tk)
        print("\n###", tk, (r.get("src") if r else "NO DATA"))
        if r:
            print(render_body(r))
