"""
MADRRY Tier-A Forward Tracker & Win/Loss Study
==============================================

Goal
----
Track every name that appears on our Tier-A list (A+/A/A-) and watch how it
behaves AFTER the day we picked it, then label the outcome:

  WIN   - prints a FRESH 52-week high after the pick day (a forward bar's HIGH
          exceeds the prior 252-bar high as-of the pick date).
  LOSS  - intraday LOW touches entry x 0.92  (-8% from the breakout entry).
  OPEN  - neither has happened yet and < WINDOW trading days have elapsed.
  EXP   - WINDOW trading days elapsed with no win/loss (expired / no-result).

First event by date wins.  If a win-high and the -8% line are both touched on the
SAME bar, we record LOSS (conservative) and flag it (`same_bar_ambiguous`).

Dedup rule (v1):  ONE tracker per ticker, anchored at its FIRST Tier-A
appearance in the snapshot set.  Re-appearances on later days do not restart it.

Every feature is frozen as-of the pick day (META score + component breakdown,
RS, industry RS, sector/theme/industry, ANTS, HTF, ADR, dist-52w, status).  The
labeled records are written to tier_a_tracking.json so we can study which
features separate winners from losers and re-calibrate the META scoring so a
higher META score yields a higher win rate.

Run:  python3 madrry_tier_a_tracker.py            # build + grade + study
      python3 madrry_tier_a_tracker.py --study    # re-run study on existing db
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

WORKSPACE = "/Users/boundbythese/.openclaw/workspace"
DB_PATH = os.path.join(WORKSPACE, "tier_a_tracking.json")
WINDOW = 40            # trading-day follow window
LOSS_MULT = 0.92       # entry x 0.92  => -8%
LOOKBACK_52W = 252     # bars used for the prior 52-week high

TIER_A = {"A+", "A", "A-"}
_META_RE = re.compile(r"^\s*(.+?):\s*.*?\((\d+)\s*/\s*(\d+)\)\s*$")


# ---------------------------------------------------------------------------
# 1. BUILD TRACKERS  (first-appearance dedup, frozen features)
# ---------------------------------------------------------------------------
def parse_meta_details(details):
    """['Trend: Strong (10/15)', ...] -> {'Trend': {'pts':10,'max':15}}"""
    out = {}
    for line in details or []:
        m = _META_RE.match(line)
        if m:
            out[m.group(1).strip()] = {"pts": int(m.group(2)), "max": int(m.group(3))}
    return out


ARCHIVE_DIR = os.path.join(WORKSPACE, "snapshots_archive")


def snapshot_files():
    """All dated snapshot files from the DURABLE archive + the live workspace, deduped by
    data date (archive preferred — it is never pruned), sorted chronologically. Points the
    study at the FULL signal history so cohorts no longer vanish when the scanner prunes
    dated snapshots to the newest 14 (IMPROVEMENT_PLAN Phase 0/3 — grows the resolved
    sample the auto-apply / refit gates wait on). Additive: same first-appearance dedup."""
    import re
    rx = re.compile(r"latest_setups_(\d{4}-\d{2}-\d{2})\.json$")
    by_date = {}
    for base in (ARCHIVE_DIR, WORKSPACE):          # archive first -> archive wins ties
        for f in glob.glob(os.path.join(base, "latest_setups_*.json")):
            m = rx.search(f)
            if m:
                by_date.setdefault(m.group(1), f)
    return [by_date[d] for d in sorted(by_date)]


def build_trackers():
    files = snapshot_files()
    seen = {}
    for f in files:
        dt = os.path.basename(f).split("latest_setups_")[1].replace(".json", "")
        try:
            rows = json.load(open(f))
        except Exception:
            continue
        for r in rows:
            if (r.get("tier") or "") not in TIER_A:
                continue
            t = r.get("ticker")
            if not t or t in seen:
                continue
            entry = r.get("entry")
            if entry is None:
                continue
            seen[t] = {
                "ticker": t,
                "pick_date": dt,
                "tier": r.get("tier"),
                "entry": float(entry),
                "loss_line": round(float(entry) * LOSS_MULT, 4),
                # --- frozen features ---
                "meta_score": r.get("meta_score"),
                "meta_components": parse_meta_details(r.get("meta_details")),
                "rs_rating": r.get("rs_rating"),
                "ind_rs": r.get("ind_rs"),
                "rs_new_high": r.get("rs_new_high"),
                "sector": r.get("sector"),
                "theme": r.get("theme"),
                "ind_name": r.get("ind_name"),
                "is_htf": bool(r.get("is_htf")),
                "power_score": r.get("power_score"),
                "ants_label": r.get("ants_label"),
                "ants_level": r.get("ants_level"),
                "ants_ok": r.get("ants_ok"),
                "ants_3m_days": r.get("ants_3m_days"),
                "ants_3m_peak": r.get("ants_3m_peak"),
                "ants_rs_rising": r.get("ants_rs_rising"),
                "adr": r.get("adr"),
                # Basis marker: records from ~2026-06-30 on store the canonical 20-day ADR%
                # (TradingView ADRP / 100·(mean(H/L)−1)); a MISSING field = the older 1-day
                # Volatility.D basis (~40% larger). Any ADR-conditioned study must segregate
                # on this before comparing adr across the boundary.
                "adr_basis": "adrp20",
                "dist_52w": r.get("dist_52w"),
                "at_high": bool(r.get("at_high")),
                "nh_1m": r.get("nh_1m"),
                "nh_3m": r.get("nh_3m"),
                "days_since_high": r.get("days_since_high"),
                "status_labels": r.get("status_labels"),
            }
    return seen


# ---------------------------------------------------------------------------
# 2. PRICE DATA
# ---------------------------------------------------------------------------
def fetch_prices(tickers):
    """Batched daily OHLC, raw (auto_adjust=False) so highs/entry compare 1:1."""
    data = yf.download(
        tickers=sorted(tickers), period="2y", interval="1d",
        auto_adjust=False, group_by="ticker", threads=True, progress=False,
    )
    out = {}
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(how="all")
            if not df.empty:
                out[t] = df
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# 3. GRADE
# ---------------------------------------------------------------------------
def grade(tr, df):
    """Resolve one tracker against its price history."""
    pick = pd.Timestamp(tr["pick_date"])
    hist = df[df.index <= pick]
    fwd = df[df.index > pick].head(WINDOW)
    if hist.empty or fwd.empty:
        tr["outcome"] = "nodata"
        return tr

    prior_52w_high = float(hist["High"].tail(LOOKBACK_52W).max())
    tr["prior_52w_high"] = round(prior_52w_high, 4)

    win_date = loss_date = None
    same_bar = False
    for ts, row in fwd.iterrows():
        hi, lo = float(row["High"]), float(row["Low"])
        is_win = hi > prior_52w_high
        is_loss = lo <= tr["loss_line"]
        if is_win and is_loss:           # same bar -> conservative LOSS
            loss_date, same_bar = ts, True
            break
        if is_loss:
            loss_date = ts
            break
        if is_win:
            win_date = ts
            break

    # max favorable / adverse excursion over the followed window (context)
    tr["mfe_pct"] = round((float(fwd["High"].max()) / tr["entry"] - 1) * 100, 2)
    tr["mae_pct"] = round((float(fwd["Low"].min()) / tr["entry"] - 1) * 100, 2)
    tr["days_followed"] = int(len(fwd))

    if loss_date is not None:
        tr["outcome"] = "loss"
        res = loss_date
    elif win_date is not None:
        tr["outcome"] = "win"
        res = win_date
    elif len(fwd) >= WINDOW:
        tr["outcome"] = "expired"
        res = None
    else:
        tr["outcome"] = "open"
        res = None

    tr["same_bar_ambiguous"] = same_bar
    tr["resolve_date"] = res.date().isoformat() if res is not None else None
    tr["days_to_resolve"] = (
        int(fwd.index.get_loc(res)) + 1 if res is not None else None
    )
    return tr


# ---------------------------------------------------------------------------
# 4. STUDY
# ---------------------------------------------------------------------------
def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _wr(records):
    w = sum(r["outcome"] == "win" for r in records)
    l = sum(r["outcome"] == "loss" for r in records)
    n = w + l
    return w, l, n, (round(100 * w / n) if n else None)


def _bucket_table(records, keyfn, title, order=None):
    groups = defaultdict(list)
    for r in records:
        k = keyfn(r)
        if k is not None:
            groups[k].append(r)
    print(f"\n{title}")
    print(f"  {'bucket':<22}{'W':>4}{'L':>4}{'N':>5}{'win%':>7}")
    keys = order if order else sorted(groups, key=lambda k: -_wr(groups[k])[3] if _wr(groups[k])[3] is not None else 0)
    for k in keys:
        if k not in groups:
            continue
        w, l, n, wr = _wr(groups[k])
        if n == 0:
            continue
        print(f"  {str(k):<22}{w:>4}{l:>4}{n:>5}{(str(wr)+'%' if wr is not None else '-'):>7}")


def study(records):
    resolved = [r for r in records if r["outcome"] in ("win", "loss")]
    print("=" * 60)
    print("MADRRY TIER-A WIN/LOSS STUDY")
    print("=" * 60)
    oc = Counter(r["outcome"] for r in records)
    print(f"\nTrackers: {len(records)}   outcomes: {dict(oc)}")
    w, l, n, wr = _wr(records)
    print(f"Resolved: {n}  (win {w} / loss {l})   overall win-rate: {wr}%")
    open_n = oc.get("open", 0) + oc.get("expired", 0)
    print(f"Unresolved (open+expired): {open_n}   nodata: {oc.get('nodata',0)}")

    if n < 8:
        print("\n[!] Too few resolved names for a stable study yet — "
              "forward data is still young. Re-run as days accumulate.")

    # META score buckets  -> the headline question
    def meta_bucket(r):
        m = _num(r.get("meta_score"))
        if m is None:
            return None
        for lo in (60, 50, 40, 30, 20, 0):
            if m >= lo:
                return f">={lo}"
    _bucket_table(resolved, meta_bucket, "WIN-RATE BY META SCORE",
                  order=[">=60", ">=50", ">=40", ">=30", ">=20", ">=0"])

    # RS rating buckets
    def rs_bucket(r):
        v = _num(r.get("rs_rating"))
        if v is None:
            return None
        for lo in (90, 80, 70, 0):
            if v >= lo:
                return f"RS>={lo}"
    _bucket_table(resolved, rs_bucket, "WIN-RATE BY RS RATING",
                  order=["RS>=90", "RS>=80", "RS>=70", "RS>=0"])

    _bucket_table(resolved, lambda r: r.get("tier"), "WIN-RATE BY TIER",
                  order=["A+", "A", "A-"])
    _bucket_table(resolved, lambda r: r.get("sector"), "WIN-RATE BY SECTOR")
    _bucket_table(resolved, lambda r: "HTF" if r.get("is_htf") else "non-HTF",
                  "WIN-RATE BY HTF FLAG")
    _bucket_table(resolved, lambda r: "ANTS ok" if r.get("ants_ok") else "no ANTS",
                  "WIN-RATE BY ANTS")
    _bucket_table(resolved, lambda r: "at_high" if r.get("at_high") else "below_high",
                  "WIN-RATE BY AT-HIGH STATUS")

    # META COMPONENT correlation with winning
    print("\nMETA COMPONENT vs WIN (avg points: winners vs losers)")
    comps = defaultdict(lambda: {"win": [], "loss": []})
    for r in resolved:
        for cname, cv in (r.get("meta_components") or {}).items():
            comps[cname][r["outcome"]].append(cv["pts"])
    print(f"  {'component':<20}{'win avg':>9}{'loss avg':>10}{'edge':>8}")
    rows = []
    for cname, d in comps.items():
        wavg = np.mean(d["win"]) if d["win"] else None
        lavg = np.mean(d["loss"]) if d["loss"] else None
        edge = (wavg - lavg) if (wavg is not None and lavg is not None) else None
        rows.append((cname, wavg, lavg, edge))
    for cname, wavg, lavg, edge in sorted(rows, key=lambda x: -(x[3] if x[3] is not None else -99)):
        wa = f"{wavg:.1f}" if wavg is not None else "-"
        la = f"{lavg:.1f}" if lavg is not None else "-"
        ed = f"{edge:+.1f}" if edge is not None else "-"
        print(f"  {cname:<20}{wa:>9}{la:>10}{ed:>8}")
    print("\n(edge>0 => component scores higher on winners => deserves MORE weight;")
    print(" edge<=0 => not separating winners from losers => candidate to down-weight)")


# ---------------------------------------------------------------------------
# 5. META RECALIBRATION  (data-driven reweighting, validated on the labels)
# ---------------------------------------------------------------------------
# Original production max-points per component (scanner.py defaults).
# Used only as the fractional-normalisation fallback for old records.
CUR_MAX = {
    "Trend": 15, "Proximity": 10, "10MA Quality": 15, "Vol Contraction": 15,
    "Vol Expansion": 10, "Flag": 10, "Candle": 10, "Base Quality": 15,
    "RS": 15, "Volatility": 10, "Supply Shock": 10, "Risk": 15,
}
# Flag and Candle are the same slot (one or the other fires), max 10.

META_WEIGHTS_PATH = os.path.join(WORKSPACE, "meta_weights.json")


def active_weights():
    """Currently-LIVE component weights (meta_weights.json), else the original
    defaults. The weekend recalibration iterates from these, not from scratch."""
    base = dict(CUR_MAX)
    try:
        blob = json.load(open(META_WEIGHTS_PATH))
        for k, v in (blob.get("weights", blob) or {}).items():
            if k in base and isinstance(v, (int, float)):
                base[k] = v
        base["Candle"] = base["Flag"]   # shared slot
    except Exception:
        pass
    return base


def _frac_components(rec):
    """Return {component: pts/max} for a record's frozen META breakdown."""
    out = {}
    for c, d in (rec.get("meta_components") or {}).items():
        mx = d.get("max") or CUR_MAX.get(c)
        if mx:
            out[c] = d["pts"] / mx
    return out


def _compute_recal(records, lam=1.4, blend=0.5):
    """Pure compute: derive next-gen weights + re-scored buckets. Returns a dict.
    Iterates from the currently-active weights (meta_weights.json) so each
    weekend nudges the live weights rather than recomputing from defaults."""
    base = active_weights()
    resolved = [r for r in records if r["outcome"] in ("win", "loss")]
    fr = {"win": defaultdict(list), "loss": defaultdict(list)}
    for r in resolved:
        for c, f in _frac_components(r).items():
            fr[r["outcome"]][c].append(f)
    edge = {}
    for c in base:
        wv = np.mean(fr["win"][c]) if fr["win"].get(c) else 0.0
        lv = np.mean(fr["loss"][c]) if fr["loss"].get(c) else 0.0
        edge[c] = wv - lv
    mean_abs = np.mean([abs(e) for e in edge.values()]) or 1.0
    new_max = {}
    for c in base:
        mult = min(2.0, max(0.25, 1 + lam * edge.get(c, 0.0) / mean_abs))
        new_max[c] = blend * base[c] * mult + (1 - blend) * base[c]
    shared = round((new_max["Flag"] + new_max["Candle"]) / 2)
    # Absolute cap so no single component can vanish or dominate over many weeks.
    new_int = {c: min(40, max(3, round(v))) for c, v in new_max.items()}
    new_int["Flag"] = new_int["Candle"] = min(40, max(3, shared))
    denom_cur = sum(base[c] for c in base if c != "Candle")
    denom_new = sum(new_int[c] for c in new_int if c != "Candle")

    def score(rec, maxes, denom):
        s = sum(f * maxes.get(c, 0) for c, f in _frac_components(rec).items())
        return 100 * s / denom if denom else 0

    def buckets(key, maxes, denom):
        out = []
        for lo, hi in [(60, 999), (50, 60), (40, 50), (30, 40), (0, 30)]:
            grp = [r for r in resolved if lo <= score(r, maxes, denom) < hi]
            w = sum(x["outcome"] == "win" for x in grp)
            n = len(grp)
            out.append({"bucket": f"{lo}-{hi if hi < 999 else '+'}", "w": w,
                        "l": n - w, "n": n, "wr": round(100 * w / n) if n else None})
        return out

    def spread(maxes, denom):
        # DISPLAY ONLY — top(>=50) minus bottom(<30) win-rate. NOT the apply gate:
        # for a high-META Tier-A pool the <30 bucket is usually empty, so this
        # collapses to top-bucket win-rate (it does not measure separation).
        top = [r for r in resolved if score(r, maxes, denom) >= 50]
        bot = [r for r in resolved if score(r, maxes, denom) < 30]
        tw = sum(r["outcome"] == "win" for r in top) / len(top) if top else 0
        bw = sum(r["outcome"] == "win" for r in bot) / len(bot) if bot else 0
        return round(100 * (tw - bw))

    def corr(maxes, denom):
        # Point-biserial corr(score, win) over ALL resolved — boundary-free
        # discrimination metric. This is the apply gate (no empty-bucket trap).
        # NOTE: still IN-SAMPLE (fit and measured on the same names); a proper
        # decision needs an out-of-sample holdout. Treat as advisory only.
        xs = [score(r, maxes, denom) for r in resolved]
        ys = [1.0 if r["outcome"] == "win" else 0.0 for r in resolved]
        if len(xs) < 3:
            return 0.0
        sx, sy = np.std(xs), np.std(ys)
        if sx == 0 or sy == 0:
            return 0.0
        return round(float(np.corrcoef(xs, ys)[0, 1]), 4)

    weights = [{"comp": c, "cur": base[c], "new": new_int[c],
                "edge": round(edge.get(c, 0.0), 3)}
               for c in base if c != "Candle"]
    return {
        "weights": weights, "denom_cur": denom_cur, "denom_new": denom_new,
        "v1_buckets": buckets("v1", base, denom_cur),
        "v2_buckets": buckets("v2", new_int, denom_new),
        # spread_* are DISPLAY-ONLY and now consistently scored with `base`
        # (was a CUR_MAX-numerator / live-denom hybrid bug).
        "spread_v1": spread(base, denom_cur), "spread_v2": spread(new_int, denom_new),
        # corr_* is the real gate metric.
        "corr_v1": corr(base, denom_cur), "corr_v2": corr(new_int, denom_new),
    }


def compute_summary(records):
    """Structured study summary for the report tab (no printing)."""
    resolved = [r for r in records if r["outcome"] in ("win", "loss")]
    oc = Counter(r["outcome"] for r in records)
    w, l, n, wr = _wr(records)

    def bucket_rows(keyfn, order=None):
        groups = defaultdict(list)
        for r in resolved:
            k = keyfn(r)
            if k is not None:
                groups[k].append(r)
        keys = order or sorted(groups, key=lambda k: -(_wr(groups[k])[3] or 0))
        rows = []
        for k in keys:
            if k not in groups:
                continue
            ww, ll, nn, rr = _wr(groups[k])
            if nn:
                rows.append({"k": str(k), "w": ww, "l": ll, "n": nn, "wr": rr})
        return rows

    def meta_b(r):
        m = _num(r.get("meta_score"))
        return next((f">={lo}" for lo in (60, 50, 40, 30, 20, 0) if m is not None and m >= lo), None)

    def rs_b(r):
        v = _num(r.get("rs_rating"))
        return next((f"RS>={lo}" for lo in (90, 80, 70, 0) if v is not None and v >= lo), None)

    comps = defaultdict(lambda: {"win": [], "loss": []})
    for r in resolved:
        for cn, cv in (r.get("meta_components") or {}).items():
            comps[cn][r["outcome"]].append(cv["pts"])
    comp_rows = []
    for cn, d in comps.items():
        wa = float(np.mean(d["win"])) if d["win"] else None
        la = float(np.mean(d["loss"])) if d["loss"] else None
        comp_rows.append({"comp": cn, "win_avg": round(wa, 1) if wa is not None else None,
                          "loss_avg": round(la, 1) if la is not None else None,
                          "edge": round(wa - la, 1) if (wa is not None and la is not None) else None})
    comp_rows.sort(key=lambda x: -(x["edge"] if x["edge"] is not None else -99))

    return {
        "asof": max((r["pick_date"] for r in records), default=""),
        "overall": {"n": n, "w": w, "l": l, "wr": wr,
                    "open": oc.get("open", 0) + oc.get("expired", 0),
                    "nodata": oc.get("nodata", 0), "total": len(records)},
        "by_meta": bucket_rows(meta_b, [">=60", ">=50", ">=40", ">=30", ">=20", ">=0"]),
        "by_rs": bucket_rows(rs_b, ["RS>=90", "RS>=80", "RS>=70", "RS>=0"]),
        "by_tier": bucket_rows(lambda r: r.get("tier"), ["A+", "A", "A-"]),
        "by_sector": bucket_rows(lambda r: r.get("sector")),
        "by_htf": bucket_rows(lambda r: "HTF" if r.get("is_htf") else "non-HTF"),
        "by_ants": bucket_rows(lambda r: "ANTS ok" if r.get("ants_ok") else "no ANTS"),
        "by_athigh": bucket_rows(lambda r: "at_high" if r.get("at_high") else "below_high"),
        "components": comp_rows,
        "recal": _compute_recal(records),
    }


def recalibrate(records, lam=1.4, blend=0.5):
    """Derive v2 component weights from the labeled winners/losers.

    Method (interpretable, regularised — NOT a black box):
      edge_c   = mean(frac_c | win) - mean(frac_c | loss)      # predictive separation
      mult_c   = clip(1 + lam * edge_c / mean|edge|, 0.25, 2.0) # bounded nudge
      new_max  = blend*cur_max*mult_c + (1-blend)*cur_max       # shrink toward current
    Then re-score every resolved name with v2 and check monotonicity.
    """
    resolved = [r for r in records if r["outcome"] in ("win", "loss")]
    fr = {"win": defaultdict(list), "loss": defaultdict(list)}
    for r in resolved:
        for c, f in _frac_components(r).items():
            fr[r["outcome"]][c].append(f)

    comps = [c for c in CUR_MAX if fr["win"].get(c) or fr["loss"].get(c)]
    # Merge Flag/Candle into one slot for weighting (same scoring position).
    edge = {}
    for c in comps:
        wv = np.mean(fr["win"][c]) if fr["win"].get(c) else 0.0
        lv = np.mean(fr["loss"][c]) if fr["loss"].get(c) else 0.0
        edge[c] = wv - lv
    mean_abs = np.mean([abs(e) for e in edge.values()]) or 1.0

    new_max = {}
    for c in CUR_MAX:
        e = edge.get(c, 0.0)
        mult = min(2.0, max(0.25, 1 + lam * e / mean_abs))
        nm = blend * CUR_MAX[c] * mult + (1 - blend) * CUR_MAX[c]
        new_max[c] = nm
    # Round to clean integers; keep Flag==Candle equal (shared slot).
    shared = round((new_max["Flag"] + new_max["Candle"]) / 2)
    new_int = {c: max(0, round(v)) for c, v in new_max.items()}
    new_int["Flag"] = new_int["Candle"] = shared

    print("\n" + "=" * 60)
    print("META RECALIBRATION  (v2 proposed weights)")
    print("=" * 60)
    print(f"  {'component':<16}{'cur max':>8}{'edge':>8}{'new max':>9}{'Δ':>6}")
    denom_cur = denom_new = 0
    for c in CUR_MAX:
        if c == "Candle":
            continue  # shared with Flag, shown once
        cm, nm, e = CUR_MAX[c], new_int[c], edge.get(c, 0.0)
        denom_cur += cm
        denom_new += nm
        print(f"  {c:<16}{cm:>8}{e:>+8.2f}{nm:>9}{nm-cm:>+6}")
    print(f"  {'TOTAL (denom)':<16}{denom_cur:>8}{'':>8}{denom_new:>9}")

    # Re-score every resolved name under v1 and v2 (from stored fractions).
    def score(rec, maxes):
        s = tot = 0.0
        for c, f in _frac_components(rec).items():
            m = maxes.get(c, 0)
            s += f * m
            tot += m
        return 100 * s / denom_new if maxes is new_int else 100 * s / denom_cur

    for r in resolved:
        r["_v1"] = score(r, CUR_MAX)
        r["_v2"] = score(r, new_int)

    def mono(scorekey, label):
        print(f"\nWIN-RATE BY {label} (re-scored)")
        print(f"  {'bucket':<12}{'W':>4}{'L':>4}{'N':>5}{'win%':>7}")
        order = [(60, 999), (50, 60), (40, 50), (30, 40), (0, 30)]
        for lo, hi in order:
            grp = [r for r in resolved if lo <= r[scorekey] < hi]
            w = sum(x["outcome"] == "win" for x in grp)
            n = len(grp)
            wr = round(100 * w / n) if n else None
            tag = f"{lo}-{hi if hi<999 else '+'}"
            print(f"  {tag:<12}{w:>4}{n-w:>4}{n:>5}{(str(wr)+'%' if wr is not None else '-'):>7}")

    mono("_v1", "CURRENT META (v1)")
    mono("_v2", "RECALIBRATED META (v2)")

    # Spearman-ish: does v2 rank-order win-rate better? Report top-vs-bottom spread.
    def spread(key):
        top = [r for r in resolved if r[key] >= 50]
        bot = [r for r in resolved if r[key] < 30]
        tw = sum(r["outcome"] == "win" for r in top) / len(top) if top else 0
        bw = sum(r["outcome"] == "win" for r in bot) / len(bot) if bot else 0
        return round(100 * (tw - bw))
    print(f"\nTop(≥50)-minus-bottom(<30) win-rate spread:  "
          f"v1 = {spread('_v1')} pts   v2 = {spread('_v2')} pts   "
          f"(bigger = score separates winners better)")
    return new_int


def main():
    if "--study" in sys.argv and os.path.exists(DB_PATH):
        records = json.load(open(DB_PATH))["records"]
        study(records)
        if "--recalibrate" in sys.argv:
            recalibrate(records)
        return

    print("Building trackers from snapshots ...")
    trackers = build_trackers()
    print(f"  {len(trackers)} unique Tier-A names (first appearance)")

    print("Fetching price history (batched) ...")
    prices = fetch_prices(set(trackers))
    print(f"  got prices for {len(prices)}/{len(trackers)} names")

    records = []
    for t, tr in trackers.items():
        df = prices.get(t)
        if df is None:
            tr["outcome"] = "nodata"
        else:
            tr = grade(tr, df)
        records.append(tr)

    payload = {
        "generated": date.today().isoformat(),
        "window": WINDOW,
        "loss_mult": LOSS_MULT,
        "win_def": "fresh_52w_high",
        "loss_def": "entry*0.92 intraday-touch",
        "dedup": "first_appearance_per_ticker",
        "study_summary": compute_summary(records),
        "records": records,
    }
    tmp = DB_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, DB_PATH)
    print(f"Wrote {DB_PATH}  ({len(records)} records)")

    study(records)


if __name__ == "__main__":
    main()
