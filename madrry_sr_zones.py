"""
Support/Resistance zone engine + entry-quality grade (informational, shadow mode).

Implements the S&R playbook the user supplied (2026-07-04 tutorial), distilled to
mechanical rules:

  1. MEMORY      - zones come from swing pivots anywhere in the lookback ("look left");
                   old untouched levels still count.
  2. ZONES       - S/R is a price BAND, not a line: pivots cluster into [lo, hi] zones
                   (ATR-scaled tolerance); stops belong OUTSIDE the zone.
  3. FLIP        - broken support acts as resistance and vice versa; a flipped zone is
                   stronger protection than a plain pivot cluster.
  4. WEAR        - each test consumes a zone's strength; >=4 recent tests -> likely to
                   break (bad as protection, good as the barrier a breakout must clear).
  5. SHAKEOUT    - a fast pierce through a zone that snaps back (overshoot/trap) is
                   fuel for the 180-degree move, not a failure of the zone.
  6. TIMEFRAMES  - weekly-chart zones dominate daily ones: overlap = confluence.
  7. CONFLUENCE  - MAs (10/20/50/200) sitting inside the protecting zone add edge, as
                   does a LOW-VOLUME pullback into it.
  8. R:R         - entry near protection, stop just outside it, headroom to the next
                   opposing zone; the grade leans on that asymmetry.

Calibrated against the tutorial's own printed trades, point-in-time (THC 76-77.81
short 2022-04, XPO ~37.2 short 2023-02, SOXX ~398 pullback 2023-03, STAA 67.36/64.45
2023-01, AMD 2023-03, MRNA ~211 short 2022-01, COIN 50-53 vs 84.6-87.6 2023-03,
^GSPC/^RUT 2022-10); see tests/test_sr_zones.py + VERIFICATION.md.

STATUS RULING (WINNER_RADAR_CONTINUATION.md 2.7): this ships as an INFORMATIONAL
grade + logged features only. It filters nothing and changes no printed entry/stop
until the ledger accumulates era-consistent evidence that the grade discriminates.

Public API (neither ever raises - they degrade to []/{} and stash the error):

    zones(df, asof=None, timeframe="D")                       -> list of zone dicts
    analyze(df, entry, direction="long", asof=None)           -> sr_* feature dict

Output values are plain Python floats/ints/bools/strs (the scanner's snapshot
json.dumps has no default= handler - numpy scalars would kill the report).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# --- daily-timeframe tuning ---
PIVOT_K_DAILY = 3          # swing pivot needs k bars each side (daily)
LOOKBACK_D = 300           # daily bars of memory ("look left")
TOL_MULT_D = 1.0           # cluster tolerance = 1.0 x daily ATR% ...
TOL_MIN_D, TOL_MAX_D = 0.8, 2.5   # ...clamped (pct) - keeps zones 1-4% wide
PAD_ATR_D = 0.15           # zone edge padding, x daily ATR
# --- weekly-timeframe tuning ---
PIVOT_K_WEEKLY = 2
LOOKBACK_W = 160           # ~3y of weekly memory (tutorial: memory spans years)
TOL_MULT_W = 0.5
TOL_MIN_W, TOL_MAX_W = 1.5, 4.0
PAD_ATR_W = 0.10
# --- behaviour ---
TOUCH_ATR = 0.25           # a bar "touches" a zone if it comes within 0.25 x ATR
TEST_GAP = 3               # bars fully away from the zone before a new test counts
WEAR_WINDOW = 120          # wear = tests within the last N bars
WORN_TESTS = 4             # tutorial: a zone tested this often is about to fail
SHAKEOUT_BARS = 12         # look for overshoot-and-reclaim traps in the last N bars
LOWVOL_RATIO = 0.90        # pullback volume below 0.90 x 50d avg = "low volume"
NEAR_ATR = 1.5             # entry within 1.5 x ATR of protection = "not a chase"


def _coerce_dt_index(out: pd.DataFrame) -> pd.DataFrame:
    """CSV round-trips with MIXED utc offsets (EST/EDT) parse into an OBJECT
    index under pandas 2.x - the no-asof path shrugs but _asof_trim's .tz
    access dies, silently breaking point-in-time purity. Coerce wall-clock-
    preserving (tz_localize(None), never tz_convert - HK sessions must keep
    their local date). Strings/Timestamps only: ints must NOT become 1970
    nanosecond dates. Best-effort - failure leaves the index as-is."""
    try:
        if not isinstance(out.index, pd.DatetimeIndex) and out.index.dtype == object \
                and all(isinstance(t, (str, pd.Timestamp)) for t in out.index):
            out.index = pd.DatetimeIndex([
                pd.Timestamp(t).tz_localize(None)
                if pd.Timestamp(t).tzinfo is not None else pd.Timestamp(t)
                for t in out.index])
    except Exception:
        pass
    return out


def _norm_ohlcv(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Return a frame with lowercase o/h/l/c/v float columns or None.
    Passes through frames that are already in normalized form."""
    if df is None or len(df) < 60:
        return None
    if all(c in df.columns for c in ("o", "h", "l", "c")):
        out = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["o", "h", "l", "c"])
        out = _coerce_dt_index(out)
        return out if len(out) >= 60 else None
    cols = {str(c).lower(): c for c in df.columns}
    need = {"open": "o", "high": "h", "low": "l", "close": "c"}
    out = pd.DataFrame(index=df.index)
    for src, dst in need.items():
        if src not in cols:
            return None
        out[dst] = pd.to_numeric(df[cols[src]], errors="coerce")
    vol = cols.get("volume")
    out["v"] = pd.to_numeric(df[vol], errors="coerce") if vol else np.nan
    # +/-inf must die here too - one inf tick becomes an inf pivot, an inf zone,
    # and a literal Infinity token in the snapshot JSON (invalid RFC-8259).
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["o", "h", "l", "c"])
    out = _coerce_dt_index(out)
    return out if len(out) >= 60 else None


def _asof_trim(fr: pd.DataFrame, asof: Optional[str]) -> pd.DataFrame:
    if asof is None:
        return fr
    ts = pd.Timestamp(asof)
    if fr.index.tz is not None and ts.tz is None:
        ts = ts.tz_localize(fr.index.tz)
    return fr[fr.index <= ts]


def _atr_abs(fr: pd.DataFrame, n: int = 14) -> float:
    h, l, c = fr["h"].values, fr["l"].values, fr["c"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return float(np.mean(tr[-n:]))


def _pivots(fr: pd.DataFrame, k: int) -> List[dict]:
    """Swing highs/lows: bar whose high (low) is the unique extreme of +-k bars."""
    h, l = fr["h"].values, fr["l"].values
    idx = fr.index
    out = []
    for i in range(k, len(fr) - k):
        win_h = h[i - k:i + k + 1]
        win_l = l[i - k:i + k + 1]
        # keep-first tie-break: an exact double-top within k bars is the STRONGEST
        # kind of level - strict uniqueness would silently delete both pivots
        # (pervasive on coarse-tick / low-priced names).
        if h[i] >= win_h.max() and int(np.argmax(win_h)) == k:
            out.append({"i": i, "date": idx[i], "price": float(h[i]), "kind": "high"})
        if l[i] <= win_l.min() and int(np.argmin(win_l)) == k:
            out.append({"i": i, "date": idx[i], "price": float(l[i]), "kind": "low"})
    return out


def _cluster(pivots: List[dict], tol_pct: float) -> List[List[dict]]:
    """1-D agglomerative: sort by price, extend the cluster while the next pivot is
    within tol_pct percent of the cluster's running mean."""
    if not pivots:
        return []
    piv = sorted(pivots, key=lambda p: p["price"])
    groups: List[List[dict]] = [[piv[0]]]
    for p in piv[1:]:
        ref = float(np.mean([q["price"] for q in groups[-1]]))
        if ref > 0 and (p["price"] - ref) / ref * 100.0 <= tol_pct:
            groups[-1].append(p)
        else:
            groups.append([p])
    return groups


def _count_tests(fr: pd.DataFrame, lo: float, hi: float, atr: float):
    """Distinct tests of a zone: touch-runs separated by >= TEST_GAP bars fully
    outside the touch band. Returns (tests_total, tests_recent, last_touch_i)."""
    near_lo, near_hi = lo - TOUCH_ATR * atr, hi + TOUCH_ATR * atr
    hh, ll = fr["h"].values, fr["l"].values
    touch = (hh >= near_lo) & (ll <= near_hi)
    tests, tests_recent, last_i = 0, 0, None
    away = TEST_GAP                      # start "away" so the first touch counts
    n = len(fr)
    for i in range(n):
        if touch[i]:
            if away >= TEST_GAP:
                tests += 1
                if i >= n - WEAR_WINDOW:
                    tests_recent += 1
            away = 0
            last_i = i
        else:
            away += 1
    return tests, tests_recent, last_i


def _zone_from_group(group: List[dict], fr: pd.DataFrame, atr: float, pad_mult: float) -> dict:
    prices = [p["price"] for p in group]
    lo = min(prices) - pad_mult * atr
    hi = max(prices) + pad_mult * atr
    kinds = {p["kind"] for p in group}
    cl = fr["c"].values
    c_now = float(cl[-1])
    if c_now > hi:
        role = "support"
    elif c_now < lo:
        role = "resistance"
    else:
        # price is INSIDE the zone (a retest in progress): role = the side price
        # approached from - last close outside the band tells us which.
        role = "inside"
        outside = np.where((cl < lo) | (cl > hi))[0]
        if len(outside):
            # approached from below => the zone is being hit as RESISTANCE;
            # approached from above => being sat on as SUPPORT.
            role = "resistance" if cl[outside[-1]] < lo else "support"
    flipped = (role == "resistance" and "low" in kinds) or (role == "support" and "high" in kinds)
    tests, tests_recent, last_i = _count_tests(fr, lo, hi, atr)
    return {
        "lo": round(float(lo), 2), "hi": round(float(hi), 2),
        "mid": round(float(lo + hi) / 2.0, 2),
        "role": role, "flipped": bool(flipped),
        "pivots": int(len(group)), "tests": int(tests),
        "tests_recent": int(tests_recent),
        "kinds": sorted(kinds),
        "last_test_i": (int(last_i) if last_i is not None else None),
        "first_pivot_date": str(min(p["date"] for p in group).date()),
        "last_pivot_date": str(max(p["date"] for p in group).date()),
    }


def zones(df: pd.DataFrame, asof: Optional[str] = None, timeframe: str = "D") -> List[dict]:
    """All S/R zones visible in the (optionally as-of-truncated) frame."""
    try:
        fr = _norm_ohlcv(df)
        if fr is None:
            return []
        fr = _asof_trim(fr, asof)
        if len(fr) < 60:
            return []
        if timeframe == "W":
            fr = pd.DataFrame({
                "o": fr["o"].resample("W-FRI").first(),
                "h": fr["h"].resample("W-FRI").max(),
                "l": fr["l"].resample("W-FRI").min(),
                "c": fr["c"].resample("W-FRI").last(),
                "v": fr["v"].resample("W-FRI").sum(),
            }).dropna(subset=["o", "h", "l", "c"])
            fr = fr.tail(LOOKBACK_W)
            k, tol_mult, tol_min, tol_max, pad = (
                PIVOT_K_WEEKLY, TOL_MULT_W, TOL_MIN_W, TOL_MAX_W, PAD_ATR_W)
        else:
            fr = fr.tail(LOOKBACK_D)
            k, tol_mult, tol_min, tol_max, pad = (
                PIVOT_K_DAILY, TOL_MULT_D, TOL_MIN_D, TOL_MAX_D, PAD_ATR_D)
        if len(fr) < 40:
            return []
        atr = _atr_abs(fr)
        last = float(fr["c"].iloc[-1])
        if atr <= 0 or last <= 0:
            return []
        atr_pct = atr / last * 100.0
        tol = float(np.clip(tol_mult * atr_pct, tol_min, tol_max))
        piv = _pivots(fr, k)
        out = []
        for g in _cluster(piv, tol):
            z = _zone_from_group(g, fr, atr, pad)
            z["timeframe"] = timeframe
            out.append(z)
        return sorted(out, key=lambda z: z["mid"])
    except Exception:
        return []


def _overlaps(a: dict, b: dict) -> bool:
    return a["lo"] <= b["hi"] and b["lo"] <= a["hi"]


def _detect_shakeout(fr: pd.DataFrame, zone: dict, long_side: bool) -> bool:
    """Trap rule: within the last SHAKEOUT_BARS bars a bar pierced through the
    protecting zone (low under a support's lo / high over a resistance's hi) yet the
    market reclaimed it - a close back on the right side within <=2 bars."""
    n = len(fr)
    lo_i = max(0, n - SHAKEOUT_BARS)
    h, l, c = fr["h"].values, fr["l"].values, fr["c"].values
    for i in range(lo_i, n):
        if long_side:
            if l[i] < zone["lo"]:                      # overshoot below support
                for j in range(i, min(i + 3, n)):
                    if c[j] > zone["lo"]:              # reclaimed
                        return True
        else:
            if h[i] > zone["hi"]:                      # overshoot above resistance
                for j in range(i, min(i + 3, n)):
                    if c[j] < zone["hi"]:
                        return True
    return False


def _ma_confluence(fr: pd.DataFrame, zone: dict) -> List[str]:
    out = []
    c = fr["c"]
    for n in (10, 20, 50, 200):
        if len(c) >= n:
            ma = float(c.rolling(n).mean().iloc[-1])
            if zone["lo"] <= ma <= zone["hi"]:
                out.append(str(n) + "MA")
    return out


def _lowvol_pullback(fr: pd.DataFrame, long_side: bool) -> Optional[bool]:
    """Are the recent counter-trend bars quiet? (long: down-days in the last 5;
    short: up-days). Low volume against you = the tutorial's ideal retest."""
    v = fr["v"].values.astype(float)
    c = fr["c"].values
    if len(fr) < 55 or np.isnan(v[-50:]).any():
        return None
    v50 = float(np.mean(v[-50:]))
    if v50 <= 0:
        return None
    chg = np.diff(c)
    recent = []
    for i in range(len(fr) - 5, len(fr)):
        if i <= 0:
            continue
        counter = chg[i - 1] < 0 if long_side else chg[i - 1] > 0
        if counter:
            recent.append(v[i])
    if not recent:
        return None
    return bool(np.mean(recent) < LOWVOL_RATIO * v50)


def analyze(df: pd.DataFrame, entry: Any, direction: str = "long",
            asof: Optional[str] = None) -> Dict[str, Any]:
    """Entry-quality features for a planned entry price. {} when the chart is
    unusable; never raises. All values are plain Python types (json-safe)."""
    try:
        entry = float(entry)
        if not np.isfinite(entry) or entry <= 0:
            return {}
        fr = _norm_ohlcv(df)
        if fr is None:
            return {}
        fr = _asof_trim(fr, asof)
        if len(fr) < 60:
            return {}
        zd = zones(fr, timeframe="D")
        zw = zones(fr, timeframe="W")
        if not zd:
            return {}
        fr = fr.tail(LOOKBACK_D)
        atr = _atr_abs(fr)
        if atr <= 0:
            return {}

        long_side = (direction != "short")
        # protection = nearest zone on the risk side of the planned entry; the zone
        # may straddle the entry (retest in progress - stop still goes outside it).
        # Selection is purely geometric: once price trades at the entry, everything
        # below (long) / above (short) protects, whatever its role was before.
        if long_side:
            cands = [z for z in zd if z["lo"] < entry]
            prot = max(cands, key=lambda z: z["hi"]) if cands else None
            bars_ = [z for z in zd if z["lo"] > entry]
            barrier = min(bars_, key=lambda z: z["lo"]) if bars_ else None
        else:
            cands = [z for z in zd if z["hi"] > entry]
            prot = min(cands, key=lambda z: z["lo"]) if cands else None
            bars_ = [z for z in zd if z["hi"] < entry]
            barrier = max(bars_, key=lambda z: z["hi"]) if bars_ else None
        if prot is None:
            return {"sr_grade": "C", "sr_score": 30.0, "sr_flags": ["no_protection"]}

        flags: List[str] = []
        score = 50.0

        # suggested stop: OUTSIDE the protecting zone (tutorial rule) with an
        # ATR-fraction buffer, floored at min(0.75 ATR, 2%) from the entry - a
        # stop pinned a fraction of an ATR away is shaken out by pure noise, but
        # on high-ATR names 0.75 ATR alone would out-widen a perfectly good zone
        # stop (tutorial's tight stops are the "2-3% paper cut"; workable 2-5%).
        # Informational only - the printed stop is untouched.
        floor_d = min(0.75 * atr, 0.02 * entry)
        if long_side:
            stop_sug = min(prot["lo"] - 0.25 * atr, entry - floor_d)
        else:
            stop_sug = max(prot["hi"] + 0.25 * atr, entry + floor_d)
        # the floor/padding can land the stop INSIDE a neighboring zone - exactly
        # the noise-edge placement rule 2 forbids; walk it out (bounded)...
        stop0 = stop_sug
        for _ in range(8):
            z = next((q for q in zd if q["lo"] < stop_sug < q["hi"]), None)
            if z is None:
                break
            stop_sug = (z["lo"] - 0.25 * atr) if long_side else (z["hi"] + 0.25 * atr)
        risk = (entry - stop_sug) if long_side else (stop_sug - entry)
        # ...but never widen a workable stop past the tutorial's own ~8% hard
        # limit just to clear a stacked band (e.g. COIN's 48-51 shelf under the
        # 51-54 protection): in that conflict the 8% rule wins, revert.
        risk0 = (entry - stop0) if long_side else (stop0 - entry)
        if risk / entry * 100.0 > 8.0 and risk0 / entry * 100.0 <= 8.0:
            stop_sug, risk = stop0, risk0
        risk_pct = risk / entry * 100.0
        # the protection STACK = everything between the stop and the entry
        band = ({"lo": min(stop_sug, prot["lo"]), "hi": entry} if long_side
                else {"lo": entry, "hi": max(stop_sug, prot["hi"])})

        # flip is judged relative to the TRADE: a long protected by prior HIGHS is
        # sitting on a flipped (resistance->support) zone; a short capped by prior
        # LOWS is hitting a flipped (support->resistance) zone. A flip visible on
        # a WEEKLY zone inside the protection stack counts too (higher timeframe
        # dominates - e.g. the tutorial's COIN "50 was resistance, now support").
        need_kind = "high" if long_side else "low"
        prot_flip = need_kind in prot["kinds"] or any(
            need_kind in z["kinds"] for z in zw if _overlaps(z, band))
        if prot_flip:
            flags.append("flip")
            score += 15
        if prot["tests_recent"] >= WORN_TESTS:
            flags.append("prot_worn")           # protection tested too often lately
            score -= 10
        if _detect_shakeout(fr, prot, long_side):
            flags.append("shakeout")
            score += 10
        wk = [z for z in zw if _overlaps(z, prot)]
        if wk:
            flags.append("wk_confl")
            score += 10
        mas = _ma_confluence(fr, prot)
        if mas:
            flags.append("ma_confl:" + "+".join(mas))
            score += 8
        lv = _lowvol_pullback(fr, long_side)
        if lv:
            flags.append("lowvol_pullback")
            score += 7

        # chase / extension: how far past the protection is the entry?
        edge = prot["hi"] if long_side else prot["lo"]
        ext_abs = (entry - edge) if long_side else (edge - entry)
        ext_atr = ext_abs / atr
        if ext_atr > NEAR_ATR:
            flags.append("extended")
            score -= 12

        headroom = None
        rr = None
        no_headroom = False
        if barrier is not None:
            tgt = barrier["lo"] if long_side else barrier["hi"]
            headroom = (tgt - entry) if long_side else (entry - tgt)
            if risk > 0:
                rr = headroom / risk
            # wear on the barrier counts ALL tests (memory): a much-tested
            # opposing zone is the one the tutorial expects to break.
            if barrier["tests"] >= WORN_TESTS:
                flags.append("barrier_worn")
                score += 5
            if headroom < 1.0 * atr:
                flags.append("no_headroom")
                no_headroom = True
                score -= 15
        else:
            flags.append("blue_sky")            # nothing in the way at all
            score += 5

        # higher-timeframe target (tutorial lesson 6): the weekly barrier sets the
        # realistic exit for the swing, even when daily zones sit in between.
        rr_wk = None
        if zw and risk > 0:
            if long_side:
                wb = [z for z in zw if z["lo"] > entry]
                wb = min(wb, key=lambda z: z["lo"]) if wb else None
            else:
                wb = [z for z in zw if z["hi"] < entry]
                wb = max(wb, key=lambda z: z["hi"]) if wb else None
            if wb is not None:
                wtgt = wb["lo"] if long_side else wb["hi"]
                rr_wk = ((wtgt - entry) if long_side else (entry - wtgt)) / risk

        best_rr = max(x for x in (rr, rr_wk, float("-inf")) if x is not None)
        if np.isfinite(best_rr):
            if best_rr >= 3:
                score += 10
            elif not no_headroom:               # <1-ATR headroom already charged
                if best_rr < 1:
                    score -= 15
                elif best_rr < 2:
                    score -= 5
        if risk_pct > 8.0:
            flags.append("wide_zone_stop")      # tutorial: >7-8% risk is the limit
            score -= 8

        score = float(np.clip(score, 0.0, 100.0))
        grade = "A" if score >= 75 else ("B" if score >= 55 else "C")
        out: Dict[str, Any] = {
            "sr_grade": grade,
            "sr_score": round(score, 1),
            "sr_flags": [str(f) for f in flags],
            "sr_prot_lo": float(prot["lo"]), "sr_prot_hi": float(prot["hi"]),
            "sr_prot_tests": int(prot["tests"]), "sr_prot_flip": bool(prot_flip),
            "sr_stop_suggest": round(float(stop_sug), 2),
            "sr_risk_pct": round(float(risk_pct), 2),
            "sr_ext_atr": round(float(ext_atr), 2),
            "sr_wk_confl": bool(wk),
        }
        # USER 2026-07-06: DRAW-ONLY zone band for the 60-bar chart — the nearest
        # zone built from the LAST 60 bars first, widening the pivot lookback 60
        # bars at a time (120, 180, ...) until a zone exists. Kept SEPARATE from
        # sr_prot_* (the trade/gate zone) so grades, stops, risk and scores stay
        # byte-for-byte unchanged; the chart draws sr_draw_* when present and
        # falls back to sr_prot_* otherwise.
        try:
            _px = float(fr["c"].iloc[-1])
            _n = len(fr)
            for _win in (60, 120, 180, 240, 300):
                _zz = zones(fr.tail(_win), timeframe="D")
                if _zz:
                    _zd = min(_zz, key=lambda z: abs(z["mid"] - _px))
                    out["sr_draw_lo"] = float(_zd["lo"])
                    out["sr_draw_hi"] = float(_zd["hi"])
                    break
                if _win >= _n:
                    break
        except Exception:  # noqa: BLE001 — never break a scan over the chart hint
            pass
        if barrier is not None:
            out["sr_barrier_lo"] = float(barrier["lo"])
            out["sr_barrier_hi"] = float(barrier["hi"])
            out["sr_barrier_tests"] = int(barrier["tests"])
        if headroom is not None:
            out["sr_headroom_pct"] = round(float(headroom) / entry * 100.0, 2)
        if rr is not None:
            out["sr_rr"] = round(float(rr), 2)
        if rr_wk is not None:
            out["sr_rr_wk"] = round(float(rr_wk), 2)
        # defense in depth: a non-finite value anywhere = a broken chart -> {}
        # (the snapshot json.dumps would otherwise emit an invalid Infinity token)
        if any(isinstance(v, float) and not np.isfinite(v) for v in out.values()):
            return {}
        return out
    except Exception as exc:                     # never let this kill a scan
        return {"sr_error": str(exc)[:120]}
