"""
Trendline engine v2 (tutorial #3, 2026-07-04) - shadow mode.

Diagonal support/resistance, mechanized from the trendline tutorial. The scanner
already carries a first-generation 4-type block (calculate_trendline_analysis,
untouched - its badges/scoring are existing behavior); THIS module adds the
tutorial's refinements as tl_* shadow features:

  1. TWO-POINT construction through swing highs/lows; four kinds:
       UTL  rising  line through lows  = support in an uptrend
       TRL  rising  line through highs = overhead target in an uptrend
       DTL  falling line through highs = resistance in a downtrend
       TSL  falling line through lows  = support in a downtrend
  2. VALIDITY ("make sense"): between and after the anchors no bar may pierce
     the line beyond an ATR-scaled tolerance without consequence - a line is a
     ZONE, and overshoots that snap back within <=3 bars are shakeouts (a
     POSITIVE signal), while a decisive close beyond it BREAKS the line.
  3. FLIP: a broken support line acts as resistance afterwards (and vice
     versa) - the break is recorded, the line stays visible as its mirror.
  4. STEEPNESS: the steeper the line, the weaker (nothing on its left side) -
     steep lines are flagged; after they break, price seeks the next
     shallower line.
  5. WEAR: like horizontal zones, many tests weaken a line (>=4 touches).
  6. MICRO-ADJUSTMENT: lines re-anchor as new swing points print - all valid
     recent lines coexist; the report reads the NEAREST governing lines.
  7. HIGHER TIMEFRAME dominates: weekly (and monthly, where history allows)
     lines are computed from resampled bars and outrank daily ones.

STATUS: informational shadow mode (handbook ruling 2.7) - filters nothing,
never raises, json-safe plain-Python output. Same contract as madrry_sr_zones
and madrry_pullback_buy (whose _norm_ohlcv/_asof_trim/_atr_abs are reused).

    lines(df, asof=None, timeframe="D")               -> list of line dicts
    analyze_lines(df, entry, direction="long", asof=None) -> tl_* feature dict
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from madrry_sr_zones import _asof_trim, _atr_abs, _norm_ohlcv, _pivots

# --- construction ---
LOOKBACK = {"D": 300, "W": 160, "M": 400}
PIVOT_K = {"D": 3, "W": 2, "M": 2}
MIN_SPAN = {"D": 10, "W": 4, "M": 6}      # min bars between the two anchors
# cap per kind (sorted by quality) - decades-long monthly frames legitimately
# host more coexisting lines than a 1y daily frame
MAX_PER_KIND = {"D": 6, "W": 6, "M": 12}
# --- zone / behaviour (multiples of the frame's ATR) ---
VALID_TOL = 0.50          # anchors-to-now pierce tolerance while line "holds"
TOUCH_TOL = 0.35          # a bar within this of the line = a touch
OVERSHOOT_MAX = 1.00      # pierce deeper than this (on a close) = break...
SNAPBACK_BARS = 3         # ...unless the close reclaims within 0-3 bars
WORN_TOUCHES = 4          # tutorial: the 5th test tends to fail
STEEP_PCT_DAY = {"D": 0.45, "W": 1.10, "M": 2.00}   # %/bar slope = "steep"
NEAR_ATR = 0.75           # "at the line" proximity for flags


def _line_value(p0, slope, i):
    return p0["price"] + slope * (i - p0["i"])


def _classify(kind: str, slope: float) -> str:
    if kind == "high":
        return "TRL" if slope > 0 else "DTL"
    return "UTL" if slope > 0 else "TSL"


def _scan_line(fr, p0, p1, atr) -> Optional[dict]:
    """Validate one candidate line and measure its post-anchor life."""
    h, l, c = fr["h"].values, fr["l"].values, fr["c"].values
    n = len(fr)
    slope = (p1["price"] - p0["price"]) / (p1["i"] - p0["i"])
    kind = p0["kind"]
    is_support = kind == "low"
    # between the anchors the line must hold as drawn (zone tolerance)
    for i in range(p0["i"] + 1, p1["i"]):
        v = _line_value(p0, slope, i)
        if is_support and l[i] < v - VALID_TOL * atr:
            return None
        if not is_support and h[i] > v + VALID_TOL * atr:
            return None
    touches, snapbacks = 0, 0
    broken_i = None
    # away starts at 0 (NOT the "start away" idiom): the scan begins right
    # after anchor #2, which sits ON the line by construction - price must
    # first LEAVE the line before a post-anchor bar counts as a new test,
    # else the anchor's own +-k swing double-counts as touch #1.
    away = 0
    i = p1["i"] + 1
    while i < n:
        v = _line_value(p0, slope, i)
        pierced = (c[i] < v - OVERSHOOT_MAX * atr) if is_support \
            else (c[i] > v + OVERSHOOT_MAX * atr)
        if pierced:
            # decisive close beyond the zone: shakeout only if reclaimed fast
            reclaimed = False
            for j in range(i + 1, min(i + 1 + SNAPBACK_BARS, n)):
                vj = _line_value(p0, slope, j)
                if (is_support and c[j] > vj) or (not is_support and c[j] < vj):
                    reclaimed = True
                    snapbacks += 1
                    i = j          # resume after the reclaim
                    break
            if not reclaimed:
                broken_i = i
                break
            i += 1
            continue
        near = (l[i] <= v + TOUCH_TOL * atr and h[i] >= v - TOUCH_TOL * atr)
        if near:
            # a counted test must HOLD: the close stays on the correct side of
            # the line (or within the touch band beyond it - zone rule). A wide
            # bar that closes decisively past the band merely passed THROUGH
            # the line, not evidence it was defended.
            held = (c[i] >= v - TOUCH_TOL * atr) if is_support \
                else (c[i] <= v + TOUCH_TOL * atr)
            if held and away >= 3:
                touches += 1
            away = 0
        else:
            away += 1
        i += 1
    v_today = _line_value(p0, slope, n - 1)
    # steepness as SEEN on the chart at the right edge = slope relative to the
    # line's own level (normalizing by current price misgrades lines far from
    # price - a TSL at 31 vs price 256 read 8x too shallow).
    ref = abs(float(v_today))
    if ref < 1e-9:
        ref = float(c[-1])
    slope_pct = slope / ref * 100.0 if ref > 0 else 0.0
    return {
        "kind": _classify(kind, slope),
        "anchors": [str(p0["date"].date()), str(p1["date"].date())],
        "slope_pct_bar": round(float(slope_pct), 3),
        "slope_abs": float(slope),                 # $/own-frame-bar, unrounded (chart drawing)
        "value_today": round(float(v_today), 2),
        "value_today_raw": float(v_today),         # unrounded (chart drawing)
        "value_next": round(float(_line_value(p0, slope, n)), 2),
        "touches": int(touches), "snapbacks": int(snapbacks),
        "broken": broken_i is not None,
        "bars_since_break": (int(n - 1 - broken_i) if broken_i is not None else None),
        "worn": bool(touches >= WORN_TOUCHES),
        "age_bars": int(n - 1 - p1["i"]),
    }


def lines(df: pd.DataFrame, asof: Optional[str] = None, timeframe: str = "D",
          max_per_kind: Optional[int] = None) -> List[dict]:
    """All valid trendlines visible in the (optionally as-of-truncated) frame."""
    try:
        fr = _norm_ohlcv(df)
        if fr is None:
            return []
        fr = _asof_trim(fr, asof)
        if len(fr) < 60:
            return []
        if timeframe in ("W", "M"):
            rule = "W-FRI" if timeframe == "W" else "ME"
            vol = fr["v"] if "v" in fr.columns else pd.Series(np.nan, index=fr.index)
            fr = pd.DataFrame({
                "o": fr["o"].resample(rule).first(),
                "h": fr["h"].resample(rule).max(),
                "l": fr["l"].resample(rule).min(),
                "c": fr["c"].resample(rule).last(),
                "v": vol.resample(rule).sum(),
            }).dropna(subset=["o", "h", "l", "c"])
        fr = fr.tail(LOOKBACK.get(timeframe, 300))
        if len(fr) < 40:
            return []
        atr = _atr_abs(fr)
        if atr <= 0:
            return []
        piv = _pivots(fr, PIVOT_K.get(timeframe, 3))
        highs = [p for p in piv if p["kind"] == "high"]
        lows = [p for p in piv if p["kind"] == "low"]
        steep_cut = STEEP_PCT_DAY.get(timeframe, 0.45)
        out = []
        for grp in (highs, lows):
            for a in range(len(grp) - 1):
                for b in range(a + 1, len(grp)):
                    p0, p1 = grp[a], grp[b]
                    if p1["i"] - p0["i"] < MIN_SPAN.get(timeframe, 10):
                        continue
                    ln = _scan_line(fr, p0, p1, atr)
                    if ln is None:
                        continue
                    ln["timeframe"] = timeframe
                    ln["steep"] = bool(abs(ln["slope_pct_bar"]) > steep_cut)
                    out.append(ln)
        # keep the strongest PER KIND (alive first, then touches, then freshness)
        # PLUS the lines nearest to the current price per kind - an old many-
        # touch line must never crowd out the line that GOVERNS price today
        out.sort(key=lambda x: (x["broken"], -x["touches"], x["age_bars"]))
        px = float(fr["c"].iloc[-1])
        cap = max_per_kind if max_per_kind is not None else MAX_PER_KIND.get(timeframe, 6)
        kept: List[dict] = []
        counts: Dict[str, int] = {}
        for q in out:
            k = q["kind"]
            if counts.get(k, 0) < cap:
                kept.append(q)
                counts[k] = counts.get(k, 0) + 1
        seen = {id(q) for q in kept}
        by_dist = sorted([q for q in out if not q["broken"]],
                         key=lambda q: abs(q["value_today"] - px))
        near_n: Dict[str, int] = {}
        for q in by_dist:
            k = q["kind"]
            if near_n.get(k, 0) >= 3:
                continue
            near_n[k] = near_n.get(k, 0) + 1
            if id(q) not in seen:
                kept.append(q)
                seen.add(id(q))
        # never let the cap evict a JUST-broken line: alive-first sorting plus
        # the alive-only re-add above would drop breaks 0-3 bars old - the
        # tutorial's most actionable event and the sole source of the
        # fresh_break_* flags in analyze_lines. Bounded: at most 2 per kind.
        fresh = sorted(
            [q for q in out if q["broken"] and q["bars_since_break"] is not None
             and q["bars_since_break"] <= SNAPBACK_BARS],
            key=lambda q: (q["bars_since_break"], -q["touches"]))
        fb_n: Dict[str, int] = {}
        for q in fresh:
            k = q["kind"]
            if fb_n.get(k, 0) >= 2:
                continue
            fb_n[k] = fb_n.get(k, 0) + 1
            if id(q) not in seen:
                kept.append(q)
                seen.add(id(q))
        return kept
    except Exception:
        return []


def _nearest(cands, entry, below: bool):
    if below:
        xs = [q for q in cands if q["value_next"] < entry]
        return max(xs, key=lambda q: q["value_next"]) if xs else None
    xs = [q for q in cands if q["value_next"] > entry]
    return min(xs, key=lambda q: q["value_next"]) if xs else None


def analyze_lines(df: pd.DataFrame, entry: Any, direction: str = "long",
                  asof: Optional[str] = None) -> Dict[str, Any]:
    """Governing-trendline features for a planned entry. {} when unusable;
    never raises; plain-Python json-safe values only."""
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
        ld = lines(fr, timeframe="D")
        lw = lines(fr, timeframe="W")
        if not ld and not lw:
            return {}
        atr = _atr_abs(fr.tail(300))
        if atr <= 0:
            return {}
        long_side = direction != "short"
        alive = [q for q in ld + lw if not q["broken"]]
        sup_kinds = ("UTL", "TSL")
        res_kinds = ("TRL", "DTL")
        sup = _nearest([q for q in alive if q["kind"] in sup_kinds], entry, below=True)
        res = _nearest([q for q in alive if q["kind"] in res_kinds], entry, below=False)
        # higher timeframe DOMINATES (tutorial rule 7): when a weekly line sits
        # at effectively the same level as the daily winner (within 0.5 ATR),
        # the weekly read governs - surface it, not the daily shadow of it.
        for sel, kinds, below in ((sup, sup_kinds, True), (res, res_kinds, False)):
            if sel is None or sel["timeframe"] != "D":
                continue
            wk = _nearest([q for q in alive if q["kind"] in kinds
                           and q["timeframe"] == "W"], entry, below=below)
            if wk is not None and abs(wk["value_next"] - sel["value_next"]) <= 0.5 * atr:
                if below:
                    sup = wk
                else:
                    res = wk

        flags: List[str] = []
        out: Dict[str, Any] = {}
        # $/trading-day slope divisor per the line's own timeframe (a weekly
        # line advances one of ITS bars every ~5 trading days) - lets the chart
        # renderer draw the true diagonal across daily candles.
        _tf_days = {"D": 1.0, "W": 5.0, "M": 21.0}
        if sup is not None:
            out.update({
                "tl_sup_kind": sup["kind"], "tl_sup_tf": sup["timeframe"],
                "tl_sup_at": float(sup["value_next"]),
                "tl_sup_now": float(sup["value_today_raw"]),
                "tl_sup_slope_d": float(sup["slope_abs"]) / _tf_days.get(sup["timeframe"], 1.0),
                "tl_sup_dist_atr": round((entry - sup["value_next"]) / atr, 2),
                "tl_sup_touches": int(sup["touches"]),
                "tl_sup_anchors": [str(a) for a in sup["anchors"]],
            })
            if (entry - sup["value_next"]) <= NEAR_ATR * atr:
                flags.append("at_" + sup["kind"])       # danger-point zone
            if sup["steep"]:
                flags.append("sup_steep")               # weak diagonal support
            if sup["worn"]:
                flags.append("sup_worn")
            if sup["snapbacks"] > 0:
                flags.append("sup_shakeout")            # overshoot & reclaim
            if sup["timeframe"] != "D":
                flags.append("sup_higher_tf")
        if res is not None:
            out.update({
                "tl_res_kind": res["kind"], "tl_res_tf": res["timeframe"],
                "tl_res_at": float(res["value_next"]),
                "tl_res_now": float(res["value_today_raw"]),
                "tl_res_slope_d": float(res["slope_abs"]) / _tf_days.get(res["timeframe"], 1.0),
                "tl_res_dist_atr": round((res["value_next"] - entry) / atr, 2),
                "tl_res_touches": int(res["touches"]),
                "tl_res_anchors": [str(a) for a in res["anchors"]],
            })
            out["tl_res_headroom_pct"] = round(
                (res["value_next"] - entry) / entry * 100.0, 2)
            if (res["value_next"] - entry) <= NEAR_ATR * atr:
                # longs: into the diagonal lid (take-profit zone); shorts: the
                # protected danger-point at overhead resistance
                flags.append("at_" + res["kind"])
            if res["worn"]:
                flags.append("res_worn")                # lid likely to give way
            if res["timeframe"] != "D":
                flags.append("res_higher_tf")
        # fresh break events (tutorial: broken support flips to resistance;
        # a DTL broken upward is the pullback-recovery / breakout confirmation).
        # freshness is measured in TRADING DAYS: bars_since_break is in the
        # line's OWN bars, so one weekly bar counts as ~5 daily bars - only a
        # break in the CURRENT week is fresh, and it gets a _W marker because
        # that "decisive weekly close" is the in-progress W-FRI bar until Friday
        for q in ld + lw:
            if q["broken"] and q["bars_since_break"] is not None:
                mult = 5 if q["timeframe"] == "W" else 1
                if q["bars_since_break"] * mult > 3:
                    continue
                tf_tag = "" if q["timeframe"] == "D" else "_" + str(q["timeframe"])
                if q["kind"] in sup_kinds:
                    flags.append("fresh_break_down_" + q["kind"] + tf_tag)
                else:
                    flags.append("fresh_break_up_" + q["kind"] + tf_tag)
        # --- SALIENT lines for the CHART (USER 2026-07-06) -------------------
        # analyze_lines above selects the trade-geometry lines (nearest alive
        # support below entry / resistance above entry). Those are correct for
        # the gate + text, but they hide the human-obvious multi-touch / recent
        # lines and can surface a 1-touch line 5 ATR away. Draw the SALIENT line
        # instead: most construction+retest touches, nearest to price, daily
        # preferred, INCLUDING a just-broken line (a fresh break IS the event —
        # EPD's June-highs DTL). Emitted as tl_draw_* so the gate/flags/text
        # (tl_sup_at, tl_sup_dist_atr, ...) are completely untouched.
        try:
            px = float(fr["c"].iloc[-1])
            _tf_d = {"D": 1.0, "W": 5.0, "M": 21.0}

            def _valnow(q):
                return q.get("value_today_raw", q["value_today"])

            def _sal(q):
                dist = abs(_valnow(q) - px) / atr if atr else 0.0
                eff = q["touches"] + q["snapbacks"] + 2   # 2 construction anchors always count
                return eff * 2.0 - dist - (2.0 if q.get("steep") else 0.0)

            def _drawable(q):
                # a line the chart can draw: alive (or broken <=3 bars ago) and
                # sitting within 4 ATR of price (near the visible action)
                alive = (not q["broken"]) or (q["bars_since_break"] is not None
                                              and q["bars_since_break"] <= SNAPBACK_BARS)
                return bool(alive and abs(_valnow(q) - px) <= 4.0 * atr)

            def _pick(pool, kinds):
                c = [q for q in pool if q["kind"] in kinds and _drawable(q)]
                return max(c, key=_sal, default=None)

            # USER 2026-07-06: anchor the DRAWN trendlines on the LAST 60 bars
            # first; if no salient line of a side can be constructed there, widen
            # the pivot lookback 60 bars at a time (120, 180, ...) until one
            # appears. The chart still DISPLAYS 60 bars — only the anchoring
            # pivots may come from further left, and the line is extrapolated
            # back into the window via value-today + slope. Daily lines are
            # primary; the weekly higher-TF pool is the final backstop when even
            # full daily history yields no drawable line of a side.
            _sup2 = _res2 = None
            _n = len(fr)
            for _win in (60, 120, 180, 240, 300):
                _ldw = lines(fr.tail(_win), timeframe="D")
                if _sup2 is None:
                    _sup2 = _pick(_ldw, sup_kinds)
                if _res2 is None:
                    _res2 = _pick(_ldw, res_kinds)
                if (_sup2 is not None and _res2 is not None) or _win >= _n:
                    break
            if _sup2 is None or _res2 is None:            # weekly higher-TF backstop
                _lww = lines(fr, timeframe="W")
                if _sup2 is None:
                    _sup2 = _pick(_lww, sup_kinds)
                if _res2 is None:
                    _res2 = _pick(_lww, res_kinds)
            if _sup2 is not None:
                out["tl_draw_sup_now"] = float(_valnow(_sup2))
                out["tl_draw_sup_slope_d"] = float(_sup2["slope_abs"]) / _tf_d.get(_sup2["timeframe"], 1.0)
                out["tl_draw_sup_kind"] = _sup2["kind"]
            if _res2 is not None:
                out["tl_draw_res_now"] = float(_valnow(_res2))
                out["tl_draw_res_slope_d"] = float(_res2["slope_abs"]) / _tf_d.get(_res2["timeframe"], 1.0)
                out["tl_draw_res_kind"] = _res2["kind"]
        except Exception:  # noqa: BLE001 — never break a scan over the chart hint
            pass

        if not out and not flags:
            return {}
        out["tl_flags"] = sorted(set(str(f) for f in flags))
        if any(isinstance(v, float) and not np.isfinite(v) for v in out.values()):
            return {}
        return out
    except Exception as exc:                              # never kill a scan
        return {"tl_error": str(exc)[:120]}
