"""
Parallel-channel engine (tutorial #4, 2026-07-04) - shadow mode.

Mechanizes the channel playbook: price runs inside PARALLEL rails, and the
projected rail tells you where to take profit (long at the top of an up
channel), where to cover (short at the bottom of a down channel), and where an
entry has no edge (long into the top = short-edge territory).

  1. 2+1 CONSTRUCTION: a channel is a validated two-anchor trendline (the
     base) parallel-copied through ONE opposite-side swing pivot - explicitly
     NOT the common 3-point method. Either base works: two lows + one high or
     two highs + one low. The 2+1 copy predicts earlier, and the tutorial's
     core claim is that the projection is MOST accurate on its first touch.
  2. TREND PREREQUISITE: no channel without an actual trend. Up needs a higher
     high AND a higher low among the window's pivots; down needs lower high
     AND lower low (388.HK lesson: lower highs alone are not a downtrend).
     A breach-in-progress (price trading beyond the last pivot) counts for
     the side the pivot lag hasn't confirmed yet.
  3. ZONE SEMANTICS: rails are zones. An overshoot that closes back inside
     within <=3 bars is a shakeout (fuel for the 180-degree move), while a
     decisive close >1 ATR beyond a rail without reclaim BREAKS that side.
  4. FIRST-TOUCH FRESHNESS: touches on the projected rail are counted;
     <2 completed tests = "fresh" (the forecast still carries the 2+1 edge),
     >=4 = worn (accuracy decays from the 3rd touch on - overshoots, early
     turns, slope changes).
  5. REASONABLENESS (2333.HK lesson): a channel top that lands inside a
     horizontal resistance zone is the RIGHT channel to draw - flagged as SR
     confluence. A channel whose lower rail price already violated is dead.
  6. FAKE-BREAK WATCH (SEDG lesson): an upside break OUT of a down channel in
     a downtrend is squeeze fuel until proven - flagged, never assumed real.
  7. HIGHER TIMEFRAME dominates, same as the trendline engine.

STATUS: informational shadow mode (handbook ruling 2.7) - filters nothing,
never raises, json-safe plain-Python output. NOT part of the user-ratified
Stage-4 support gate (SR/PB/TL); promotion needs explicit user sign-off.

    channels(df, asof=None, timeframe="D")                  -> list of dicts
    analyze_channels(df, entry, direction="long", asof=None) -> ch_* dict
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from madrry_sr_zones import _asof_trim, _atr_abs, _norm_ohlcv, _pivots
import madrry_sr_zones as SR
from madrry_trendlines import (LOOKBACK, MIN_SPAN, OVERSHOOT_MAX, PIVOT_K,
                               SNAPBACK_BARS, STEEP_PCT_DAY, TOUCH_TOL,
                               VALID_TOL)

# --- channel-specific tuning ---
MIN_WIDTH_ATR = 1.5        # thinner than this is line noise, not a channel
MAX_WIDTH_ATR = {"D": 12.0, "W": 14.0, "M": 18.0}
MAX_PER_DIR = 4            # kept channels per direction (up/down) per frame
FRESH_PROJ_MAX = 2         # <2 completed projected-rail tests = still "fresh"
WORN_TOUCHES = 4           # same wear rule as horizontal zones / lines
NEAR_ATR = 0.75            # "at the rail" proximity (daily ATR)
NEAR_POS = 12.0            # ...or within this % of the channel's height
SHAKEOUT_BARS = 12         # recent-overshoot window (daily bars)
CONTAIN_SLACK = 0.5        # rail slack (x frame ATR) when testing containment
SR_CONFL_ATR = 0.35        # top-vs-horizontal-zone confluence tolerance


def _scan_rail(fr, a_i: int, a_price: float, slope: float, is_upper: bool,
               atr: float) -> Tuple[int, int, Optional[int], Optional[int]]:
    """Post-anchor life of one rail: (touches, snapbacks, broken_i, last_snap_i).
    Same zone semantics as the trendline engine: away>=3 + held-close touches,
    decisive close >1 ATR beyond = break unless reclaimed within <=3 bars."""
    h, l, c = fr["h"].values, fr["l"].values, fr["c"].values
    n = len(fr)
    touches = snapbacks = 0
    broken_i = last_snap_i = None
    away = 0
    i = a_i + 1
    while i < n:
        v = a_price + slope * (i - a_i)
        pierced = (c[i] > v + OVERSHOOT_MAX * atr) if is_upper \
            else (c[i] < v - OVERSHOOT_MAX * atr)
        if pierced:
            reclaimed = False
            for j in range(i + 1, min(i + 1 + SNAPBACK_BARS, n)):
                vj = a_price + slope * (j - a_i)
                if (is_upper and c[j] < vj) or (not is_upper and c[j] > vj):
                    reclaimed = True
                    snapbacks += 1
                    last_snap_i = j
                    i = j
                    break
            if not reclaimed:
                broken_i = i
                break
            i += 1
            continue
        near = (l[i] <= v + TOUCH_TOL * atr and h[i] >= v - TOUCH_TOL * atr)
        if near:
            held = (c[i] <= v + TOUCH_TOL * atr) if is_upper \
                else (c[i] >= v - TOUCH_TOL * atr)
            if held and away >= 3:
                touches += 1
            away = 0
        else:
            away += 1
        i += 1
    return touches, snapbacks, broken_i, last_snap_i


TREND_STEP_ATR = 0.30      # a wave step must be MATERIAL, not tick noise


def _trend_ok(piv: List[dict], h, l, i0: int, want_up: bool, atr: float) -> bool:
    """388.HK prerequisite: an up channel needs a higher high AND a higher low
    in the window; down needs lower high AND lower low - and the steps must be
    MATERIAL (>= 0.3 ATR): the tutorial's counter-example is a range whose
    lows sat flat (a 0.26-ATR dip between equal lows is not 一浪低於一浪).
    The wave relationship is CURRENT, not historical: each side is read by
    its LATEST MATERIAL step - walk the adjacent swing-pivot pairs from the
    newest backwards and let the first step >= 0.3 ATR decide (immaterial
    wiggles are ignored, so a pullback doesn't erase the trend, but a crash
    leg's internal dip months ago can't outvote weeks of rising lows - 388's
    Jul-Sep counter-example). The pivot +-k confirmation lag is covered by a
    raw breach beyond the LAST pivot of the trending side."""
    step = TREND_STEP_ATR * atr

    def _latest_material(seq):
        for a, b in zip(seq[-2::-1], seq[::-1]):
            d = b["price"] - a["price"]
            if abs(d) >= step:
                return 1 if d > 0 else -1
        return 0

    highs = [p for p in piv if p["kind"] == "high" and p["i"] >= i0]
    lows = [p for p in piv if p["kind"] == "low" and p["i"] >= i0]
    if want_up:
        breach = (float(h[highs[-1]["i"] + 1:].max()) - highs[-1]["price"]) \
            if (highs and h[highs[-1]["i"] + 1:].size > 0) else -1.0
        hh = _latest_material(highs) == 1 or breach >= step
        hl = _latest_material(lows) == 1
        # runaway move: a DEEP breach (>= 1 ATR beyond the last swing high)
        # is the trend itself - relentless legs print no +-k pivots at all
        # (388 weekly Aug-Oct 2022), so pair evidence goes silent exactly
        # when the trend is most obvious
        return (hh and hl) or breach >= atr
    breach = (lows[-1]["price"] - float(l[lows[-1]["i"] + 1:].min())) \
        if (lows and l[lows[-1]["i"] + 1:].size > 0) else -1.0
    ll = _latest_material(lows) == -1 or breach >= step
    lh = _latest_material(highs) == -1
    return (ll and lh) or breach >= atr


def channels(df: pd.DataFrame, asof: Optional[str] = None,
             timeframe: str = "D") -> List[dict]:
    """All valid parallel channels (2+1) visible in the frame. Never raises."""
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
        h, l, c = fr["h"].values, fr["l"].values, fr["c"].values
        n = len(fr)
        piv = _pivots(fr, PIVOT_K.get(timeframe, 3))
        highs = [p for p in piv if p["kind"] == "high"]
        lows = [p for p in piv if p["kind"] == "low"]
        span_min = MIN_SPAN.get(timeframe, 10)
        w_max = MAX_WIDTH_ATR.get(timeframe, 12.0)
        steep_cut = STEEP_PCT_DAY.get(timeframe, 0.45)
        out: List[dict] = []
        for grp, is_low_base in ((lows, True), (highs, False)):
            opp = highs if is_low_base else lows
            for a in range(len(grp) - 1):
                for b in range(a + 1, len(grp)):
                    p0, p1 = grp[a], grp[b]
                    if p1["i"] - p0["i"] < span_min:
                        continue
                    slope = (p1["price"] - p0["price"]) / (p1["i"] - p0["i"])
                    if slope == 0:
                        continue
                    direction = "up" if slope > 0 else "down"
                    # NO base-level dedup: near-collinear anchors with slightly
                    # different slopes diverge by whole ATRs once extrapolated
                    # to today (reviewer-measured material channel loss on ~1/3
                    # of real snapshots under a bucketed skip, still present
                    # under an anchor-tolerance skip). Every valid pair is
                    # scanned - measured cost is ~7ms/ticker - and the output
                    # stays bounded by the rail-level _dup + caps below.
                    v_base_today = p1["price"] + slope * (n - 1 - p1["i"])
                    # base validity between the anchors (zone tolerance)
                    ok = True
                    for i in range(p0["i"] + 1, p1["i"]):
                        v = p0["price"] + slope * (i - p0["i"])
                        if is_low_base and l[i] < v - VALID_TOL * atr:
                            ok = False
                            break
                        if not is_low_base and h[i] > v + VALID_TOL * atr:
                            ok = False
                            break
                    if not ok:
                        continue
                    # trend prerequisite for the channel's window
                    if not _trend_ok(piv, h, l, p0["i"], direction == "up", atr):
                        continue
                    # base rail life after anchor #2
                    b_t, b_s, b_broken, b_snap_i = _scan_rail(
                        fr, p1["i"], p1["price"], slope, not is_low_base, atr)
                    # 2+1 projection: EVERY opposite-side pivot in the window is
                    # a candidate rail, not just the most extreme one - the
                    # 2333.HK micro-adjustment lesson draws through the NEWEST
                    # swing high and treats the older spike as overshoot. Each
                    # candidate rail is judged from the moment the channel is
                    # COMPLETE (both anchors + projection printed): the
                    # micro-adjusted channel describes the current regime and
                    # is not invalidated by the regime it replaced.
                    cands = []
                    for q in opp:
                        if q["i"] < p0["i"]:
                            continue
                        off = q["price"] - (p0["price"] + slope * (q["i"] - p0["i"]))
                        if is_low_base and off <= 0:
                            continue
                        if not is_low_base and off >= 0:
                            continue
                        width = abs(off)
                        if width < MIN_WIDTH_ATR * atr or width > w_max * atr:
                            continue
                        cands.append((q, off))
                    # near-identical rails collapse to one scan (the skipped
                    # pivot registers as a touch of the kept rail anyway)
                    cands.sort(key=lambda t: -abs(t[1]))
                    picked = []
                    for q, off in cands:
                        if any(abs(off - o2) <= TOUCH_TOL * atr for _, o2 in picked):
                            continue
                        picked.append((q, off))
                        if len(picked) >= 6:
                            break
                    for best, best_off in picked:
                        width = abs(best_off)
                        start_i = max(p1["i"], best["i"])
                        p_t, p_s, p_broken, p_snap_i = _scan_rail(
                            fr, start_i,
                            best["price"] + slope * (start_i - best["i"]),
                            slope, is_low_base, atr)
                        if is_low_base:
                            bot_i, bot_p = p1["i"], p1["price"]
                            top_i, top_p = best["i"], best["price"]
                            bot_t, bot_s, bot_broken, bot_snap = b_t, b_s, b_broken, b_snap_i
                            top_t, top_s, top_broken, top_snap = p_t, p_s, p_broken, p_snap_i
                        else:
                            top_i, top_p = p1["i"], p1["price"]
                            bot_i, bot_p = best["i"], best["price"]
                            top_t, top_s, top_broken, top_snap = b_t, b_s, b_broken, b_snap_i
                            bot_t, bot_s, bot_broken, bot_snap = p_t, p_s, p_broken, p_snap_i
                        broken_side = None
                        broken_i = None
                        if top_broken is not None and (bot_broken is None
                                                       or top_broken <= bot_broken):
                            broken_side, broken_i = "top", top_broken
                        elif bot_broken is not None:
                            broken_side, broken_i = "bottom", bot_broken
                        top_today = top_p + slope * (n - 1 - top_i)
                        bot_today = bot_p + slope * (n - 1 - bot_i)
                        # steepness as seen at the right edge, normalized by
                        # the BASE LINE's own level (the trendline engine's
                        # convention) with its subnormal guard - a collapse
                        # close must not overflow slope_pct_bar to +-inf
                        ref = abs(float(v_base_today))
                        if ref < 1e-9:
                            ref = abs(float(c[-1]))
                        if ref < 1e-9:
                            ref = 1.0
                        pos = (float(c[-1]) - bot_today) / (top_today - bot_today) * 100.0 \
                            if top_today > bot_today else None
                        base_kind = ("UTL" if direction == "up" else "TSL") if is_low_base \
                            else ("TRL" if direction == "up" else "DTL")
                        out.append({
                            "dir": direction, "timeframe": timeframe,
                            "base_kind": base_kind,
                            "base_anchors": [str(p0["date"].date()), str(p1["date"].date())],
                            "proj_anchor": str(best["date"].date()),
                            "slope_pct_bar": round(float(slope / ref * 100.0), 3),
                            "slope_abs": float(slope),          # $/own-frame-bar, unrounded (chart drawing)
                            "steep": bool(abs(slope / ref * 100.0) > steep_cut),
                            "width_atr": round(float(width / atr), 2),
                            "frame_atr": round(float(atr), 4),
                            "top_today": round(float(top_today), 2),
                            "top_today_raw": float(top_today),  # unrounded (chart drawing)
                            "top_next": round(float(top_p + slope * (n - top_i)), 2),
                            "bot_today": round(float(bot_today), 2),
                            "bot_today_raw": float(bot_today),  # unrounded (chart drawing)
                            "bot_next": round(float(bot_p + slope * (n - bot_i)), 2),
                            "top_touches": int(top_t), "bot_touches": int(bot_t),
                            "top_snapbacks": int(top_s), "bot_snapbacks": int(bot_s),
                            "base_touches": int(b_t),
                            "proj_touches": int(p_t), "proj_snapbacks": int(p_s),
                            "last_top_snap_ago": (int(n - 1 - top_snap) if top_snap is not None else None),
                            "last_bot_snap_ago": (int(n - 1 - bot_snap) if bot_snap is not None else None),
                            "broken_side": broken_side,
                            "bars_since_break": (int(n - 1 - broken_i) if broken_i is not None else None),
                            "pos_pct": (round(float(pos), 1) if pos is not None else None),
                            "age_bars": int(n - 1 - max(p1["i"], best["i"])),
                        })
        # quality = how well BOTH rails have been respected; the projected
        # rail's tests weigh extra (they are the 2+1 forecast coming true)
        def _score(q):
            return (q["base_touches"] + 1.5 * q["proj_touches"]
                    + 0.5 * (q["top_snapbacks"] + q["bot_snapbacks"]))
        out.sort(key=lambda q: (q["broken_side"] is not None, -_score(q),
                                q["width_atr"], q["age_bars"]))
        kept: List[dict] = []
        counts: Dict[str, int] = {}

        def _dup(q):
            return any(k["dir"] == q["dir"]
                       and abs(k["top_today"] - q["top_today"]) <= 0.5 * atr
                       and abs(k["bot_today"] - q["bot_today"]) <= 0.5 * atr
                       for k in kept)

        for q in out:
            if q["broken_side"] is not None:
                continue
            # dedup: same direction + both rails within 0.5 ATR of a kept one
            if _dup(q):
                continue
            if counts.get(q["dir"], 0) >= MAX_PER_DIR:
                continue
            counts[q["dir"]] = counts.get(q["dir"], 0) + 1
            kept.append(q)
        # the score cap must never evict the rails that GOVERN price today:
        # re-add the alive channels whose nearest rail sits closest to the
        # current close (the inner lid/floor the MDB lesson trades against)
        px = float(c[-1])
        near = sorted([q for q in out if q["broken_side"] is None],
                      key=lambda q: min(abs(q["top_today"] - px),
                                        abs(q["bot_today"] - px)))
        near_n: Dict[str, int] = {}
        for q in near:
            k = q["dir"]
            if near_n.get(k, 0) >= 3:
                continue
            near_n[k] = near_n.get(k, 0) + 1
            if q not in kept and not _dup(q):
                kept.append(q)
        # retain freshly-broken channels (<=3 own-frame bars): the break IS the
        # event (SEDG fake-break lesson) - the cap must not silence it
        fb: Dict[str, int] = {}
        for q in sorted([q for q in out if q["broken_side"] is not None
                         and q["bars_since_break"] is not None
                         and q["bars_since_break"] <= SNAPBACK_BARS],
                        key=lambda q: (q["bars_since_break"], -_score(q))):
            key = q["dir"] + q["broken_side"]
            if fb.get(key, 0) >= 1:
                continue
            dup = any(k["dir"] == q["dir"] and k["broken_side"] == q["broken_side"]
                      and abs(k["top_today"] - q["top_today"]) <= 0.5 * atr
                      for k in kept)
            if dup:
                continue
            fb[key] = 1
            kept.append(q)
        # hard json contract: no non-finite float may ever leave this module
        return [q for q in kept
                if all(np.isfinite(v) for v in q.values() if isinstance(v, float))]
    except Exception:
        return []


def _contains(q: dict, entry: float) -> bool:
    slack = CONTAIN_SLACK * q["frame_atr"]
    return (q["bot_next"] - slack) <= entry <= (q["top_next"] + slack)


def _q_score(q: dict) -> float:
    return (q["base_touches"] + 1.5 * q["proj_touches"]
            + 0.5 * (q["top_snapbacks"] + q["bot_snapbacks"]))


def analyze_channels(df: pd.DataFrame, entry: Any, direction: str = "long",
                     asof: Optional[str] = None) -> Dict[str, Any]:
    """Governing-channel features for a planned entry. {} when unusable;
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
        chd = channels(fr, timeframe="D")
        chw = channels(fr, timeframe="W")
        if not chd and not chw:
            return {}
        atr = _atr_abs(fr.tail(300))
        if atr <= 0:
            return {}
        alive_d = [q for q in chd if q["broken_side"] is None and _contains(q, entry)]
        alive_w = [q for q in chw if q["broken_side"] is None and _contains(q, entry)]
        # the trader draws the channel FOR the trade: a long reads the up
        # channel, a short the down channel (牛市做長倉 熊市做短倉); the
        # counter-trend structure only governs when it is the only one, and
        # ties go to the YOUNGEST structure (micro-adjustment: the newest
        # valid drawing describes the current regime)
        want = "down" if direction == "short" else "up"

        def _pick(cands):
            if not cands:
                return None
            pref = [q for q in cands if q["dir"] == want]
            pool = pref or cands
            return max(pool, key=lambda q: (_q_score(q), -q["age_bars"]))

        gov = _pick(alive_d)
        wk = _pick(alive_w)
        # higher timeframe dominates ONLY when it describes the same
        # structure - BOTH rails must coincide, else a tall weekly channel
        # sharing one rail would flip a near-top read into near-bottom
        if gov is not None and wk is not None:
            if (abs(wk["top_next"] - gov["top_next"]) <= 0.5 * atr
                    and abs(wk["bot_next"] - gov["bot_next"]) <= 0.5 * atr):
                gov = wk
        elif gov is None:
            gov = wk

        flags: List[str] = []
        out: Dict[str, Any] = {}
        if gov is not None:
            # rails come from the ENSEMBLE of alive containing channels: the
            # first lid above the entry and the first floor below it (the
            # MDB lesson - lock profit at the INNER line, the nearest rail
            # governs the trade even when a wider drawing scores higher).
            # Fields are STRICT (a target must be beyond the entry); the
            # at-the-rail flags tolerate the containment slack.
            pool = alive_d + alive_w
            slack = CONTAIN_SLACK * atr
            g_top, g_bot = float(gov["top_next"]), float(gov["bot_next"])
            tops = [float(q["top_next"]) for q in pool if q["top_next"] > entry]
            bots = [float(q["bot_next"]) for q in pool if q["bot_next"] < entry]
            top = min(tops) if tops else g_top
            bot = max(bots) if bots else g_bot
            lid_near = min((float(q["top_next"]) - entry for q in pool
                            if q["top_next"] >= entry - slack), default=None)
            flr_near = min((entry - float(q["bot_next"]) for q in pool
                            if q["bot_next"] <= entry + slack), default=None)
            g_h = g_top - g_bot
            pos = (entry - g_bot) / g_h * 100.0 if g_h > 0 else None
            if pos is not None and not np.isfinite(pos):
                # NaN would slip through min/max clamping as a silent 125.0
                pos = None
            out.update({
                "ch_dir": gov["dir"], "ch_tf": gov["timeframe"],
                "ch_base_kind": gov["base_kind"],
                "ch_top_at": round(top, 2),
                "ch_top_dist_atr": round((top - entry) / atr, 2),
                "ch_top_headroom_pct": round((top - entry) / entry * 100.0, 2),
                "ch_bot_at": round(bot, 2),
                "ch_bot_dist_atr": round((entry - bot) / atr, 2),
                "ch_width_atr": gov["width_atr"],
                "ch_proj_touches": int(gov["proj_touches"]),
                "ch_touches": int(gov["base_touches"] + gov["proj_touches"]),
                "ch_anchors": [str(a) for a in gov["base_anchors"]],
                "ch_proj_anchor": str(gov["proj_anchor"]),
                # the GOVERNING channel's own parallel rails at the last bar +
                # $/trading-day slope - the drawable geometry. ch_top_at /
                # ch_bot_at above are ensemble trade levels (nearest lid/floor
                # across ALL alive channels, possibly different slopes) and
                # must never be used to draw the channel.
                "ch_gov_top_now": float(gov["top_today_raw"]),
                "ch_gov_bot_now": float(gov["bot_today_raw"]),
                "ch_gov_slope_d": float(gov["slope_abs"]) / (5.0 if gov["timeframe"] == "W" else 1.0),
            })
            if pos is not None:
                out["ch_pos_pct"] = round(max(-25.0, min(125.0, pos)), 1)
            near_top = (lid_near is not None and lid_near <= NEAR_ATR * atr) or (
                pos is not None and pos >= 100.0 - NEAR_POS)
            near_bot = (flr_near is not None and flr_near <= NEAR_ATR * atr) or (
                pos is not None and pos <= NEAR_POS)
            # both can be true at once (a daily lid overhead while price sits
            # on a weekly floor, or a narrow channel) - report both facts,
            # never suppress a warning because a comfort exists
            if near_top:
                flags.append("near_top")        # long: take-profit zone, no edge
            if near_bot:
                flags.append("near_bottom")     # long edge / short cover zone
            if gov["proj_touches"] < FRESH_PROJ_MAX:
                flags.append("fresh_projection")   # 2+1 first-touch reliability
            if gov["proj_touches"] >= WORN_TOUCHES:
                flags.append("proj_worn")
            if gov["steep"]:
                flags.append("steep")
            if gov["timeframe"] != "D":
                flags.append("higher_tf")
            if (direction != "short" and gov["dir"] == "down") or (
                    direction == "short" and gov["dir"] == "up"):
                flags.append("counter_trend")
            mult = 5 if gov["timeframe"] == "W" else 1
            if gov["last_top_snap_ago"] is not None \
                    and gov["last_top_snap_ago"] * mult <= SHAKEOUT_BARS:
                flags.append("top_overshoot_recent")   # 180-degree fuel at the lid
            if gov["last_bot_snap_ago"] is not None \
                    and gov["last_bot_snap_ago"] * mult <= SHAKEOUT_BARS:
                flags.append("bot_overshoot_recent")
            # 2333.HK reasonableness: does the top land in horizontal resistance?
            try:
                zs = SR.zones(fr, timeframe="D") + SR.zones(fr, timeframe="W")
                for z in zs:
                    if z.get("lo") is None or z.get("hi") is None:
                        continue
                    if z["lo"] - SR_CONFL_ATR * atr <= top <= z["hi"] + SR_CONFL_ATR * atr:
                        flags.append("top_sr_confluence")
                        break
            except Exception:
                pass
        # fresh channel breaks are events even when no alive channel contains
        for q in chd + chw:
            if q["broken_side"] is None or q["bars_since_break"] is None:
                continue
            mult = 5 if q["timeframe"] == "W" else 1
            if q["bars_since_break"] * mult > 3:
                continue
            tf_tag = "" if q["timeframe"] == "D" else "_" + str(q["timeframe"])
            if q["broken_side"] == "top":
                flags.append("fresh_ch_break_up" + tf_tag)
                if q["dir"] == "down":
                    # SEDG lesson: breakout OF a down channel = squeeze bait
                    flags.append("fake_break_watch")
            else:
                flags.append("fresh_ch_break_down" + tf_tag)
        if not out and not flags:
            return {}
        out["ch_flags"] = sorted(set(str(f) for f in flags))
        if any(isinstance(v, float) and not np.isfinite(v) for v in out.values()):
            return {}
        return out
    except Exception as exc:                              # never kill a scan
        return {"ch_error": str(exc)[:120]}
