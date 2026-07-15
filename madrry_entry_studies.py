"""
Phase-4 entry-point studies (1 = breakout vs pullback, 5 = stop quality).

The Phase-3 diagnostic proved the SCORE is fine and the GEOMETRY leaks the edge (60% of
structural winners only reach +1R; 62% touch the stop). These studies test whether a
different ENTRY (pullback/limit into the 9-21 EMA) or a different STOP (ATR-based) recovers
it — simulating every variant on the IDENTICAL signal cohort.

Reviewer guardrails baked in:
  * ADVERSE-SELECTION: compare PER-SIGNAL expectancy over the SAME cohort. Every variant is
    simulated on every signal; a non-fill counts as 0R (no trade), and fill rates are
    reported SEPARATELY. (A limit fills preferentially on names that weaken and misses
    runaway winners — per-fill stats are biased, per-signal ones are not.)
  * LEAD WITH EXPECTANCY (mean R) + TAIL CAPTURE (share reaching +3R via r_max), NOT win%.
  * STOP-WIDTH changes the meaning of R: 1R is defined as a FIXED risk budget, so mean r_final
    is already dollar-normalized across stop rules (position ∝ 1/stop-distance).
  * WALK-FORWARD: a rule is a candidate FILTER only if it beats the baseline with the SAME
    SIGN pre-2026 AND in 2026+live; a sign flip -> ship as an informational flag only.
  * SHORTS: no verdicts from backtest (survivorship most acute there); coil only here.

Run: python3 madrry_entry_studies.py
"""
from __future__ import annotations
import json, os, sqlite3
from collections import defaultdict

import numpy as np

import madrry_replay as R
import madrry_ledger as L

WS = os.path.dirname(os.path.abspath(__file__))
BT_DB = os.path.join(WS, "madrry_ledger_backtest.db")
LIVE_DB = os.path.join(WS, "madrry_ledger.db")
OUT = os.path.join(WS, "entry_studies.json")


def _load_coil(db):
    if not os.path.exists(db):
        return []
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    out = []
    for r in c.execute("SELECT signal_id, ticker, section, first_seen_date, entry, stop, features "
                       "FROM signals WHERE section='coil'"):
        d = dict(r)
        try: d["feat"] = json.loads(d.get("features") or "{}")
        except Exception: d["feat"] = {}
        if d["feat"].get("is_etf"):
            continue    # ETF coil rows (2026-07-15) stay out of the entry-engine cohorts
        out.append(d)
    c.close()
    return out


def _per_signal_R(sim):
    """Per-signal realized R: 0 for a non-fill (didn't trade), r_final if resolved/expired,
    None if still OPEN/immature or NODATA (excluded from the cohort)."""
    st = sim["status"]
    if st == "EXPIRED_UNTRIGGERED":
        return 0.0, False
    if st in ("RESOLVED", "EXPIRED"):
        return sim["r_final"], True
    return None, False   # OPEN / NODATA -> exclude


def _agg(recs):
    """recs: list of dicts with keys r (per-signal R or None), filled, win1r, r_max."""
    incl = [x for x in recs if x["r"] is not None]
    filled = [x for x in incl if x["filled"]]
    if not incl:
        return None
    rs = [x["r"] for x in incl]
    return {
        "n_cohort": len(incl),
        "fill_rate": round(100 * len(filled) / len(incl), 1),
        "expectancy_per_signal_R": round(float(np.mean(rs)), 4),   # non-fills counted as 0R
        "expectancy_per_fill_R": round(float(np.mean([x["r"] for x in filled])), 4) if filled else None,
        "win_1r_rate_of_filled": round(100 * np.mean([x["win1r"] for x in filled]), 1) if filled else None,
        "tail_share_3R_of_filled": round(100 * np.mean([1.0 if (x["r_max"] or 0) >= 3 else 0.0 for x in filled]), 1) if filled else None,
    }


def _era(date_str):
    return "pre_2026" if date_str < "2026-01-01" else "y2026plus"


def run(db, tag):
    sigs = _load_coil(db)
    tickers = sorted({s["ticker"] for s in sigs})
    # convert each ticker's frame to the bars-list ONCE (not per signal) — the hot path
    bars_by = {}
    for t in tickers:
        df = R.load_bars(t)
        if df is not None and len(df) >= R.MIN_BARS:
            bars_by[t] = L._df_to_bars(df)
    # variant -> era -> list of per-signal records
    data = defaultdict(lambda: defaultdict(list))
    ATR_MULT = 1.5
    for s in sigs:
        bars = bars_by.get(s["ticker"])
        if bars is None:
            continue
        pos = L._pick_pos(bars, s["first_seen_date"])
        if pos < 0 or pos >= len(bars):
            continue
        feat = s["feat"]
        entry, stop = s["entry"], s["stop"]
        pb_entry, pb_stop = feat.get("pb_entry"), feat.get("pb_stop")
        adr = feat.get("adr")
        era = _era(s["first_seen_date"])
        # Require all three variants to EXIST for this signal, else skip it (identical cohort).
        if pb_entry is None or pb_stop is None or not adr or adr <= 0:
            continue
        atr_stop = round(entry * (1 - ATR_MULT * adr / 100.0), 4)
        if atr_stop >= entry:
            continue
        variants = {
            "breakout": L.simulate_trade(bars, pos, entry, stop, "long", "stop"),
            "pullback": L.simulate_trade(bars, pos, pb_entry, pb_stop, "long", "limit"),
            "breakout_atr_stop": L.simulate_trade(bars, pos, entry, atr_stop, "long", "stop"),
        }
        # INTERSECTION cohort: include the signal only if EVERY variant has matured
        # (a variant still OPEN/immature would bias the comparison). All-or-nothing.
        outs = {v: _per_signal_R(sim) for v, sim in variants.items()}
        if any(r is None for (r, _filled) in outs.values()):
            continue
        for vname, sim in variants.items():
            r, filled = outs[vname]
            rec = {"r": r, "filled": filled, "win1r": sim.get("win_1r") or 0, "r_max": sim.get("r_max")}
            data[vname]["all"].append(rec)
            data[vname][era].append(rec)
    result = {}
    for vname, eras in data.items():
        result[vname] = {e: _agg(recs) for e, recs in eras.items() if _agg(recs)}
    return result


def _sign_consistent(base, cand, key="expectancy_per_signal_R"):
    """True if cand beats base with the SAME sign in BOTH pre_2026 and y2026plus."""
    def delta(era):
        b = (base.get(era) or {}).get(key)
        c = (cand.get(era) or {}).get(key)
        if b is None or c is None:
            return None
        return c - b
    d1, d2 = delta("pre_2026"), delta("y2026plus")
    if d1 is None or d2 is None:
        return None
    return (d1 > 0 and d2 > 0)


def _bucket(edges, v):
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return f"{edges[i]}-{edges[i + 1]}"
    return "other"


def study2_chase(db):
    """Study 2 — 'don't chase' cutoff. Bucket by extension at the signal (coil: min_dist to
    the nearest EMA; NH52: ext9 above the 9-EMA) and report per-signal expectancy under BOTH
    stop regimes. A chase entry with a TIGHT stop is doubly bad, so the cutoff typically sits
    at a different extension under the ATR stop — that interaction is the whole point."""
    if not os.path.exists(db):
        return {}
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    bcache = {}
    def get_bars(tk):
        if tk not in bcache:
            df = R.load_bars(tk)
            bcache[tk] = L._df_to_bars(df) if (df is not None and len(df) >= R.MIN_BARS) else None
        return bcache[tk]
    specs = [("coil", "min_dist", [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 1.0]),
             ("nh52", "ext9", [-1e9, 0, 3, 6, 10, 15, 25, 1e9])]
    result = {}
    for section, fkey, edges in specs:
        buckets = defaultdict(lambda: {"struct": [], "atr": [], "atr_pre": [], "atr_post": []})
        for r in c.execute("SELECT ticker,first_seen_date,entry,stop,features FROM signals WHERE section=?", (section,)):
            d = dict(r)
            try: feat = json.loads(d["features"] or "{}")
            except Exception: feat = {}
            if feat.get("is_etf"):
                continue    # ETF rows (2026-07-15) stay out of the stock studies
            ext, adr, entry, stop = feat.get(fkey), feat.get("adr"), d["entry"], d["stop"]
            if ext is None or adr is None or adr <= 0 or entry is None or stop is None:
                continue
            bars = get_bars(d["ticker"])
            if bars is None:
                continue
            pos = L._pick_pos(bars, d["first_seen_date"])
            if pos < 0 or pos >= len(bars):
                continue
            atr_stop = round(entry * (1 - 1.5 * adr / 100.0), 4)
            if atr_stop >= entry:
                continue
            r1, _ = _per_signal_R(L.simulate_trade(bars, pos, entry, stop, "long", "stop"))
            r2, _ = _per_signal_R(L.simulate_trade(bars, pos, entry, atr_stop, "long", "stop"))
            if r1 is None or r2 is None:
                continue
            b = _bucket(edges, ext)
            buckets[b]["struct"].append(r1); buckets[b]["atr"].append(r2)
            buckets[b]["atr_pre" if _era(d["first_seen_date"]) == "pre_2026" else "atr_post"].append(r2)
        def mean(x): return round(float(np.mean(x)), 3) if x else None
        result[section] = {}
        for b, v in buckets.items():
            pre, post = mean(v["atr_pre"]), mean(v["atr_post"])
            # a bucket only earns a RULE if the ATR-stop sign holds in BOTH eras (else a flag)
            same_sign = (pre is not None and post is not None and
                         ((pre > 0 and post > 0) or (pre < 0 and post < 0)))
            result[section][b] = {"n": len(v["struct"]), "exp_struct": mean(v["struct"]),
                                  "exp_atr": mean(v["atr"]), "exp_atr_pre": pre, "exp_atr_post": post,
                                  "rule_eligible_both_eras": same_sign}
    c.close()
    return result


def _coil_iter(db):
    """Yield (bars, pos, entry, atr_stop, adr, era) for every labelable coil signal under the
    ATR geometry — the shared cohort for the policy studies 3 & 4."""
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    bcache = {}
    def get_bars(tk):
        if tk not in bcache:
            df = R.load_bars(tk)
            bcache[tk] = L._df_to_bars(df) if (df is not None and len(df) >= R.MIN_BARS) else None
        return bcache[tk]
    for r in c.execute("SELECT ticker,first_seen_date,entry,features FROM signals WHERE section='coil'"):
        d = dict(r)
        try: feat = json.loads(d["features"] or "{}")
        except Exception: feat = {}
        if feat.get("is_etf"):
            continue    # ETF rows (2026-07-15) stay out of the stock studies
        adr, entry = feat.get("adr"), d["entry"]
        if adr is None or adr <= 0 or entry is None:
            continue
        bars = get_bars(d["ticker"])
        if bars is None:
            continue
        pos = L._pick_pos(bars, d["first_seen_date"])
        if pos < 0 or pos >= len(bars):
            continue
        atr_stop = round(entry * (1 - 1.5 * adr / 100.0), 4)
        if atr_stop >= entry:
            continue
        yield bars, pos, entry, atr_stop, adr, _era(d["first_seen_date"])
    c.close()


def study3_validity_window(db):
    """Study 3 as a POLICY (not 'days-to-trigger among winners', which conditions on the
    outcome): 'cancel unfilled orders after N days'. Re-simulate the FULL coil cohort under
    the ATR stop for each N, per-signal expectancy (non-fills=0R), walk-forward. Best N is the
    validity window; only adopt if it beats N=3 with the same sign in both eras."""
    Ns = [1, 2, 3, 4, 5, 7, 10]
    acc = {n: {"all": [], "pre_2026": [], "y2026plus": []} for n in Ns}
    for bars, pos, entry, atr_stop, adr, era in _coil_iter(db):
        for n in Ns:
            r, _ = _per_signal_R(L.simulate_trade(bars, pos, entry, atr_stop, "long", "stop", trigger_window=n))
            if r is not None:
                acc[n]["all"].append(r); acc[n][era].append(r)
    def mean(x): return round(float(np.mean(x)), 4) if x else None
    return {n: {e: mean(v) for e, v in eras.items()} for n, eras in acc.items()}


def study4_gap_filter(db):
    """Study 4 as a POLICY: 'skip fills where the trigger-day OPEN gapped > entry + 1×ADR'
    (the SYNA gap-up-chase case). Re-simulate the FULL coil cohort under the ATR stop WITH vs
    WITHOUT the filter, per-signal expectancy, walk-forward."""
    acc = {"nofilter": {"all": [], "pre_2026": [], "y2026plus": []},
           "gap_filtered": {"all": [], "pre_2026": [], "y2026plus": []},
           "gapped_only": {"all": [], "pre_2026": [], "y2026plus": []}}
    for bars, pos, entry, atr_stop, adr, era in _coil_iter(db):
        sim = L.simulate_trade(bars, pos, entry, atr_stop, "long", "stop")
        r, _ = _per_signal_R(sim)
        if r is None:
            continue
        gapped = False
        if sim["triggered"] and sim.get("days_to_trigger"):
            tp = pos + sim["days_to_trigger"]
            if 0 <= tp < len(bars):
                gapped = bars[tp]["o"] > entry * (1 + adr / 100.0)   # >1-ADR gap-up chase
        r_filtered = 0.0 if gapped else r     # policy: skip the gapped fill
        for tag, val in (("nofilter", r), ("gap_filtered", r_filtered)):
            acc[tag]["all"].append(val); acc[tag][era].append(val)
        if gapped:
            acc["gapped_only"]["all"].append(r); acc["gapped_only"][era].append(r)
    def mean(x): return round(float(np.mean(x)), 4) if x else None
    def n(x): return len(x)
    out = {tag: {e: mean(v) for e, v in eras.items()} for tag, eras in acc.items()}
    out["gapped_only"]["n"] = n(acc["gapped_only"]["all"])
    return out


def main():
    report = {"note": "expectancy_per_signal_R counts non-fills as 0R (the fair per-signal metric); "
                      "fill_rate reported separately; lead with expectancy + tail, not win%."}
    for tag, db in (("backtest", BT_DB), ("live_ledger", LIVE_DB)):
        res = run(db, tag)
        report[tag] = res
    report["study2_chase_backtest"] = study2_chase(BT_DB)
    report["study3_validity_window"] = study3_validity_window(BT_DB)
    report["study4_gap_filter"] = study4_gap_filter(BT_DB)
    report["walk_forward_verdicts"] = {}
    bt = report.get("backtest", {})
    base = bt.get("breakout", {})
    for cand_name in ("pullback", "breakout_atr_stop"):
        cand = bt.get(cand_name)
        if cand:
            sc = _sign_consistent(base, cand)
            report["walk_forward_verdicts"][cand_name] = {
                "beats_breakout_same_sign_both_eras": sc,
                "verdict": ("ADOPT as filter" if sc else "informational flag only (sign not consistent)"),
            }
    with open(OUT + ".tmp", "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(OUT + ".tmp", OUT)

    # console summary
    print("=" * 84)
    print("PHASE-4 ENTRY STUDIES — per-SIGNAL expectancy (non-fills=0R), lead with R + tail")
    print("=" * 84)
    for tag in ("backtest", "live_ledger"):
        print(f"\n[{tag}] coil:")
        for v, eras in report[tag].items():
            a = eras.get("all")
            if not a: continue
            print(f"  {v:20s} n={a['n_cohort']:5d} fill={a['fill_rate']}% "
                  f"exp/signal={a['expectancy_per_signal_R']:+.3f}R exp/fill={a['expectancy_per_fill_R']} "
                  f"win1r={a['win_1r_rate_of_filled']}% tail3R={a['tail_share_3R_of_filled']}%")
    print("\nWALK-FORWARD VERDICTS (require same-sign improvement pre-2026 AND 2026+):")
    for k, v in report["walk_forward_verdicts"].items():
        print(f"  {k:20s} -> {v['verdict']}")
    print("\nSTUDY 2 — chase cutoff (per-signal exp R by extension × stop; ATR era-split for rule-eligibility):")
    for section, buckets in report["study2_chase_backtest"].items():
        print(f"  [{section}] bucket -> struct / ATR (ATR pre|post, rule-eligible both eras?):")
        for b in sorted(buckets, key=lambda x: (x == "other", x)):
            v = buckets[b]
            print(f"    {b:16s} n={v['n']:5d} struct {str(v['exp_struct']):>7} ATR {str(v['exp_atr']):>7} "
                  f"(pre {str(v['exp_atr_pre']):>7}|post {str(v['exp_atr_post']):>7}) rule={v['rule_eligible_both_eras']}")
    print("\nSTUDY 3 — validity window POLICY (cancel unfilled after N days; ATR stop; per-signal exp R):")
    for n in sorted(report["study3_validity_window"]):
        v = report["study3_validity_window"][n]
        print(f"    N={n:2d} days: all {v['all']:+.4f}  pre {v['pre_2026']:+.4f}  2026+ {v['y2026plus']:+.4f}")
    print("\nSTUDY 4 — gap filter POLICY (skip trigger-open > entry+1×ADR; ATR stop):")
    g = report["study4_gap_filter"]
    print(f"    no-filter   : all {g['nofilter']['all']:+.4f}  pre {g['nofilter']['pre_2026']:+.4f}  2026+ {g['nofilter']['y2026plus']:+.4f}")
    print(f"    gap-filtered: all {g['gap_filtered']['all']:+.4f}  pre {g['gap_filtered']['pre_2026']:+.4f}  2026+ {g['gap_filtered']['y2026plus']:+.4f}")
    print(f"    gapped-only : all {g['gapped_only']['all']}  (n={g['gapped_only']['n']}) -- the trades the filter removes")
    _append_report_md(report)
    print(f"\nwrote {OUT}")
    return 0


def _append_report_md(report):
    """Gap 1 — write the Phase-4 findings into BACKTEST_REPORT.md (idempotent: replaces the
    section between the markers)."""
    md = os.path.join(WS, "BACKTEST_REPORT.md")
    START, END = "<!-- PHASE4:START -->", "<!-- PHASE4:END -->"
    L4 = [START, "", "## Phase 4 — entry-point studies (1 breakout-vs-pullback, 5 stop-quality, 2 chase-cutoff)", "",
          "Per-SIGNAL expectancy on the INTERSECTION cohort (every variant matured); non-fills=0R; "
          "fill rate separate; lead with expectancy + tail; walk-forward pre-2026 vs 2026+. "
          "Survivorship-biased -> RELATIVE ranking only; shorts get NO verdict here.", ""]
    bt = report.get("backtest", {})
    if bt.get("breakout"):
        L4 += ["**Studies 1 + 5 — entry style × stop (coil, intersection cohort):**", "",
               "| variant | n | fill% | exp/signal R | win_1r% | tail3R% |", "|---|--:|--:|--:|--:|--:|"]
        for v in ("breakout", "pullback", "breakout_atr_stop"):
            a = (bt.get(v) or {}).get("all")
            if a:
                L4.append(f"| {v} | {a['n_cohort']} | {a['fill_rate']} | {a['expectancy_per_signal_R']} "
                          f"| {a['win_1r_rate_of_filled']} | {a['tail_share_3R_of_filled']} |")
        L4 += ["", "**Walk-forward verdicts:** " +
               "; ".join(f"{k} → {v['verdict']}" for k, v in report.get("walk_forward_verdicts", {}).items()), ""]
    for section, buckets in report.get("study2_chase_backtest", {}).items():
        if not buckets:
            continue
        L4 += [f"**Study 2 — chase cutoff, `{section}` (expectancy R by extension × stop regime):**", "",
               "| extension bucket | n | struct-stop R | ATR-stop R |", "|---|--:|--:|--:|"]
        for b in sorted(buckets, key=lambda x: (x == "other", x)):
            v = buckets[b]
            L4.append(f"| {b} | {v['n']} | {v['exp_struct']} | {v['exp_atr']} |")
        L4.append("")
    s3 = report.get("study3_validity_window", {})
    if s3:
        L4 += ["**Study 3 — validity-window POLICY** (cancel unfilled after N days, ATR stop; per-signal exp R):", "",
               "| N days | all | pre-2026 | 2026+ |", "|--:|--:|--:|--:|"]
        for n in sorted(s3):
            v = s3[n]
            L4.append(f"| {n} | {v['all']} | {v['pre_2026']} | {v['y2026plus']} |")
        L4.append("")
    g = report.get("study4_gap_filter", {})
    if g:
        L4 += ["**Study 4 — gap-filter POLICY** (skip trigger-open > entry+1×ADR, ATR stop):", "",
               "| policy | all | pre-2026 | 2026+ |", "|---|--:|--:|--:|",
               f"| no filter | {g['nofilter']['all']} | {g['nofilter']['pre_2026']} | {g['nofilter']['y2026plus']} |",
               f"| gap-filtered | {g['gap_filtered']['all']} | {g['gap_filtered']['pre_2026']} | {g['gap_filtered']['y2026plus']} |",
               f"| (gapped trades removed: n={g['gapped_only']['n']}, their avg R = {g['gapped_only']['all']}) | | | |", ""]
    L4 += ["*Geometry decision (1.5×ADR stop primary, pullback secondary) is ratified as DIRECTION; "
           "the printed-stop switch is a user decision after studies 2-4 — it moves IBKR draft sizing "
           "and the Tracking win/loss definition. Live OOS labeling of both geometries started 2026-07-03.*", "", END]
    block = "\n".join(L4)
    try:
        txt = open(md).read() if os.path.exists(md) else ""
    except Exception:
        txt = ""
    import re
    if START in txt and END in txt:
        txt = re.sub(re.escape(START) + r".*?" + re.escape(END), block, txt, flags=re.S)
    else:
        txt = (txt + "\n\n" + block) if txt else block
    with open(md + ".tmp", "w") as fh:
        fh.write(txt)
    os.replace(md + ".tmp", md)


if __name__ == "__main__":
    raise SystemExit(main())
