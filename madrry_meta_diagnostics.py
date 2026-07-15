"""
Phase-3 task 1: META score diagnostics — is the flat decile a SCORE problem or a
GEOMETRY problem?  (the reviewer's guardrail #2)

For each score (raw v4 probability, legacy component score, and the displayed v4
percentile on the live ledger) it computes, per section and per regime, the decile
win-rate table AND the point-biserial correlation against EVERY available label:
  - win_1r / win_2r  : trade-sim outcome FROM the printed breakout entry (geometry-laden)
  - label_2adr       : the +2ADR-close barrier from pick-close = v4's OWN training target
  - legacy_struct    : the 40-bar structural label (fresh 52w high) = the basis of the
                       historical 44pp / corr-0.30 META evidence (meta_benchmark.json)
  - r_final          : mean realized R (expectancy — lead with THIS, not win%)

Reading: if a score has real lift on label_2adr / legacy_struct but NOT on win_1r, the
SCORE is fine and the ENTRY/STOP GEOMETRY is the problem (Phase 4) — do not "fix" the
score by re-fitting. Runs on BOTH the live ledger and the survivorship-biased backtest;
the backtest is for RELATIVE comparison only.

Run: python3 madrry_meta_diagnostics.py
"""
from __future__ import annotations
import json, os, sqlite3
from collections import defaultdict

import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
LIVE_DB = os.path.join(WS, "madrry_ledger.db")
BT_DB = os.path.join(WS, "madrry_ledger_backtest.db")
OUT_JSON = os.path.join(WS, "meta_diagnostics.json")

LABELS = ["win_1r", "win_2r", "label_2adr", "legacy_struct_win", "r_final"]


def _regime_map():
    try:
        from build_backtest_report import spy_regime
        return spy_regime()
    except Exception:
        return {}


def _rows(db, reg):
    if not os.path.exists(db):
        return []
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    out = []
    for r in c.execute("SELECT s.section, s.first_seen_date, s.features, o.* "
                       "FROM signals s JOIN outcomes o USING(signal_id) "
                       "WHERE o.status IN ('RESOLVED','EXPIRED')"):
        d = dict(r)
        try: f = json.loads(d.get("features") or "{}")
        except Exception: f = {}
        if f.get("is_etf"):
            continue    # ETF coil rows (2026-07-15) stay out of stock META diagnostics
        d["legacy_struct_win"] = (1 if d.get("legacy_struct") == "win"
                                  else (0 if d.get("legacy_struct") in ("loss", "expired") else None))
        d["v4_prob_raw"] = f.get("v4_prob_raw")
        d["legacy_score_raw"] = f.get("legacy_score_raw")
        d["v4_pctile_disp"] = f.get("meta_score")          # live displayed score (v4 percentile)
        d["regime"] = reg.get(d["first_seen_date"], "unknown")
        out.append(d)
    c.close()
    return out


def _pointbiserial(scores, labels):
    s = np.array(scores, float); y = np.array(labels, float)
    m = ~(np.isnan(s) | np.isnan(y))
    s, y = s[m], y[m]
    if len(s) < 30 or s.std() == 0 or y.std() == 0 or len(set(y)) < 2:
        return None, len(s)
    return round(float(np.corrcoef(s, y)[0, 1]), 3), len(s)


def _decile(rows, score_key, label_key):
    pts = [(r[score_key], r[label_key]) for r in rows
           if r.get(score_key) is not None and r.get(label_key) is not None]
    if len(pts) < 50:
        return None
    vals = np.array([p[0] for p in pts], float)
    order = np.argsort(vals)
    q = np.array_split(order, 10)
    top = [pts[j][1] for j in q[-1]]
    bot = [pts[j][1] for j in q[0]]
    def rate(g): return round(100 * np.mean([x for x in g if x is not None]), 1) if g else None
    return {"n": len(pts), "bottom_decile": rate(bot), "top_decile": rate(top),
            "spread_pp": (round(rate(top) - rate(bot), 1) if rate(top) is not None and rate(bot) is not None else None)}


def analyse(name, rows):
    res = {"n_resolved": len(rows), "by_score": {}}
    scores = [("v4_prob_raw", "raw v4 probability"),
              ("legacy_score_raw", "legacy component score"),
              ("v4_pctile_disp", "displayed v4 percentile")]
    for sk, slabel in scores:
        if not any(r.get(sk) is not None for r in rows):
            continue
        entry = {"label_corr": {}, "decile": {}}
        for lk in LABELS:
            corr, n = _pointbiserial([r.get(sk) for r in rows], [r.get(lk) for r in rows])
            entry["label_corr"][lk] = {"corr": corr, "n": n}
            if lk != "r_final":
                d = _decile(rows, sk, lk)
                if d: entry["decile"][lk] = d
        res["by_score"][sk] = {"label": slabel, **entry}
    return res


def main():
    reg = _regime_map()
    report = {"generated_note": "expectancy (mean R) leads; win% can diverge from R (see bear regime)"}
    for tag, db in (("live_ledger", LIVE_DB), ("backtest", BT_DB)):
        rows = _rows(db, reg)
        report[tag] = {"overall": analyse(tag, rows), "by_section": {}, "by_regime": {}}
        for sec in sorted({r["section"] for r in rows}):
            report[tag]["by_section"][sec] = analyse(sec, [r for r in rows if r["section"] == sec])
        for g in ("bull", "bear"):
            gr = [r for r in rows if r["regime"] == g]
            if gr:
                report[tag]["by_regime"][g] = analyse(g, gr)
    with open(OUT_JSON + ".tmp", "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(OUT_JSON + ".tmp", OUT_JSON)

    # ---- console summary: the KEY question ----
    print("=" * 78)
    print("META DIAGNOSTICS — is the flat decile a SCORE problem or a GEOMETRY problem?")
    print("=" * 78)
    for tag in ("backtest", "live_ledger"):
        block = report[tag]
        coil = block["by_section"].get("coil") or block["overall"]
        print(f"\n[{tag}] coil, n_resolved={coil['n_resolved']}")
        for sk, sv in coil["by_score"].items():
            print(f"  score={sv['label']}:")
            for lk in LABELS:
                lc = sv["label_corr"].get(lk, {})
                dd = sv["decile"].get(lk, {})
                extra = (f"  decile spread {dd.get('spread_pp')}pp "
                         f"({dd.get('bottom_decile')}%->{dd.get('top_decile')}%)" if dd else "")
                print(f"    corr vs {lk:18s} = {str(lc.get('corr')):>6} (n={lc.get('n')}){extra}")
    print(f"\nwrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
