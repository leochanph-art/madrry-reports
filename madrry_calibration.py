"""
Phase-3: score_calibration.json — turn the raw v4 probability into a calibrated win-rate +
expected-R table, keyed on raw v4 prob × regime. PARAMETERIZED BY SPEC VERSION so switching
the ratified trade geometry (stop/window) is a RE-RUN, not a rewrite.

Locked spec (reviewer): the ADOPTED geometry is the 1.5×ADR stop, so the default spec keys on
the ATR labels (win_1r_atr / r_final_atr) over FULLY-ATR-RESOLVED rows only. The tight-stop
spec is retained for comparison / until the user ratifies the switch.

Live ledger is primary; the survivorship-biased backtest FILLS sparse buckets (n_live<20),
flagged `_backtest: true`. Regenerate weekly and from scratch after any promoted v4 refit
(the score's meaning changes). Today the live+v4_prob sample is thin (v4_prob_raw only exists
on post-2026-07-02 snapshots), so most buckets are backtest-filled — expected, not a bug.

Run: python3 madrry_calibration.py [--spec atr_3day|tight_3day]
"""
from __future__ import annotations
import argparse, json, os, sqlite3
from collections import defaultdict

import numpy as np

WS = os.path.dirname(os.path.abspath(__file__))
LIVE_DB = os.path.join(WS, "madrry_ledger.db")
BT_DB = os.path.join(WS, "madrry_ledger_backtest.db")
OUT = os.path.join(WS, "score_calibration.json")

SPECS = {
    "atr_5day": {"win": "win_1r_atr5", "win2": "win_2r_atr5", "r": "r_final_atr5", "status": "status_atr5",
                 "desc": "1.5xADR stop, 5-day trigger window (CANONICAL ratified geometry, coil, from 2026-07-06)"},
    "atr_3day": {"win": "win_1r_atr", "win2": "win_2r_atr", "r": "r_final_atr", "status": "status_atr",
                 "desc": "1.5xADR stop, 3-day trigger window (pre-ratification comparison)"},
    "tight_3day": {"win": "win_1r", "win2": "win_2r", "r": "r_final", "status": "status",
                   "desc": "printed tight MA/PDL stop, 3-day window (legacy; kept for series continuity)"},
}
PROB_EDGES = [0.0, 0.35, 0.45, 0.50, 0.55, 0.60, 0.65, 1.01]
MIN_LIVE_BUCKET = 20


def _regime():
    try:
        from build_backtest_report import spy_regime
        return spy_regime()
    except Exception:
        return {}


def _bucket(p):
    for i in range(len(PROB_EDGES) - 1):
        if PROB_EDGES[i] <= p < PROB_EDGES[i + 1]:
            return f"{PROB_EDGES[i]:.2f}-{PROB_EDGES[i + 1]:.2f}"
    return None


def _rows(db, spec, reg):
    """coil rows with a raw v4 prob AND a fully-resolved outcome under `spec`'s geometry."""
    if not os.path.exists(db):
        return []
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    q = (f"SELECT s.first_seen_date, s.features, o.{spec['win']} AS w1, o.{spec['win2']} AS w2, "
         f"o.{spec['r']} AS rr, o.{spec['status']} AS st "
         f"FROM signals s JOIN outcomes o USING(signal_id) WHERE s.section='coil'")
    out = []
    for r in c.execute(q):
        d = dict(r)
        if d["st"] not in ("RESOLVED", "EXPIRED"):
            continue
        try:
            f = json.loads(d["features"] or "{}")
        except Exception:
            f = {}
        if f.get("is_etf"):
            continue    # ETF coil rows (2026-07-15) stay out of the stock calibration cells
        p = f.get("v4_prob_raw")
        if p is None:
            continue
        out.append({"p": float(p), "w1": d["w1"], "w2": d["w2"], "r": d["rr"],
                    "regime": reg.get(d["first_seen_date"], "unknown")})
    c.close()
    return out


def _agg(rows):
    if not rows:
        return None
    w1 = [x["w1"] for x in rows if x["w1"] is not None]
    w2 = [x["w2"] for x in rows if x["w2"] is not None]
    rr = [x["r"] for x in rows if x["r"] is not None]
    return {"n": len(rows),
            "p_win_1r": round(100 * np.mean(w1), 1) if w1 else None,
            "p_win_2r": round(100 * np.mean(w2), 1) if w2 else None,
            "expected_R": round(float(np.mean(rr)), 3) if rr else None}


def build(spec_version="atr_5day"):
    spec = SPECS[spec_version]
    reg = _regime()
    live = _rows(LIVE_DB, spec, reg)
    bt = _rows(BT_DB, spec, reg)
    regimes = ("bull", "bear", "unknown")
    table = {}
    for rg in regimes:
        for i in range(len(PROB_EDGES) - 1):
            key = f"{PROB_EDGES[i]:.2f}-{PROB_EDGES[i + 1]:.2f}"
            lrows = [x for x in live if x["regime"] == rg and _bucket(x["p"]) == key]
            brows = [x for x in bt if x["regime"] == rg and _bucket(x["p"]) == key]
            live_agg = _agg(lrows)
            if live_agg and live_agg["n"] >= MIN_LIVE_BUCKET:
                cell = {**live_agg, "source": "live"}
            else:
                bt_agg = _agg(brows)
                if bt_agg:
                    cell = {**bt_agg, "source": "backtest", "_backtest": True,
                            "n_live": (live_agg["n"] if live_agg else 0)}
                elif live_agg:
                    cell = {**live_agg, "source": "live_sparse"}
                else:
                    continue
            table[f"{rg}|{key}"] = cell
    # monotonicity check (within each regime, does p_win_1r rise with the prob bucket?)
    mono = {}
    for rg in regimes:
        cells = [(PROB_EDGES[i], table.get(f"{rg}|{PROB_EDGES[i]:.2f}-{PROB_EDGES[i+1]:.2f}"))
                 for i in range(len(PROB_EDGES) - 1)]
        seq = [c["p_win_1r"] for _lo, c in cells if c and c.get("p_win_1r") is not None and c.get("n", 0) >= MIN_LIVE_BUCKET]
        viol = sum(1 for i in range(1, len(seq)) if seq[i] < seq[i - 1] - 1e-9)
        mono[rg] = {"n_buckets_ge20": len(seq), "monotonic_violations": viol}
    payload = {"spec_version": spec_version, "spec": spec, "prob_edges": PROB_EDGES,
               "n_live": len(live), "n_backtest": len(bt),
               "note": "keyed on raw v4 prob x regime; live primary, backtest fills n_live<20 "
                       "(flagged _backtest); expected_R leads (win% and R diverge in bear).",
               "monotonicity": mono, "table": table}
    tmp = OUT + ".tmp"; json.dump(payload, open(tmp, "w"), indent=1); os.replace(tmp, OUT)
    print(f"[calibration] spec={spec_version} n_live={len(live)} n_backtest={len(bt)} "
          f"cells={len(table)} -> {OUT}")
    for rg in regimes:
        print(f"  {rg}: monotonicity {mono[rg]}")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="atr_5day", choices=list(SPECS))
    args = ap.parse_args(argv)
    build(args.spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
