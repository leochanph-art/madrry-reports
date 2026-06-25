"""
MADRRY v4 forward tracker — logs each pick's 45 v4 FEATURES (frozen at pick date)
plus the v4 SCORE, then grades the outcome forward on v4's own training basis:

  WIN  = a forward CLOSE >= entry * (1 + 2*ADR/100)   (a real +2ADR move up)
  LOSS = a forward LOW   <= entry * 0.92              (-8%)
  first event in a 40-bar window; else open/expired.

Also records barrier return + forward 20d/40d returns. This builds a growing
SURVIVORSHIP-FREE, LIVE labeled dataset (v4_tracking.json) — the input the weekly
re-fit needs to know whether v4 actually holds up on real forward picks (not the
biased backtest). Features are computed from the same bars used for grading via
madrry_meta_v4.compute_features (parity-verified to the deployed model).

Universe: Tier-A picks (A+/A/A-) at first appearance, from latest_setups_*.json.
Idempotent: re-grades 'open' records each run; resolved records stay frozen.

Price basis: RAW (auto_adjust=False), matching the LIVE scanner (fetch_histories_batch
also raw). The deployed model was trained on adjusted data10y, so the current live
v4_score carries a small dividend-skew on high-yield names; the FIRST re-fit must
re-train on THIS tracker's raw features so the whole stack becomes raw-on-raw and the
skew is eliminated. Selection note: only Tier-A picks are tracked (a gated subset, NOT
the broad universe v4 was trained on) -> a re-fit on this data yields a Tier-A-conditional
model. MATURITY: do NOT re-fit until enough picks have a FULL 40-bar window resolved
(early short-window data is loss-censored: the -8% stop triggers faster than +2ADR).

Run:  python3 madrry_v4_tracker.py
"""
from __future__ import annotations
import glob, json, os
from collections import Counter
from datetime import date
import numpy as np
import pandas as pd

import madrry_meta_v4 as V4
import madrry_tier_a_tracker as T   # reuse fetch_prices + WORKSPACE

WORKSPACE = T.WORKSPACE
DB_PATH = os.path.join(WORKSPACE, "v4_tracking.json")
WINDOW = 40
LOSS_MULT = 0.92
TARGET_K = 2.0
TIER_A = {"A+", "A", "A-"}


def build_picks():
    """First-appearance Tier-A picks from the dated snapshots."""
    files = sorted(glob.glob(os.path.join(WORKSPACE, "latest_setups_2026-*.json")))
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
            seen[t] = {"ticker": t, "pick_date": dt, "tier": r.get("tier"),
                       "sector": r.get("sector"), "legacy_meta_score": r.get("meta_score")}
    return seen


def grade(df, pos, entry, adr):
    """Forward +2ADR / -8% barrier grade + returns. Returns dict or None.

    Label convention matches the model's training labeler (madrry_features_extract):
      win barrier -> 1 ; loss barrier -> 0 ; window-end (expired) -> sign of the
      window-end return (1 if up else 0) ; still-open (immature) -> None (excluded
      from any re-fit). barrier_return keeps the training convention (loss = -0.08
      flat); realized_return additionally records the gap-aware low for honest
      economics."""
    fwd = df.iloc[pos + 1:pos + 1 + WINDOW]
    if fwd.empty:
        return None
    win_line = entry * (1 + TARGET_K * adr / 100.0)
    loss_line = entry * LOSS_MULT
    outcome, resolve_date, days = "open", None, None
    bret = realized = None
    for k, (_, row) in enumerate(fwd.iterrows()):
        if float(row["Low"]) <= loss_line:                       # loss barrier
            outcome, resolve_date, days = "loss", fwd.index[k], k + 1
            bret = LOSS_MULT - 1.0                                # training-consistent -0.08
            realized = float(row["Low"]) / entry - 1.0           # gap-aware actual
            break
        if float(row["Close"]) >= win_line:                      # win barrier
            outcome, resolve_date, days = "win", fwd.index[k], k + 1
            bret = realized = float(row["Close"]) / entry - 1.0
            break
    if outcome == "open" and len(fwd) >= WINDOW:                 # 40 bars, no touch -> resolved
        outcome = "expired"
        bret = realized = float(fwd["Close"].iloc[-1]) / entry - 1.0
    # label: win=1, loss=0, expired=sign(window-end), open=None (immature -> excluded)
    if outcome == "win":
        label = 1
    elif outcome == "loss":
        label = 0
    elif outcome == "expired":
        label = 1 if bret > 0 else 0
    else:
        label = None
    cl = df["Close"]
    def fret(n): return round(float(cl.iloc[pos + n]) / entry - 1.0, 4) if pos + n < len(cl) else None
    return {
        "outcome": outcome, "label_2adr": label,
        "barrier_return": round(bret, 4) if bret is not None else None,
        "realized_return": round(realized, 4) if realized is not None else None,
        "fwd20_return": fret(20), "fwd40_return": fret(40),
        "win_target": round(win_line, 4), "loss_line": round(loss_line, 4),
        "resolve_date": resolve_date.date().isoformat() if resolve_date is not None else None,
        "days_to_resolve": days, "days_followed": int(len(fwd)),
    }


def main():
    picks = build_picks()
    print(f"Tier-A picks (first appearance): {len(picks)}")
    prices = T.fetch_prices(set(picks))
    print(f"  prices for {len(prices)}/{len(picks)}")

    db = {}
    if os.path.exists(DB_PATH):
        try:
            db = json.load(open(DB_PATH)).get("records", {})
        except Exception:
            db = {}

    n_new = n_regrade = 0
    for t, p in picks.items():
        key = f"{t}|{p['pick_date']}"
        existing = db.get(key)
        # skip if already resolved (frozen)
        if existing and existing.get("outcome") in ("win", "loss", "expired"):
            continue
        df = prices.get(t)
        if df is None:
            db[key] = {**p, **(existing or {}), "outcome": "nodata"}
            continue
        pick = pd.Timestamp(p["pick_date"])
        pos = int(df.index.searchsorted(pick, side="right")) - 1
        # require >=252 prior bars so perf_252 / 52w features are TRUE full-window
        # values (avoids feeding out-of-distribution perf_252=0 into the re-fit).
        if pos < 252 or pos >= len(df) - 1:
            db[key] = {**p, **(existing or {}), "outcome": "short_history"}
            continue
        hist = df.iloc[:pos + 1]
        feats = V4.compute_features(hist)
        v4 = V4.meta_v4_score(hist)
        entry = float(df["Close"].iloc[pos]); adr = feats["adr"]
        if adr <= 0:
            db[key] = {**p, **(existing or {}), "outcome": "nodata"}
            continue
        graded = grade(df, pos, entry, adr)
        if graded is None:
            db[key] = {**p, **(existing or {}), "outcome": "nodata"}
            continue
        rec = {**p, "entry": round(entry, 4), "adr": round(adr, 3),
               "v4_score": v4, "v4_features": {k: round(float(v), 5) for k, v in feats.items()},
               **graded}
        if existing is None:
            n_new += 1
        else:
            n_regrade += 1
        db[key] = rec

    payload = {"generated": date.today().isoformat(), "window": WINDOW,
               "win_def": "fwd close >= entry*(1+2*ADR/100)", "loss_def": "fwd low <= entry*0.92",
               "n": len(db), "records": db}
    tmp = DB_PATH + ".tmp"
    json.dump(payload, open(tmp, "w"), indent=1)
    os.replace(tmp, DB_PATH)

    # summary
    oc = Counter(r.get("outcome") for r in db.values())
    res = [r for r in db.values() if r.get("outcome") in ("win", "loss")]
    w = sum(r["outcome"] == "win" for r in res)
    wr = round(100 * w / len(res)) if res else None
    print(f"new {n_new} · re-graded {n_regrade} · total {len(db)}")
    print(f"outcomes: {dict(oc)}")
    print(f"live v4-pick win-rate (resolved): {wr}%  ({w}W/{len(res)-w}L of {len(res)})")
    if res:
        # does v4_score rank the live outcomes? (correlation, advisory — tiny sample early)
        s = np.array([r["v4_score"] for r in res if r.get("v4_score") is not None], float)
        yv = np.array([1.0 if r["outcome"] == "win" else 0.0 for r in res if r.get("v4_score") is not None])
        if len(s) > 5 and s.std() and yv.std():
            print(f"corr(v4_score, win) on live picks: {np.corrcoef(s, yv)[0,1]:+.3f}  (advisory; sample small)")
    print(f"wrote {DB_PATH}")


if __name__ == "__main__":
    main()
