"""
MADRRY META Weekend Recalibration
=================================
Re-fit the M.E.T.A. component weights from the labeled Tier-A win/loss data and
write them to meta_weights.json, which the scanner loads on its next run.

Guardrails (so a noisy/young sample can't wreck live scoring):
  * MIN_RESOLVED gate — refuse to update with too few resolved win/loss names.
  * Bounded changes — _compute_recal already clips each component's multiplier
    to [0.25, 2.0] and shrinks 50% toward the current weight (blend=0.5).
  * Every write is appended to meta_weights_history.jsonl and the json is
    git-committed by the morning loop, so any change is reviewable / revertible.

Run:  python3 madrry_recalibrate_meta.py            # apply if gate passes
      python3 madrry_recalibrate_meta.py --dry-run  # print, don't write
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date

import madrry_tier_a_tracker as T

WORKSPACE = "/Users/boundbythese/.openclaw/workspace"
DB_PATH = os.path.join(WORKSPACE, "tier_a_tracking.json")
WEIGHTS_PATH = os.path.join(WORKSPACE, "meta_weights.json")
HISTORY_PATH = os.path.join(WORKSPACE, "meta_weights_history.jsonl")
MIN_RESOLVED = 30          # don't tune live scoring on fewer than this many W/L
CORR_MARGIN = 0.02         # min IN-SAMPLE corr(score,win) gain to apply new weights


def main():
    dry = "--dry-run" in sys.argv
    if not os.path.exists(DB_PATH):
        print("no tier_a_tracking.json — run the tracker first")
        return 1
    records = json.load(open(DB_PATH))["records"]
    resolved = [r for r in records if r.get("outcome") in ("win", "loss")]
    n = len(resolved)
    recal = T._compute_recal(records)
    weights = {w["comp"]: w["new"] for w in recal["weights"]}

    c1, c2 = recal.get("corr_v1", 0.0), recal.get("corr_v2", 0.0)
    print(f"resolved win/loss names: {n}  (gate: >= {MIN_RESOLVED})")
    print(f"GATE metric  corr(score,win):  v1={c1:+.4f}  v2={c2:+.4f}  "
          f"(need v2 >= v1 + {CORR_MARGIN})   [IN-SAMPLE — advisory]")
    print(f"display only spread  v1={recal['spread_v1']}  v2={recal['spread_v2']} pts "
          f"(empty-bottom-bucket caveat applies)")
    for w in recal["weights"]:
        print(f"  {w['comp']:<16} {w['cur']:>3} -> {w['new']:>3}  (edge {w['edge']:+.2f})")

    if n < MIN_RESOLVED:
        print(f"\nGATE NOT MET ({n} < {MIN_RESOLVED}) — keeping current weights.")
        return 0
    # Boundary-free gate: require a non-trivial IN-SAMPLE corr improvement.
    # (A proper decision needs an out-of-sample holdout; this is a safety floor,
    #  not a validation. The Sunday cron is intentionally disabled until then.)
    if c2 < c1 + CORR_MARGIN:
        print(f"\nProposed weights do not improve corr(score,win) by >= {CORR_MARGIN} "
              f"({c2:+.4f} vs {c1:+.4f}) — keeping current weights.")
        return 0

    payload = {
        "version": date.today().isoformat(),
        "generated": date.today().isoformat(),
        "n_resolved": n,
        "denom": recal["denom_new"],
        "corr_v1": c1,
        "corr_v2": c2,
        "spread_v1_display": recal["spread_v1"],
        "spread_v2_display": recal["spread_v2"],
        "in_sample": True,
        "weights": weights,
    }
    if dry:
        print("\n[dry-run] would write:", json.dumps(payload["weights"]))
        return 0

    tmp = WEIGHTS_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1)
    os.replace(tmp, WEIGHTS_PATH)
    with open(HISTORY_PATH, "a") as fh:
        fh.write(json.dumps(payload) + "\n")
    print(f"\nWROTE {WEIGHTS_PATH}  (v{payload['version']}, n={n}, denom={payload['denom']})")
    print("Scanner will pick these up on its next run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
