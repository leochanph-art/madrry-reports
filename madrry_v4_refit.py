"""
MADRRY v4 WEEKLY RE-FIT — armed, waiting for maturity, promote-only-if-better.
=============================================================================
Closes the self-improvement loop. Retrains the v4 model on the LIVE, survivorship-
free forward results in v4_tracking.json (raw-on-raw, matching the live scanner),
validates OUT-OF-SAMPLE, and ONLY promotes a new meta_v4_model.json if it beats the
currently-deployed model on a held-out set. Non-destructive: any gate failure ->
keep the current model untouched.

GATES (all must pass to even attempt a promotion):
  1. MATURITY  — only use picks with a FULL 40-bar resolved window (days_followed>=40,
     label in {0,1}). Until enough exist the job idles ("ARMED — waiting").
  2. SAMPLE    — >= MIN_MATURE mature records AND >= MIN_TICKERS distinct tickers.
  3. PROMOTION — candidate (fit on train tickers) must beat the INCUMBENT on the SAME
     held-out test tickers by >= PROMOTE_MARGIN (corr), else keep current.

Design notes (from the audits):
  * Ticker-GROUPED train/test split (a ticker never in both) — no leakage.
  * Tier-A-conditional by construction (the tracker only logs gated picks) — that is
    the deployment context (v4 ranks WITHIN tiers), so it is the right object to fit.
  * Promotion compares candidate vs incumbent on the SAME test set (fair), then ships
    a full-data refit (validate the method, deploy the full fit).
  * The 10y base (adjusted) is NOT mixed in (basis mismatch); the deployed model is the
    incumbent baseline to beat. Blending a raw-regenerated 10y base is future work.

Run:  python3 madrry_v4_refit.py            # gated; idles until mature
      python3 madrry_v4_refit.py --dry-run  # never writes
      python3 madrry_v4_refit.py --force    # bypass maturity/sample gates (TEST ONLY)
"""
from __future__ import annotations
import json, os, sys, shutil
from datetime import date
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings("ignore")

WS = "/Users/boundbythese/.openclaw/workspace"
TRACK = os.path.join(WS, "v4_tracking.json")
MODEL = os.path.join(WS, "meta_v4_model.json")
HISTORY = os.path.join(WS, "meta_v4_refit_history.jsonl")
ARCHIVE = os.path.join(WS, "model_archive")

MATURE_BARS = 40          # FULL window must elapse for EVERY included pick (no censoring:
                          # barrier-resolved records are NOT short-cut in early, else the
                          # sample over-represents fast losses and biases the fit).
MIN_MATURE = 150          # min mature resolved records before re-fitting (was 300 — too slow;
                          # 150 is ample for a 45-feature/L1 logit and ~halves the cold start)
MIN_TICKERS = 40          # min distinct tickers (for ticker-grouped CV)
PROMOTE_MARGIN = 0.05     # corr gain to ship (was 0.02 — inside the fold-noise band)
FOLD_WIN_FRAC = 0.7       # candidate must also win this fraction of folds (was bare majority)
SEED = 13


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def load_mature(min_bars=MATURE_BARS):
    """Mature, fully-resolved live records -> (X, y, tickers, feature_names).
    min_bars is relaxed under --force so the pipeline can be smoke-tested early."""
    feats = json.load(open(MODEL))["features"]
    recs = json.load(open(TRACK)).get("records", {})
    X, y, tick = [], [], []
    for r in recs.values():
        if r.get("outcome") not in ("win", "loss", "expired"):
            continue
        if (r.get("days_followed") or 0) < min_bars:
            continue
        if r.get("label_2adr") not in (0, 1):
            continue
        f = r.get("v4_features") or {}
        if len(f) != len(feats):
            continue
        X.append([float(f.get(k, 0.0)) for k in feats])
        y.append(int(r["label_2adr"]))
        tick.append(r["ticker"])
    return np.array(X), np.array(y), np.array(tick), feats


def build_model(X, y, feats):
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.3, penalty="l1", solver="saga", max_iter=8000,
                             random_state=SEED).fit(sc.transform(X), y)
    if int(np.max(clf.n_iter_)) >= 8000:      # did not converge -> unsafe to trust
        raise RuntimeError("logistic fit did not converge")
    probs = clf.predict_proba(sc.transform(X))[:, 1]
    cal = np.quantile(probs, np.linspace(0, 1, 101)).tolist()
    scale = [s if s != 0 else 1.0 for s in sc.scale_.tolist()]   # no 0-scale (div-by-zero guard)
    return {"features": feats, "mean": sc.mean_.tolist(), "scale": scale,
            "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
            "calib_pctile": cal}


def apply_model(m, X):
    z = (X - np.array(m["mean"])) / np.array(m["scale"])
    return 1.0 / (1.0 + np.exp(-(z @ np.array(m["coef"]) + m["intercept"])))


def main():
    dry = "--dry-run" in sys.argv
    force = "--force" in sys.argv
    if not os.path.exists(TRACK):
        print("no v4_tracking.json yet — nothing to re-fit."); return 0
    X, y, tick, feats = load_mature(min_bars=1 if force else MATURE_BARS)
    n, nt = len(y), len(set(tick.tolist()))
    print(f"mature resolved records: {n} (need {MIN_MATURE}) · distinct tickers: {nt} (need {MIN_TICKERS})")

    # GATE 1+2: maturity / sample size
    if not force and (n < MIN_MATURE or nt < MIN_TICKERS or len(set(y.tolist())) < 2):
        print(f"ARMED — waiting for maturity ({n}/{MIN_MATURE} mature, {nt}/{MIN_TICKERS} tickers). "
              "Keeping current model. (Each pick needs a full 40-bar window.)")
        if not dry:
            _log({"event": "wait", "n_mature": n, "n_tickers": nt})
        return 0
    if force and (n < 30 or nt < 3):
        print("--force but too few records/tickers (need >=30 rows, >=3 tickers). Aborting."); return 0

    # 5-fold ticker-grouped comparison: in EACH fold fit the candidate on train
    # tickers and score BOTH candidate and incumbent on the same held-out test
    # tickers. Robust to an unlucky single split (this runs unattended).
    incumbent = json.load(open(MODEL))
    cand_folds, inc_folds = [], []
    try:
        for k, (tr, te) in enumerate(GroupKFold(min(5, nt)).split(X, y, tick)):
            if len(set(y[tr].tolist())) < 2 or len(te) < 5:
                continue
            cm = build_model(X[tr], y[tr], feats)
            cand_folds.append(corr(apply_model(cm, X[te]), y[te]))
            inc_folds.append(corr(apply_model(incumbent, X[te]), y[te]))
    except Exception as exc:  # noqa: BLE001 — any fit/CV failure -> never promote
        print(f"fold fit failed ({exc}) — KEEPING current model.")
        if not dry:
            _log({"event": "no_promote", "reason": f"fit_error:{exc}", "n_mature": n})
        return 0
    if len(cand_folds) < 3:
        print("Too few valid folds to decide safely — KEEPING current model.")
        if not dry:
            _log({"event": "no_promote", "reason": "few_folds", "n_mature": n})
        return 0
    cmean, imean = float(np.mean(cand_folds)), float(np.mean(inc_folds))
    wins = sum(c > i for c, i in zip(cand_folds, inc_folds))
    need_wins = int(np.ceil(FOLD_WIN_FRAC * len(cand_folds)))
    print(f"{len(cand_folds)}-fold OOS corr: candidate mean={cmean:+.3f}  incumbent mean={imean:+.3f}  "
          f"candidate wins {wins}/{len(cand_folds)} (need {need_wins})  margin needs +{PROMOTE_MARGIN}")

    # GATE 3: promote only on a robust margin AND a strong fold majority
    if not (cmean >= imean + PROMOTE_MARGIN and wins >= need_wins):
        print("Candidate does NOT robustly beat incumbent — KEEPING current model.")
        if not dry:
            _log({"event": "no_promote", "n_mature": n, "cand_mean": round(cmean, 4),
                  "inc_mean": round(imean, 4), "wins": wins, "folds": len(cand_folds)})
        return 0
    c_cand, c_inc = cmean, imean
    if dry:
        print(f"[dry-run] WOULD promote (cand {c_cand:+.3f} > inc {c_inc:+.3f}). No write.")
        return 0

    # promote: ship a full-data refit, archive the old model, log + (wrapper commits)
    final = build_model(X, y, feats)
    final["version"] = date.today().isoformat()
    final["note"] = (f"v4 weekly re-fit on {n} mature LIVE Tier-A picks (raw). "
                     f"Promoted: held-out corr {c_cand:+.3f} vs incumbent {c_inc:+.3f}. "
                     "Tier-A-conditional, survivorship-free.")
    os.makedirs(ARCHIVE, exist_ok=True)
    shutil.copy(MODEL, os.path.join(ARCHIVE, f"meta_v4_model.{date.today().isoformat()}.json"))
    tmp = MODEL + ".tmp"; json.dump(final, open(tmp, "w")); os.replace(tmp, MODEL)
    _log({"event": "promote", "n_mature": n, "n_tickers": nt,
          "cand_test": round(c_cand, 4), "inc_test": round(c_inc, 4), "version": final["version"]})
    print(f"PROMOTED new meta_v4_model.json (v{final['version']}). The next daily scan will use it.")
    return 0


def _log(d):
    d = {"date": date.today().isoformat(), **d}
    with open(HISTORY, "a") as fh:
        fh.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
