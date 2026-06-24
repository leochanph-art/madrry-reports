"""
MADRRY v3 — fit ALL components on the non-tautological label, OUT-OF-SAMPLE
==========================================================================
Lessons applied from the session audit:
  * Out-of-sample: ticker-GROUPED 5-fold CV (a ticker is never in train & test together).
  * Honest baselines: v1/v2 are fixed weightings (no fit) so their corr is already OOS.
  * Permutation null: shuffle labels, refit -> OOS corr must collapse to ~0 (no leakage).
  * Report a constrained, SHIPPABLE v3 (non-negative integer maxes, production architecture)
    AND the unconstrained logistic 'ceiling' (best linear combo incl. signs).
  * Disclose: data10y is survivorship-biased; Risk/Supply-Shock-lowfloat carry no variance.

Primary training label: +2ADR (a real, ~balanced move up). Generalization checked on +1/+3 ADR.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import pointbiserialr  # noqa
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

Z = np.load("/Users/boundbythese/.openclaw/workspace/meta_dataset_10y.npz")
META = json.load(open("/Users/boundbythese/.openclaw/workspace/meta_dataset_10y.meta.json"))
COMPS = META["comps"]
X = Z["fracs"].astype(float)             # N x 11 component fractions
G = Z["tickid"]                          # ticker group ids
V1, V2 = Z["v1"].astype(float), Z["v2"].astype(float)
PRIMARY = "lab_w2adr"
LABELS = {"+1ADR": "lab_w1adr", "+2ADR": "lab_w2adr", "+3ADR": "lab_w3adr"}


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def oos_logistic(Xall, y, groups, C=1.0, seed_shuffle=False):
    """Pooled 5-fold ticker-grouped OOS predictions from logistic regression."""
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=5)
    yy = y.copy()
    for tr, te in gkf.split(Xall, yy, groups):
        ytr = yy[tr]
        if seed_shuffle:
            rng = np.random.default_rng(len(tr))     # deterministic per-fold shuffle
            ytr = rng.permutation(ytr)
        sc = StandardScaler().fit(Xall[tr])
        clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xall[tr]), ytr)
        pred[te] = clf.decision_function(sc.transform(Xall[te]))
    return pred


def fit_full_weights(Xall, y, C=1.0):
    """Refit on ALL data -> non-negative integer production weights + signed coefs."""
    sc = StandardScaler().fit(Xall)
    clf = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xall), y)
    beta_std = clf.coef_[0]
    std = Xall.std(0)
    raw = np.where(std > 1e-9, beta_std / std, 0.0)   # convert to raw-fraction weight
    signed = {COMPS[i]: round(float(raw[i]), 3) for i in range(len(COMPS))}
    pos = np.clip(raw, 0, None)
    # scale positive part to ~125 total (Risk held at its v1 default 15 -> ~140 arch)
    fitt = pos.sum()
    w = {}
    for i, c in enumerate(COMPS):
        if c == "Risk":
            w[c] = 15                                  # unfittable here (const) -> hold v1
        else:
            w[c] = int(round(125 * pos[i] / fitt)) if fitt > 0 else 0
    return w, signed


def main():
    yP = Z[PRIMARY]
    mP = yP >= 0
    Xp, yp, gp = X[mP], yP[mP].astype(int), G[mP]
    print(f"Primary label +2ADR: {mP.sum()} resolved, base win-rate {round(100*yp.mean())}%, "
          f"{len(set(gp.tolist()))} tickers")

    # ---- fit v3 (OOS) on primary, plus ceiling & null ----
    pred_v3 = oos_logistic(Xp, yp, gp, C=1.0)
    pred_null = oos_logistic(Xp, yp, gp, C=1.0, seed_shuffle=True)
    c_v3 = corr(pred_v3, yp)
    c_null = corr(pred_null, yp)
    auc_v3 = roc_auc_score(yp, pred_v3)
    print(f"\nOOS (ticker-grouped 5-fold) on +2ADR:")
    print(f"  logistic v3 corr={c_v3:.3f}  AUC={auc_v3:.3f}")
    print(f"  permutation NULL corr={c_null:.3f}  (must be ~0 -> no leakage)")

    # ---- ship weights (refit on all) ----
    w3, signed = fit_full_weights(Xp, yp, C=1.0)
    d3 = sum(w3.values())
    print(f"\nv3 weights (non-neg integer, denom {d3}):")
    print("  " + json.dumps(w3))
    print("  signed raw coefs (sign = direction of real-return signal):")
    for c in COMPS:
        print(f"    {c:<16}{signed[c]:+.3f}   v1={dict_v1[c]:>2}  v2={dict_v2.get(c,'-'):>2}  v3={w3[c]:>2}")

    # ---- v3 fixed-weight score (shippable, applied like v1/v2) ----
    def score(w, denom):
        s = X @ np.array([w[c] for c in COMPS], float)
        return np.clip(s, 0, None) / denom * 100
    v3score = score(w3, d3)

    # ---- head-to-head OOS on all 3 return labels ----
    print("\n" + "=" * 78)
    print("v1 vs v2 vs v3  —  corr(score, win)  on RETURN labels  (v3 = OOS pooled CV)")
    print("=" * 78)
    print(f"  {'label':<8}{'Nres':>7}{'base%':>7}{'v1':>8}{'v2':>8}{'v3_ship':>9}{'v3_OOS':>9}{'ceiling':>9}")
    out = {}
    for name, key in LABELS.items():
        yk = Z[key]; m = yk >= 0
        yy = yk[m].astype(int)
        c1, c2 = corr(V1[m], yy), corr(V2[m], yy)
        c3ship = corr(v3score[m], yy)
        # OOS fitted corr on THIS label (refit per label, grouped CV)
        p = oos_logistic(X[m], yy, G[m], C=1.0)
        c3oos = corr(p, yy)
        out[name] = dict(n=int(m.sum()), base=round(100*yy.mean()),
                         v1=round(c1, 3), v2=round(c2, 3),
                         v3_ship=round(c3ship, 3), v3_oos=round(c3oos, 3))
        print(f"  {name:<8}{m.sum():>7}{round(100*yy.mean()):>7}"
              f"{c1:>8.3f}{c2:>8.3f}{c3ship:>9.3f}{c3oos:>9.3f}{c3oos:>9.3f}")

    json.dump({"v3_weights": w3, "v3_denom": d3, "signed_coefs": signed,
               "oos_corr_v3_primary": round(c_v3, 4), "auc_v3": round(auc_v3, 4),
               "null_corr": round(c_null, 4), "by_label": out,
               "train_label": "+2ADR", "data": "data10y (survivorship-biased)",
               "note": "Risk held at v1=15 (no variance to fit); float/mcap constant"},
              open("/Users/boundbythese/.openclaw/workspace/meta_v3_fit.json", "w"), indent=1)
    print("\nWrote meta_v3_fit.json")


dict_v1 = {"Trend": 15, "Proximity": 10, "10MA Quality": 15, "Vol Contraction": 15,
           "Vol Expansion": 10, "Flag": 10, "Base Quality": 15, "RS": 15,
           "Volatility": 10, "Supply Shock": 10, "Risk": 15}
dict_v2 = json.load(open("/Users/boundbythese/.openclaw/workspace/meta_weights.v2_20260624.json"))["weights"]

if __name__ == "__main__":
    main()
