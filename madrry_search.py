"""Search for an ENHANCED meta score: signed weights + rich features.
Evaluates every model OUT-OF-SAMPLE (ticker-grouped 5-fold) on BOTH:
  (a) win-rate discrimination  corr(pred, +2ADR win)
  (b) ECONOMIC edge  = top-decile mean fwd-20d return MINUS universe (the real gate)
Reports which NEW features actually matter. No leakage (permutation null checked).
"""
import json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

Z = np.load("meta_features_10y.npz", allow_pickle=True)
M = json.load(open("meta_features_10y.meta.json"))
F = M["features"]; X = np.nan_to_num(Z["X"].astype(float), nan=0.0, posinf=0.0, neginf=0.0)
y = Z["lab2adr"].astype(int); G = Z["tickid"]; YR = Z["year"]; YM = Z["ym"]; FWD20 = Z["fwd20"].astype(float)
fidx = {f: i for i, f in enumerate(F)}
FRAC = [f for f in F if f.startswith("frac_")]
print(f"{len(y)} samples, {len(F)} features, {len(set(G.tolist()))} tickers, +2ADR base {round(100*y.mean())}%")


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def oos(model_fn, cols):
    Xc = X[:, [fidx[c] for c in cols]]
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(Xc, y, G):
        sc = StandardScaler().fit(Xc[tr])
        m = model_fn().fit(sc.transform(Xc[tr]), y[tr])
        pred[te] = m.decision_function(sc.transform(Xc[te])) if hasattr(m, "decision_function") else m.predict_proba(sc.transform(Xc[te]))[:, 1]
    return pred


def econ_edge(pred, mask):
    p, r, ym = pred[mask], FWD20[mask], YM[mask]
    ok = ~np.isnan(r); p, r, ym = p[ok], r[ok], ym[ok]
    td, un = [], []
    for mo in sorted(set(ym.tolist())):
        mm = ym == mo
        if mm.sum() < 20: continue
        pp, rr = p[mm], r[mm]
        td.append(rr[pp >= np.quantile(pp, 0.9)].mean()); un.append(rr.mean())
    return 100*(np.mean(td)-np.mean(un)), 100*np.mean(td), 100*np.mean(un)


def winrate_top(pred, mask):
    p, yy = pred[mask], y[mask]
    r = np.argsort(np.argsort(p)); return round(100*yy[r*10//len(p) == 9].mean())


# reference: v1 and v3 fixed scores from the existing fractions
W = {"v1": {'Trend':15,'Proximity':10,'10MA Quality':15,'Vol Contraction':15,'Vol Expansion':10,'Flag':10,'Base Quality':15,'RS':15,'Volatility':10,'Supply Shock':10,'Risk':15},
     "v3": json.load(open("meta_v3_final.json"))["weights"]}
def fixed_score(w):
    cols = {("frac_"+k.replace(" ", "_")): v for k, v in w.items()}
    s = np.zeros(len(y))
    for c, v in cols.items(): s += X[:, fidx[c]]*v
    return np.clip(s, 0, None)


EXPERIMENTS = {
    "v1 (fixed)":        ("ref", W["v1"]),
    "v3 (fixed,nonneg)": ("ref", W["v3"]),
    "signed-components": (lambda: LogisticRegression(C=1.0, max_iter=3000), FRAC),
    "rich-L2":           (lambda: LogisticRegression(C=1.0, max_iter=3000), F),
    "rich-L1-sparse":    (lambda: LogisticRegression(C=0.3, penalty="l1", solver="saga", max_iter=4000), F),
    "GBM (ceiling)":     (lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=200, learning_rate=0.05), F),
}

full = np.ones(len(y), bool); oos24 = YR >= 2024
print(f"\n{'model':<20}{'OOScorr':>8}{'econ_full':>10}{'econ_24-26':>11}{'top10%win':>10}")
results = {}
for name, (fn, arg) in EXPERIMENTS.items():
    pred = fixed_score(arg) if fn == "ref" else oos(fn, arg)
    c = corr(pred, y); ef, _, _ = econ_edge(pred, full); e24, _, _ = econ_edge(pred, oos24)
    results[name] = dict(corr=round(c, 3), econ_full=round(ef, 2), econ_24=round(e24, 2), topwin=winrate_top(pred, oos24))
    print(f"  {name:<18}{c:>8.3f}{ef:>+10.2f}{e24:>+11.2f}{winrate_top(pred, oos24):>9}%")

# permutation null on the best rich model
yp = np.random.default_rng(0).permutation(y)
pn = np.full(len(y), np.nan)
for tr, te in GroupKFold(5).split(X, yp, G):
    sc = StandardScaler().fit(X[tr]); m = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[tr]), yp[tr]); pn[te] = m.decision_function(sc.transform(X[te]))
print(f"\n  permutation NULL (rich-L2) corr = {corr(pn, yp):.3f}  (must be ~0)")

# feature importances: which features the sparse model keeps (signed)
sc = StandardScaler().fit(X); clf = LogisticRegression(C=0.3, penalty="l1", solver="saga", max_iter=5000).fit(sc.transform(X), y)
coefs = sorted([(F[i], clf.coef_[0][i]) for i in range(len(F)) if abs(clf.coef_[0][i]) > 1e-3], key=lambda x: -abs(x[1]))
print(f"\nSPARSE signed model kept {len(coefs)}/{len(F)} features (standardized coef = importance & SIGN):")
for f, co in coefs[:20]:
    print(f"   {f:<22}{co:+.3f}   {'(predicts UP)' if co>0 else '(predicts DOWN / subtract)'}")
json.dump({"results": results, "sparse_coefs": [(f, round(c, 4)) for f, c in coefs]}, open("meta_search.json", "w"), indent=1)
print("\nwrote meta_search.json")
