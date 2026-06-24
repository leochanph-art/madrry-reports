"""
MADRRY v3 — recent-2026 holdout + simple economic backtest, then finalize weights.
==================================================================================
(1) Recent test: fit v3 on year<=2025, evaluate corr + decile win-rate on UNTOUCHED 2026.
(2) Economic backtest: fit v3 on year<=2023 (no look-ahead), trade top-decile each month
    2024-01..2026-04, equal-weight, barrier exit (+2ADR target / -8% stop / window-end),
    compare equity / CAGR / maxDD / MAR / win% for v1 vs v2 vs v3 vs whole-universe baseline.
(3) Finalize v3 on ALL data -> meta_v3_final.json (capped, sane weights).

Caveats (disclosed): data10y is survivorship-biased; no costs/slippage; monthly equal-weight
rebalance; barrier return is a proxy; risk_pct/float/mcap constant (Risk/Base held at 15).
"""
from __future__ import annotations
import json, glob, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")
import madrry_relabel_10y as R   # reuse load/precompute/fractions/score_from_fracs/COMPS/V1..

DATA_DIR = R.DATA_DIR; COMPS = R.COMPS; L52 = R.L52; WINDOW = R.WINDOW
V1, V1D, V2, V2D = R.V1, R.V1D, R.V2, R.V2D
MONTHS = pd.date_range("2017-06-01", "2026-04-01", freq="MS")
TARGET_K = 2.0; STOP = 0.92
HELD = {"Risk": 15, "Base Quality": 15}   # near-constant here -> hold at v1 default


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return 0.0 if a.std() == 0 or b.std() == 0 else float(np.corrcoef(a, b)[0, 1])


def build():
    """One pass: per-sample fracs, +2ADR label, barrier trade return, year, ym, ticker."""
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    FR, YR, YM, TID, LAB, RET = [], [], [], [], [], []
    targets = MONTHS.to_numpy()
    tids = {}
    for n, f in enumerate(files):
        try: df = R.load(f)
        except Exception: continue
        if len(df) < L52 + WINDOW: continue
        pc = R.precompute(df); idx = pc["idx"]; t = os.path.basename(f)[:-4]
        for pos in np.unique(idx.searchsorted(targets, side="right") - 1):
            if pos < L52 or pos >= len(df) - 1: continue
            close = pc["close"][pos]; avgvol = pc["avgvol30"][pos]; sma200 = pc["sma200"][pos]; adr = pc["adr"][pos]
            if not (close > 10 and avgvol > 500000 and close > sma200): continue
            if not np.isfinite(adr) or adr <= 0: continue
            c21, c63 = pc["c21"][pos], pc["c63"][pos]
            sd = {"perf_1m": (close/c21-1)*100 if c21 else 0.0, "perf_3m": (close/c63-1)*100 if c63 else 0.0,
                  "adr": adr, "close": close, "sma10": pc["ema9"][pos], "sma20": pc["ema21"][pos],
                  "vol_pct": df["Volume"].iloc[pos]/avgvol*100 if avgvol else 100.0, "risk_pct": 5.0,
                  "day_range_pct": pc["drp"][pos], "dist_52w": (pc["hi52"][pos]-close)/pc["hi52"][pos]*100,
                  "mcap": 10.0, "float_shares": 0}
            h20 = df.iloc[pos-19:pos+1]
            isflag = R.M.is_tight_flag(h20, adr, days=3, max_range_pct=10.0) if adr > 0 else False
            rh = h20["High"].max(); rl = h20["Low"].min(); bd = (rh-rl)/rh*100 if rh > 0 else 25.0
            fr = R.fractions(sd, isflag, bd)
            # barrier trade return
            fh = pc["high"][pos+1:pos+1+WINDOW]; fl = pc["low"][pos+1:pos+1+WINDOW]; fc = pc["close"][pos+1:pos+1+WINDOW]
            if len(fc) == 0: continue
            tgt = close*(1+TARGET_K*adr/100.0); stp = close*STOP
            ret = None
            for j in range(len(fc)):
                if fl[j] <= stp: ret = STOP-1.0; break          # -8% stop
                if fc[j] >= tgt: ret = fc[j]/close-1.0; break   # target (actual close)
            if ret is None: ret = fc[-1]/close-1.0              # window-end exit
            lab = 1 if ret > 0 else 0
            FR.append([fr[c] for c in COMPS]); YR.append(int(idx[pos].year))
            YM.append(idx[pos].strftime("%Y-%m")); TID.append(tids.setdefault(t, len(tids)))
            LAB.append(lab); RET.append(ret)
        if (n+1) % 500 == 0: print(f"  build {n+1}/{len(files)} rows={len(FR)}")
    return (np.array(FR), np.array(YR), np.array(YM), np.array(TID), np.array(LAB), np.array(RET, float))


def fit_v3(Xtr, ytr, cap=40, total=110):
    sc = StandardScaler().fit(Xtr); clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(Xtr), ytr)
    raw = np.where(Xtr.std(0) > 1e-9, clf.coef_[0]/Xtr.std(0), 0.0)
    pos = np.array([0.0 if COMPS[i] in HELD else max(0.0, raw[i]) for i in range(len(COMPS))])
    sc_w = pos/pos.sum()*total if pos.sum() > 0 else pos
    w = {}
    for i, c in enumerate(COMPS):
        w[c] = HELD[c] if c in HELD else int(min(cap, round(sc_w[i])))
    return w


def score(X, w):
    d = sum(w.values())
    return np.clip(X @ np.array([w[c] for c in COMPS], float), 0, None)/d*100


def deciles(s):
    r = np.argsort(np.argsort(s)); return r*10//len(s)


def main():
    print("Building economic dataset (one data10y pass) ...")
    X, YR, YM, TID, LAB, RET = build()
    print(f"  {len(X)} samples, {len(set(TID))} tickers, years {YR.min()}-{YR.max()}")

    # ---------- (1) recent 2026 holdout ----------
    tr = YR <= 2025; te = YR == 2026
    w_25 = fit_v3(X[tr], LAB[tr])
    v3s = score(X, w_25)
    print("\n" + "="*68)
    print("(1) RECENT 2026 HOLDOUT  (v3 fit on <=2025, tested on untouched 2026)")
    print("="*68)
    print(f"  2026 test samples: {te.sum()}  base win-rate {round(100*LAB[te].mean())}%")
    V1W = {'Trend':15,'Proximity':10,'10MA Quality':15,'Vol Contraction':15,'Vol Expansion':10,'Flag':10,'Base Quality':15,'RS':15,'Volatility':10,'Supply Shock':10,'Risk':15}
    c1 = corr(score(X[te], V1W), LAB[te])
    v2w = json.load(open("meta_weights.v2_20260624.json"))["weights"]
    c2 = corr(score(X[te], v2w), LAB[te]); c3 = corr(v3s[te], LAB[te])
    print(f"  corr(score,win):  v1={c1:.3f}   v2={c2:.3f}   v3={c3:.3f}")
    d1, d3 = deciles(score(X[te], V1W)), deciles(v3s[te])
    print(f"  top-decile win-rate:  v1={round(100*LAB[te][d1==9].mean())}%   v3={round(100*LAB[te][d3==9].mean())}%   (bottom v3={round(100*LAB[te][d3==0].mean())}%)")

    # ---------- (2) economic backtest (v3 fit <=2023, trade 2024..2026-04) ----------
    w_23 = fit_v3(X[YR <= 2023], LAB[YR <= 2023])
    sc = {"v1": score(X, V1W), "v2": score(X, v2w), "v3": score(X, w_23)}
    months = sorted(set(YM[(YR >= 2024)]))
    def equity(scores):
        mr = []
        for mo in months:
            mask = (YM == mo)
            if mask.sum() < 20: continue
            s = scores[mask]; r = RET[mask]
            top = s >= np.quantile(s, 0.9)
            mr.append(r[top].mean())
        mr = np.array(mr); eq = np.cumprod(1+mr)
        cagr = eq[-1]**(12/len(mr))-1
        peak = np.maximum.accumulate(eq); dd = (eq/peak-1).min()
        return dict(months=len(mr), cagr=round(100*cagr,1), maxdd=round(100*dd,1),
                    mar=round(cagr/abs(dd),2) if dd<0 else None, avg=round(100*mr.mean(),2),
                    winm=round(100*(mr>0).mean()))
    base_r = []
    for mo in months:
        mask = YM == mo
        if mask.sum() >= 20: base_r.append(RET[mask].mean())
    base_r = np.array(base_r); beq = np.cumprod(1+base_r); bpeak = np.maximum.accumulate(beq)
    print("\n" + "="*68)
    print("(2) ECONOMIC BACKTEST  top-decile, 2024-01..2026-04, barrier exit")
    print("="*68)
    print(f"  {'strat':<10}{'mons':>5}{'CAGR%':>8}{'maxDD%':>8}{'MAR':>6}{'avgTrd%':>9}{'win-mon%':>9}")
    res = {}
    for k in ("v1","v2","v3"):
        e = equity(sc[k]); res[k]=e
        print(f"  {k+' top10%':<10}{e['months']:>5}{e['cagr']:>8}{e['maxdd']:>8}{str(e['mar']):>6}{e['avg']:>9}{e['winm']:>9}")
    bcagr=beq[-1]**(12/len(base_r))-1; bdd=(beq/bpeak-1).min()
    print(f"  {'universe':<10}{len(base_r):>5}{round(100*bcagr,1):>8}{round(100*bdd,1):>8}{round(bcagr/abs(bdd),2):>6}{round(100*base_r.mean(),2):>9}{round(100*(base_r>0).mean()):>9}")

    # ---------- (3) finalize v3 on ALL data ----------
    w_final = fit_v3(X, LAB)
    full_corr = corr(score(X, w_final), LAB)
    json.dump({"weights": w_final, "denom": sum(w_final.values()), "full_corr_2adr": round(full_corr,3),
               "recent2026_corr_v1_v2_v3": [round(c1,3), round(c2,3), round(c3,3)],
               "backtest": res, "trained_on": "data10y +2ADR barrier label (survivorship-biased)"},
              open("meta_v3_final.json","w"), indent=1)
    print("\nFINAL v3 weights (denom %d, full-sample corr %.3f):"%(sum(w_final.values()), full_corr))
    print("  "+json.dumps(w_final))
    print("Wrote meta_v3_final.json")


if __name__ == "__main__":
    main()
