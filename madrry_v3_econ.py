"""Decisive economics: does a high META score predict higher FORWARD RETURN?
Barrier-free test (removes the +2ADR/-8% asymmetry artifact). Mean forward 20d & 40d
return of top-decile-by-score vs the whole universe, for v1/v2/v3, full 10y and 2024-26 OOS.
"""
import json, glob, os
import numpy as np
import pandas as pd
import madrry_relabel_10y as R

COMPS = R.COMPS
V1W = {'Trend':15,'Proximity':10,'10MA Quality':15,'Vol Contraction':15,'Vol Expansion':10,'Flag':10,'Base Quality':15,'RS':15,'Volatility':10,'Supply Shock':10,'Risk':15}
V2W = json.load(open("meta_weights.v2_20260624.json"))["weights"]
V3W = json.load(open("meta_v3_final.json"))["weights"]
MONTHS = pd.date_range("2017-06-01", "2026-03-01", freq="MS")


def sc(fr, w):
    d = sum(w.values()); s = sum(fr[c]*w[c] for c in COMPS); return 100*max(0.0, s)/d


rows = []
files = sorted(glob.glob(os.path.join(R.DATA_DIR, "*.csv")))
for n, f in enumerate(files):
    try: df = R.load(f)
    except Exception: continue
    if len(df) < R.L52 + 41: continue
    pc = R.precompute(df); idx = pc["idx"]; cl = pc["close"]
    for pos in np.unique(idx.searchsorted(MONTHS.to_numpy(), side="right") - 1):
        if pos < R.L52 or pos >= len(df) - 21: continue
        close = cl[pos]; avgvol = pc["avgvol30"][pos]; sma200 = pc["sma200"][pos]; adr = pc["adr"][pos]
        if not (close > 10 and avgvol > 500000 and close > sma200) or not np.isfinite(adr) or adr <= 0: continue
        c21, c63 = pc["c21"][pos], pc["c63"][pos]
        sd = {"perf_1m":(close/c21-1)*100 if c21 else 0,"perf_3m":(close/c63-1)*100 if c63 else 0,"adr":adr,
              "close":close,"sma10":pc["ema9"][pos],"sma20":pc["ema21"][pos],
              "vol_pct":df["Volume"].iloc[pos]/avgvol*100 if avgvol else 100,"risk_pct":5.0,
              "day_range_pct":pc["drp"][pos],"dist_52w":(pc["hi52"][pos]-close)/pc["hi52"][pos]*100,"mcap":10.0,"float_shares":0}
        h20 = df.iloc[pos-19:pos+1]
        isflag = R.M.is_tight_flag(h20, adr, days=3, max_range_pct=10.0) if adr > 0 else False
        rh, rl = h20["High"].max(), h20["Low"].min(); bd = (rh-rl)/rh*100 if rh > 0 else 25.0
        fr = R.fractions(sd, isflag, bd)
        f20 = cl[pos+20]/close-1 if pos+20 < len(cl) else np.nan
        f40 = cl[pos+40]/close-1 if pos+40 < len(cl) else np.nan
        rows.append((idx[pos].year, idx[pos].strftime("%Y-%m"), sc(fr,V1W), sc(fr,V2W), sc(fr,V3W), f20, f40))
    if (n+1) % 500 == 0: print(f"  {n+1}/{len(files)} rows={len(rows)}")

A = np.array([(r[0],)+r[2:] for r in rows], float)   # year,v1,v2,v3,f20,f40
YM = np.array([r[1] for r in rows]); YR = A[:,0]
print(f"\n{len(A)} samples")

def topdecile_vs_universe(scol, rcol, mask):
    s = A[mask, scol]; r = A[mask, rcol]; ym = YM[mask]
    good = ~np.isnan(r); s, r, ym = s[good], r[good], ym[good]
    td, un = [], []
    for mo in sorted(set(ym)):
        mm = ym == mo
        if mm.sum() < 20: continue
        ss, rr = s[mm], r[mm]
        td.append(rr[ss >= np.quantile(ss, 0.9)].mean()); un.append(rr.mean())
    return np.mean(td), np.mean(un), np.mean(td)-np.mean(un)

for label, rcol in [("fwd-20d", 4), ("fwd-40d", 5)]:
    print(f"\n=== mean {label} return: TOP-DECILE-by-score vs UNIVERSE (monthly avg) ===")
    for period, mask in [("full 2017-26", np.ones(len(A), bool)), ("OOS 2024-26", YR >= 2024)]:
        print(f"  [{period}]")
        for name, col in [("v1", 1), ("v2", 2), ("v3", 3)]:
            td, un, sp = topdecile_vs_universe(col, rcol, mask)
            verdict = "score ADDS value" if sp > 0.001 else ("≈neutral" if abs(sp) <= 0.001 else "score HURTS")
            print(f"    {name}: top10%={100*td:+.2f}%  universe={100*un:+.2f}%  edge={100*sp:+.2f}pp  -> {verdict}")
