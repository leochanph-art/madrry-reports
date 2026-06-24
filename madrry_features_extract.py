"""
MADRRY rich feature extraction (point-in-time, 10y) for META-score enhancement.
Builds a wide feature matrix per sample so we can: allow SIGNED weights, add new
components (20/50/200 MA, MA alignment/slopes/spreads, support/resistance, trendline
channel, multi-lookback momentum, volume ratios), and search for a score that
actually predicts forward returns. All features use data up to & incl. the as-of bar.
Caches -> meta_features_10y.npz (+ .meta.json).
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import pandas as pd
import madrry_relabel_10y as R   # load/precompute/fractions/COMPS, M (scanner)

OUT = "/Users/boundbythese/.openclaw/workspace/meta_features_10y.npz"
SIDE = "/Users/boundbythese/.openclaw/workspace/meta_features_10y.meta.json"
L52, WINDOW = 252, 40
MONTHS = pd.date_range("2017-06-01", "2026-03-01", freq="MS")
ECOMPS = R.COMPS  # existing 11 component fractions (for the 'minus component' test)


def rich_pre(df):
    C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]
    def sma(n): return C.rolling(n).mean().to_numpy()
    def rmax(n): return H.rolling(n).max().to_numpy()
    def rmin(n): return L.rolling(n).min().to_numpy()
    def vavg(n): return V.rolling(n).mean().to_numpy()
    return dict(close=C.to_numpy(), high=H.to_numpy(), low=L.to_numpy(), vol=V.to_numpy(),
                ema9=C.ewm(span=9, adjust=False).mean().to_numpy(),
                ema21=C.ewm(span=21, adjust=False).mean().to_numpy(),
                sma10=sma(10), sma20=sma(20), sma50=sma(50), sma200=sma(200),
                adr=((H/L).rolling(20).mean()-1).to_numpy()*100,
                avgvol5=vavg(5), avgvol20=vavg(20), avgvol50=vavg(50),
                hi20=rmax(20), hi60=rmax(60), hi120=rmax(120), hi252=rmax(252),
                lo20=rmin(20), lo60=rmin(60),
                c5=C.shift(5).to_numpy(), c21=C.shift(21).to_numpy(), c63=C.shift(63).to_numpy(),
                c126=C.shift(126).to_numpy(), c252=C.shift(252).to_numpy(),
                drp=((H-L)/L*100).to_numpy(), idx=df.index)


def channel(closes):
    """linear fit to a window of closes -> (slope %/bar, residual % of last vs fit)."""
    n = len(closes); x = np.arange(n)
    s, b = np.polyfit(x, closes, 1)
    fit_last = s*(n-1)+b
    return s/closes[-1]*100, (closes[-1]-fit_last)/closes[-1]*100


def feats(pc, pos, sd_frac):
    p = pc; c = p["close"][pos]
    def d(ma): return (c/ma[pos]-1)*100 if ma[pos] > 0 else 0.0
    def slope(ma, k=10): return (ma[pos]/ma[pos-k]-1)*100 if (pos-k >= 0 and ma[pos-k] > 0) else 0.0
    sma10, sma20, sma50, sma200 = p["sma10"], p["sma20"], p["sma50"], p["sma200"]
    align = int(sma10[pos] > sma20[pos]) + int(sma20[pos] > sma50[pos]) + int(sma50[pos] > sma200[pos])
    price_above = sum(int(c > m[pos]) for m in (sma10, sma20, sma50, sma200))
    def perf(cs): return (c/cs[pos]-1)*100 if cs[pos] else 0.0
    avgvol20 = p["avgvol20"][pos]
    ch20s, ch20r = channel(p["close"][pos-19:pos+1])
    ch60s, ch60r = channel(p["close"][pos-59:pos+1]) if pos >= 59 else (0.0, 0.0)
    rng60 = p["hi60"][pos]-p["lo60"][pos]
    f = {
        # --- MA distances ---
        "d_ema9": d(p["ema9"]), "d_ema21": d(p["ema21"]),
        "d_sma10": d(sma10), "d_sma20": d(sma20), "d_sma50": d(sma50), "d_sma200": d(sma200),
        # --- MA slopes ---
        "slope_sma10": slope(sma10), "slope_sma20": slope(sma20), "slope_sma50": slope(sma50, 20),
        # --- MA alignment / spreads ---
        "ma_align": align, "price_above_ma": price_above,
        "sp_10_20": (sma10[pos]-sma20[pos])/sma20[pos]*100 if sma20[pos] else 0.0,
        "sp_20_50": (sma20[pos]-sma50[pos])/sma50[pos]*100 if sma50[pos] else 0.0,
        "sp_50_200": (sma50[pos]-sma200[pos])/sma200[pos]*100 if sma200[pos] else 0.0,
        # --- momentum multi-lookback ---
        "perf_5": perf(p["c5"]), "perf_21": perf(p["c21"]), "perf_63": perf(p["c63"]),
        "perf_126": perf(p["c126"]), "perf_252": perf(p["c252"]),
        # --- support / resistance / position ---
        "dist_52w": (p["hi252"][pos]-c)/p["hi252"][pos]*100 if p["hi252"][pos] else 0.0,
        "d_hi20": (c-p["hi20"][pos])/p["hi20"][pos]*100 if p["hi20"][pos] else 0.0,
        "d_hi60": (c-p["hi60"][pos])/p["hi60"][pos]*100 if p["hi60"][pos] else 0.0,
        "d_hi120": (c-p["hi120"][pos])/p["hi120"][pos]*100 if p["hi120"][pos] else 0.0,
        "d_lo20": (c-p["lo20"][pos])/p["lo20"][pos]*100 if p["lo20"][pos] else 0.0,
        "pos_in_range60": (c-p["lo60"][pos])/rng60 if rng60 > 0 else 0.5,
        # --- volatility / range ---
        "adr": p["adr"][pos], "day_range_pct": p["drp"][pos],
        # --- volume ---
        "rvol": p["vol"][pos]/avgvol20*100 if avgvol20 else 100.0,
        "volc_5_20": p["avgvol5"][pos]/avgvol20*100 if avgvol20 else 100.0,
        "volc_20_50": avgvol20/p["avgvol50"][pos]*100 if p["avgvol50"][pos] else 100.0,
        # --- trendline channel ---
        "ch20_slope": ch20s, "ch20_resid": ch20r, "ch60_slope": ch60s, "ch60_resid": ch60r,
    }
    # existing 11 component fractions (so we can test 'minus component' / signed)
    for c_ in ECOMPS:
        f["frac_" + c_.replace(" ", "_")] = sd_frac[c_]
    return f


def main():
    files = sorted(glob.glob(os.path.join(R.DATA_DIR, "*.csv")))
    targets = MONTHS.to_numpy()
    FEAT_NAMES = None
    ROWS, YR, YM, TID, L2, FWD20, FWD40, BRET = [], [], [], [], [], [], [], []
    tids = {}
    for n, f in enumerate(files):
        try: df = R.load(f)
        except Exception: continue
        if len(df) < L52 + WINDOW + 1: continue
        pc = rich_pre(df); idx = pc["idx"]; t = os.path.basename(f)[:-4]
        for pos in np.unique(idx.searchsorted(targets, side="right") - 1):
            if pos < L52 or pos >= len(df)-1: continue
            c = pc["close"][pos]; avgvol = pc["avgvol20"][pos]; sma200 = pc["sma200"][pos]; adr = pc["adr"][pos]
            if not (c > 10 and avgvol > 500000 and c > sma200) or not np.isfinite(adr) or adr <= 0: continue
            # existing-component fractions need the scanner-style stock_data
            sd = {"perf_1m": (c/pc["c21"][pos]-1)*100 if pc["c21"][pos] else 0.0,
                  "perf_3m": (c/pc["c63"][pos]-1)*100 if pc["c63"][pos] else 0.0, "adr": adr, "close": c,
                  "sma10": pc["ema9"][pos], "sma20": pc["ema21"][pos],
                  "vol_pct": pc["vol"][pos]/avgvol*100 if avgvol else 100.0, "risk_pct": 5.0,
                  "day_range_pct": pc["drp"][pos], "dist_52w": (pc["hi252"][pos]-c)/pc["hi252"][pos]*100,
                  "mcap": 10.0, "float_shares": 0}
            h20 = df.iloc[pos-19:pos+1]
            isflag = R.M.is_tight_flag(h20, adr, days=3, max_range_pct=10.0) if adr > 0 else False
            rh, rl = h20["High"].max(), h20["Low"].min(); bd = (rh-rl)/rh*100 if rh > 0 else 25.0
            sd_frac = R.fractions(sd, isflag, bd)
            fr = feats(pc, pos, sd_frac)
            if FEAT_NAMES is None: FEAT_NAMES = list(fr)
            # labels / returns
            fh = pc["high"][pos+1:pos+1+WINDOW]; fl = pc["low"][pos+1:pos+1+WINDOW]; fc = pc["close"][pos+1:pos+1+WINDOW]
            if len(fc) == 0: continue
            tgt = c*(1+2*adr/100.0); stp = c*0.92; bret = None; lab = -1
            for j in range(len(fc)):
                if fl[j] <= stp: lab, bret = 0, 0.92-1.0; break
                if fc[j] >= tgt: lab, bret = 1, fc[j]/c-1.0; break
            if bret is None: bret = fc[-1]/c-1.0; lab = 1 if bret > 0 else 0
            ROWS.append([fr[k] for k in FEAT_NAMES]); YR.append(int(idx[pos].year)); YM.append(idx[pos].strftime("%Y-%m"))
            TID.append(tids.setdefault(t, len(tids))); L2.append(lab); BRET.append(bret)
            FWD20.append(pc["close"][pos+20]/c-1 if pos+20 < len(pc["close"]) else np.nan)
            FWD40.append(pc["close"][pos+40]/c-1 if pos+40 < len(pc["close"]) else np.nan)
        if (n+1) % 400 == 0: print(f"  {n+1}/{len(files)} rows={len(ROWS)}")
    X = np.array(ROWS, np.float32)
    np.savez_compressed(OUT, X=X, year=np.array(YR, np.int16), tickid=np.array(TID, np.int32),
                        lab2adr=np.array(L2, np.int8), bret=np.array(BRET, np.float32),
                        fwd20=np.array(FWD20, np.float32), fwd40=np.array(FWD40, np.float32),
                        ym=np.array(YM))
    json.dump({"features": FEAT_NAMES, "n": int(X.shape[0]), "n_feat": int(X.shape[1]),
               "existing_comps": ECOMPS}, open(SIDE, "w"))
    print(f"\ncached {X.shape[0]} rows x {X.shape[1]} features -> {OUT}")
    print("features:", FEAT_NAMES)


if __name__ == "__main__":
    main()
