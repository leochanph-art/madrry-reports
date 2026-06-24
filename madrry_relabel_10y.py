"""
MADRRY 10y RELABEL — non-tautological win definition + per-component features
=============================================================================
Part 1 of the v3 work. Re-runs the 10y point-in-time scan but:
  * grades RETURN-based, non-tautological labels (no "fresh 52w high"):
      WIN(kADR)  = a forward CLOSE >= as-of_close * (1 + k*ADR/100)   (a real move up)
      LOSS       = a forward LOW  <= as-of_close * 0.92               (-8%, unchanged)
      first event in a 40-bar window wins.  k in {1,2,3}.
  * ALSO keeps the old 52w-high label for side-by-side (the tautological one).
  * extracts the 11 M.E.T.A. COMPONENT FRACTIONS per sample (0..1, Proximity can be
    -1) so v3 can fit weights on the components directly.

Outputs:
  - meta_dataset_10y.npz  (features + labels + ticker/year + v1/v2 scores)  -> feeds v3 fit
  - prints v1 vs v2 corr/spread/baseline under each label (answers: does v2's edge survive?)

DISCLOSED LIMITATIONS (from the session audit):
  - data10y has NO delisted names -> survivorship-biased (not fixable here).
  - risk_pct/float/mcap held constant -> the Risk component (const 0.667) and the
    low-float half of Supply Shock carry no variance and cannot be calibrated here.
"""
from __future__ import annotations
import glob, json, os
from collections import defaultdict
import numpy as np
import pandas as pd
import madrry_html_scanner_v2 as M

DATA_DIR = "/Users/boundbythese/Downloads/qmwork/data10y"
CACHE = "/Users/boundbythese/.openclaw/workspace/meta_dataset_10y.npz"
SIDECAR = "/Users/boundbythese/.openclaw/workspace/meta_dataset_10y.meta.json"
WINDOW, LOSS_MULT, L52 = 40, 0.92, 252
MONTHS = pd.date_range("2017-06-01", "2026-04-01", freq="MS")
KADR = [1.0, 2.0, 3.0]

COMPS = ["Trend", "Proximity", "10MA Quality", "Vol Contraction", "Vol Expansion",
         "Flag", "Base Quality", "RS", "Volatility", "Supply Shock", "Risk"]
V1 = dict(M.META_WEIGHTS_DEFAULT); V1D = sum(V1.values())
V2 = json.load(open("/Users/boundbythese/.openclaw/workspace/meta_weights.v2_20260624.json"))["weights"]; V2D = sum(V2.values())


def fractions(sd, is_flag, base_depth):
    """Replicate calculate_meta_momentum_score's per-tier fraction for each component."""
    f = {}
    p1, p3, d = sd["perf_1m"], sd["perf_3m"], sd["dist_52w"]
    f["Trend"] = 1.0 if (p1 >= 50 or p3 >= 100) else (10/15 if (p1 >= 25 or p3 >= 50) else 5/15)
    f["Proximity"] = 1.0 if d <= 5 else (0.7 if d <= 10 else (0.3 if d <= 15 else -1.0))
    cl, s10, s20 = sd["close"], sd["sma10"], sd["sma20"]
    if cl > 0 and s10 > 0 and s20 > 0:
        d10 = abs(cl-s10)/s10*100; d20 = abs(cl-s20)/s20*100
        f["10MA Quality"] = 1.0 if (d10 <= 3 and d20 > 10 and cl > s20) else (12/15 if d10 <= 3 else (8/15 if d10 <= 5 else 3/15))
    else:
        f["10MA Quality"] = 0.0
    v = sd["vol_pct"]
    f["Vol Contraction"] = 1.0 if v <= 55 else (10/15 if v <= 75 else 0.0)
    f["Vol Expansion"] = 1.0 if v >= 250 else (0.5 if v >= 150 else 0.0)
    if is_flag:
        f["Flag"] = 1.0
    else:
        drp, adr = sd["day_range_pct"], sd["adr"]
        f["Flag"] = 0.5 if (drp > 0 and adr > 0 and drp <= adr*0.5) else 0.0
    f["Base Quality"] = 1.0 if base_depth < 35 else (8/15 if base_depth <= 50 else 0.0)
    f["RS"] = 1.0 if d <= 5 else (10/15 if p3 >= 60 else 5/15)
    eer = p3/sd["adr"] if sd["adr"] > 0 else 0.0
    f["Volatility"] = 1.0 if eer >= 5 else (0.6 if eer >= 3 else 0.0)
    lowf = (sd["mcap"] < 2.0) if not sd["float_shares"] else (sd["float_shares"] < 200e6)
    hir = v >= 150
    f["Supply Shock"] = 1.0 if (lowf and hir) else (0.5 if (lowf or hir) else 0.0)
    r = sd["risk_pct"]
    f["Risk"] = 1.0 if r <= 3.5 else (10/15 if r <= 5 else 0.0)
    return f


def score_from_fracs(fr, w, denom):
    s = sum(fr[c]*w[c] for c in COMPS)
    return 100*max(0.0, s)/denom


def precompute(df):
    C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]
    return dict(close=C.to_numpy(), high=H.to_numpy(), low=L.to_numpy(),
                ema9=C.ewm(span=9, adjust=False).mean().to_numpy(),
                ema21=C.ewm(span=21, adjust=False).mean().to_numpy(),
                adr=((H/L).rolling(20).mean()-1).to_numpy()*100,
                avgvol30=V.rolling(30).mean().to_numpy(),
                sma200=C.rolling(200).mean().to_numpy(),
                hi52=H.rolling(L52).max().to_numpy(),
                c21=C.shift(21).to_numpy(), c63=C.shift(63).to_numpy(),
                drp=((H-L)/L*100).to_numpy(), idx=df.index)


def load(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)
    return df.dropna(subset=["High","Low","Close","Volume"]).set_index("date").sort_index()


def grade_labels(pc, pos, close, adr, hi52):
    """Return dict label-> 1 win / 0 loss / -1 unresolved, for each label variant."""
    loss_line = close*LOSS_MULT
    fh = pc["high"][pos+1:pos+1+WINDOW]; fl = pc["low"][pos+1:pos+1+WINDOW]; fc = pc["close"][pos+1:pos+1+WINDOW]
    n = len(fh)
    out = {}
    if n == 0:
        return None
    # old tautological label: first fresh-52w-high (intraday) vs -8%
    o = -1
    for k in range(n):
        if fl[k] <= loss_line: o = 0; break
        if fh[k] > hi52: o = 1; break
    out["w52"] = o if (o != -1 or n >= WINDOW) and o != -1 else (o if o != -1 else -1)
    out["w52"] = o
    # return-based labels: forward CLOSE >= close*(1+k*adr/100) vs -8% intraday
    for k in KADR:
        win_line = close*(1 + k*adr/100.0)
        oc = -1
        for j in range(n):
            if fl[j] <= loss_line: oc = 0; break
            if fc[j] >= win_line: oc = 1; break
        out[f"w{int(k)}adr"] = oc
    return out


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    print(f"data10y: {len(files)} tickers, monthly x {len(MONTHS)}")
    targets = MONTHS.to_numpy()
    FR, TICK, YR, V1S, V2S = [], [], [], [], []
    LAB = {k: [] for k in ["w52", "w1adr", "w2adr", "w3adr"]}
    tickers = []
    selfcheck_done = 0
    for n, f in enumerate(files):
        try: df = load(f)
        except Exception: continue
        if len(df) < L52 + WINDOW: continue
        pc = precompute(df); idx = pc["idx"]; tname = os.path.basename(f)[:-4]
        positions = np.unique(idx.searchsorted(targets, side="right") - 1)
        for pos in positions:
            if pos < L52 or pos >= len(df)-1: continue
            close = pc["close"][pos]; avgvol = pc["avgvol30"][pos]; sma200 = pc["sma200"][pos]; adr = pc["adr"][pos]
            if not (close > 10 and avgvol > 500000 and close > sma200): continue
            if not np.isfinite(adr) or adr <= 0: continue
            c21, c63 = pc["c21"][pos], pc["c63"][pos]
            sd = {"perf_1m": (close/c21-1)*100 if c21 else 0.0, "perf_3m": (close/c63-1)*100 if c63 else 0.0,
                  "adr": adr, "close": close, "sma10": pc["ema9"][pos], "sma20": pc["ema21"][pos],
                  "vol_pct": df["Volume"].iloc[pos]/avgvol*100 if avgvol else 100.0,
                  "risk_pct": 5.0, "day_range_pct": pc["drp"][pos], "dist_52w": (pc["hi52"][pos]-close)/pc["hi52"][pos]*100,
                  "mcap": 10.0, "float_shares": 0}
            hist20 = df.iloc[pos-19:pos+1]
            isflag = M.is_tight_flag(hist20, adr, days=3, max_range_pct=10.0) if adr > 0 else False
            rh = hist20["High"].max(); rl = hist20["Low"].min(); base_depth = (rh-rl)/rh*100 if rh > 0 else 25.0
            fr = fractions(sd, isflag, base_depth)
            # fidelity self-check: fraction->v1 score must equal the live scorer (a few times)
            if selfcheck_done < 50:
                live = M.calculate_meta_momentum_score(sd, hist20)["score"]
                mine = round(score_from_fracs(fr, V1, V1D), 1)
                assert abs(live-mine) < 0.05, f"fidelity {tname}: live {live} mine {mine}"
                selfcheck_done += 1
            labs = grade_labels(pc, pos, close, adr, pc["hi52"][pos])
            if labs is None: continue
            FR.append([fr[c] for c in COMPS]); TICK.append(tname); YR.append(int(idx[pos].year))
            V1S.append(score_from_fracs(fr, V1, V1D)); V2S.append(score_from_fracs(fr, V2, V2D))
            for k in LAB: LAB[k].append(labs[k])
        if (n+1) % 400 == 0: print(f"  {n+1}/{len(files)}  rows={len(FR)}")

    FR = np.array(FR, np.float32); YR = np.array(YR, np.int16)
    V1S = np.array(V1S, np.float32); V2S = np.array(V2S, np.float32)
    uniq = sorted(set(TICK)); tid = {t: i for i, t in enumerate(uniq)}
    TICKID = np.array([tid[t] for t in TICK], np.int32)
    labarr = {k: np.array(v, np.int8) for k, v in LAB.items()}
    np.savez_compressed(CACHE, fracs=FR, tickid=TICKID, year=YR, v1=V1S, v2=V2S, **{f"lab_{k}": labarr[k] for k in labarr})
    json.dump({"comps": COMPS, "tickers": uniq, "labels": list(LAB), "kadr": KADR,
               "n": int(FR.shape[0]), "v1_denom": V1D, "v2_denom": V2D},
              open(SIDECAR, "w"))
    print(f"\nfidelity self-check passed on {selfcheck_done} samples.")
    print(f"cached {FR.shape[0]} rows x {FR.shape[1]} comps, {len(uniq)} unique tickers -> {CACHE}")

    def corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0: return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    print("\n" + "="*72)
    print("PART 1 — v1 vs v2 under each label (resolved = win/loss only)")
    print("="*72)
    print(f"  {'label':<8}{'Nres':>8}{'base%':>7}{'corr_v1':>9}{'corr_v2':>9}{'Δcorr':>8}  verdict")
    for k in ["w52", "w1adr", "w2adr", "w3adr"]:
        y = labarr[k]; m = y >= 0
        yr = y[m].astype(float); v1 = V1S[m]; v2 = V2S[m]
        base = round(100*yr.mean())
        c1, c2 = corr(v1, yr), corr(v2, yr)
        verdict = "v2 better" if c2 > c1 + 0.005 else ("≈tie" if abs(c2-c1) <= 0.005 else "v1 better")
        lab = "52w-high" if k == "w52" else f"+{k[1]}ADR"
        print(f"  {lab:<8}{m.sum():>8}{base:>7}{c1:>9.3f}{c2:>9.3f}{c2-c1:>+8.3f}  {verdict}")
    print("\n(52w-high = the OLD tautological label; +kADR = return-based, non-tautological.")
    print(" If v2's Δcorr shrinks/flips going from 52w-high to +kADR, its edge was gaming the label.)")


if __name__ == "__main__":
    main()
