"""
MADRRY META v4 — enhanced score (self-contained, no scanner import to avoid cycles).
Computes the same 45 point-in-time features as madrry_features_extract.feats() from a
single OHLCV DataFrame (last bar = as-of), then applies meta_v4_model.json
(signed logistic) and percentile-calibrates the win-probability to 0-100.

The 11 legacy component fractions use the TRAINING assumptions (risk_pct=5, float=0,
mcap=10) so live scoring matches how the model was fit. All other features are pure
price/volume off the DataFrame.

Usage in the scanner:  from madrry_meta_v4 import meta_v4_score
                       score = meta_v4_score(hist_df)     # 0-100, or None if no model
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_v4_model.json")
_M = None


def _model():
    global _M
    if _M is None and os.path.exists(_MODEL_PATH):
        _M = json.load(open(_MODEL_PATH))
    return _M


def _tight_flag(h, adr, days=3, max_range_pct=10.0):
    """Exact port of the scanner's is_tight_flag (madrry_html_scanner_v2.py:893)."""
    if h is None or len(h) < days or adr <= 0:
        return False
    r = h.tail(days).dropna(subset=["High", "Low", "Close", "Volume"])
    if len(r) < days:
        return False
    dr = (r["High"] - r["Low"]) / r["Close"] * 100
    if not dr.le(adr * 1.0).all():
        return False
    total = (r["High"].max() - r["Low"].min()) / r["Close"].iloc[-1] * 100
    if total > max_range_pct:
        return False
    v = r["Volume"].values
    if v[-1] > v[0] * 1.5:
        return False
    return True


def _fractions(close, adr, ema9, ema21, vol_pct, dist_52w, perf_1m, perf_3m,
               day_range_pct, base_depth, is_flag):
    """11 legacy component fractions (training assumptions: risk=5, float=0, mcap=10)."""
    f = {}
    f["Trend"] = 1.0 if (perf_1m >= 50 or perf_3m >= 100) else (10/15 if (perf_1m >= 25 or perf_3m >= 50) else 5/15)
    f["Proximity"] = 1.0 if dist_52w <= 5 else (0.7 if dist_52w <= 10 else (0.3 if dist_52w <= 15 else -1.0))
    if close > 0 and ema9 > 0 and ema21 > 0:
        d10 = abs(close-ema9)/ema9*100; d20 = abs(close-ema21)/ema21*100
        f["10MA Quality"] = 1.0 if (d10 <= 3 and d20 > 10 and close > ema21) else (12/15 if d10 <= 3 else (8/15 if d10 <= 5 else 3/15))
    else:
        f["10MA Quality"] = 0.0
    f["Vol Contraction"] = 1.0 if vol_pct <= 55 else (10/15 if vol_pct <= 75 else 0.0)
    f["Vol Expansion"] = 1.0 if vol_pct >= 250 else (0.5 if vol_pct >= 150 else 0.0)
    f["Flag"] = 1.0 if is_flag else (0.5 if (day_range_pct > 0 and adr > 0 and day_range_pct <= adr*0.5) else 0.0)
    f["Base Quality"] = 1.0 if base_depth < 35 else (8/15 if base_depth <= 50 else 0.0)
    f["RS"] = 1.0 if dist_52w <= 5 else (10/15 if perf_3m >= 60 else 5/15)
    eer = perf_3m/adr if adr > 0 else 0.0
    f["Volatility"] = 1.0 if eer >= 5 else (0.6 if eer >= 3 else 0.0)
    f["Supply Shock"] = 0.5 if vol_pct >= 150 else 0.0   # low-float leg off (float=0)
    f["Risk"] = 10/15                                     # risk_pct=5 -> "Acceptable"
    return f


def _channel(closes):
    n = len(closes); x = np.arange(n)
    s, b = np.polyfit(x, closes, 1)
    fit_last = s*(n-1)+b
    return s/closes[-1]*100, (closes[-1]-fit_last)/closes[-1]*100


def compute_features(df: pd.DataFrame) -> dict:
    """45 features at the LAST bar of df (OHLCV, capitalized cols)."""
    C, H, L, V = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(df); i = n-1; c = float(C.iloc[i])
    def sma(k): return float(C.iloc[max(0, i-k+1):i+1].mean())
    def ema(sp): return float(C.ewm(span=sp, adjust=False).mean().iloc[i])
    def rmax(k): return float(H.iloc[max(0, i-k+1):i+1].max())
    def rmin(k): return float(L.iloc[max(0, i-k+1):i+1].min())
    def vavg(k): return float(V.iloc[max(0, i-k+1):i+1].mean())
    def cs(k): return float(C.iloc[i-k]) if i-k >= 0 else c
    sma10, sma20, sma50, sma200 = sma(10), sma(20), sma(50), sma(200)
    ema9, ema21 = ema(9), ema(21)
    adr = float((H.iloc[max(0, i-19):i+1]/L.iloc[max(0, i-19):i+1]).mean()-1)*100
    hi252 = rmax(252); avgvol20 = vavg(20)
    def d(m): return (c/m-1)*100 if m > 0 else 0.0
    def slope(curr, prev): return (curr/prev-1)*100 if prev > 0 else 0.0
    sma10_10 = float(C.iloc[max(0, i-19):i-9].mean()) if i-9 >= 0 else sma10
    sma20_10 = float(C.iloc[max(0, i-29):i-9].mean()) if i-9 >= 0 else sma20
    sma50_20 = float(C.iloc[max(0, i-69):i-19].mean()) if i-19 >= 0 else sma50
    align = int(sma10 > sma20)+int(sma20 > sma50)+int(sma50 > sma200)
    price_above = sum(int(c > m) for m in (sma10, sma20, sma50, sma200))
    perf_1m = slope(c, cs(21)); perf_3m = slope(c, cs(63))
    ch20s, ch20r = _channel(C.iloc[i-19:i+1].to_numpy())
    ch60s, ch60r = _channel(C.iloc[i-59:i+1].to_numpy()) if i >= 59 else (0.0, 0.0)
    rng60 = rmax(60)-rmin(60)
    # base_depth (20-bar) + flag for legacy fractions
    bd = (rmax(20)-rmin(20))/rmax(20)*100 if rmax(20) > 0 else 25.0
    isflag = _tight_flag(df.tail(3), adr)
    vol_pct = float(V.iloc[i])/avgvol20*100 if avgvol20 else 100.0
    dist_52w = (hi252-c)/hi252*100 if hi252 else 0.0
    f = {
        "d_ema9": d(ema9), "d_ema21": d(ema21), "d_sma10": d(sma10), "d_sma20": d(sma20),
        "d_sma50": d(sma50), "d_sma200": d(sma200),
        "slope_sma10": slope(sma10, sma10_10), "slope_sma20": slope(sma20, sma20_10), "slope_sma50": slope(sma50, sma50_20),
        "ma_align": align, "price_above_ma": price_above,
        "sp_10_20": (sma10-sma20)/sma20*100 if sma20 else 0.0, "sp_20_50": (sma20-sma50)/sma50*100 if sma50 else 0.0,
        "sp_50_200": (sma50-sma200)/sma200*100 if sma200 else 0.0,
        "perf_5": slope(c, cs(5)), "perf_21": perf_1m, "perf_63": perf_3m,
        "perf_126": slope(c, cs(126)), "perf_252": slope(c, cs(252)),
        "dist_52w": dist_52w, "d_hi20": (c-rmax(20))/rmax(20)*100 if rmax(20) else 0.0,
        "d_hi60": (c-rmax(60))/rmax(60)*100 if rmax(60) else 0.0, "d_hi120": (c-rmax(120))/rmax(120)*100 if rmax(120) else 0.0,
        "d_lo20": (c-rmin(20))/rmin(20)*100 if rmin(20) else 0.0,
        "pos_in_range60": (c-rmin(60))/rng60 if rng60 > 0 else 0.5,
        "adr": adr, "day_range_pct": (float(H.iloc[i])-float(L.iloc[i]))/float(L.iloc[i])*100,
        "rvol": vol_pct, "volc_5_20": vavg(5)/avgvol20*100 if avgvol20 else 100.0,
        "volc_20_50": avgvol20/vavg(50)*100 if vavg(50) else 100.0,
        "ch20_slope": ch20s, "ch20_resid": ch20r, "ch60_slope": ch60s, "ch60_resid": ch60r,
    }
    fr = _fractions(c, adr, ema9, ema21, vol_pct, dist_52w, perf_1m, perf_3m, f["day_range_pct"], bd, isflag)
    for k, v in fr.items():
        f["frac_"+k.replace(" ", "_")] = v
    return f


def meta_v4_score(df: pd.DataFrame):
    """0-100 enhanced score (percentile of P(+2ADR win)). None if model/data missing."""
    m = _model()
    # Require >= 200 bars so SMA200 (coef +0.20) and the 252-lookbacks are TRUE
    # full-window values, matching the training regime (pos>=252). Short-history
    # names are out-of-distribution -> return None -> scanner uses legacy score.
    if m is None or df is None or len(df) < 200:
        return None
    try:
        f = compute_features(df)
        x = np.array([f.get(k, 0.0) for k in m["features"]], float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        scale = np.array(m["scale"], float)
        scale[scale == 0] = 1.0                      # guard against a 0-scale div-by-zero
        z = (x - np.array(m["mean"])) / scale
        logit = float(np.dot(z, m["coef"]) + m["intercept"])
        prob = 1.0/(1.0+np.exp(-logit))
        return round(float(np.interp(prob, m["calib_pctile"], np.linspace(0, 100, 101))), 1)
    except Exception:
        return None
