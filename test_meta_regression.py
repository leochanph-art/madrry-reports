"""Regression: refactored calculate_meta_momentum_score with DEFAULT weights
must reproduce the original hardcoded scorer bit-for-bit (score + details)."""
import itertools, random
import pandas as pd
import madrry_html_scanner_v2 as M

# Force defaults regardless of any meta_weights.json on disk.
M.META_WEIGHTS = dict(M.META_WEIGHTS_DEFAULT)
M.META_DENOM = sum(M.META_WEIGHTS.values())
assert M.META_DENOM == 140, M.META_DENOM


def legacy(stock_data, hist_df=None):
    score = 0; badges=[]; details=[]
    adr = stock_data.get("adr", 0)
    is_flag = False
    if hist_df is not None and len(hist_df) >= 3 and adr > 0:
        is_flag = M.is_tight_flag(hist_df, adr, days=3, max_range_pct=10.0)
    perf_1m = stock_data.get("perf_1m", 0); perf_3m = stock_data.get("perf_3m", 0)
    if perf_1m >= 50 or perf_3m >= 100: score+=15; details.append("Trend: Explosive (15/15)")
    elif perf_1m >= 25 or perf_3m >= 50: score+=10; details.append("Trend: Strong (10/15)")
    else: score+=5; details.append("Trend: Established (5/15)")
    dist_52w = stock_data.get("dist_52w", 0)
    if dist_52w <= 5.0: score+=10; details.append("Proximity: Golden Zone 0-5% (10/10)")
    elif dist_52w <= 10.0: score+=7; details.append("Proximity: Near Pivot 5-10% (7/10)")
    elif dist_52w <= 15.0: score+=3; details.append("Proximity: Extended 10-15% (3/10)")
    else:
        score-=10; details.append("Proximity: Climax Run >15% (-10 penalty)"); badges.append("x")
    close=stock_data.get("close",0); sma10=stock_data.get("sma10",0); sma20=stock_data.get("sma20",0)
    if close>0 and sma10>0 and sma20>0:
        dist10=abs(close-sma10)/sma10*100; dist20=abs(close-sma20)/sma20*100
        if dist10<=3.0 and dist20>10.0 and close>sma20: score+=15; details.append("10MA Quality: Dominance (15/15)")
        elif dist10<=3.0: score+=12; details.append("10MA Quality: Hugging 10MA (12/15)")
        elif dist10<=5.0: score+=8; details.append("10MA Quality: Near 10MA (8/15)")
        else: score+=3; details.append("10MA Quality: Resting (3/15)")
    vol_pct=stock_data.get("vol_pct",100)
    if vol_pct<=55: score+=15; details.append("Vol Contraction: VooDoo <55% (15/15)")
    elif vol_pct<=75: score+=10; details.append("Vol Contraction: Contracting (10/15)")
    else: details.append("Vol Contraction: Normal (0/15)")
    if vol_pct>=250: score+=10; details.append("Vol Expansion: Massive >2.5x (10/10)")
    elif vol_pct>=150: score+=5; details.append("Vol Expansion: Moderate >1.5x (5/10)")
    else: details.append("Vol Expansion: No Surge (0/10)")
    if is_flag: score+=10; details.append("Flag: 3-Day Tight Coil (10/10)")
    else:
        day_range_pct=stock_data.get("day_range_pct",0)
        if day_range_pct>0 and adr>0:
            if day_range_pct<=adr*0.5: score+=5; details.append("Candle: Ultra Tight (5/10)")
            else: details.append("Candle: Loose (0/10)")
        else: details.append("Candle: Loose (0/10)")
    base_depth=25.0
    if hist_df is not None and len(hist_df)>=20:
        rh=hist_df["High"].iloc[-20:].max(); rl=hist_df["Low"].iloc[-20:].min()
        if rh>0: base_depth=(rh-rl)/rh*100
    if base_depth<35.0: score+=15; details.append(f"Base Quality: Tight Base {base_depth:.1f}% (15/15)")
    elif base_depth<=50.0: score+=8; details.append(f"Base Quality: Moderate Base {base_depth:.1f}% (8/15)")
    else: details.append(f"Base Quality: Wide/Loose Base {base_depth:.1f}% (0/15)")
    if dist_52w<=5.0: score+=15; details.append("RS: Leading - Near 52W High (15/15)")
    elif perf_3m>=60.0: score+=10; details.append("RS: Strong Outperformance (10/15)")
    else: score+=5; details.append("RS: Market Performer (5/15)")
    eer=0.0
    if adr>0: eer=perf_3m/adr
    if eer>=5.0: score+=10; details.append(f"Volatility: Super Efficient EER {eer:.1f} (10/10)")
    elif eer>=3.0: score+=6; details.append(f"Volatility: Clean EER {eer:.1f} (6/10)")
    else: details.append(f"Volatility: Erratic EER {eer:.1f} (0/10)")
    float_shares=stock_data.get("float_shares",0); mcap=stock_data.get("mcap",10.0)
    if float_shares>0: is_low=float_shares<200e6; fd=f"{float_shares/1e6:.1f}M"
    else: is_low=mcap<2.0; fd=f"Cap {mcap:.1f}B"
    hv=vol_pct>=150
    if is_low and hv: score+=10; details.append(f"Supply Shock: Low Float ({fd}) + RVOL Surge (10/10)")
    elif is_low or hv: score+=5; details.append("Supply Shock: Capable (5/10)")
    else: details.append("Supply Shock: Large/Quiet (0/10)")
    risk_pct=stock_data.get("risk_pct",10.0)
    if risk_pct<=3.5: score+=15; details.append(f"Risk: Super Asymmetric {risk_pct}% (15/15)")
    elif risk_pct<=5.0: score+=10; details.append(f"Risk: Acceptable {risk_pct}% (10/15)")
    else: details.append(f"Risk: Wide Stop {risk_pct}% (0/15)")
    raw=max(0,score); fs=(raw/140.0)*100.0
    if risk_pct<=3.5: fs*=1.1
    elif risk_pct>6.0: fs*=0.8
    fs=round(min(fs,100.0),1)
    return fs, details


def mk_hist(seed):
    r = random.Random(seed)
    rows = [{"High": 100+r.uniform(-5,40), "Low": 80+r.uniform(-5,30),
             "Close": 90+r.uniform(-5,35), "Open": 90+r.uniform(-5,35),
             "Volume": r.uniform(1e5, 5e6)} for _ in range(r.choice([0,3,25]))]
    return pd.DataFrame(rows) if rows else None


random.seed(0)
fails = 0; n = 0
grid_perf1 = [0,25,50,80]; grid_perf3=[0,50,60,100,150]; grid_dist=[2,7,12,20]
grid_vol=[40,60,100,160,300]; grid_risk=[3,4,6,8]
for i,(p1,p3,d,v,rk) in enumerate(itertools.product(grid_perf1,grid_perf3,grid_dist,grid_vol,grid_risk)):
    r=random.Random(i)
    sd={"adr":r.uniform(2,8),"perf_1m":p1,"perf_3m":p3,"dist_52w":d,"vol_pct":v,
        "risk_pct":rk,"close":100,"sma10":100-r.uniform(0,8),"sma20":100-r.uniform(0,20),
        "day_range_pct":r.uniform(0,6),"float_shares":r.choice([0,150e6,500e6]),
        "mcap":r.uniform(1,50)}
    hist=mk_hist(i)
    ns=M.calculate_meta_momentum_score(sd,hist)
    ls,ld=legacy(sd,hist)
    n+=1
    if abs(ns["score"]-ls)>1e-9 or ns["details"]!=ld:
        fails+=1
        if fails<=5:
            print("MISMATCH", {"p1":p1,"p3":p3,"d":d,"v":v,"rk":rk})
            print("  new",ns["score"],"legacy",ls)
            for a,b in zip(ns["details"],ld):
                if a!=b: print("   detail:",repr(a),"!=",repr(b))
print(f"\nchecked {n} cases · mismatches: {fails}")
print("PASS — refactor is behaviour-preserving at default weights" if fails==0 else "FAIL")
