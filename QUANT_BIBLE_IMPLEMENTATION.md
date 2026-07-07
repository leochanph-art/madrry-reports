# Implementing the MIT Quant Bible in the MADRRY Report

**Source:** *Quant Bible* — MIT Sloan Business Club (51 pp.)
**Target:** MADRRY Ultimate Scanner Report (`madrry_report.html`)
**Status:** Implementation plan / spec

---

## 1. What the document is — and what actually applies

The Quant Bible is a quant-finance **interview prep** guide. Roughly half of it
(§1 career resources, §7 company question banks) is not relevant to a trading
report. But four sections contain concepts that map directly onto what MADRRY
already does, and can make the report measurably more rigorous:

| Bible section | Concept | Where it lands in MADRRY |
|---|---|---|
| §2.1 Bayes' theorem, conditional probability | P(win \| condition) | Market kill-switch, tier stats |
| §2.2 Expected value & variance | EV per setup, portfolio variance | Trade Plan boxes, runbar |
| §2.5 Covariance & correlation | Pairwise ρ of candidates | New concentration-risk panel |
| §3.1–3.2 LLN, CLT, confidence intervals | CI on win rates, sample-size honesty | Tracking section |
| §4.3–4.5 Regression, t-tests, lasso/ridge | Validate & re-weight M.E.T.A. score | Scoring engine |
| §4.6 Econometrics (selection bias, OVB) | Backtest hygiene | Scoring engine / tracking |
| §6 Market making | Theo value vs. last price vs. position; CI width ∝ uncertainty | Intraday execution plan |

The single biggest unlock: **the report currently states plans but never
measures them.** Almost everything below depends on one prerequisite —
logging outcomes (Phase 0).

---

## 2. Phase 0 — Outcome logging (prerequisite for everything)

The report already has a "Yesterday's Watchlist Tracking" panel, but it only
shows one day and no resolution (win/loss). The Bible's stats machinery (LLN,
CLT, CIs — §3) is useless without samples.

**Add a per-run append-only log** (e.g. `history/outcomes.jsonl`), one row per
watchlist candidate per day:

```json
{
  "date": "2026-06-06",
  "ticker": "CHRW",
  "tier": "A",
  "meta_score": 63.5,
  "meta_components": {"trend": 5, "proximity": 7, "ma10_quality": 12,
    "vol_contraction": 15, "vol_expansion": 0, "candle": 0, "base_quality": 15,
    "rs": 5, "volatility": 0, "supply_shock": 5, "risk": 10},
  "setup": "breakout",
  "market_state": "GREEN",
  "entry": 186.71, "stop": 179.93, "risk_pct": 3.6,
  "outcome": null
}
```

`outcome` is resolved on later runs: `"triggered"` → then `"stopped"` /
`"target"` / `"open"`, plus realized R-multiple (`r = (exit − entry) /
(entry − stop)`). Untriggered rows resolve to `"no_fill"`. The scanner
already re-checks yesterday's names ("Still Coiled"), so the resolution pass
is an extension of existing logic, not new infrastructure.

---

## 3. Phase 1 — Expected value on every trade plan (Bible §2.2)

EV is the weighted average of outcomes (Bible: `E[X] = Σ x·p(x)`). In
R-multiples per setup type:

```
EV_R(setup) = p_win × avg_win_R − (1 − p_win) × 1
```

where `p_win` and `avg_win_R` come from the outcome log, grouped by setup
(`breakout`, `pullback`, `ORB`, `U&R`). With fixed $500 risk per trade the
dollar EV is simply `EV_R × $500`.

**Report change:** add one line to each entry box:

```
🟢 Breakout Plan
Buy: $186.71   Stop: $179.93   Risk: 3.6%
EV: +0.42R ≈ +$210/trade   (n=87 breakouts, win 38%, avg win +2.7R)
```

Also add a runbar chip per setup family so the day's list is framed by
whether the setup class is even positive-EV in the current regime.

**Guard:** if `n < 30` for a bucket, show `EV: — (insufficient sample, n=12)`
rather than a number. That honesty requirement comes straight from §3
(estimates converge by LLN; small n means the estimate is noise).

---

## 4. Phase 1 — Confidence intervals on win rates (Bible §3.2)

A win rate is a Bernoulli parameter estimate. The Bible gives the exact
tool: for Bernoulli data, `σ = √(p(1−p)) ≤ ½`, so a 95% CI is

```
p̂ ± 1.96 × √(p̂(1−p̂)) / √n        (or the conservative p̂ ± 0.98/√n)
```

**Report change:** everywhere a win rate appears, render the interval, not
the point:

```
Breakout win rate: 38% ± 10%  (n=87)
U&R win rate:      55% ± 30%  (n=11 — LOW SAMPLE, do not size up on this)
```

This single change prevents the most common trader error the Bible warns
about — treating a lucky 6-of-8 streak (75% "win rate") as an edge. At n=8
the 95% CI is roughly ±35%: statistically indistinguishable from a coin flip.

---

## 5. Phase 1 — Bayesian kill-switch: make the market filter quantitative (Bible §2.1)

The report already has the qualitative version: *"🚨 MARKET RED LIGHT — QQQ
below VWAP. Breakouts fail in weak tape."* The Bible's conditional
probability machinery turns that belief into a measured number:

```
P(win | GREEN tape),  P(win | YELLOW),  P(win | RED)
```

computed from the outcome log by conditioning on `market_state` at entry
(the definitional formula `P(A|B) = P(A∩B)/P(B)` is literally a filtered
group-by). Law of total probability (§2.1) gives the sanity check that the
conditioned rates recombine to the overall rate.

**Report change:** replace the static warning with the measured deltas:

```
🚨 RED LIGHT — QQQ below VWAP.
Measured: breakout win rate drops 41% → 19% in RED tape (n=214 / n=63).
Historical EV in RED tape: −0.31R. Stand down or trade ¼ size.
```

Same conditioning works per tier (does A+ actually outperform A−?) and per
badge (do "VooDoo day" names really follow through?). Each of these is a
one-line group-by once Phase 0 exists.

---

## 6. Phase 2 — Correlation & concentration risk (Bible §2.2 + §2.5)

Today's report treats each $500-risk trade as independent. The Bible's
variance-of-sums result says that's only true when correlations are zero:

```
Var(X₁+…+Xₙ) = Σ Var(Xᵢ) + 2·Σ Cov(Xᵢ,Xⱼ)
```

Momentum watchlists are systematically correlated (same themes, same tape).
Five "independent" $500 risks with average pairwise ρ ≈ 0.7 behave closer to
one $2,000 risk.

**Implementation:**
1. Compute pairwise Pearson ρ of 20-day daily returns for all tickers on
   today's list (prices are already fetched for sparklines).
2. Report average pairwise ρ and an **effective independent bets** number:
   `n_eff ≈ n / (1 + (n−1)·ρ̄)`.
3. Flag theme clusters: if ≥3 names share a theme tag and pairwise ρ > 0.7,
   badge them `🔗 CLUSTER` and suggest taking only the top M.E.T.A. name.

**Report change — new panel under Market Overview:**

```
🧮 PORTFOLIO RISK CHECK
Names on list: 27 | Avg pairwise ρ: 0.58 | Effective independent bets: ~1.9
⚠️ If you take every A-/A signal today you are effectively making 2 bets, not 27.
Clusters: [Semis: NVDA·AVGO·MU ρ̄=0.81] [Air Freight: CHRW·EXPD ρ̄=0.74]
```

(Bible caveat worth keeping in mind: ρ=0 does not imply independence — §2.5.)

---

## 7. Phase 3 — Validate the M.E.T.A. score with regression (Bible §4.3–4.6)

M.E.T.A. is a hand-weighted linear model: 11 components with weights like
15/10/15/10… chosen by intuition. Section 4 of the Bible is exactly the
toolkit for auditing such a model:

1. **Fit instead of guess.** Regress forward outcome on the 11 standardized
   component scores (linear on realized R, or logistic on win/loss —
   logistic is the better fit for a binary target):
   `β̂ = (XᵀX)⁻¹Xᵀy` (§4.3).
2. **t-test each component** (§4.3): z-score `zⱼ = β̂ⱼ/(σ̂√vⱼ)`, threshold
   |z| ≥ 2. Components that never predict anything (candidates: the ones
   that are almost always 0, like *Vol Expansion* on quiet days) are noise
   in the score.
3. **F-test the insignificant group** (§4.3) before dropping — three
   individually weak components can still be jointly significant.
4. **Expect multicollinearity** (§4.5): *10MA Quality*, *Vol Contraction*,
   and *Candle tightness* all measure "quiet pullback" and will be
   correlated. Remedies, in the Bible's order of preference: ridge (shrink),
   lasso (sparsify — good if the goal is a shorter checklist), or drop via
   subset selection. Lasso is recommended here: it yields a smaller, more
   explainable scorecard, which suits a human-read report.
5. **Control for the tape** (§4.6). Market regime is a textbook omitted
   variable: bullish tape inflates every component *and* every outcome. Add
   `market_state` as a control or the component betas absorb its effect
   (OVB formula: `β_short − β_long = π₁ × γ`). This is the difference
   between "VooDoo days predict wins" and "VooDoo days happen in uptrends,
   and uptrends predict wins."

**Report change:** none visually at first — this recalibrates the weights
behind the existing 0–100 score. Add a footnote to the M.E.T.A. breakdown:
`weights: fitted 2026-06 (logistic, n=412, AUC 0.63)` so score versions are
traceable. Refit quarterly; ~150+ resolved outcomes minimum before trusting
a fit.

---

## 8. Phase 4 — Market-making discipline in the intraday plan (Bible §6)

Ravi's three determinants of a market map one-to-one onto trade management:

| Market maker | MADRRY equivalent | Report change |
|---|---|---|
| **Theoretical value** — wider market when uncertain | Target/stop confidence. TRL/UTL are shown as point values (`TRL: +25.7%`) | Show targets as ranges scaled by ADR: `TRL: +19% ↔ +32%`. Wider range ⇒ smaller size, per §6.2's "CI should be wider when uncertainty is higher" |
| **Last price traded** — respect the market when it disagrees with your model | Price action vs. the setup thesis | Already partially done (VWAP cancel rule). Add: if the name gaps > 1 ADR past the trigger, the "market" has repriced — recompute risk instead of using stale levels |
| **Current position** — skew quotes to reduce exposure | Open positions in correlated names | If already long a cluster (Phase 2), skew: raise the effective trigger / cut size on further names in that cluster, the way a long MM lowers his ask to get flat |
| **Realistic markets** — 0@1bn never trades | Entry/stop realism | Flag plans whose risk% exceeds the setup's historical MAE distribution — a stop nobody would honor is a 0@1bn market |

Also from §6.1: for thin names (e.g. EVC at $8.77, 806 shares), the bid-ask
spread **is** a cost of the trade. Add spread as % of risk to the intraday
table; if spread > ~10% of the stop distance, badge `💸 SPREAD TAX`.

---

## 9. What we deliberately do NOT port

- §1 (career/companies/classes) and §7 (interview question bank) — out of scope.
- §5 case studies — methodology already absorbed via §4 (preprocessing,
  multicollinearity handling, log-transforms feed into Phase 3).
- Nearest-neighbor models (§4.1–4.2) — the report's sample sizes are far too
  small for k-NN on 11-dimensional component vectors (curse of
  dimensionality, §4.2, argues against it explicitly).

---

## 10. Rollout order & dependencies

```
Phase 0  Outcome log (jsonl)                ── prerequisite, ship first
Phase 1  EV lines + win-rate CIs + Bayesian kill-switch
         └─ needs ~30 resolved outcomes per bucket before numbers show
Phase 2  Correlation panel                  ── no history needed; ship anytime
Phase 3  M.E.T.A. regression refit          ── needs ~150+ resolved outcomes
Phase 4  MM-style execution upgrades        ── no history needed; ship anytime
```

Phases 2 and 4 need no historical data and can ship immediately. Phases 1
and 3 start collecting value the day Phase 0 ships and switch on
automatically as `n` crosses each threshold — which is itself the Bible's
core lesson (§3.1): **no estimate before the law of large numbers has had a
chance to work.**
