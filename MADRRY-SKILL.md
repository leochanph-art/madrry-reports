# MADRRY-SKILL — project knowledge for the MADRRY stock scanner
*(Read this first in any session touching madrry_html_scanner_v2.py — it saves
re-deriving everything from the code. Update it when thresholds/design change.)*

## What this is
A daily momentum stock scanner (J Law 超績投資客 / Minervini / Martin playbook
methodology) that generates `madrry_report.html`: market regime verdict, tiered
trade setups with entry/stop tickets, leadership reads, and self-graded
performance tracking. Owner trades from the report manually — **the system never
trades; keep it decision-support only.**

## File map (workspace = /Users/boundbythese/.openclaw/workspace)
| File | Role |
|---|---|
| `madrry_html_scanner_v2.py` | THE scanner (~3000 lines, Python 3.9 — no 3.10+ syntax) |
| `madrry_report.html` | stable latest-report alias (timestamped copies also written) |
| `madrry_trade_plan.py` | standalone live ORB/VWAP intraday plan |
| `latest_setups.json` | legacy alias: most recent run's picks |
| `latest_setups_YYYY-MM-DD.json` | picks keyed by **US-session data date** (14 kept) — the tracking source |
| `breakout_log.json` | per-session win/loss outcomes (90 days, merge-per-ticker) |
| `rs_history.json` | last-known RS percentile per ticker (carry-forward) |
| `breadth_history.json`, `hve_history.json` | breadth day-over-day, HVE event memory |
| `spy_qqq_extension_percentiles.csv` | historical extension→percentile lookup |
| `run_madrry_morning.sh` | morning loop: scan ▸ verify(errors=0+fresh) ▸ retry ▸ push+Telegram ▸ alert |
| `madrry_weekly_review.py` | Sunday review: win-rate by signal → WEEKLY_REVIEW.md + Telegram |
| `cloudflare_quote_worker.js` | keyless live-price proxy (deploy → set LIVE_PRICE_PROXY) |
| `JLaw_Trading_Guidebook.md` | the 30-video methodology distilled |

## Schedules (owner is in Taipei, UTC+8; 06:00 Taipei ≈ just after US close)
- **launchd `com.madrry.scanner`** Tue–Sat 06:00 → `run_madrry_morning.sh`
  (scan → verify → retry once → zip→Telegram → git push → alert on double failure)
- **launchd `com.madrry.weekly`** Sun 09:00 → weekly review
- **launchd `com.madrry.ibkrautostage`** Tue–Sat 06:15 → `madrry_ibkr_autostage.py`
  (SELF-OWNED, no OpenClaw; rebuilt 2026-06-17). Reads `top_picks_orders.json` →
  places each pick as a **transmit=False** order via ib_insync/TWS = a DRAFT the
  owner reviews + transmits. Fail-closed; gated by `IBKR_AUTOSTAGE_ENABLED`.
  **Requires TWS/IB Gateway running + API enabled (port 7496 live) when it fires.**
  Replaces the abandoned openclaw-cron staging path (gateway was unreliable).
- **crontab** 21:55 Mon–Fri: OpenClaw "Pre-Execution Trade Plan"; 04:05 Tue–Sat: post-market ping
- Apply plist changes with `launchctl unload` + `load` (edits don't hot-reload)

## Core design decisions (and why)
- **Data-date keying** (`asof` from QQQ's last daily bar, NOT wall clock): a 06:00
  Taipei run carries the just-closed US session. Same-session re-runs overwrite the
  same dated file, so tracking always grades the genuine previous session — fixes
  the "yesterday = my last run" bug (old archive silently never worked: missing
  shutil import).
- **Win/loss grading**: prior session's picks vs next bar — triggered (high≥entry)
  and close≥entry & not stopped = win; stopped or close<entry = loss; untriggered =
  "still coiled", not counted. Log merges per ticker (re-runs update, never erase).
- **Win-rate displays**: regime chip = rolling last-50 outcomes; watchlist line =
  accumulated all-time + 🚩HTF split (HTF tagged via `htf` flag in the log).
- **Full A- pool recorded** (uncapped, ~50-70/day) for grading; display capped at 25.

## Tier gates (coil scan; universe: >$7, vol>500k, close>SMA200, mcap>$2B, ADR≥2.0, off-52wk≤25%, within 10% of 9/21 EMA)
- **ETF leg (2026-07-15, USER: "screen ETFs as well")**: funds enter the SAME coil
  pipeline via a second TV query (`type=fund` + `typespecs=etf`); size gate =
  **AUM ≥ $2B** (funds have no market cap), all other Stage-1 filters identical.
  Stage-2 **RS 80+ is NOT exempted**: the Fred6725 feeds don't cover funds, so the
  scanner computes the same score locally (verified formula, 0.005 median err:
  `100·(1+strength)/(1+strength_SPY)`, strength `0.4q1+0.2q2+0.2q3+0.2q4` over
  63-bar windows) and percentile-ranks it against the stock cross-section — an ETF
  must rank RS≥80 *among stocks*. Uncomputable (young fund, <200 bars) ⇒ dropped.
  Rows carry `is_etf`/`etf_desc` (fund name = theme, sector chip = "ETF", cap col =
  AUM, 🧺 badge). **Report + IBKR drafts only** — tier-A tracker, v4 tracker and
  calibration all skip `is_etf` rows so stock-fit models never train on fund label
  geometry (low-ADR funds hit the +2·ADR win bar too easily; ADR is a feature).
  Excluded from HOT SECTORS and fundamentals prefetch.
- **A+**: 3-day tight flag · dist-to-EMA ≤4% · vol ≤55% · risk ≤3.5%
- **A**: tight-1d (range≤1×ADR) · dist ≤4% · vol ≤55% · risk ≤5%
- **A-**: dist ≤6% · vol not expanding (≤100% of prev day OR 50-day avg) · no tight-bar requirement (removed 2026-06-28)
- Sort: A+/A by M.E.T.A.; **A- by Multiple-Edge count, then M.E.T.A.**
- **HTF v2.4.1** (merged INTO A+, 🚩 badge, *under evaluation via weekly HTF
  win-rate*; owner-audited params 2026-06-12 — comments say what must never be
  changed): universe = cap>$3B, close>$10, day vol AND 20/30/60/90-bar avg vol
  each>500k, **ADR20>4% (never raise — 4.5 collapses)**, within 25% of 52wk hi,
  RS≥85 (full cross-section). Entry (all seven): thrust C/C[-40]>60% · 5-bar
  flag range<3.0×ADR20 · depth<30% of pole (load-bearing) · avg flag-bar range
  ≤1.0×ADR20 + max ≤1.25×ADR20 (**never tighten**) · flag vol < 10-bar mean ·
  5-bar cooldown per name. Ticket: entry=next open (close proxy), hard stop=
  close×(1−1.5×ADR20) gap-through fills at open, size=min(0.75%·eq/risk,
  20%·eq/px) eq=$100k; exits: 1st close ≥+40% vs 21EMA, ladder ≤3 legs at peak.
  ADR20 computed point-in-time from history, not the TV Volatility.D field.
- **New 52wk highs** (collapsed section): ADR>3 + history-verified new high; 🟢
  constructive / 🟡 mixed / 🔴 extended; ⭐ Persistent ≥5 / ⭐⭐ Relentless ≥9
  new-high weeks of last 13 (computed from 2y price each run, no log needed);
  persistence badges also attached to A-list rows (`attach_persistence`)
- **ANTS** (David Ryan accumulation; added 2026-06-14 — **display + Top-Picks
  boost only, NEVER the IBKR plan**): per-coil-name 0–5 ladder
  (NONE/MOM/MOM+VOL/MOM+PR/FULL/ELITE) + a trailing consecutive-bar "chain".
  Classic params (`ANTS_*` consts): 12 up-days/15, +20% price, +20% vol over the
  window, SMA10>20 trend leg, ELITE = rs_line(close/SPY) rising vs its 20-MA.
  `compute_ants()` (pure, vectorised, NaN-safe) + post-scan `attach_ants()` (2y
  batch + SPY, tags `ants_level/chain/label/ok` onto the 3 coil tiers). Sortable
  column in the coil tables (numeric `data-sort = level*1000+chain`; "—"=NONE/thin
  data sorts to bottom), chip in Top Picks cards, column in the markdown. FULL=+1
  / ELITE=+2 edge **only** via `_rank_top_picks(ants_boost=True)` (Top Picks);
  `write_order_plan` calls the default so the order plan stays byte-identical
  (verified: same `plan_id`, picks unchanged). NB: coils are tight by design, so
  most names read NONE/MOM — a high ANTS on a coil = accumulation + tight entry
  (the rare combo). ANTS rs_line is DISTINCT from the Fred6725 RS percentile.
  **2026-06-14 additions:** the ANTS column also shows a **3M sub-line** ("3M
  {peak}·{days}d" = peak level + active days over the last ~63 bars — prior
  accumulation is a positive even when today is quiet; sort = today·100000 +
  3M_peak·1000 + chain). **RS Line** (close/SPY): surfaced as ONE selective
  **🔵 RS Leader** badge = RS line at/near its 1-year high (within
  `ANTS_RS_HIGH_FRAC`=0.97 — strict new-highs too sparse for off-their-highs coil
  setups, ~2/62 vs ~8/62 near-high); `‹ Px` variant = RS near high while price is
  still < `ANTS_PX_LAG_FRAC`=0.95 of its own high (stealth, RS leads price).
  All from the same SPY-aligned `compute_ants` pass; markdown gets a 3M note + an
  RS Lead column. **Simplified 2026-06-15 (owner): dropped the redundant per-row
  RS sparkline + RS↑/↓ trend arrows — only genuine standouts light up.** Still
  display-only — IBKR order plan re-verified byte-identical (same plan_id).

## Industry-group RS (Fred6725 rs_industries.csv — added 2026-06-17)
`fetch_and_load_industry_rs()` loads the 144-group industry leaderboard (RS
percentile + 1M/3M + constituent tickers); `attach_industry_rs()` tags each pick
with its group's percentile (`ind_rs`/`ind_name`). Surfaced as the collapsed
**🏭 HOT INDUSTRY GROUPS** strip (top 12, 🎯 = a group your picks sit in) and a
per-row 🏭 badge in the coil/new-high tables. Display-only — NEVER the IBKR plan
(order plan re-verified byte-identical, same plan_id). `IND_RS_STRONG=90`.

## Audit (2026-06-17, 13-agent + adversarial-verify sweep) — fixes applied
Critical/High fixed: (C1) tz-naive TV stale-bar concat corrupted the index and
silently bypassed the HTF anti-phantom-fire guard → now tz-localized; (H1) win/
loss now gated PER-TICKER on its own bar date (was a batch-max flag → stale
self-grading); (H2) A+ overflow past 25 was silently deleted → now folded into
the tracked pool like A; (H3) failed breadth fetch scored as 2 phantom YELLOW →
now `breadth.get("ok")` guarded; (H4) U&R cleanup 5→10 calendar days (was
purging weekend day-4/5 setups before scan_ur). Sweep: scan_hve ZeroDivision
guard, RS/ index asof in ET, TV-dedup symbol-normalize (.↔-), AVWAP anchor
excl-today, dist-day vol=0 skip, _edge_count 0.0%-dist fix, dry-vs-prev 70→65,
dist-day card colors → regime thresholds, ext50 in new-high RED gate, RS final-
attempt sleep guard, NaN-bar skip in grading. NOT touched (calibration): M.E.T.A.
140-denominator + risk double-count, new-high min_periods. Backup:
`backup_pre_audit_fixes_20260617.py`.

## Regime (early-warning grid → GREEN/YELLOW/RED + allow_breakouts)
Scored: trend(QQQ/SPY 10>21) · breadth>50MA · >200MA · distribution days (≥6 r)
· climax ext (percentile-calibrated, P90 r) · leaders below 50DMA/no-new-high ·
topping-range break · T2108 (<40 r; **divergence = hard RED override**) ·
breakout win-rate (<40 r; <35 hard override; needs n≥8) · sector RS (≥2 leading
weak = r) · **VIX scored 🟡 above 20 or >15% spike; info-only when calm**.
Verdict: hard-override or ≥3 reds → RED; ≥1 red or ≥3 yellows → YELLOW.
allow_breakouts = not RED and dist<4 and no override. Runbar chip + Action Plan
reflect the verdict.

## Data sources & quirks
- **TradingView scan API** (`scanner.tradingview.com/america/scan`) — server-side filters; ~2s sleep between scans
- **Yahoo chart API / yfinance batch** — history; 6 AM Taipei = last bar is the just-closed US session
- **RS percentiles**: github Fred6725/rs-log → **use full `rs_stocks.csv`** (the
  `_1` file is only the top half!). US-only: ADR/OTC/spinoffs absent → "N/A" + 
  carry-forward from `rs_history.json` (shows `*` marker)
- **Barchart** $S5TW/$S5FI/$S5TH breadth — cookie-jar + XSRF token dance
- **Stockbee Google-Sheet CSV** — real T2108; 2nd-to-last col, S&P last col, rows newest-first
- **Live prices in report**: 🔄 button; Cloudflare Worker proxy if `LIVE_PRICE_PROXY`
  set, else per-symbol Yahoo-chart via allorigins (partial, rate-limited)

## Commands
```bash
cd /Users/boundbythese/.openclaw/workspace
python3 -m py_compile madrry_html_scanner_v2.py     # syntax gate
python3 madrry_html_scanner_v2.py                    # full run (~45s, expect "errors=0")
python3 madrry_weekly_review.py --no-notify          # review without Telegram
bash run_madrry_morning.sh                           # full morning loop (pushes + telegrams!)
```
Verify a run: `grep "DONE" /tmp/madrry_scanner.log | tail -1` → `errors=0`.

## IBKR draft orders (TOP PICKS → reviewable drafts; NEVER transmitted)
- The scanner writes `top_picks_orders.json` each run via `write_order_plan()` — deterministic
  INTENT only (no orders, no account data). ALL sizing/validation is in Python; the agent does
  ONLY `shares = floor(equity * shares_per_equity)`. Top 3 picks as LIMIT-at-breakout BUYs,
  **fail-closed gate** (`gated_out:true` unless regime ∈ {GREEN,YELLOW} AND breakouts not
  suppressed). Constants: `IBKR_TOP_N=3`, `IBKR_RISK_FRAC=0.005`, `IBKR_MAX_POS_FRAC=0.10`,
  `IBKR_MAX_SESSION_FRAC=0.35`, `IBKR_MIN_PRICE=1`, `IBKR_MIN_RPS_PCT=0.005`,
  `IBKR_ONE_PER_SECTOR=True`, `GATE_ALLOW_REGIMES={GREEN,YELLOW}`. Plan carries `plan_id`,
  `generated_at_utc`, `expected_session` for the runbook's freshness/dedup gates. Write failure
  → quarantines the file (do-not-stage).
- The brokerage is touched ONLY by the agent following **`STAGE_IBKR_DRAFTS.md`** (hardened
  per the 2026-06-14 safety audit: kill-switch file `IBKR_AUTOSTAGE_ENABLED`, mechanical
  freshness gate, single-consumption ledger `top_picks_staged.json`, atomic batch + rollback,
  full-payload readback, contract price-sanity). Runbook: kill-switch + tool preflight → read/
  freshness-gate plan → ledger dedup → `get_account_summary` → positions/orders/trades dedup →
  `search_contracts`+`get_price_snapshot` → `create_order_instruction` (DRAFT) → readback →
  Telegram links. **Unattended cron is OFF** (crontab edit blocked by OS sandbox, AND the
  kill-switch file is absent) until a supervised first run.
- **HARD RULE: drafts only.** `create_order_instruction` makes a *non-live* instruction the
  owner reviews + submits in IBKR. NEVER transmit/submit/execute an order; never click
  confirm on the owner's behalf. Fail-safe = create nothing + alert. The Python scanner
  never imports/calls IBKR. IBKR MCP server id: `3ced1ec3-…`; reconnect if "invalidated".
- IBKR limitation: MARKET/LIMIT only (no stop/bracket) → the protective stop is the owner's
  manual job at `stop_reference`; LIMIT-at-breakout can fill immediately if price < limit.

## Publishing & security
- Repo `leochanph-art/madrry-reports`, branch `madrry-reports`. **Only ever
  `git add madrry_report.html`** — NEVER `git add -A` (workspace holds personal
  files: MEMORY.md, USER.md, SOUL.md, openclaw.json with bot tokens).
- ⚠️ The git remote URL embeds a **plaintext GitHub PAT — owner should rotate it**;
  redact `ghp_*` from any printed output (`sed -E 's#ghp_[A-Za-z0-9]+#***#g'`).
- `openclaw.json` holds Telegram bot + gateway tokens in plaintext — never publish.

## Open decisions
- **HTF placement**: stays in A+ until the weekly review's HTF win-rate sample
  (n≥8) supports keep vs move-to-own-section.
- LIVE_PRICE_PROXY: Worker not yet deployed (button uses best-effort fallback).
- Committer identity is auto-generated; owner may set git user.name/email.
