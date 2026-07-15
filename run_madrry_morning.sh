#!/bin/bash
# MADRRY morning loop — scan ▸ verify ▸ retry once ▸ publish ▸ alert on failure.
# Called by launchd (com.madrry.scanner) every Tue-Sat 06:00 Taipei.
# The gate: report regenerated in the last N minutes AND the run log's last
# DONE line shows errors=0. Fail the gate twice -> Telegram alert (no silent
# stale reports — the "loop that fails quietly" fix).

WS="/Users/boundbythese/.openclaw/workspace"
LOG="/tmp/madrry_scanner.log"
OPENCLAW="/opt/homebrew/bin/openclaw"
SESSION="telegram:214505789"
FRESH_MIN=10        # report must be newer than this many minutes to count

cd "$WS" || exit 1

scrub() { sed -E -e 's#(gh[pousr]|github_pat)_[A-Za-z0-9_]+#***TOKEN***#g' -e 's#(https?://)[^@/[:space:]]+@#\1***@#g'; }

run_scan() {
    python3 "$WS/madrry_html_scanner_v2.py" >> "$LOG" 2>&1
}

verify() {
    # 1) report file regenerated recently
    [ -n "$(find "$WS/madrry_report.html" -mmin -"$FRESH_MIN" 2>/dev/null)" ] || return 1
    # 2) the most recent DONE line reports zero errors
    tail -20 "$LOG" | grep -E "DONE in .*errors=0" >/dev/null || return 1
    return 0
}

notify() {
    "$OPENCLAW" sessions send --sessionKey "$SESSION" --message "$1" >/dev/null 2>&1
}

echo "=== $(date '+%F %T') morning loop start ===" >> "$LOG"

# --- Trilogy feed freshness gate (2026-07-10 fix) ---------------------------
# The Trilogy feed (feeds/trilogy_ready_to_buy.json) is produced by the
# trilogy-nightly agent task (07:45 Tue-Sat). That scheduler can fire hours
# late when its host isn't running in the morning (observed 2026-07-09 18:04
# and 2026-07-10 21:24 fires), which left the 08:16 report one bar stale.
# Gate: if the feed's asof predates the last completed US session, ask the
# gateway to run the pure-python pipeline now and wait for the export.
# On a morning after a US holiday the refresh is a safe no-op (the pipeline
# self-gates with "no new bar" and the old asof is correct).
TRILOGY_FEED="$WS/feeds/trilogy_ready_to_buy.json"
trilogy_fresh() {
    python3 - "$TRILOGY_FEED" <<'PY'
import json, sys, datetime as dt
try:
    asof = dt.date.fromisoformat(json.load(open(sys.argv[1]))["asof"])
except Exception:
    sys.exit(1)                      # missing/unreadable -> treat as stale
d = dt.date.today() - dt.timedelta(days=1)
while d.weekday() >= 5:              # weekend -> walk back to Friday
    d -= dt.timedelta(days=1)
sys.exit(0 if asof >= d else 1)
PY
}
if ! trilogy_fresh; then
    echo "--- $(date '+%F %T') trilogy feed STALE; requesting refresh ---" >> "$LOG"
    if "$OPENCLAW" sessions send --sessionKey "$SESSION" --message \
"MADRRY autotask: the Trilogy feed is stale (feeds/trilogy_ready_to_buy.json asof is behind the last US close; the 07:45 trilogy-nightly task has not run yet). Run exactly: cd \"/Users/boundbythese/Downloads/Chart learning project claude\" && python3 webapp/pipeline_nightly.py — then confirm the 'feed export' line appears at the end of webapp/reports/pipeline.log. Reply with one line: the new asof, or the exit reason (e.g. no new bar)." \
        >> "$LOG" 2>&1; then
        for _i in $(seq 1 24); do    # poll up to 12 min for the export to land
            sleep 30
            trilogy_fresh && break
        done
    fi
    if trilogy_fresh; then
        echo "--- $(date '+%F %T') trilogy feed refreshed ---" >> "$LOG"
    else
        echo "--- $(date '+%F %T') trilogy feed STILL stale; proceeding ---" >> "$LOG"
        notify "⚠️ MADRRY: Trilogy tab in today's report is one day STALE (feed refresh didn't land in time — trilogy-nightly scheduler likely down, or US holiday). Report generated anyway."
    fi
fi

run_scan
if ! verify; then
    echo "--- first run failed the gate; retrying in 60s ---" >> "$LOG"
    sleep 60
    run_scan
fi

if verify; then
    # Durable snapshot archive (IMPROVEMENT_PLAN Phase 0). The scanner prunes dated
    # snapshots to the newest 14, so cohorts roll off before their outcome window
    # matures. Copy every surviving dated snapshot into snapshots_archive/ (gitignored,
    # never pruned) so the ledger has a permanent source. Non-fatal.
    mkdir -p "$WS/snapshots_archive"
    if cp "$WS"/latest_setups_20*.json "$WS/snapshots_archive/" 2>/dev/null; then
        echo "--- $(date '+%F %T') snapshots archived ---" >> "$LOG"
    else
        echo "snapshot archive copy skipped (non-fatal)" >> "$LOG"
    fi

    # Forward Tier-A tracker: re-grade every tracked pick against fresh prices
    # (fresh snapshot just landed). Non-fatal — never block the report on it.
    echo "--- $(date '+%F %T') tier-A tracker ---" >> "$LOG"
    python3 "$WS/madrry_tier_a_tracker.py" >> "$LOG" 2>&1 || \
        echo "tier-A tracker errored (non-fatal)" >> "$LOG"

    # v4 forward tracker: freeze each pick's 45 v4 features + grade on the +2ADR
    # basis -> v4_tracking.json (the live, survivorship-free dataset for the re-fit).
    echo "--- $(date '+%F %T') v4 tracker ---" >> "$LOG"
    python3 "$WS/madrry_v4_tracker.py" >> "$LOG" 2>&1 || \
        echo "v4 tracker errored (non-fatal)" >> "$LOG"

    # Unified signal ledger (IMPROVEMENT_PLAN Phase 1d): ingest the day's snapshot into
    # madrry_ledger.db (all six sections), then trade-simulation-label OPEN signals via
    # yfinance. Non-fatal; a Python-internal deadline guards a hung fetch (no `timeout`
    # command on this Mac). Assert the DB mtime advanced and log loudly if not.
    echo "--- $(date '+%F %T') ledger ingest+label ---" >> "$LOG"
    _ledger_before=$(date -r "$WS/madrry_ledger.db" +%s 2>/dev/null || echo 0)
    if python3 "$WS/madrry_ledger.py" ingest >> "$LOG" 2>&1 && \
       python3 "$WS/madrry_ledger.py" label --source yfinance --deadline 420 >> "$LOG" 2>&1; then
        _ledger_after=$(date -r "$WS/madrry_ledger.db" +%s 2>/dev/null || echo 0)
        if [ "$_ledger_after" -gt "$_ledger_before" ]; then
            echo "ledger DB advanced ($_ledger_before -> $_ledger_after)" >> "$LOG"
        else
            echo "⚠️ ledger DB did NOT advance (mtime unchanged) — inspect /tmp/madrry_scanner.log" >> "$LOG"
        fi
    else
        echo "ledger ingest/label errored (non-fatal)" >> "$LOG"
    fi

    # WEEKLY v4 re-fit — ONLY on Saturday's run (which covers Friday's close, i.e. the
    # full trading week just ended). Gated/promote-only-if-better; any new model lands
    # NOW so the NEXT scan (Monday's data, Tue run) already ranks with the updated score.
    if [ "$(date +%u)" = "6" ]; then
        echo "--- $(date '+%F %T') v4 WEEKLY re-fit (Saturday) ---" >> "$LOG"
        python3 "$WS/madrry_v4_refit.py" >> "$LOG" 2>&1 || \
            echo "v4 weekly re-fit errored (non-fatal)" >> "$LOG"

        # WEEKLY universe data refresh (IMPROVEMENT_PLAN Phase 2 upkeep): extend every
        # universe ticker's daily bars to the just-closed session so stock_data_current/
        # + universe_manifest.json don't rot (the backtest + its reproduction test rely on
        # complete, current coverage). Runs post-close (08:16 Taipei covers the prior US
        # session), so bars are COMPLETE. Non-fatal, Python-internal deadline (no `timeout`).
        echo "--- $(date '+%F %T') weekly universe refresh (Saturday) ---" >> "$LOG"
        python3 "$WS/refresh_universe_data.py" --period 2y --deadline 900 >> "$LOG" 2>&1 || \
            echo "universe refresh errored (non-fatal)" >> "$LOG"

        # Phase-3: gated legacy-weight auto-apply (OOS-validated on tier_a_tracking.json; emits
        # 'wait' until >4 weeks of resolved history exists — correct, do not force) + weekly
        # calibration-table refresh (keyed to the ADOPTED ATR geometry, backtest-filled until
        # live matures). Both non-fatal; neither is in the auto-commit add-list.
        echo "--- $(date '+%F %T') meta auto-apply + calibration (Saturday) ---" >> "$LOG"
        python3 "$WS/madrry_meta_autoapply.py" --apply >> "$LOG" 2>&1 || \
            echo "meta auto-apply errored (non-fatal)" >> "$LOG"
        python3 "$WS/madrry_calibration.py" --spec atr_5day >> "$LOG" 2>&1 || \
            echo "calibration refresh errored (non-fatal)" >> "$LOG"
    fi

    # Intraday monitor data (madrry_monitor.html reads this on the Pages site)
    echo "--- $(date '+%F %T') monitor data export ---" >> "$LOG"
    python3 "$WS/build_monitor_data.py" >> "$LOG" 2>&1 || \
        echo "monitor data export errored (non-fatal)" >> "$LOG"

    zip -j /tmp/madrry_report.zip "$WS/madrry_report.html" >/dev/null 2>&1
    # Per-file add so a missing/late artifact never aborts the whole publish (git add
    # is atomic on a bad pathspec — one absent file would stage NOTHING).
    for f in madrry_report.html tier_a_tracking.json v4_tracking.json \
             meta_v4_model.json meta_v4_refit_history.jsonl model_archive \
             madrry_monitor.html monitor_data.json; do
        [ -e "$WS/$f" ] && git -C "$WS" add "$f" 2>&1 | scrub >> "$LOG"
    done
    git -C "$WS" commit -m "Daily report $(date +%Y-%m-%d) (auto)" 2>&1 | scrub >> "$LOG"
    git -C "$WS" push origin madrry-reports    2>&1 | scrub >> "$LOG"
    notify "MEDIA:/tmp/madrry_report.zip"
    echo "=== $(date '+%F %T') morning loop OK ===" >> "$LOG"
else
    ERRS=$(grep -iE "ERROR|Traceback" "$LOG" | tail -3 | scrub)
    notify "🚨 MADRRY morning scan FAILED twice — today's report is STALE (GitHub + Telegram not updated). Last errors: ${ERRS:-none captured}. Log: /tmp/madrry_scanner.log"
    echo "=== $(date '+%F %T') morning loop FAILED (alert sent) ===" >> "$LOG"
    exit 1
fi
