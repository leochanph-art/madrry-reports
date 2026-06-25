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

scrub() { sed -E 's#ghp_[A-Za-z0-9]+#***TOKEN***#g'; }

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

run_scan
if ! verify; then
    echo "--- first run failed the gate; retrying in 60s ---" >> "$LOG"
    sleep 60
    run_scan
fi

if verify; then
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

    zip -j /tmp/madrry_report.zip "$WS/madrry_report.html" >/dev/null 2>&1
    # Per-file add so a missing/late artifact never aborts the whole publish (git add
    # is atomic on a bad pathspec — one absent file would stage NOTHING).
    for f in madrry_report.html tier_a_tracking.json v4_tracking.json; do
        [ -f "$WS/$f" ] && git -C "$WS" add "$f" 2>&1 | scrub >> "$LOG"
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
