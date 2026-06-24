#!/bin/bash
# MADRRY weekend META recalibration — re-fit component weights from the week's
# accumulated Tier-A win/loss labels and (if they separate strictly better)
# write meta_weights.json, which the scanner loads on its next run.
# Called by launchd (com.madrry.recalibrate) every Sunday 10:00.
# Gated + bounded inside madrry_recalibrate_meta.py; every change is committed.

WS="/Users/boundbythese/.openclaw/workspace"
LOG="/tmp/madrry_recalibrate.log"
cd "$WS" || exit 1

echo "=== $(date '+%F %T') weekend recalibration start ===" >> "$LOG"

# Refresh the labels first (re-grade open names against the latest prices).
python3 "$WS/madrry_tier_a_tracker.py"      >> "$LOG" 2>&1 || \
    echo "tracker errored (continuing with existing labels)" >> "$LOG"

# Re-fit weights (gate + caps live inside the script).
python3 "$WS/madrry_recalibrate_meta.py"    >> "$LOG" 2>&1

# Commit whatever changed (weights, history, tracking db) so it is reviewable.
git -C "$WS" add meta_weights.json meta_weights_history.jsonl tier_a_tracking.json 2>>"$LOG"
git -C "$WS" commit -m "Weekend META recalibration $(date +%Y-%m-%d)" >> "$LOG" 2>&1 \
    && git -C "$WS" push origin madrry-reports >> "$LOG" 2>&1

echo "=== $(date '+%F %T') weekend recalibration done ===" >> "$LOG"
