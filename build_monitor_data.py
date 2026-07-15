#!/usr/bin/env python3
"""build_monitor_data.py — export monitor_data.json for the intraday monitor.

Reads latest_setups.json (the scanner's snapshot), drops excluded tickers,
computes the default-watch flag, and writes the compact monitor_data.json
contract that madrry_monitor.html + madrry_intraday_watch.py consume.

Stdlib only. Safe to run any time; it only reads the snapshot and aux text
files and rewrites monitor_data.json.
"""

import json
import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

WS = os.path.dirname(os.path.abspath(__file__))
SETUPS = os.path.join(WS, "latest_setups.json")
EXCLUDED = os.path.join(WS, "excluded_tickers.txt")
PROXY = os.path.join(WS, "live_price_proxy.txt")
OUT = os.path.join(WS, "monitor_data.json")
TAIPEI = ZoneInfo("Asia/Taipei")


def die(msg):
    print(f"build_monitor_data.py: {msg}", file=sys.stderr)
    sys.exit(1)


def num(v):
    """Return v as float if it is a real number, else None (NaN guarded)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if v != v:  # NaN
        return None
    return float(v)


def r2(v):
    v = num(v)
    return None if v is None else round(v, 2)


def load_excluded():
    excl = set()
    try:
        with open(EXCLUDED) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line:
                    excl.add(line.upper())
    except OSError:
        pass  # no exclusion file -> nothing excluded
    return excl


def load_proxy():
    try:
        with open(PROXY) as f:
            return f.read().strip()
    except OSError:
        return ""


def is_watch(row):
    """Default-watch rule from the monitor spec."""
    if row["tier"] in ("A+", "A"):
        return True
    if row["top"]:
        return True
    if row["status"] in ("COILED", "TRIGGERED_TODAY"):
        return True
    p2p = row["p2p"]
    if p2p is not None and 0 <= p2p <= 5:
        return True
    if row["line_state"] == "break":
        return True
    if row["line_state"] == "watch":
        line, close = row["line"], row["close"]
        if line is not None and close is not None and close > 0:
            dist = (line / close - 1) * 100
            if 0 <= dist <= 4:
                return True
    return False


def main():
    if not os.path.exists(SETUPS):
        die(f"{SETUPS} missing")
    try:
        with open(SETUPS) as f:
            setups = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        die(f"cannot parse {SETUPS}: {e}")
    if not isinstance(setups, list) or not setups:
        die(f"{SETUPS} has no tickers")

    excluded = load_excluded()
    rows = []
    for s in setups:
        if not isinstance(s, dict):
            continue
        ticker = str(s.get("ticker") or "").strip().upper()
        if not ticker or ticker in excluded:
            continue
        rs = num(s.get("rs_rating"))
        row = {
            "t": ticker,
            "sec": s.get("section"),
            "tier": s.get("tier"),
            "top": s.get("top_pick") is True,
            "close": r2(s.get("close")),
            "adr": r2(s.get("adr")),
            "entry": r2(s.get("entry")),
            "stop": r2(s.get("stop")),
            "risk": r2(s.get("risk_pct")),
            "line": r2(s.get("lbw_line_at")),
            "line_state": s.get("lbw_state"),
            "p2p": r2(s.get("pct_to_pivot")),
            "rs": None if rs is None else int(round(rs)),
            "theme": s.get("theme"),
            "status": s.get("status"),
        }
        row["watch"] = is_watch(row)
        rows.append(row)

    if not rows:
        die("0 tickers after exclusions")

    # watch first, then p2p ascending with nulls last
    rows.sort(key=lambda r: (not r["watch"],
                             r["p2p"] is None,
                             r["p2p"] if r["p2p"] is not None else 0.0))

    asof = datetime.fromtimestamp(os.path.getmtime(SETUPS), TAIPEI).strftime("%Y-%m-%d")
    watch_n = sum(1 for r in rows if r["watch"])
    payload = {
        "asof": asof,
        "generated": datetime.now(TAIPEI).isoformat(timespec="seconds"),
        "proxy": load_proxy(),
        "counts": {"total": len(rows), "watch": watch_n},
        "tickers": rows,
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    os.replace(tmp, OUT)
    print(f"monitor_data.json: {len(rows)} tickers ({watch_n} watch) asof {asof}")


if __name__ == "__main__":
    main()
