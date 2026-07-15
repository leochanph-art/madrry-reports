#!/usr/bin/env python3
"""MADRRY intraday watcher — Telegram push alerts during US regular trading hours.

Reads monitor_data.json (written by build_monitor_data.py in the morning pipeline),
watches every ticker with watch==true (capped at 80), polls Yahoo via ONE batched
yf.download per poll, and pushes compact alert batches over the same openclaw
Telegram session the morning script uses. Alert semantics mirror the client-side
engine in madrry_monitor.html (Component 4):

  1. TRIGGER     — cross: prev poll < entry and now >= entry.
                   First observation of the day already >= entry -> "GAP OVER TRIGGER".
  2. LINE BREAK  — same cross logic against `line` when line_state == "watch".
  4. MOVE        — |day change %| >= max(3, 1.5 * ADR); day change vs monitor_data close.
  5. VOLUME      — projected RVOL = (cum vol / avgVol50) / paceFraction >= 2.5,
                   only after 15 minutes of elapsed RTH.
(Types 3 NEAR and 6 STOP are page-only — they need user attention/starring context.)

Per-ticker per-day dedupe via /tmp/madrry_intraday_state_YYYYMMDD.json so restarts
don't re-alert. Lockfile so a duplicate launchd start (21:15 + 22:15 Taipei covers
both EDT and EST opens) exits immediately.

CLI:
  --once      single poll then exit (testing; ignores session gating + holiday guard)
  --dry-run   print alerts to stdout instead of sending Telegram (state not persisted)
  --data PATH override monitor_data.json location
"""

import argparse
import datetime as dt
import fcntl
import json
import math
import os
import subprocess
import sys
import time
import warnings
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")  # keep urllib3/yfinance noise out of the log

WS = "/Users/boundbythese/.openclaw/workspace"
DATA_DEFAULT = os.path.join(WS, "monitor_data.json")
LOG_PATH = "/tmp/madrry_intraday.log"
LOCK_PATH = "/tmp/madrry_intraday.lock"
OPENCLAW = "/opt/homebrew/bin/openclaw"
SESSION = "telegram:214505789"
ET = ZoneInfo("America/New_York")

WATCH_CAP = 80          # max tickers in the polled set
POLL_SEC = 90           # poll cadence during RTH
CHECK_SYMBOL = "SPY"    # holiday guard: no SPY bars by 09:45 ET -> holiday, exit
FAIL_WARN_AT = 10       # consecutive fully-failed polls before one Telegram warning
TG_CHUNK = 3400         # max chars per Telegram message (hard API limit is 4096)

# --- thresholds: MUST stay identical to madrry_monitor.html's alert engine ----
MOVE_MIN_PCT = 3.0      # MOVE fires at |dayChg%| >= max(3, 1.5*ADR)
MOVE_ADR_MULT = 1.5
RVOL_ALERT = 2.5        # VOLUME fires at projected RVOL >= 2.5
RVOL_MIN_ELAPSED = 15   # ...but only after 15 min of RTH
NEAR_PCT = 0.5          # (page-only NEAR threshold, kept here for reference)

# U-shaped intraday cumulative-volume curve (minutes into RTH -> fraction of a
# full day's volume normally traded by then). Simple documented approximation:
# ~12% in the first 30 min, slow midday, ~15% in the last 30 min. Same anchors
# as the monitor page so RVOL_projected matches across surfaces.
PACE_ANCHORS = [(0, 0.0), (30, 0.12), (60, 0.19), (120, 0.30), (195, 0.44),
                (270, 0.60), (330, 0.75), (360, 0.85), (390, 1.0)]


def pace_fraction(elapsed_min):
    e = max(0.0, min(390.0, float(elapsed_min)))
    for (m0, f0), (m1, f1) in zip(PACE_ANCHORS, PACE_ANCHORS[1:]):
        if e <= m1:
            return max(0.02, f0 + (f1 - f0) * (e - m0) / (m1 - m0))
    return 1.0


def log(msg):
    line = "[%s] %s" % (time.strftime("%F %T"), msg)
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def num(x):
    """Return float(x) or None (NaN guarded)."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# --- lockfile ----------------------------------------------------------------
_lock_fd = None  # kept open for process lifetime


def acquire_lock():
    global _lock_fd
    _lock_fd = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another instance holds %s — exiting (duplicate launchd start is expected)" % LOCK_PATH)
        sys.exit(0)
    _lock_fd.seek(0)
    _lock_fd.truncate()
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()


# --- state file (per-day dedupe, survives restarts) ---------------------------
def state_path(day_et):
    return "/tmp/madrry_intraday_state_%s.json" % day_et.strftime("%Y%m%d")


def load_state(day_et):
    try:
        with open(state_path(day_et)) as f:
            st = json.load(f)
        if isinstance(st, dict):
            st.setdefault("fired", {})
            st.setdefault("last_price", {})
            return st
    except (OSError, ValueError):
        pass
    return {"fired": {}, "last_price": {}, "armed_sent": False, "fail_warned": False}


def save_state(day_et, st, enabled=True):
    if not enabled:
        return
    try:
        tmp = state_path(day_et) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(st, f)
        os.replace(tmp, state_path(day_et))
    except OSError as e:
        log("state save failed (non-fatal): %s" % e)


# --- delivery -----------------------------------------------------------------
def send_telegram(text, dry_run):
    if dry_run:
        print("--- DRY-RUN message (not sent) ---")
        print(text)
        print("--- end message ---")
        return True
    try:
        r = subprocess.run([OPENCLAW, "sessions", "send", "--sessionKey", SESSION,
                            "--message", text],
                           capture_output=True, timeout=60)
        if r.returncode != 0:
            log("openclaw send failed rc=%s: %s" % (r.returncode, r.stderr.decode(errors="replace")[:200]))
            return False
        return True
    except Exception as e:  # noqa: BLE001 — delivery must never kill the watcher
        log("openclaw send exception (non-fatal): %s" % e)
        return False


# --- data ---------------------------------------------------------------------
def load_watchlist(path):
    """Returns (meta, tickers) or (None, []) when the data file is unusable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        log("cannot read %s: %s" % (path, e))
        return None, []
    seen, out = {}, []
    for tk in data.get("tickers") or []:
        t = str(tk.get("t") or "").upper().strip()
        if not t or not tk.get("watch"):
            continue
        if t in seen:
            # Same ticker can appear in two sections (e.g. minervini + coil) with
            # different entries, and only one record carries the LINE level. Merge:
            # watch the earliest (lowest) buy trigger, keep the line wherever it lives.
            base = seen[t]
            if num(tk.get("entry")) is not None and (
                    num(base.get("entry")) is None or tk["entry"] < base["entry"]):
                for k in ("entry", "stop", "risk", "sec", "tier", "status", "p2p"):
                    if tk.get(k) is not None:
                        base[k] = tk[k]
            if num(base.get("line")) is None and num(tk.get("line")) is not None:
                base["line"] = tk["line"]
                base["line_state"] = tk.get("line_state")
            base["top"] = bool(base.get("top") or tk.get("top"))
            continue
        if len(out) >= WATCH_CAP:
            continue           # cap new names, but keep merging dups of included ones
        rec = dict(tk)
        seen[t] = rec
        out.append(rec)
    return data, out


def fetch_avg_vol(symbols):
    """One batched 3mo daily download -> {ticker: avgVol50}. Non-fatal per ticker.
    avgVol50 = mean of the last up-to-50 non-null volumes EXCLUDING the final bar
    if it's today's (partial) — same rule as the quote worker's mode=base."""
    import yfinance as yf
    out = {}
    if not symbols:
        return out
    try:
        df = yf.download(symbols, period="3mo", interval="1d", threads=True,
                         progress=False, group_by="ticker")
    except Exception as e:  # noqa: BLE001
        log("avgVol50 base download failed (volume alerts disabled): %s" % e)
        return out
    if df is None or df.empty:
        log("avgVol50 base download empty (volume alerts disabled)")
        return out
    today = dt.datetime.now(ET).date()
    for t in symbols:
        try:
            sub = df[t] if hasattr(df.columns, "levels") else df
            vols = sub["Volume"].dropna()
            if len(vols) and vols.index[-1].date() >= today:
                vols = vols.iloc[:-1]          # drop today's partial bar
            vols = vols.iloc[-50:]
            if len(vols):
                v = num(vols.mean())
                if v and v > 0:
                    out[t] = v
        except Exception:  # noqa: BLE001 — per-ticker failures are non-fatal
            continue
    return out


def poll_quotes(symbols):
    """ONE batched yf.download per poll. Returns ({t: {price, vol, bars}}, err).
    Only bars stamped with today's ET date count — pre-open, Yahoo returns the
    prior session for period='1d', which must not feed the alert engine."""
    import yfinance as yf
    try:
        df = yf.download(symbols, period="1d", interval="5m", prepost=False,
                         threads=True, progress=False, group_by="ticker")
    except Exception as e:  # noqa: BLE001
        return {}, str(e)
    if df is None or df.empty:
        return {}, "empty frame"
    today = dt.datetime.now(ET).date()
    quotes = {}
    for t in symbols:
        try:
            sub = df[t] if hasattr(df.columns, "levels") else df
            sub = sub.dropna(subset=["Close"])
            if sub.empty:
                continue
            idx = sub.index
            try:
                dates = idx.tz_convert(ET).date
            except (TypeError, AttributeError):
                dates = idx.date  # tz-naive: assume exchange time
            mask = dates == today
            day = sub[mask]
            if day.empty:
                quotes[t] = {"price": None, "vol": None, "bars": 0}
                continue
            quotes[t] = {
                "price": num(day["Close"].iloc[-1]),
                "vol": num(day["Volume"].sum()),
                "bars": int(len(day)),
            }
        except Exception:  # noqa: BLE001 — per-ticker failures are non-fatal
            continue
    return quotes, None


# --- alert engine (mirrors madrry_monitor.html) --------------------------------
def fmt_line(parts):
    return " | ".join(p for p in parts if p)


def evaluate(tickers, quotes, avg_vol, state, elapsed_min):
    """Returns list of (dedupe_key, alert_line) for this poll; mutates state
    (fired/last_price). If a send later fails, the caller rolls the keys back
    out of state['fired'] so the alert retries on the next poll."""
    alerts = []
    fired = state["fired"]
    lastp = state["last_price"]

    for tk in tickers:
        t = tk["t"] if isinstance(tk.get("t"), str) else str(tk.get("t", "")).upper()
        q = quotes.get(t)
        if not q:
            continue
        price = q.get("price")
        if price is None or price <= 0:
            continue
        prev = num(lastp.get(t))
        entry = num(tk.get("entry"))
        line_at = num(tk.get("line"))
        stop = num(tk.get("stop"))
        adr = num(tk.get("adr"))
        risk = num(tk.get("risk"))
        close = num(tk.get("close"))
        tier = tk.get("tier")

        day_chg = (price / close - 1.0) * 100.0 if close and close > 0 else None

        rvol = None
        av = avg_vol.get(t)
        vol = q.get("vol")
        if av and vol and elapsed_min >= RVOL_MIN_ELAPSED:
            rvol = (vol / av) / pace_fraction(elapsed_min)

        chg_s = ("day %+.1f%%" % day_chg) if day_chg is not None else ""
        rvol_s = ("RVOL %.1f" % rvol) if rvol is not None else ""
        stop_s = ""
        if stop is not None:
            stop_s = "stop $%.2f" % stop + ((" (risk %.1f%%)" % risk) if risk is not None else "")
        tier_s = ("tier %s" % tier) if tier else ""

        # 1. TRIGGER — any un-alerted observation at/over entry fires: a cross, a
        # first-observation gap, or a retry after a failed-send rollback (requiring
        # a fresh cross would strand rolled-back alerts, since prev is already
        # above entry by the next poll). Once/day dedupe still holds via `fired`.
        if entry is not None and "%s|TRIGGER" % t not in fired and price >= entry:
            fired["%s|TRIGGER" % t] = time.time()
            head = "🎯 GAP OVER TRIGGER" if prev is None else "🎯 TRIGGER"
            alerts.append(("%s|TRIGGER" % t, fmt_line([
                "%s %s $%.2f ≥ entry $%.2f" % (head, t, price, entry),
                chg_s, rvol_s, stop_s, tier_s])))

        # 2. LINE BREAK — same semantics vs the watched resistance line
        if (line_at is not None and tk.get("line_state") == "watch"
                and "%s|LINE" % t not in fired and price >= line_at):
            fired["%s|LINE" % t] = time.time()
            head = "📈 LINE BREAK (gap)" if prev is None else "📈 LINE BREAK"
            alerts.append(("%s|LINE" % t, fmt_line([
                "%s %s $%.2f ≥ line $%.2f" % (head, t, price, line_at),
                chg_s, rvol_s, tier_s])))

        # 4. MOVE (unusual price move vs ADR)
        if day_chg is not None and "%s|MOVE" % t not in fired:
            thr = max(MOVE_MIN_PCT, MOVE_ADR_MULT * adr) if adr else MOVE_MIN_PCT
            if abs(day_chg) >= thr:
                fired["%s|MOVE" % t] = time.time()
                arrow = "▲" if day_chg > 0 else "▼"
                alerts.append(("%s|MOVE" % t, fmt_line([
                    "🚨 MOVE %s %s %+.1f%% (thr %.1f%%) $%.2f" % (arrow, t, day_chg, thr, price),
                    rvol_s, tier_s])))

        # 5. VOLUME (projected RVOL)
        if rvol is not None and rvol >= RVOL_ALERT and "%s|VOLUME" % t not in fired:
            fired["%s|VOLUME" % t] = time.time()
            alerts.append(("%s|VOLUME" % t, fmt_line([
                "🔊 VOLUME %s RVOL %.1f (≥%.1f) $%.2f" % (t, rvol, RVOL_ALERT, price),
                chg_s, tier_s])))

        lastp[t] = price
    return alerts


# --- session helpers ------------------------------------------------------------
def now_et():
    return dt.datetime.now(ET)


def et_at(day, h, m):
    return dt.datetime(day.year, day.month, day.day, h, m, tzinfo=ET)


def elapsed_rth_min(now):
    return (now - et_at(now.date(), 9, 30)).total_seconds() / 60.0


# --- main -----------------------------------------------------------------------
def run_poll(tickers, avg_vol, state, dry_run, day_et):
    syms = [tk["t"] for tk in tickers]
    quotes, err = poll_quotes(sorted(set(syms + [CHECK_SYMBOL])))
    if err is not None:
        return None, 0, err
    now = now_et()
    alerts = evaluate(tickers, quotes, avg_vol, state, elapsed_rth_min(now))
    if alerts:
        # Chunk to stay under Telegram's 4096-char hard limit (a market-wide gap
        # day can fire dozens of alerts in one poll). A failed chunk rolls its
        # dedupe keys back so those alerts retry on the next poll.
        header = "MADRRY intraday %s ET" % now.strftime("%H:%M")
        batches, cur_lines, cur_keys, cur_len = [], [], [], 0
        for key, line in alerts:
            if cur_lines and cur_len + len(line) + 1 > TG_CHUNK:
                batches.append((cur_lines, cur_keys))
                cur_lines, cur_keys, cur_len = [], [], 0
            cur_lines.append(line)
            cur_keys.append(key)
            cur_len += len(line) + 1
        batches.append((cur_lines, cur_keys))
        for i, (lines, keys) in enumerate(batches):
            head = header if len(batches) == 1 else "%s (%d/%d)" % (header, i + 1, len(batches))
            if not send_telegram(head + "\n" + "\n".join(lines), dry_run):
                for k in keys:
                    state["fired"].pop(k, None)
    live = sum(1 for t in syms if quotes.get(t, {}).get("price") is not None)
    spy_bars = quotes.get(CHECK_SYMBOL, {}).get("bars") or 0
    log("poll %s ET: quotes %d/%d with today bars (SPY bars=%d), alerts=%d"
        % (now.strftime("%H:%M:%S"), live, len(syms), spy_bars, len(alerts)))
    return quotes, len(alerts), None


def main():
    ap = argparse.ArgumentParser(description="MADRRY intraday Telegram watcher")
    ap.add_argument("--once", action="store_true",
                    help="single poll then exit (ignores session gating + holiday guard)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print alerts instead of sending Telegram; state not persisted")
    ap.add_argument("--data", default=DATA_DEFAULT, help="monitor_data.json path")
    args = ap.parse_args()

    persist = not args.dry_run
    if not args.once:
        acquire_lock()   # duplicate launchd start (21:15 + 22:15 Taipei) exits here

    day = now_et().date()
    meta, tickers = load_watchlist(args.data)
    if not tickers:
        log("no watchable tickers in %s — nothing to do (exit 0)" % args.data)
        if args.once:
            print("poll summary: 0 tickers (no monitor_data), alerts=0")
        return 0
    n_entries = sum(1 for tk in tickers if num(tk.get("entry")) is not None)
    log("watch set: %d tickers (%d with buy triggers), asof %s, data %s"
        % (len(tickers), n_entries, (meta or {}).get("asof"), args.data))

    avg_vol = fetch_avg_vol([tk["t"] for tk in tickers])
    log("avgVol50 ready for %d/%d tickers" % (len(avg_vol), len(tickers)))

    state = load_state(day)

    if args.once:
        quotes, n_alerts, err = run_poll(tickers, avg_vol, state, args.dry_run, day)
        if err is not None:
            log("poll failed: %s" % err)
            print("poll summary: %d tickers, poll FAILED (%s), alerts=0" % (len(tickers), err))
            return 0
        save_state(day, state, enabled=persist)
        live = sum(1 for tk in tickers if quotes.get(tk["t"], {}).get("price") is not None)
        note = "" if live else " (no today bars — market pre-open or closed)"
        print("poll summary: %d tickers, %d with live quotes%s, alerts=%d"
              % (len(tickers), live, note, n_alerts))
        return 0

    # --- session-gated loop ---
    now = now_et()
    if now.weekday() >= 5:
        log("weekend in ET — exiting")
        return 0
    open_929 = et_at(now.date(), 9, 29)
    close_1602 = et_at(now.date(), 16, 2)
    if now >= close_1602:
        log("started after 16:02 ET — exiting")
        return 0
    if now < open_929:
        wait = (open_929 - now).total_seconds()
        log("pre-open: sleeping %.0fs until 09:29 ET" % wait)
        time.sleep(wait)

    spy_seen = False
    consec_fail = 0
    while now_et() < close_1602:
        quotes, _n, err = run_poll(tickers, avg_vol, state, args.dry_run, day)
        if err is not None:
            consec_fail += 1
            log("poll failed (%d consecutive): %s" % (consec_fail, err))
            if consec_fail >= FAIL_WARN_AT and not state.get("fail_warned"):
                send_telegram("⚠️ MADRRY intraday watch: %d consecutive polls failed "
                              "(Yahoo/network?). Still trying. Log: %s"
                              % (consec_fail, LOG_PATH), args.dry_run)
                state["fail_warned"] = True
        else:
            consec_fail = 0
            if (quotes.get(CHECK_SYMBOL, {}).get("bars") or 0) > 0:
                spy_seen = True
                # Arm message only once the market is verifiably open (SPY has
                # today-bars) — never on a holiday, never on a dead network.
                if not state.get("armed_sent"):
                    ok = send_telegram(
                        "📡 MADRRY intraday watch live — %d tickers, %d with buy triggers. "
                        "Alerts: trigger/LINE/move/volume." % (len(tickers), n_entries),
                        args.dry_run)
                    if ok:
                        state["armed_sent"] = True
        save_state(day, state, enabled=persist)

        # Holiday guard: only a SUCCESSFUL poll (err is None) still showing zero
        # SPY today-bars at/after 09:45 ET means a US holiday. Failed polls
        # (Yahoo/network down) must never trip this — keep trying all day.
        if not spy_seen and err is None and now_et() >= et_at(day, 9, 45):
            log("no SPY bars by 09:45 ET — US market holiday; exiting")
            return 0

        remaining = (close_1602 - now_et()).total_seconds()
        if remaining <= 0:
            break
        time.sleep(min(POLL_SEC, max(1, remaining)))

    log("16:02 ET reached — session done, exiting")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(130)
