// MADRRY live-price proxy v2 — keyless Cloudflare Worker.
// Relays Yahoo's chart endpoint and adds CORS, so the report's
// "🔄 Refresh Prices" button and the intraday monitor page work from both
// the local file and the public GitHub link. No API key, no secrets.
//
// Deploy (free, ~5 min):
//   1. cloudflare.com -> sign up (free).
//   2. Workers & Pages -> Create application -> Create Worker.
//   3. Replace the default code with everything in this file -> Deploy.
//   4. Copy the worker URL (e.g. https://madrry-quotes.<you>.workers.dev).
//   5. Paste the worker URL into the monitor page's ⚙ settings (it persists
//      in the browser), and optionally into live_price_proxy.txt in the
//      workspace so the daily export bakes it into monitor_data.json.
//   6. Done. The monitor now fetches all tickers in ONE call per poll.
//
// Usage:
//   GET https://<worker>/?symbols=AAPL,QQQ,NVDA
//     -> {"result":[{"symbol":"AAPL","regularMarketPrice":291.07}, ...]}   (legacy, unchanged)
//   GET https://<worker>/?symbols=...&mode=full
//     -> {"result":[{"symbol","price","prevClose","dayHigh","dayLow","vol","state","t"}, ...]}
//   GET https://<worker>/?symbols=...&mode=base
//     -> {"result":[{"symbol","avgVol50","prevClose"}, ...]}   (from 3mo daily bars,
//        excluding today's partial bar; used once per session for RVOL baselines)
//
// Symbols are capped at 45 per request (Cloudflare free plan allows 50
// subrequests per request; the monitor page chunks accordingly).

export default {
  async fetch(request) {
    const cors = {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Cache-Control": "no-store",
      "Content-Type": "application/json",
    };
    if (request.method === "OPTIONS") return new Response(null, { headers: cors });

    const url = new URL(request.url);
    const mode = url.searchParams.get("mode") || "";
    const symbols = (url.searchParams.get("symbols") || "")
      .split(",").map((s) => s.trim()).filter(Boolean).slice(0, 45);
    if (!symbols.length) return new Response(JSON.stringify({ result: [] }), { headers: cors });

    const num = (v) => (typeof v === "number" && isFinite(v)) ? v : null;
    const chart = async (sym, range) => {
      const y = "https://query1.finance.yahoo.com/v8/finance/chart/" +
                encodeURIComponent(sym) + "?interval=1d&range=" + range;
      const r = await fetch(y, { headers: { "User-Agent": "Mozilla/5.0" } });
      const d = await r.json();
      return (d && d.chart && d.chart.result && d.chart.result[0]) || null;
    };

    let result;
    if (mode === "full") {
      // Full intraday snapshot from the 1d chart meta (one subrequest per symbol).
      result = await Promise.all(symbols.map(async (sym) => {
        try {
          const c = await chart(sym, "1d");
          const m = (c && c.meta) || {};
          return {
            symbol: sym,
            price: num(m.regularMarketPrice),
            prevClose: num(m.chartPreviousClose) != null ? num(m.chartPreviousClose) : num(m.previousClose),
            dayHigh: num(m.regularMarketDayHigh),
            dayLow: num(m.regularMarketDayLow),
            vol: num(m.regularMarketVolume),
            state: m.marketState != null ? m.marketState : null,
            t: num(m.regularMarketTime),
          };
        } catch (e) {
          return { symbol: sym, price: null, prevClose: null, dayHigh: null,
                   dayLow: null, vol: null, state: null, t: null };
        }
      }));
    } else if (mode === "base") {
      // 50-day average volume + last completed close from 3mo daily bars.
      // The final bar is excluded when it is TODAY in the exchange's timezone
      // (a live session bar is partial and would poison the volume baseline).
      result = await Promise.all(symbols.map(async (sym) => {
        try {
          const c = await chart(sym, "3mo");
          const m = (c && c.meta) || {};
          const ts = Array.isArray(c && c.timestamp) ? c.timestamp : [];
          const q = (c && c.indicators && c.indicators.quote && c.indicators.quote[0]) || {};
          const gmtoff = num(m.gmtoffset) != null ? m.gmtoffset : 0;
          const dayOf = (t) => Math.floor((t + gmtoff) / 86400);
          let n = ts.length;
          if (n && dayOf(ts[n - 1]) === dayOf(Math.floor(Date.now() / 1000))) n -= 1;
          const closes = (Array.isArray(q.close) ? q.close : []).slice(0, n);
          const vols = (Array.isArray(q.volume) ? q.volume : []).slice(0, n)
            .filter((v) => num(v) != null);
          const last50 = vols.slice(-50);
          const avgVol50 = last50.length
            ? Math.round(last50.reduce((a, b) => a + b, 0) / last50.length) : null;
          let prevClose = null;
          for (let i = closes.length - 1; i >= 0; i--) {
            if (num(closes[i]) != null) { prevClose = closes[i]; break; }
          }
          return { symbol: sym, avgVol50, prevClose };
        } catch (e) {
          return { symbol: sym, avgVol50: null, prevClose: null };
        }
      }));
    } else {
      // Legacy shape — the report's Refresh Prices button depends on this.
      result = await Promise.all(symbols.map(async (sym) => {
        try {
          const c = await chart(sym, "1d");
          const m = c && c.meta;
          return { symbol: sym, regularMarketPrice: (m && m.regularMarketPrice != null) ? m.regularMarketPrice : null };
        } catch (e) {
          return { symbol: sym, regularMarketPrice: null };
        }
      }));
    }

    return new Response(JSON.stringify({ result }), { headers: cors });
  },
};
