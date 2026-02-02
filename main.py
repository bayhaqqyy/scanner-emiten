import io
import os
import threading
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template_string

APP_TITLE = "Scanner Emiten: Scalping & Swing"
LOCAL_TICKERS_CSV = "all.csv"
REMOTE_TICKERS_CSV = "https://raw.githubusercontent.com/carmensyva/list-emiten/refs/heads/main/all.csv"

# Tunable parameters (override via environment variables)
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "60"))
SCALPING_SCAN_SECONDS = int(os.getenv("SCALPING_SCAN_SECONDS", "60"))
SWING_SCAN_SECONDS = int(os.getenv("SWING_SCAN_SECONDS", "300"))
UI_POLL_SECONDS = int(os.getenv("UI_POLL_SECONDS", "1"))

# Scalping rules (1m data)
SCALP_EMA_FAST = int(os.getenv("SCALP_EMA_FAST", "9"))
SCALP_EMA_SLOW = int(os.getenv("SCALP_EMA_SLOW", "21"))
SCALP_RSI_LEN = int(os.getenv("SCALP_RSI_LEN", "14"))
SCALP_RSI_MIN = float(os.getenv("SCALP_RSI_MIN", "50"))
SCALP_RSI_MAX = float(os.getenv("SCALP_RSI_MAX", "70"))
SCALP_VOL_SPIKE = float(os.getenv("SCALP_VOL_SPIKE", "1.5"))
SCALP_ATR_LEN = int(os.getenv("SCALP_ATR_LEN", "14"))
SCALP_R_MULT = float(os.getenv("SCALP_R_MULT", "2.0"))

# Swing rules (1d data)
SWING_EMA_FAST = int(os.getenv("SWING_EMA_FAST", "20"))
SWING_EMA_SLOW = int(os.getenv("SWING_EMA_SLOW", "50"))
SWING_RSI_LEN = int(os.getenv("SWING_RSI_LEN", "14"))
SWING_RSI_MIN = float(os.getenv("SWING_RSI_MIN", "45"))
SWING_RSI_MAX = float(os.getenv("SWING_RSI_MAX", "65"))
SWING_VOL_SPIKE = float(os.getenv("SWING_VOL_SPIKE", "1.2"))
SWING_ATR_LEN = int(os.getenv("SWING_ATR_LEN", "14"))
SWING_SL_ATR = float(os.getenv("SWING_SL_ATR", "1.5"))
SWING_R_MULT = float(os.getenv("SWING_R_MULT", "2.0"))

app = Flask(__name__)

_lock = threading.Lock()
_scalping_cache = {"items": [], "updated_at": None, "error": None}
_swing_cache = {"items": [], "updated_at": None, "error": None}


def _now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_tickers():
    if os.path.exists(LOCAL_TICKERS_CSV):
        df_emiten = pd.read_csv(LOCAL_TICKERS_CSV)
    else:
        response = requests.get(REMOTE_TICKERS_CSV, timeout=15)
        response.raise_for_status()
        df_emiten = pd.read_csv(io.StringIO(response.text))

    tickers = df_emiten.iloc[:, 0].dropna().astype(str).str.strip().tolist()
    return tickers[:MAX_TICKERS]


def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()


def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(high, low, close, length):
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def vwap(high, low, close, volume):
    typical_price = (high + low + close) / 3
    pv = typical_price * volume
    return pv.cumsum() / volume.cumsum()


def _format_price(value):
    if value is None or pd.isna(value):
        return None
    return float(round(value, 2))


def _trade_plan(entry, atr_value, sl_atr, r_mult):
    if atr_value is None or pd.isna(atr_value) or atr_value <= 0:
        return None, None, None, None
    sl = entry - (atr_value * sl_atr)
    risk = entry - sl
    tp = entry + (risk * r_mult)
    return _format_price(entry), _format_price(sl), _format_price(tp), _format_price(risk)


def scan_scalping():
    results = []
    tickers = load_tickers()

    for symbol in tickers:
        ticker_jk = f"{symbol}.JK"
        try:
            df = yf.download(
                ticker_jk,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=True,
            )
            if df.empty or len(df) < max(SCALP_EMA_SLOW, SCALP_RSI_LEN, SCALP_ATR_LEN) + 5:
                continue

            close_s = df["Close"].squeeze()
            high_s = df["High"].squeeze()
            low_s = df["Low"].squeeze()
            vol_s = df["Volume"].squeeze()

            last_close = float(close_s.iloc[-1])
            prev_close = float(close_s.iloc[-2])
            change = ((last_close - prev_close) / prev_close) * 100

            ema_fast = ema(close_s, SCALP_EMA_FAST).iloc[-1]
            ema_slow = ema(close_s, SCALP_EMA_SLOW).iloc[-1]
            rsi_val = rsi(close_s, SCALP_RSI_LEN).iloc[-1]
            vwap_val = vwap(high_s, low_s, close_s, vol_s).iloc[-1]
            vwap_diff = (last_close / vwap_val - 1) * 100
            avg_vol = vol_s.tail(20).mean()
            vol_spike = (vol_s.iloc[-1] / avg_vol) if avg_vol and avg_vol > 0 else 0
            atr_val = atr(high_s, low_s, close_s, SCALP_ATR_LEN).iloc[-1]

            if (
                last_close > vwap_val
                and ema_fast > ema_slow
                and SCALP_RSI_MIN <= rsi_val <= SCALP_RSI_MAX
                and vol_spike >= SCALP_VOL_SPIKE
            ):
                entry, sl, tp, risk = _trade_plan(
                    last_close, atr_val, sl_atr=1.0, r_mult=SCALP_R_MULT
                )
                results.append(
                    {
                        "ticker": symbol,
                        "close": _format_price(last_close),
                        "change_pct": round(change, 2),
                        "vwap_diff_pct": round(vwap_diff, 2),
                        "ema_fast": _format_price(ema_fast),
                        "ema_slow": _format_price(ema_slow),
                        "rsi": round(float(rsi_val), 2),
                        "vol_spike": round(float(vol_spike), 2),
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk": risk,
                    }
                )
        except Exception:
            continue

        time.sleep(0.05)

    results.sort(key=lambda x: x["vol_spike"], reverse=True)
    return results[:20]


def scan_swing():
    results = []
    tickers = load_tickers()

    for symbol in tickers:
        ticker_jk = f"{symbol}.JK"
        try:
            df = yf.download(
                ticker_jk,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df.empty or len(df) < max(SWING_EMA_SLOW, SWING_RSI_LEN, SWING_ATR_LEN) + 5:
                continue

            close_s = df["Close"].squeeze()
            high_s = df["High"].squeeze()
            low_s = df["Low"].squeeze()
            vol_s = df["Volume"].squeeze()

            last_close = float(close_s.iloc[-1])
            prev_close = float(close_s.iloc[-2])
            change = ((last_close - prev_close) / prev_close) * 100

            ema_fast = ema(close_s, SWING_EMA_FAST).iloc[-1]
            ema_slow = ema(close_s, SWING_EMA_SLOW).iloc[-1]
            rsi_val = rsi(close_s, SWING_RSI_LEN).iloc[-1]
            avg_vol = vol_s.tail(20).mean()
            vol_spike = (vol_s.iloc[-1] / avg_vol) if avg_vol and avg_vol > 0 else 0
            atr_val = atr(high_s, low_s, close_s, SWING_ATR_LEN).iloc[-1]

            if (
                last_close > ema_fast
                and ema_fast > ema_slow
                and SWING_RSI_MIN <= rsi_val <= SWING_RSI_MAX
                and vol_spike >= SWING_VOL_SPIKE
            ):
                entry, sl, tp, risk = _trade_plan(
                    last_close, atr_val, sl_atr=SWING_SL_ATR, r_mult=SWING_R_MULT
                )
                results.append(
                    {
                        "ticker": symbol,
                        "close": _format_price(last_close),
                        "change_pct": round(change, 2),
                        "ema_fast": _format_price(ema_fast),
                        "ema_slow": _format_price(ema_slow),
                        "rsi": round(float(rsi_val), 2),
                        "vol_spike": round(float(vol_spike), 2),
                        "entry": entry,
                        "sl": sl,
                        "tp": tp,
                        "risk": risk,
                    }
                )
        except Exception:
            continue

        time.sleep(0.05)

    results.sort(key=lambda x: x["change_pct"], reverse=True)
    return results[:20]


def _scalping_worker():
    while True:
        try:
            items = scan_scalping()
            with _lock:
                _scalping_cache["items"] = items
                _scalping_cache["updated_at"] = _now_iso()
                _scalping_cache["error"] = None
        except Exception as exc:
            with _lock:
                _scalping_cache["error"] = str(exc)
        time.sleep(SCALPING_SCAN_SECONDS)


def _swing_worker():
    while True:
        try:
            items = scan_swing()
            with _lock:
                _swing_cache["items"] = items
                _swing_cache["updated_at"] = _now_iso()
                _swing_cache["error"] = None
        except Exception as exc:
            with _lock:
                _swing_cache["error"] = str(exc)
        time.sleep(SWING_SCAN_SECONDS)


def start_workers():
    scalping_thread = threading.Thread(target=_scalping_worker, daemon=True)
    swing_thread = threading.Thread(target=_swing_worker, daemon=True)
    scalping_thread.start()
    swing_thread.start()


@app.route("/")
def index():
    return render_template_string(
        """
<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ title }}</title>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f1d1a;
      --bg2: #132622;
      --panel: #f3f1e9;
      --ink: #142015;
      --accent: #f99b1d;
      --muted: #6c7a72;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Space Grotesk", sans-serif;
      color: var(--ink);
      background: radial-gradient(circle at 20% 20%, #204336 0%, var(--bg) 40%, #0a1412 100%);
      min-height: 100vh;
    }
    header {
      padding: 28px 6vw 18px;
      color: #f7f4ea;
    }
    h1 {
      font-family: "Fraunces", serif;
      font-size: clamp(2rem, 3vw, 3rem);
      margin: 0 0 8px;
      letter-spacing: 0.5px;
    }
    .sub {
      color: #c7d1ca;
      font-size: 0.95rem;
    }
    main {
      padding: 0 6vw 40px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 20px;
    }
    .panel {
      background: var(--panel);
      border-radius: 18px;
      padding: 18px 18px 14px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.35);
      position: relative;
      overflow: hidden;
    }
    .panel::after {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(249,155,29,0.12), rgba(19,38,34,0.0));
      pointer-events: none;
    }
    h2 {
      margin: 0 0 6px;
      font-size: 1.2rem;
    }
    .meta {
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }
    th, td {
      text-align: left;
      padding: 8px 6px;
      border-bottom: 1px solid #e2dfd5;
    }
    th {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #59645c;
    }
    .badge {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: var(--accent);
      color: #121212;
      font-weight: 600;
      font-size: 0.75rem;
    }
    .empty {
      color: var(--muted);
      font-size: 0.9rem;
      padding: 12px 0;
    }
    footer {
      color: #c7d1ca;
      padding: 18px 6vw 30px;
      font-size: 0.85rem;
    }
    @media (max-width: 720px) {
      table { font-size: 0.8rem; }
      th, td { padding: 6px 4px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>{{ title }}</h1>
    <div class="sub">Auto-refresh UI tiap {{ ui_poll }} detik • Universe: max {{ max_tickers }} emiten</div>
  </header>
  <main>
    <section class="panel">
      <h2>Scalping Signals <span class="badge">1m</span></h2>
      <div class="meta">Update terakhir: <span id="scalping-updated">-</span></div>
      <div id="scalping-table"></div>
    </section>
    <section class="panel">
      <h2>Swing Signals <span class="badge">1d</span></h2>
      <div class="meta">Update terakhir: <span id="swing-updated">-</span></div>
      <div id="swing-table"></div>
    </section>
  </main>
  <footer>
    Rule ringkas: Scalping = harga > VWAP, EMA9 > EMA21, RSI 50-70, volume spike ≥ 1.5x.
    Swing = harga > EMA20 > EMA50, RSI 45-65, volume spike ≥ 1.2x.
  </footer>
  <script>
    const fmt = (v) => (v === null || v === undefined) ? "-" : v;

    function buildTable(items) {
      if (!items || items.length === 0) {
        return '<div class="empty">Belum ada sinyal sesuai rule.</div>';
      }
      const header = `
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Close</th>
              <th>Change%</th>
              <th>RSI</th>
              <th>Vol</th>
              <th>Entry</th>
              <th>SL</th>
              <th>TP</th>
            </tr>
          </thead>
          <tbody>
      `;
      const rows = items.map((item) => `
        <tr>
          <td>${item.ticker}</td>
          <td>${fmt(item.close)}</td>
          <td>${fmt(item.change_pct)}</td>
          <td>${fmt(item.rsi)}</td>
          <td>${fmt(item.vol_spike)}</td>
          <td>${fmt(item.entry)}</td>
          <td>${fmt(item.sl)}</td>
          <td>${fmt(item.tp)}</td>
        </tr>
      `).join("");
      return header + rows + "</tbody></table>";
    }

    async function refresh() {
      const [scalping, swing] = await Promise.all([
        fetch("/api/scalping").then(r => r.json()),
        fetch("/api/swing").then(r => r.json())
      ]);

      document.getElementById("scalping-updated").textContent = scalping.updated_at || "-";
      document.getElementById("swing-updated").textContent = swing.updated_at || "-";
      document.getElementById("scalping-table").innerHTML = buildTable(scalping.items);
      document.getElementById("swing-table").innerHTML = buildTable(swing.items);
    }

    refresh();
    setInterval(refresh, {{ ui_poll }} * 1000);
  </script>
</body>
</html>
        """,
        title=APP_TITLE,
        max_tickers=MAX_TICKERS,
        ui_poll=UI_POLL_SECONDS,
    )


@app.route("/api/scalping")
def api_scalping():
    with _lock:
        payload = dict(_scalping_cache)
    return jsonify(payload)


@app.route("/api/swing")
def api_swing():
    with _lock:
        payload = dict(_swing_cache)
    return jsonify(payload)


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok"})


def main():
    start_workers()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)


if __name__ == "__main__":
    main()
