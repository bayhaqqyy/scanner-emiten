import io
import os
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from flask import Flask, jsonify, render_template

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
SCALP_RSI_MIN = float(os.getenv("SCALP_RSI_MIN", "45"))
SCALP_RSI_MAX = float(os.getenv("SCALP_RSI_MAX", "75"))
SCALP_VOL_SPIKE = float(os.getenv("SCALP_VOL_SPIKE", "1.1"))
SCALP_ATR_LEN = int(os.getenv("SCALP_ATR_LEN", "14"))
SCALP_R_MULT = float(os.getenv("SCALP_R_MULT", "2.0"))
SCALP_MIN_SCORE = int(os.getenv("SCALP_MIN_SCORE", "3"))
SCALP_WATCH_SCORE = int(os.getenv("SCALP_WATCH_SCORE", "2"))

# Swing rules (1d data)
SWING_EMA_FAST = int(os.getenv("SWING_EMA_FAST", "20"))
SWING_EMA_SLOW = int(os.getenv("SWING_EMA_SLOW", "50"))
SWING_RSI_LEN = int(os.getenv("SWING_RSI_LEN", "14"))
SWING_RSI_MIN = float(os.getenv("SWING_RSI_MIN", "40"))
SWING_RSI_MAX = float(os.getenv("SWING_RSI_MAX", "70"))
SWING_VOL_SPIKE = float(os.getenv("SWING_VOL_SPIKE", "1.0"))
SWING_ATR_LEN = int(os.getenv("SWING_ATR_LEN", "14"))
SWING_SL_ATR = float(os.getenv("SWING_SL_ATR", "1.5"))
SWING_R_MULT = float(os.getenv("SWING_R_MULT", "2.0"))
SWING_MIN_SCORE = int(os.getenv("SWING_MIN_SCORE", "3"))
SWING_WATCH_SCORE = int(os.getenv("SWING_WATCH_SCORE", "2"))

app = Flask(__name__)

_lock = threading.Lock()
_scalping_cache = {"items": [], "updated_at": None, "error": None}
_swing_cache = {"items": [], "updated_at": None, "error": None}

_LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Asia/Jakarta"))


def _now_iso():
    now = datetime.now(_LOCAL_TZ)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


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


def _score_conditions(conditions):
    score = sum(1 for _, ok in conditions if ok)
    reasons = [label for label, ok in conditions if ok]
    return score, reasons


def scan_scalping():
    signals = []
    candidates = []
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

            momentum_3 = close_s.iloc[-1] > close_s.iloc[-2] > close_s.iloc[-3]
            conditions = [
                ("price_above_vwap", last_close > vwap_val),
                ("ema_trend", ema_fast > ema_slow),
                ("rsi_ok", SCALP_RSI_MIN <= rsi_val <= SCALP_RSI_MAX),
                ("vol_spike", vol_spike >= SCALP_VOL_SPIKE),
                ("momentum_3", momentum_3),
                ("green_bar", change >= 0),
            ]
            score, reasons = _score_conditions(conditions)

            if score >= SCALP_WATCH_SCORE:
                entry, sl, tp, risk = _trade_plan(
                    last_close, atr_val, sl_atr=1.0, r_mult=SCALP_R_MULT
                )
                item = {
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
                    "score": score,
                    "reasons": reasons,
                    "status": "signal" if score >= SCALP_MIN_SCORE else "watch",
                }
                candidates.append(item)
                if score >= SCALP_MIN_SCORE:
                    signals.append(item)
        except Exception:
            continue

        time.sleep(0.05)

    if signals:
        signals.sort(key=lambda x: (x["score"], x["vol_spike"]), reverse=True)
        return signals[:20]

    candidates.sort(key=lambda x: (x["score"], x["vol_spike"]), reverse=True)
    return candidates[:20]


def scan_swing():
    signals = []
    candidates = []
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

            above_fast = last_close > ema_fast
            up_3 = close_s.iloc[-1] > close_s.iloc[-2] > close_s.iloc[-3]
            conditions = [
                ("above_ema_fast", above_fast),
                ("ema_trend", ema_fast > ema_slow),
                ("rsi_ok", SWING_RSI_MIN <= rsi_val <= SWING_RSI_MAX),
                ("vol_spike", vol_spike >= SWING_VOL_SPIKE),
                ("momentum_3", up_3),
                ("green_bar", change >= 0),
            ]
            score, reasons = _score_conditions(conditions)

            if score >= SWING_WATCH_SCORE:
                entry, sl, tp, risk = _trade_plan(
                    last_close, atr_val, sl_atr=SWING_SL_ATR, r_mult=SWING_R_MULT
                )
                item = {
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
                    "score": score,
                    "reasons": reasons,
                    "status": "signal" if score >= SWING_MIN_SCORE else "watch",
                }
                candidates.append(item)
                if score >= SWING_MIN_SCORE:
                    signals.append(item)
        except Exception:
            continue

        time.sleep(0.05)

    if signals:
        signals.sort(key=lambda x: (x["score"], x["change_pct"]), reverse=True)
        return signals[:20]

    candidates.sort(key=lambda x: (x["score"], x["change_pct"]), reverse=True)
    return candidates[:20]


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
    return render_template(
        "index.html",
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
