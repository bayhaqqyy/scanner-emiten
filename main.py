import io
import json
import os
import re
import threading
import time
from datetime import datetime
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv()

APP_TITLE = "Scanner Emiten: Scalping & Swing"
LOCAL_TICKERS_CSV = "all.csv"
REMOTE_TICKERS_CSV = "https://raw.githubusercontent.com/carmensyva/list-emiten/refs/heads/main/all.csv"

# Tunable parameters (override via environment variables)
MAX_TICKERS = int(os.getenv("MAX_TICKERS", "60"))
SCALPING_SCAN_SECONDS = int(os.getenv("SCALPING_SCAN_SECONDS", "60"))
SWING_SCAN_SECONDS = int(os.getenv("SWING_SCAN_SECONDS", "300"))
BSJP_SCAN_SECONDS = int(os.getenv("BSJP_SCAN_SECONDS", "600"))
BPJS_SCAN_SECONDS = int(os.getenv("BPJS_SCAN_SECONDS", "600"))
BSJP_VOL_MULT = float(os.getenv("BSJP_VOL_MULT", "1.5"))
BSJP_VALUE_MIN = float(os.getenv("BSJP_VALUE_MIN", "10000000000"))
BSJP_RET_MIN = float(os.getenv("BSJP_RET_MIN", "0.5"))
BPJS_VALUE_MIN = float(os.getenv("BPJS_VALUE_MIN", "2000000000"))
UI_POLL_SECONDS = int(os.getenv("UI_POLL_SECONDS", "1"))
AI_ENABLED = os.getenv("AI_ENABLED", "1") == "1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free").strip()
OPENROUTER_REFERER = os.getenv("OPENROUTER_REFERER", "").strip()
OPENROUTER_TITLE = os.getenv("OPENROUTER_TITLE", "").strip()
AI_COOLDOWN_MINUTES = int(os.getenv("AI_COOLDOWN_MINUTES", "20"))
AI_MAX_ITEMS = int(os.getenv("AI_MAX_ITEMS", "5"))

# Scalping rules (1m data)
SCALP_EMA_FAST = int(os.getenv("SCALP_EMA_FAST", "9"))
SCALP_EMA_SLOW = int(os.getenv("SCALP_EMA_SLOW", "21"))
SCALP_RSI_LEN = int(os.getenv("SCALP_RSI_LEN", "14"))
SCALP_RSI_MIN = float(os.getenv("SCALP_RSI_MIN", "50"))
SCALP_RSI_MAX = float(os.getenv("SCALP_RSI_MAX", "70"))
SCALP_VOL_SPIKE = float(os.getenv("SCALP_VOL_SPIKE", "1.3"))
SCALP_ATR_LEN = int(os.getenv("SCALP_ATR_LEN", "14"))
SCALP_R_MULT = float(os.getenv("SCALP_R_MULT", "2.0"))
SCALP_BREAKOUT_LOOKBACK = int(os.getenv("SCALP_BREAKOUT_LOOKBACK", "5"))
SCALP_SOLID_BODY_MIN = float(os.getenv("SCALP_SOLID_BODY_MIN", "0.6"))
SCALP_TX_VALUE_MIN = float(os.getenv("SCALP_TX_VALUE_MIN", "5000000000"))
SCALP_MIN_SCORE = int(os.getenv("SCALP_MIN_SCORE", "7"))
SCALP_WATCH_SCORE = int(os.getenv("SCALP_WATCH_SCORE", "5"))

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
SWING_BREAKOUT_LOOKBACK = int(os.getenv("SWING_BREAKOUT_LOOKBACK", "20"))
SWING_SOLID_BODY_MIN = float(os.getenv("SWING_SOLID_BODY_MIN", "0.55"))
SWING_UPPER_SHADOW_MAX = float(os.getenv("SWING_UPPER_SHADOW_MAX", "0.4"))
SWING_EMA_SLOPE_LOOKBACK = int(os.getenv("SWING_EMA_SLOPE_LOOKBACK", "3"))
SWING_MIN_SCORE = int(os.getenv("SWING_MIN_SCORE", "6"))
SWING_WATCH_SCORE = int(os.getenv("SWING_WATCH_SCORE", "4"))

CA_CACHE_MINUTES = int(os.getenv("CA_CACHE_MINUTES", "15"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

_lock = threading.Lock()
_scalping_cache = {"items": [], "updated_at": None, "error": None}
_swing_cache = {"items": [], "updated_at": None, "error": None}
_bsjp_cache = {"items": [], "updated_at": None, "error": None}
_bpjs_cache = {"items": [], "updated_at": None, "error": None}
_ai_cache = {}
_ca_cache = {"items": [], "updated_at": None, "error": None, "at": None}
_scalp_call_base = {}
_scalp_day = None
_scalp_active = {}
_scalp_feedback = {"outcomes": [], "loss_rate": 0.0, "tighten": False}
_scalp_ai_bust = False
SCALP_ACTIVE_LIMIT = int(os.getenv("SCALP_ACTIVE_LIMIT", "10"))
SCALP_RESET_HOUR = int(os.getenv("SCALP_RESET_HOUR", "8"))
SCALP_FEEDBACK_WINDOW = int(os.getenv("SCALP_FEEDBACK_WINDOW", "20"))
SCALP_LOSS_RATE_TIGHTEN = float(os.getenv("SCALP_LOSS_RATE_TIGHTEN", "0.6"))
SCALP_TIGHTEN_SCORE = int(os.getenv("SCALP_TIGHTEN_SCORE", "1"))
SCALP_TIGHTEN_VOL = float(os.getenv("SCALP_TIGHTEN_VOL", "0.1"))
SCALP_TIGHTEN_TX = float(os.getenv("SCALP_TIGHTEN_TX", "1000000000"))

_LOCAL_TZ = ZoneInfo(os.getenv("LOCAL_TZ", "Asia/Jakarta"))


def _now_iso():
    now = datetime.now(_LOCAL_TZ)
    return now.strftime("%Y-%m-%d %H:%M:%S %Z")


def _today_label():
    now = datetime.now(_LOCAL_TZ)
    months = [
        "Januari",
        "Februari",
        "Maret",
        "April",
        "Mei",
        "Juni",
        "Juli",
        "Agustus",
        "September",
        "Oktober",
        "November",
        "Desember",
    ]
    return f"{now.day:02d} {months[now.month - 1]} {now.year}"


def _is_market_open(now=None):
    now = now or datetime.now(_LOCAL_TZ)
    weekday = now.weekday()  # Mon=0 ... Sun=6
    if weekday >= 5:
        return False
    open_time = now.replace(hour=9, minute=0, second=0, microsecond=0)
    close_hour = 16
    close_minute = 30 if weekday == 4 else 0  # Friday 16:30
    close_time = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
    return open_time <= now <= close_time


def _sleep_until_next_open():
    now = datetime.now(_LOCAL_TZ)
    weekday = now.weekday()
    if weekday >= 5:
        days_ahead = (7 - weekday)  # to next Monday
    else:
        close_hour = 16
        close_minute = 30 if weekday == 4 else 0
        close_time = now.replace(hour=close_hour, minute=close_minute, second=0, microsecond=0)
        if now < close_time:
            time.sleep(60)
            return
        days_ahead = 1
        if weekday == 4:
            days_ahead = 3
    next_open = (now + pd.Timedelta(days=days_ahead)).replace(
        hour=9, minute=0, second=0, microsecond=0
    )
    sleep_seconds = max(60, int((next_open - now).total_seconds()))
    time.sleep(sleep_seconds)


def _today_key():
    return datetime.now(_LOCAL_TZ).strftime("%Y-%m-%d")


def _reset_scalping_daily():
    global _scalp_day
    today = _today_key()
    now = datetime.now(_LOCAL_TZ)
    if _scalp_day != today and now.hour >= SCALP_RESET_HOUR:
        _scalp_day = today
        _scalp_call_base.clear()
        _scalp_active.clear()
        _scalp_feedback["outcomes"] = []
        _scalp_feedback["loss_rate"] = 0.0
        _scalp_feedback["tighten"] = False
        for key in list(_ai_cache.keys()):
            if key.startswith("scalping:"):
                _ai_cache.pop(key, None)


def _record_scalp_outcome(result):
    global _scalp_ai_bust
    outcomes = _scalp_feedback["outcomes"]
    outcomes.append(result)
    if len(outcomes) > SCALP_FEEDBACK_WINDOW:
        outcomes.pop(0)
    losses = sum(1 for o in outcomes if o == "sl")
    _scalp_feedback["loss_rate"] = losses / len(outcomes) if outcomes else 0.0
    tighten = _scalp_feedback["loss_rate"] >= SCALP_LOSS_RATE_TIGHTEN
    if tighten and not _scalp_feedback["tighten"]:
        _scalp_ai_bust = True
    _scalp_feedback["tighten"] = tighten


def _scalp_thresholds():
    if not _scalp_feedback["tighten"]:
        return SCALP_MIN_SCORE, SCALP_WATCH_SCORE, SCALP_VOL_SPIKE, SCALP_TX_VALUE_MIN
    return (
        SCALP_MIN_SCORE + SCALP_TIGHTEN_SCORE,
        SCALP_WATCH_SCORE + SCALP_TIGHTEN_SCORE,
        SCALP_VOL_SPIKE + SCALP_TIGHTEN_VOL,
        SCALP_TX_VALUE_MIN + SCALP_TIGHTEN_TX,
    )


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


def _ai_cache_key(kind, key):
    return f"{kind}:{key}"


def _ai_allowed():
    return AI_ENABLED and bool(OPENROUTER_API_KEY)


def _ai_recent(entry):
    if not entry:
        return False
    age = datetime.now(_LOCAL_TZ) - entry["at"]
    return age.total_seconds() < AI_COOLDOWN_MINUTES * 60


def _ai_prompt(item, kind):
    base = (
        "You are a trading assistant. Provide a short, structured analysis in JSON only. "
        "Fields: bias (bullish|neutral|bearish), setup, risk, confidence (1-10), summary, action. "
        "Answer in Bahasa Indonesia. Keep it concise and grounded in the numbers.\n\n"
    )
    if kind in ("scalping", "swing"):
        return (
            base
            + f"Context: {kind} signal on IDX.\n"
            + f"Ticker: {item['ticker']}\n"
            + f"Close: {item['close']}\n"
            + f"Change%: {item['change_pct']}\n"
            + f"RSI: {item['rsi']}\n"
            + f"Vol spike: {item['vol_spike']}\n"
            + f"Entry: {item['entry']}, SL: {item['sl']}\n"
            + f"TP1: {item.get('tp1')}, TP2: {item.get('tp2')}, TP3: {item.get('tp3')}\n"
            + f"Score: {item['score']}\n"
        )
    if kind in ("bsjp", "bpjs"):
        return (
            base
            + f"Context: {kind.upper()} screener on IDX.\n"
            + f"Ticker: {item.get('ticker')}\n"
            + f"Close: {item.get('close')}\n"
            + f"Change%: {item.get('change_pct')}\n"
            + f"Volume: {item.get('volume')}\n"
            + f"Value: {item.get('tx_value')}\n"
            + "Explain why it qualifies and whether it looks good for the next session.\n"
        )
    if kind == "corporate_action":
        return (
            base
            + "Context: Corporate action news (IDX/KSEI).\n"
            + f"Title: {item.get('title')}\n"
            + f"Date: {item.get('date')}\n"
            + f"Tag: {item.get('tag')}\n"
            + f"Content: {item.get('content')}\n"
            + "Summarize what the news is about and the likely impact. Provide action guidance.\n"
        )
    if kind == "fundamental":
        return (
            base
            + "Context: Fundamental snapshot for IDX.\n"
            + f"Ticker: {item.get('ticker')}\n"
            + f"Market Cap: {item.get('market_cap')}\n"
            + f"PBV: {item.get('pbv')}\n"
            + f"PER: {item.get('per')}\n"
            + f"EPS: {item.get('eps')}\n"
            + f"ROE: {item.get('roe')}\n"
            + f"DER: {item.get('der')}\n"
            + f"Net Profit Margin: {item.get('npm')}\n"
            + "Give a short valuation comment and action guidance.\n"
        )
    return base + "Context: General analysis.\n"


def _ai_analyze(item, kind, key_override=None):
    if not _ai_allowed():
        return None
    key = key_override if key_override is not None else item.get("ticker", "unknown")
    cache_key = _ai_cache_key(kind, key)
    cached = _ai_cache.get(cache_key)
    if _ai_recent(cached):
        return cached["data"]

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_REFERER
    if OPENROUTER_TITLE:
        headers["X-Title"] = OPENROUTER_TITLE
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "Return valid JSON only. Use Bahasa Indonesia."},
            {"role": "user", "content": _ai_prompt(item, kind)},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.2,
        "max_tokens": 240,
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
        try:
            parsed = json.loads(content)
        except Exception:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else None
        if not isinstance(parsed, dict):
            return None
        _ai_cache[cache_key] = {"at": datetime.now(_LOCAL_TZ), "data": parsed}
        return parsed
    except Exception:
        return None


def _attach_ai(items, kind):
    if not _ai_allowed():
        return items
    count = 0
    global _scalp_ai_bust
    if kind == "scalping" and _scalp_ai_bust:
        for key in list(_ai_cache.keys()):
            if key.startswith("scalping:"):
                _ai_cache.pop(key, None)
        _scalp_ai_bust = False
    for item in items:
        if count >= AI_MAX_ITEMS:
            break
        if item.get("status") == "watch":
            pnl = item.get("pnl_pct")
            if pnl is not None and pnl <= 0:
                cache_key = _ai_cache_key(kind, item.get("ticker", "unknown"))
                _ai_cache.pop(cache_key, None)
        item["ai"] = _ai_analyze(item, kind)
        count += 1
    return items


def _fetch_html(url):
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def _normalize_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_date(text):
    if not text:
        return None
    match = re.search(r"\d{1,2}\s+[A-Za-z]+?\s+\d{4}", text)
    return match.group(0) if match else None


def _parse_ksei_table(html, base_url, category):
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if not table:
        return []

    items = []
    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 2:
            continue
        text_cells = [_normalize_space(cell.get_text(" ", strip=True)) for cell in cells]
        link = row.find("a")
        href = urljoin(base_url, link.get("href")) if link and link.get("href") else None
        title = text_cells[1] if len(text_cells) > 1 else text_cells[0]
        date = text_cells[-1]
        if not title or title.lower().startswith("perihal"):
            continue
        items.append(
            {
                "title": title,
                "date": date,
                "source": "KSEI Corporate Action",
                "category": category,
                "url": href,
            }
        )
    return items


def _extract_article_data(url):
    try:
        html = _fetch_html(url)
    except Exception:
        return {"content": None, "image": None}
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    og_image = soup.find("meta", property="og:image")
    image_url = og_image.get("content") if og_image else None
    text = _normalize_space(soup.get_text(" ", strip=True))
    if not text:
        return {"content": None, "image": image_url}
    return {"content": text[:2000], "image": image_url}


def _parse_ksei_today(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    items = []
    current_date = None

    for node in soup.find_all(["h2", "h3", "a", "p", "li"]):
        if node.name in ("h2", "h3"):
            date_text = _extract_date(node.get_text(" ", strip=True))
            if date_text:
                current_date = date_text
            continue

        if node.name == "a":
            title = _normalize_space(node.get_text(" ", strip=True))
            href = node.get("href")
            if not title:
                continue
            items.append(
                {
                    "title": title,
                    "date": current_date,
                    "source": "KSEI Today's Announcements",
                    "category": "Announcements",
                    "url": urljoin(base_url, href) if href else None,
                }
            )
    return items


def fetch_corporate_actions():
    now = datetime.now(_LOCAL_TZ)
    if _ca_cache["at"] and (now - _ca_cache["at"]).total_seconds() < CA_CACHE_MINUTES * 60:
        return _ca_cache["items"]

    rights_url = "https://web.ksei.co.id/publications/corporate-action-schedules/rights-distribution"
    today_url = "https://web.ksei.co.id/todays-announcements?setLocale=id-ID"
    keywords = [
        "hmetd",
        "right issue",
        "rights issue",
        "pmhmetd",
        "pmthmetd",
        "private placement",
        "penambahan modal",
    ]

    items = []
    try:
        html = _fetch_html(rights_url)
        items.extend(_parse_ksei_table(html, rights_url, "Rights Distribution (HMETD)"))
    except Exception:
        pass

    try:
        html = _fetch_html(today_url)
        today_items = _parse_ksei_today(html, today_url)
        for item in today_items:
            title_lc = (item["title"] or "").lower()
            if any(key in title_lc for key in keywords):
                items.append(item)
    except Exception:
        pass

    for item in items:
        title_lc = (item["title"] or "").lower()
        if "private placement" in title_lc or "pmthmetd" in title_lc:
            item["tag"] = "Private Placement"
        elif "right issue" in title_lc or "rights issue" in title_lc or "hmetd" in title_lc:
            item["tag"] = "Right Issue"
        elif "penambahan modal" in title_lc:
            item["tag"] = "Penambahan Modal"
        else:
            item["tag"] = "Corporate Action"
        if item.get("url"):
            article = _extract_article_data(item["url"])
            item["content"] = article.get("content")
            item["image"] = article.get("image")

    _ca_cache["items"] = items[:30]
    _ca_cache["updated_at"] = _now_iso()
    _ca_cache["at"] = now
    _ca_cache["error"] = None
    return _ca_cache["items"]


def _attach_ai_corporate(items):
    if not _ai_allowed():
        return items
    count = 0
    for item in items:
        if count >= AI_MAX_ITEMS:
            break
        key = item.get("url") or item.get("title") or str(count)
        item["ai"] = _ai_analyze(item, "corporate_action", key_override=key)
        count += 1
    return items


def _safe_div(numerator, denominator):
    if numerator is None or denominator in (None, 0):
        return None
    try:
        return float(numerator) / float(denominator)
    except Exception:
        return None


def _latest_value(df, keys):
    if df is None or df.empty:
        return None
    for key in keys:
        if key in df.index:
            value = df.loc[key].iloc[0]
            return float(value) if value is not None else None
    return None


def get_fundamentals(ticker, pe_wajar=None, target_market_cap=None):
    ticker = ticker.upper().replace(".JK", "").strip()
    ticker_jk = f"{ticker}.JK"
    tkr = None
    info = {}
    try:
        tkr = yf.Ticker(ticker_jk)
        info = tkr.info or {}
    except Exception:
        info = {}

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares = info.get("sharesOutstanding")
    market_cap = info.get("marketCap") or (_safe_div(price * shares, 1) if price and shares else None)

    fin = None
    bal = None
    try:
        if tkr is None:
            tkr = yf.Ticker(ticker_jk)
        fin = tkr.financials
        bal = tkr.balance_sheet
    except Exception:
        fin = None
        bal = None

    net_income = _latest_value(
        fin,
        ["Net Income", "Net Income Common Stockholders", "Net Income Applicable To Common Shares"],
    )
    revenue = _latest_value(fin, ["Total Revenue", "Total Revenue"])
    equity = _latest_value(
        bal,
        ["Total Stockholder Equity", "Total Equity Gross Minority Interest", "Total Equity"],
    )
    liabilities = _latest_value(
        bal,
        ["Total Liab", "Total Liabilities Net Minority Interest", "Total Liabilities"],
    )

    book_value_per_share = info.get("bookValue") or _safe_div(equity, shares)
    eps = info.get("trailingEps") or _safe_div(net_income, shares)

    pbv = _safe_div(price, book_value_per_share)
    per = _safe_div(price, eps)
    bvps = book_value_per_share
    npm = _safe_div(net_income, revenue)
    der = _safe_div(liabilities, equity)
    roe = _safe_div(net_income, equity)

    fair_value_market_cap = None
    if pe_wajar is not None and net_income is not None:
        fair_value_market_cap = float(pe_wajar) * float(net_income)

    fair_price_from_target_mc = None
    if target_market_cap is not None and shares not in (None, 0):
        fair_price_from_target_mc = float(target_market_cap) / float(shares)

    ratios = {
        "market_cap": _format_price(market_cap),
        "pbv": _format_price(pbv),
        "bvps": _format_price(bvps),
        "per": _format_price(per),
        "eps": _format_price(eps),
        "net_profit_margin": _format_price(npm * 100 if npm is not None else None),
        "der": _format_price(der),
        "roe": _format_price(roe),
        "fair_value_market_cap": _format_price(fair_value_market_cap),
        "fair_price_target_mc": _format_price(fair_price_from_target_mc),
        "pbv_band": None,
    }

    inputs = {
        "price": _format_price(price),
        "shares_outstanding": shares,
        "net_income": _format_price(net_income),
        "revenue": _format_price(revenue),
        "equity": _format_price(equity),
        "liabilities": _format_price(liabilities),
        "pe_wajar": _format_price(pe_wajar) if pe_wajar is not None else None,
        "target_market_cap": _format_price(target_market_cap) if target_market_cap is not None else None,
    }

    return {
        "ticker": ticker,
        "as_of": _now_iso(),
        "inputs": inputs,
        "ratios": ratios,
        "source": "yfinance",
        "note": "Net profit margin dan ROE ditampilkan dalam persen. Nilai wajar dihitung dari PE wajar x laba bersih.",
    }


def scan_scalping():
    signals = []
    candidates = []
    tickers = load_tickers()
    _reset_scalping_daily()
    active_keys = list(_scalp_active.keys())
    active_mode = len(active_keys) >= SCALP_ACTIVE_LIMIT
    min_score, watch_score, vol_spike_min, tx_value_min = _scalp_thresholds()

    scan_list = active_keys if active_mode else tickers

    for symbol in scan_list:
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
            open_s = df["Open"].squeeze()
            vol_s = df["Volume"].squeeze()

            last_close = float(close_s.iloc[-1])
            prev_close = float(close_s.iloc[-2])
            change = ((last_close - prev_close) / prev_close) * 100

            ema_fast_series = ema(close_s, SCALP_EMA_FAST)
            ema_slow_series = ema(close_s, SCALP_EMA_SLOW)
            ema_fast = ema_fast_series.iloc[-1]
            ema_slow = ema_slow_series.iloc[-1]
            rsi_val = rsi(close_s, SCALP_RSI_LEN).iloc[-1]
            vwap_series = vwap(high_s, low_s, close_s, vol_s)
            vwap_val = vwap_series.iloc[-1]
            vwap_diff = (last_close / vwap_val - 1) * 100
            avg_vol = vol_s.tail(20).mean()
            vol_spike = (vol_s.iloc[-1] / avg_vol) if avg_vol and avg_vol > 0 else 0
            atr_val = atr(high_s, low_s, close_s, SCALP_ATR_LEN).iloc[-1]

            tx_value = float((close_s * vol_s).sum())
            vwap_rising = vwap_series.iloc[-1] > vwap_series.iloc[-2]
            ema_gap = ema_fast - ema_slow
            ema_gap_prev = ema_fast_series.iloc[-2] - ema_slow_series.iloc[-2]
            ema_widening = ema_gap > ema_gap_prev

            lookback = max(2, SCALP_BREAKOUT_LOOKBACK)
            recent_high = high_s.iloc[-lookback - 1 : -1].max()
            break_high = last_close > recent_high if pd.notna(recent_high) else False

            def solid_green(idx):
                body = abs(close_s.iloc[idx] - open_s.iloc[idx])
                rng = high_s.iloc[idx] - low_s.iloc[idx]
                if rng <= 0:
                    return False
                return close_s.iloc[idx] > open_s.iloc[idx] and (body / rng) >= SCALP_SOLID_BODY_MIN

            momentum_2 = solid_green(-1) and solid_green(-2)

            conditions = [
                ("tx_value_min", tx_value >= tx_value_min),
                ("price_above_vwap", last_close > vwap_val),
                ("vwap_rising", vwap_rising),
                ("ema_trend", ema_fast > ema_slow),
                ("ema_widening", ema_widening),
                ("rsi_ok", SCALP_RSI_MIN <= rsi_val <= SCALP_RSI_MAX),
                ("vol_spike", vol_spike >= vol_spike_min),
                ("break_high_minor", break_high),
                ("momentum_2_green", momentum_2),
            ]
            score, reasons = _score_conditions(conditions)

            if symbol in _scalp_active:
                item = _scalp_active[symbol]
                item["close"] = _format_price(last_close)
                item["change_pct"] = round(change, 2)
                item["rsi"] = round(float(rsi_val), 2)
                item["vol_spike"] = round(float(vol_spike), 2)
                item["tx_value"] = _format_price(tx_value)
                item["entry_now"] = _format_price(last_close)
                entry_plan = item.get("entry_plan")
                item["pnl_pct"] = _format_price(
                    ((last_close - entry_plan) / entry_plan) * 100 if entry_plan else None
                )
                if entry_plan:
                    item["tp1"] = _format_price(entry_plan * 1.03)
                    item["tp2"] = _format_price(entry_plan * 1.05)
                    item["tp3"] = _format_price(entry_plan * 1.10)
                # close position if TP1 or SL hit
                if item.get("tp1") and last_close >= item["tp1"]:
                    _record_scalp_outcome("tp1")
                    _scalp_active.pop(symbol, None)
                    _scalp_call_base.pop(symbol, None)
                    continue
                if item.get("sl") and last_close <= item["sl"]:
                    _record_scalp_outcome("sl")
                    _scalp_active.pop(symbol, None)
                    _scalp_call_base.pop(symbol, None)
                    continue
                continue

            if not active_mode and score >= watch_score and len(_scalp_active) < SCALP_ACTIVE_LIMIT:
                entry, sl, tp, risk = _trade_plan(
                    last_close, atr_val, sl_atr=1.0, r_mult=SCALP_R_MULT
                )
                call_key = symbol
                if call_key not in _scalp_call_base:
                    _scalp_call_base[call_key] = {
                        "entry_price": last_close,
                        "at": _now_iso(),
                    }
                call_base = _scalp_call_base[call_key]
                entry_plan = call_base["entry_price"]
                pnl_pct = ((last_close - entry_plan) / entry_plan) * 100 if entry_plan else None

                tp1 = _format_price(entry_plan * 1.03) if entry_plan else None
                tp2 = _format_price(entry_plan * 1.05) if entry_plan else None
                tp3 = _format_price(entry_plan * 1.10) if entry_plan else None

                item = {
                    "ticker": symbol,
                    "close": _format_price(last_close),
                    "change_pct": round(change, 2),
                    "vwap_diff_pct": round(vwap_diff, 2),
                    "ema_fast": _format_price(ema_fast),
                    "ema_slow": _format_price(ema_slow),
                    "rsi": round(float(rsi_val), 2),
                    "vol_spike": round(float(vol_spike), 2),
                    "tx_value": _format_price(tx_value),
                    "entry_plan": _format_price(entry_plan),
                    "entry_plan_at": call_base["at"],
                    "pnl_pct": _format_price(pnl_pct),
                    "entry_now": _format_price(last_close),
                    "entry": entry,
                    "sl": sl,
                    "tp": tp,
                    "tp1": tp1,
                    "tp2": tp2,
                    "tp3": tp3,
                    "risk": risk,
                    "score": score,
                    "reasons": reasons,
                    "status": "signal" if score >= min_score else "watch",
                }
                _scalp_active[symbol] = item
        except Exception:
            continue

        time.sleep(0.05)

    results = list(_scalp_active.values())
    results.sort(key=lambda x: (x["score"], x.get("vol_spike", 0)), reverse=True)
    return results[:SCALP_ACTIVE_LIMIT]


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
            open_s = df["Open"].squeeze()
            vol_s = df["Volume"].squeeze()

            last_close = float(close_s.iloc[-1])
            prev_close = float(close_s.iloc[-2])
            change = ((last_close - prev_close) / prev_close) * 100

            ema_fast_series = ema(close_s, SWING_EMA_FAST)
            ema_slow_series = ema(close_s, SWING_EMA_SLOW)
            ema_fast = ema_fast_series.iloc[-1]
            ema_slow = ema_slow_series.iloc[-1]
            rsi_val = rsi(close_s, SWING_RSI_LEN).iloc[-1]
            avg_vol = vol_s.tail(20).mean()
            vol_spike = (vol_s.iloc[-1] / avg_vol) if avg_vol and avg_vol > 0 else 0
            atr_val = atr(high_s, low_s, close_s, SWING_ATR_LEN).iloc[-1]

            above_fast = last_close > ema_fast
            ema_slope = ema_fast - ema_fast_series.iloc[-SWING_EMA_SLOPE_LOOKBACK]

            lookback = max(5, SWING_BREAKOUT_LOOKBACK)
            recent_high = high_s.iloc[-lookback - 1 : -1].max()
            break_res = last_close > recent_high if pd.notna(recent_high) else False

            body = abs(close_s.iloc[-1] - open_s.iloc[-1])
            rng = high_s.iloc[-1] - low_s.iloc[-1]
            body_ratio = body / rng if rng > 0 else 0
            upper_shadow = high_s.iloc[-1] - max(close_s.iloc[-1], open_s.iloc[-1])
            upper_shadow_ratio = upper_shadow / rng if rng > 0 else 0

            conditions = [
                ("above_ema_fast", above_fast),
                ("ema_trend", ema_fast > ema_slow),
                ("ema_slope_up", ema_slope > 0),
                ("rsi_ok", SWING_RSI_MIN <= rsi_val <= SWING_RSI_MAX),
                ("vol_spike", vol_spike >= SWING_VOL_SPIKE),
                ("break_resistance", break_res),
                ("solid_body", body_ratio >= SWING_SOLID_BODY_MIN),
                ("no_long_upper", upper_shadow_ratio <= SWING_UPPER_SHADOW_MAX),
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


def scan_bpjs():
    results = []
    tickers = load_tickers()
    for symbol in tickers:
        ticker_jk = f"{symbol}.JK"
        try:
            df = yf.download(
                ticker_jk,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df.empty or len(df) < 2:
                continue
            close_s = df["Close"].squeeze()
            open_s = df["Open"].squeeze()
            vol_s = df["Volume"].squeeze()
            last_close = float(close_s.iloc[-1])
            prev_close = float(close_s.iloc[-2])
            last_open = float(open_s.iloc[-1])
            last_vol = float(vol_s.iloc[-1])
            change = ((last_close - prev_close) / prev_close) * 100
            tx_value = last_close * last_vol

            if last_close >= prev_close and last_close >= last_open and tx_value > BPJS_VALUE_MIN:
                results.append(
                    {
                        "ticker": symbol,
                        "close": _format_price(last_close),
                        "change_pct": round(change, 2),
                        "volume": _format_price(last_vol),
                        "tx_value": _format_price(tx_value),
                    }
                )
        except Exception:
            continue
        time.sleep(0.05)
    results.sort(key=lambda x: x["tx_value"], reverse=True)
    return results[:30]


def scan_bsjp():
    results = []
    tickers = load_tickers()
    for symbol in tickers:
        ticker_jk = f"{symbol}.JK"
        try:
            df = yf.download(
                ticker_jk,
                period="30d",
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df.empty or len(df) < 21:
                continue
            close_s = df["Close"].squeeze()
            vol_s = df["Volume"].squeeze()
            last_close = float(close_s.iloc[-1])
            prev_close = float(close_s.iloc[-2])
            last_vol = float(vol_s.iloc[-1])
            vol_ma20 = float(vol_s.tail(20).mean())
            change = ((last_close - prev_close) / prev_close) * 100
            tx_value = last_close * last_vol

            if last_vol >= BSJP_VOL_MULT * vol_ma20 and tx_value > BSJP_VALUE_MIN and change > BSJP_RET_MIN:
                results.append(
                    {
                        "ticker": symbol,
                        "close": _format_price(last_close),
                        "change_pct": round(change, 2),
                        "volume": _format_price(last_vol),
                        "tx_value": _format_price(tx_value),
                    }
                )
        except Exception:
            continue
        time.sleep(0.05)
    results.sort(key=lambda x: x["tx_value"], reverse=True)
    return results[:30]


def _scalping_worker():
    while True:
        try:
            if not _is_market_open():
                _sleep_until_next_open()
                continue
            items = scan_scalping()
            items = _attach_ai(items, "scalping")
            with _lock:
                _scalping_cache["items"] = items
                _scalping_cache["updated_at"] = _now_iso()
                _scalping_cache["stats"] = {
                    "loss_rate": _scalp_feedback["loss_rate"],
                    "tighten": _scalp_feedback["tighten"],
                    "window": len(_scalp_feedback["outcomes"]),
                }
                _scalping_cache["error"] = None
        except Exception as exc:
            with _lock:
                _scalping_cache["error"] = str(exc)
        time.sleep(SCALPING_SCAN_SECONDS)


def _swing_worker():
    while True:
        try:
            if not _is_market_open():
                _sleep_until_next_open()
                continue
            items = scan_swing()
            items = _attach_ai(items, "swing")
            with _lock:
                _swing_cache["items"] = items
                _swing_cache["updated_at"] = _now_iso()
                _swing_cache["error"] = None
        except Exception as exc:
            with _lock:
                _swing_cache["error"] = str(exc)
        time.sleep(SWING_SCAN_SECONDS)


def _bpjs_worker():
    while True:
        try:
            if not _is_market_open():
                _sleep_until_next_open()
                continue
            items = scan_bpjs()
            items = _attach_ai(items, "bpjs")
            with _lock:
                _bpjs_cache["items"] = items
                _bpjs_cache["updated_at"] = _now_iso()
                _bpjs_cache["error"] = None
        except Exception as exc:
            with _lock:
                _bpjs_cache["error"] = str(exc)
        time.sleep(BPJS_SCAN_SECONDS)


def _bsjp_worker():
    while True:
        try:
            if not _is_market_open():
                _sleep_until_next_open()
                continue
            items = scan_bsjp()
            items = _attach_ai(items, "bsjp")
            with _lock:
                _bsjp_cache["items"] = items
                _bsjp_cache["updated_at"] = _now_iso()
                _bsjp_cache["error"] = None
        except Exception as exc:
            with _lock:
                _bsjp_cache["error"] = str(exc)
        time.sleep(BSJP_SCAN_SECONDS)


def start_workers():
    scalping_thread = threading.Thread(target=_scalping_worker, daemon=True)
    swing_thread = threading.Thread(target=_swing_worker, daemon=True)
    bsjp_thread = threading.Thread(target=_bsjp_worker, daemon=True)
    bpjs_thread = threading.Thread(target=_bpjs_worker, daemon=True)
    scalping_thread.start()
    swing_thread.start()
    bsjp_thread.start()
    bpjs_thread.start()


@app.route("/")
def index():
    return render_template(
        "index.html",
        title=APP_TITLE,
        max_tickers=MAX_TICKERS,
        ui_poll=UI_POLL_SECONDS,
        today_label=_today_label(),
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


@app.route("/api/bsjp")
def api_bsjp():
    with _lock:
        payload = dict(_bsjp_cache)
    return jsonify(payload)


@app.route("/api/bpjs")
def api_bpjs():
    with _lock:
        payload = dict(_bpjs_cache)
    return jsonify(payload)


@app.route("/api/corporate-actions")
def api_corporate_actions():
    try:
        items = fetch_corporate_actions()
        items = _attach_ai_corporate(items)
        payload = {"items": items, "updated_at": _ca_cache["updated_at"], "error": None}
    except Exception as exc:
        payload = {"items": [], "updated_at": _ca_cache["updated_at"], "error": str(exc)}
    return jsonify(payload)


@app.route("/api/fundamentals/<ticker>")
def api_fundamentals(ticker):
    pe_wajar = request.args.get("pe_wajar")
    target_mc = request.args.get("target_market_cap")
    try:
        pe_wajar = float(pe_wajar) if pe_wajar is not None and pe_wajar != "" else None
    except Exception:
        pe_wajar = None
    try:
        target_mc = float(target_mc) if target_mc is not None and target_mc != "" else None
    except Exception:
        target_mc = None

    data = get_fundamentals(ticker, pe_wajar=pe_wajar, target_market_cap=target_mc)
    if _ai_allowed():
        summary = {
            "ticker": data.get("ticker"),
            "market_cap": data.get("ratios", {}).get("market_cap"),
            "pbv": data.get("ratios", {}).get("pbv"),
            "per": data.get("ratios", {}).get("per"),
            "eps": data.get("ratios", {}).get("eps"),
            "roe": data.get("ratios", {}).get("roe"),
            "der": data.get("ratios", {}).get("der"),
            "npm": data.get("ratios", {}).get("net_profit_margin"),
        }
        data["ai"] = _ai_analyze(summary, "fundamental", key_override=data.get("ticker"))
    return jsonify(data)


@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "market_open": _is_market_open(), "now": _now_iso()})


def main():
    start_workers()
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)


if __name__ == "__main__":
    main()
