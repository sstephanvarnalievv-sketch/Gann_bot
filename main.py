# gann_bot_v2.py
# Товарищ Ганн v2 — один файл.
# Углы Ганна (веер с фиксированной шкалой цена/бар) + зоны схождения + CCI + облако EMA + 4H (Т-Инвестиции)
# + цепочка ИИ (Grok -> Gemini) + журнал прогнозов с автоматической проверкой (/stats).
#
# Деплой: Amvera (long polling, без вебхука) или любой хостинг. Файл в репозитории называется main.py.
# Переменные окружения (минимум): BOT_TOKEN, а также XAI_API_KEY, GEMINI_API_KEY и/или OPENROUTER_API_KEY.
# Важно: ALLOWED_USER_IDS (через запятую; свой ID покажет /id), WEBHOOK_SECRET (случайная строка).
# Необязательно: TINKOFF_TOKEN, AI_PROXY (прокси только для запросов к ИИ), AI_ORDER, XAI_MODELS, GEMINI_MODELS,
# TELEGRAM_PROXY, OPENROUTER_MODELS, JOURNAL_PATH (по умолчанию /data/journal.json, если есть /data), REDIS_URL, WEBHOOK_URL.

import os
import re
import ssl
import json
import math
import time
import uuid
import secrets
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple

import httpx
import numpy as np
import pandas as pd
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web

# ====================== CONFIG ======================
BOT_TOKEN = os.getenv("BOT_TOKEN")
XAI_API_KEY = os.getenv("XAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
AI_PROXY = os.getenv("AI_PROXY") or None      # например http://user:pass@host:port — только для запросов к ИИ
TINKOFF_TOKEN = os.getenv("TINKOFF_TOKEN")
REDIS_URL = os.getenv("REDIS_URL")

AI_ORDER = [x.strip().lower() for x in os.getenv("AI_ORDER", "grok,gemini,openrouter").split(",") if x.strip()]
# Названия моделей меняются — держим их в переменных окружения; пробуются по порядку.
XAI_MODELS = [x.strip() for x in os.getenv("XAI_MODELS", os.getenv("XAI_MODEL", "grok-4.7,grok-4,grok-3-mini")).split(",") if x.strip()]
GEMINI_MODELS = [x.strip() for x in os.getenv("GEMINI_MODELS", "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-2.0-flash").split(",") if x.strip()]
# Бесплатные модели OpenRouter меняются — сверьтесь с каталогом (фильтр :free); нерабочие названия просто пропускаются.
OPENROUTER_MODELS = [x.strip() for x in os.getenv("OPENROUTER_MODELS", "google/gemini-2.0-flash-exp:free,deepseek/deepseek-chat-v3-0324:free").split(",") if x.strip()]

def _ids(name: str) -> set:
    return {int(x) for x in re.findall(r"\d+", os.getenv(name, ""))}

ALLOWED_USER_IDS = _ids("ALLOWED_USER_IDS")
ADMIN_IDS = _ids("ADMIN_IDS") or ALLOWED_USER_IDS
USER_LIMIT_PER_HOUR = int(os.getenv("USER_LIMIT_PER_HOUR", "6"))
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "150"))

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET") or secrets.token_urlsafe(32)   # без переменной — случайный на время работы процесса
WEBHOOK_PATH = "/webhook"
PORT = int(os.getenv("PORT", 8080))
RAILWAY_DOMAIN = os.getenv("RAILWAY_PUBLIC_DOMAIN") or os.getenv("RAILWAY_STATIC_URL")
WEBHOOK_URL = os.getenv("WEBHOOK_URL") or (f"https://{RAILWAY_DOMAIN}{WEBHOOK_PATH}" if RAILWAY_DOMAIN else None)   # нет URL -> long polling

TI_BASE = "https://invest-public-api.tinkoff.ru/rest/tinkoff.public.invest.api.contract.v1."
TI_CA_BUNDLE = os.getenv("TINKOFF_CA_BUNDLE")                      # путь к корневому сертификату Минцифры (безопасный вариант)
TI_SSL_INSECURE = os.getenv("TINKOFF_SSL_INSECURE", "0") == "1"    # крайний случай: отключить проверку сертификата

GANN_UNIT_ATR = float(os.getenv("GANN_UNIT_ATR", "0.1"))           # шкала: 1x1 = 0.1 ATR за дневной бар
COST_PCT = float(os.getenv("COST_PCT", "0.12"))                    # издержки круга, % (для статистики журнала)
CACHE_TTL = 20 * 60
MSK = timezone(timedelta(hours=3))
TICKER_ALIAS = {"YNDX": "YDEX"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gann")

# ====================== КЭШ ======================
_cache: Dict[str, Tuple[float, object]] = {}

def cache_get(key: str):
    item = _cache.get(key)
    if item and time.time() - item[0] < CACHE_TTL:
        return item[1]
    _cache.pop(key, None)
    return None

def cache_set(key: str, data):
    if len(_cache) > 300:
        for k in sorted(_cache, key=lambda x: _cache[x][0])[:100]:
            _cache.pop(k, None)
    _cache[key] = (time.time(), data)

# ====================== HTTP ======================
async def _http(method: str, url: str, *, headers=None, json_body=None, params=None,
                timeout: float = 30.0, verify=True, proxy: Optional[str] = None) -> Tuple[int, Optional[dict], str]:
    async with httpx.AsyncClient(timeout=timeout, verify=verify, proxy=proxy) as c:
        r = await c.request(method, url, headers=headers, json=json_body, params=params)
        try:
            data = r.json()
        except Exception:
            data = None
        return r.status_code, data, (r.text or "")[:300]

def _ti_verify():
    if TI_CA_BUNDLE:
        return ssl.create_default_context(cafile=TI_CA_BUNDLE)
    return False if TI_SSL_INSECURE else True

async def ti_call(method: str, body: dict) -> Tuple[int, Optional[dict], str]:
    try:
        return await _http("POST", TI_BASE + method, headers={"Authorization": f"Bearer {TINKOFF_TOKEN}"},
                           json_body=body, timeout=30, verify=_ti_verify())
    except Exception as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            logger.error("Т-Инвестиции: сертификат не принят (корневой сертификат Минцифры). Укажите TINKOFF_CA_BUNDLE "
                         "или, как крайнюю меру, TINKOFF_SSL_INSECURE=1 (токен тогда передаётся без проверки сертификата).")
        raise

def _q(v: dict) -> float:
    return float(v.get("units", 0)) + float(v.get("nano", 0)) / 1e9

# ====================== MOEX ISS (дневки) ======================
async def moex_daily(ticker: str, days: int = 900) -> Optional[pd.DataFrame]:
    key = f"moex:{ticker}:{days}"
    hit = cache_get(key)
    if hit is not None:
        return hit
    till = datetime.now(MSK).strftime("%Y-%m-%d")
    frm = (datetime.now(MSK) - timedelta(days=days)).strftime("%Y-%m-%d")
    url = f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json"
    rows, cols, start = [], None, 0
    for _ in range(8):                              # ISS отдаёт максимум 500 строк за запрос
        try:
            st, data, _t = await _http("GET", url, params={"from": frm, "till": till, "interval": 24, "start": start}, timeout=35)
        except Exception as e:
            logger.error(f"MOEX {ticker}: {e}")
            return None
        if st != 200 or not data:
            break
        block = data.get("candles") or {}
        cols = block.get("columns") or cols
        part = block.get("data") or []
        if not part:
            break
        rows += part
        start += len(part)
        if len(part) < 500:
            break
    if not rows or not cols:
        return None
    df = pd.DataFrame(rows, columns=cols).rename(columns={"begin": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "open", "high", "low", "close", "volume"]].sort_values("time").drop_duplicates("time").reset_index(drop=True)
    cache_set(key, df)
    return df

# ====================== Т-ИНВЕСТИЦИИ (4H) ======================
async def ti_instrument(ticker: str) -> Optional[dict]:
    """Точный поиск акции на TQBR в рублях. Никаких «первый попавшийся инструмент»."""
    if not TINKOFF_TOKEN:
        return None
    key = f"inst:{ticker}"
    hit = cache_get(key)
    if hit is not None:
        return hit or None
    def ok(i: dict) -> bool:
        return (i.get("ticker", "").upper() == ticker and i.get("classCode") == "TQBR"
                and str(i.get("currency", "")).lower() == "rub")
    inst = None
    try:
        st, data, _t = await ti_call("InstrumentsService/ShareBy",
                                     {"idType": "INSTRUMENT_ID_TYPE_TICKER", "classCode": "TQBR", "id": ticker})
        cand = (data or {}).get("instrument") if st == 200 else None
        if cand and ok(cand):
            inst = cand
        if inst is None:
            st, data, _t = await ti_call("InstrumentsService/FindInstrument",
                                         {"query": ticker, "instrumentKind": "INSTRUMENT_TYPE_SHARE"})
            for i in (data or {}).get("instruments", []) if st == 200 else []:
                if ok(i):
                    inst = i
                    break
    except Exception as e:
        logger.error(f"instrument {ticker}: {e}")
        return None
    res = {"figi": inst.get("figi"), "uid": inst.get("uid"), "lot": inst.get("lot")} if inst else None
    cache_set(key, res or {})
    return res

async def ti_4h(ticker: str, days: int = 120) -> Optional[pd.DataFrame]:
    """4-часовые свечи. Для CANDLE_INTERVAL_4_HOUR максимум периода запроса — 1 месяц, поэтому окна по 28 дней."""
    if not TINKOFF_TOKEN:
        return None
    key = f"4h:{ticker}:{days}"
    hit = cache_get(key)
    if hit is not None:
        return hit
    inst = await ti_instrument(ticker)
    if not inst:
        return None
    iid = inst.get("uid") or inst.get("figi")
    end = datetime.now(timezone.utc)
    rows: Dict[str, dict] = {}
    step = 28
    for k in range(math.ceil(days / step)):
        to = end - timedelta(days=step * k)
        frm = to - timedelta(days=step)
        try:
            st, data, _t = await ti_call("MarketDataService/GetCandles", {
                "instrumentId": iid, "from": frm.strftime("%Y-%m-%dT%H:%M:%SZ"), "to": to.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "interval": "CANDLE_INTERVAL_4_HOUR"})
        except Exception as e:
            logger.error(f"4H {ticker}: {e}")
            break
        if st != 200 or not data:
            logger.warning(f"4H {ticker}: HTTP {st}")
            break
        for c in data.get("candles", []):
            rows[c["time"]] = {"time": pd.to_datetime(c["time"]), "open": _q(c["open"]), "high": _q(c["high"]),
                               "low": _q(c["low"]), "close": _q(c["close"]), "volume": int(c.get("volume", 0))}
    if not rows:
        return None
    df = pd.DataFrame(list(rows.values())).sort_values("time").reset_index(drop=True)
    cache_set(key, df)
    return df

# ====================== ИНДИКАТОРЫ ======================
def calc_ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()

def calc_atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def calc_cci(df: pd.DataFrame, n: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - sma) / (0.015 * mad)

def find_pivots(df: pd.DataFrame, left: int = 5, right: int = 5) -> List[dict]:
    """Только подтверждённые пивоты (нужно right баров справа), поэтому последние бары пивотами не бывают."""
    out, hi, lo = [], df["high"].values, df["low"].values
    for i in range(left, len(df) - right):
        if hi[i] == max(hi[i - left:i + right + 1]):
            out.append({"type": "high", "idx": i, "price": float(hi[i]), "time": df["time"].iloc[i]})
        if lo[i] == min(lo[i - left:i + right + 1]):
            out.append({"type": "low", "idx": i, "price": float(lo[i]), "time": df["time"].iloc[i]})
    return out

RATIOS = {"4x1": 4.0, "2x1": 2.0, "1x1": 1.0, "1x2": 0.5, "1x4": 0.25}

def fan_lines(df: pd.DataFrame, piv: List[dict], unit: float, max_age: int = 250) -> List[dict]:
    """Веер Ганна с ФИКСИРОВАННОЙ шкалой: цена линии на текущем баре = пивот ± k * unit * (баров от пивота).
    unit — цена за бар для 1x1 (задаётся как доля ATR). Линии растут от минимумов и падают от максимумов."""
    last = len(df) - 1
    highs = [p for p in piv if p["type"] == "high" and last - p["idx"] <= max_age][-4:]
    lows = [p for p in piv if p["type"] == "low" and last - p["idx"] <= max_age][-4:]
    out = []
    for p in highs + lows:
        bars = last - p["idx"]
        if bars < 3:
            continue
        sign = 1 if p["type"] == "low" else -1
        for name, k in RATIOS.items():
            out.append({"pivot": p["idx"], "from": p["type"], "name": name, "price": p["price"] + sign * k * unit * bars,
                        "date": p["time"].strftime("%d.%m.%Y")})
    return out

def confluence_zones(lines: List[dict], price: float, atr: float, span: float = 0.06, top: int = 3,
                      tol_atr: float = 0.15, min_strength: int = 4) -> List[dict]:
    """Зона схождения: >=4 линии от РАЗНЫХ пивотов в пределах 0.15 ATR; рядом с ценой (±span).
    Пороги подобраны так, чтобы на случайном блуждании такая зона была примерно в 1 дне из 4 (при мягких порогах она есть почти всегда)."""
    near = sorted((l for l in lines if abs(l["price"] - price) / price <= span), key=lambda x: x["price"])
    zones, cur = [], []
    tol = tol_atr * atr
    for l in near:
        if cur and l["price"] - cur[-1]["price"] > tol:
            zones.append(cur); cur = []
        cur.append(l)
    if cur:
        zones.append(cur)
    res = []
    for z in zones:
        if len({l["pivot"] for l in z}) >= min_strength:
            res.append({"price": float(np.mean([l["price"] for l in z])), "strength": len({l["pivot"] for l in z}),
                        "lines": [f"{l['name']} от {l['from']} {l['date']}" for l in z][:4]})
    res.sort(key=lambda z: (-z["strength"], abs(z["price"] - price)))
    return res[:top]

def cci_state(df: pd.DataFrame, cci: pd.Series, lookback: int = 25) -> str:
    if len(df) < lookback + 5 or pd.isna(cci.iloc[-1]):
        return "мало данных"
    p, c = df["close"].iloc[-lookback:], cci.iloc[-lookback:]
    if p.iloc[-1] > p.iloc[0] and c.iloc[-1] < c.iloc[0] and c.iloc[-1] > 80:
        return "медвежья дивергенция CCI"
    if p.iloc[-1] < p.iloc[0] and c.iloc[-1] > c.iloc[0] and c.iloc[-1] < -80:
        return "бычья дивергенция CCI"
    if c.iloc[-1] > 140:
        return "сильная перекупленность CCI"
    if c.iloc[-1] < -140:
        return "сильная перепроданность CCI"
    return "явных дивергенций нет"

def cloud_state(emas: Dict[str, float], price: float) -> str:
    above = sum(1 for v in emas.values() if price > v)
    if above >= 3:
        return "цена выше большей части облака → приоритет лонг"
    if above <= 1:
        return "цена ниже большей части облака → приоритет шорт"
    return "цена внутри облака → зона баланса"

# ====================== ИИ: Grok -> Gemini ======================
class AIError(Exception):
    pass

_breaker: Dict[str, Tuple[int, float]] = {}      # провайдер -> (число подряд ошибок, до какого времени пропускаем)

def _breaker_open(p: str) -> bool:
    return _breaker.get(p, (0, 0.0))[1] > time.time()

def _breaker_fail(p: str, hard: bool = False):
    n = _breaker.get(p, (0, 0.0))[0] + 1
    if hard or n >= 3:
        _breaker[p] = (n, time.time() + (1800 if hard else 600))
    else:
        _breaker[p] = (n, 0.0)

async def _grok(system: str, user: str) -> str:
    last = "нет моделей"
    for model in XAI_MODELS:
        for attempt in range(2):
            st, data, txt = await _http("POST", "https://api.x.ai/v1/chat/completions",
                                        headers={"Authorization": f"Bearer {XAI_API_KEY}"},
                                        json_body={"model": model, "temperature": 0.3, "max_tokens": 1300,
                                                   "messages": [{"role": "system", "content": system},
                                                                {"role": "user", "content": user}]}, timeout=60, proxy=AI_PROXY)
            if st == 200 and data:
                out = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
                if out:
                    return out.strip()
                last = f"{model}: пустой ответ"
                break
            last = f"{model}: HTTP {st}"
            if st in (401, 403):
                raise AIError(f"Grok: ключ не принят (HTTP {st})")
            if st in (402, 429):
                raise AIError(f"Grok: лимит/кредиты (HTTP {st})")
            if st in (400, 404):
                break                                  # нет такой модели — пробуем следующую
            if st >= 500 and attempt == 0:
                await asyncio.sleep(1.5)
                continue
            break
    raise AIError(f"Grok: {last}")

async def _gemini(system: str, user: str) -> str:
    last = "нет моделей"
    for model in GEMINI_MODELS:
        gen = {"temperature": 0.3, "maxOutputTokens": 1500}
        if "2.5-flash" in model:
            gen["thinkingConfig"] = {"thinkingBudget": 0}      # без «раздумий»: быстрее и не съедает лимит токенов
        st, data, txt = await _http("POST", f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                                    headers={"x-goog-api-key": GEMINI_API_KEY},
                                    json_body={"systemInstruction": {"parts": [{"text": system}]},
                                               "contents": [{"role": "user", "parts": [{"text": user}]}],
                                               "generationConfig": gen}, timeout=60, proxy=AI_PROXY)
        if st == 200 and data:
            parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
            out = "".join(p.get("text", "") for p in parts).strip()
            if out:
                return out
            last = f"{model}: пустой ответ"
            continue
        last = f"{model}: HTTP {st}"
        if st == 429 or st == 404 or st >= 500:
            continue                                   # квота/нет модели/сбой — пробуем следующую модель
        raise AIError(f"Gemini: {last} {txt[:80]}")
    raise AIError(f"Gemini: {last}")

async def _openrouter(system: str, user: str) -> str:
    last = "нет моделей"
    for model in OPENROUTER_MODELS:
        st, data, txt = await _http("POST", "https://openrouter.ai/api/v1/chat/completions",
                                    headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                                    json_body={"model": model, "temperature": 0.3, "max_tokens": 1300,
                                               "messages": [{"role": "system", "content": system},
                                                            {"role": "user", "content": user}]}, timeout=60, proxy=AI_PROXY)
        if st == 200 and data:
            out = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            if out:
                return out.strip()
            last = f"{model}: пустой ответ"
            continue
        last = f"{model}: HTTP {st}"
        if st in (401, 403):
            raise AIError(f"OpenRouter: ключ не принят (HTTP {st})")
        if st == 402:
            raise AIError(f"OpenRouter: лимит/кредиты (HTTP {st})")
        continue                                       # 400/404/429/5xx — пробуем следующую модель
    raise AIError(f"OpenRouter: {last}")

async def ai_complete(system: str, user: str) -> Tuple[str, str]:
    errors = []
    for prov in AI_ORDER:
        if prov == "grok" and not XAI_API_KEY:
            continue
        if prov == "gemini" and not GEMINI_API_KEY:
            continue
        if prov == "openrouter" and not OPENROUTER_API_KEY:
            continue
        if prov not in ("grok", "gemini", "openrouter") or _breaker_open(prov):
            continue
        try:
            text = await {"grok": _grok, "gemini": _gemini, "openrouter": _openrouter}[prov](system, user)
            _breaker[prov] = (0, 0.0)
            return text, prov
        except AIError as e:
            errors.append(str(e))
            _breaker_fail(prov, hard=("ключ не принят" in str(e) or "лимит/кредиты" in str(e)))
        except Exception as e:
            errors.append(f"{prov}: {type(e).__name__}")
            _breaker_fail(prov)
    raise AIError("; ".join(errors) or "нет доступных ИИ-провайдеров (не заданы ключи)")

# ====================== ПРОГНОЗ: разбор и проверка ======================
FORECAST_RE = re.compile(r"FORECAST_JSON:\s*(\{.*?\})\s*(?:```)?\s*$", re.S | re.M)

def parse_forecast(text: str, price: float, atr: float) -> Tuple[str, Optional[dict], str]:
    """Возвращает (текст без JSON, прогноз или None, причина отказа)."""
    m = None
    for m in FORECAST_RE.finditer(text):
        pass
    clean = FORECAST_RE.sub("", text).strip() if m else text.strip()
    clean = re.sub(r"```\w*\s*$", "", clean).strip()
    if not m:
        return clean, None, "ИИ не вернул структурированный прогноз"
    try:
        f = json.loads(m.group(1))
        d = str(f.get("direction", "")).lower()
        if d == "flat":
            return clean, None, "сценарий «вне рынка» (в журнал не пишется)"
        if d not in ("long", "short"):
            return clean, None, "некорректное направление"
        t1, inv = float(f["target1"]), float(f["invalidation"])
        t2 = float(f["target2"]) if f.get("target2") not in (None, "") else None
        hz = int(f.get("horizon_days") or 10)
        vals = [t1, inv] + ([t2] if t2 else [])
        if not all(math.isfinite(v) and v > 0 for v in vals) or atr <= 0:
            return clean, None, "некорректные числа"
        sg = 1 if d == "long" else -1
        if not (sg * (t1 - price) > 0 and sg * (price - inv) > 0):
            return clean, None, "цель или отмена не с той стороны от цены"
        if t2 is not None and sg * (t2 - t1) < 0:
            return clean, None, "target2 ближе target1"
        if not (0.5 <= abs(t1 - price) / atr <= 6.0):
            return clean, None, "цель вне разумной дистанции (0.5–6 ATR)"
        if not (0.4 <= abs(price - inv) / atr <= 3.0):
            return clean, None, "отмена вне разумной дистанции (0.4–3 ATR)"
        return clean, {"dir": d, "t1": t1, "t2": t2, "inv": inv, "horizon": max(1, min(30, hz))}, ""
    except Exception:
        return clean, None, "не удалось разобрать JSON прогноза"

def score_forecast(rec: dict, bars: List[dict], today: str) -> Optional[dict]:
    """Проверка по дневным барам строго ПОСЛЕ дня прогноза и до сегодняшнего (незавершённый бар не берём).
    Консервативно: если за один бар задеты и отмена, и цель, засчитываем отмену."""
    p, inv, t1, t2 = rec["price"], rec["inv"], rec["t1"], rec.get("t2")
    risk = abs(p - inv)
    if risk <= 0:
        return None
    sg = 1 if rec["dir"] == "long" else -1
    n = 0
    for b in bars:
        if b["date"] <= rec["date"] or b["date"] >= today:
            continue
        n += 1
        stop = b["l"] <= inv if sg == 1 else b["h"] >= inv
        if stop:
            return {"status": "stop", "R": -1.0, "bars": n, "closed": b["date"]}
        h1 = b["h"] >= t1 if sg == 1 else b["l"] <= t1
        if h1:
            h2 = t2 is not None and ((b["h"] >= t2) if sg == 1 else (b["l"] <= t2))
            tgt = t2 if h2 else t1
            return {"status": "t2" if h2 else "t1", "R": round(sg * (tgt - p) / risk, 2), "bars": n, "closed": b["date"]}
        if n >= rec["horizon"]:
            return {"status": "expired", "R": round(sg * (b["c"] - p) / risk, 2), "bars": n, "closed": b["date"]}
    return None

# ====================== ЖУРНАЛ ======================
class Store:
    def __init__(self, path: Optional[str] = None):
        path = path or os.getenv("JOURNAL_PATH") or ("/data/journal.json" if os.path.isdir("/data") else "journal.json")
        self.path, self.lock, self.redis, self.key = path, asyncio.Lock(), None, "gann:journal"
        if REDIS_URL:
            try:
                import redis.asyncio as aioredis
                self.redis = aioredis.from_url(REDIS_URL, decode_responses=True)
            except Exception as e:
                logger.warning(f"Redis недоступен, журнал в файле: {e}")

    async def load(self) -> List[dict]:
        try:
            if self.redis:
                raw = await self.redis.get(self.key)
                return json.loads(raw) if raw else []
            if os.path.exists(self.path):
                with open(self.path, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"journal load: {e}")
        return []

    async def save(self, items: List[dict]):
        items = items[-1500:]
        try:
            if self.redis:
                await self.redis.set(self.key, json.dumps(items, ensure_ascii=False))
            else:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(items, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"journal save: {e}")

store = Store()

async def journal_add(rec: dict):
    async with store.lock:
        items = await store.load()
        items.append(rec)
        await store.save(items)

async def journal_update_once():
    async with store.lock:
        items = await store.load()
        open_recs = [r for r in items if r.get("status") == "open"]
        if not open_recs:
            return 0
        today = datetime.now(MSK).strftime("%Y-%m-%d")
        changed = 0
        for tk in sorted({r["ticker"] for r in open_recs}):
            df = await moex_daily(tk, days=120)
            if df is None:
                continue
            bars = [{"date": t.strftime("%Y-%m-%d"), "o": o, "h": h, "l": l, "c": c}
                    for t, o, h, l, c in zip(df["time"], df["open"], df["high"], df["low"], df["close"])]
            for r in open_recs:
                if r["ticker"] != tk:
                    continue
                res = score_forecast(r, bars, today)
                if res:
                    r.update(res)
                    r["status"] = res["status"]
                    changed += 1
        if changed:
            await store.save(items)
        return changed

async def journal_loop():
    while True:
        await asyncio.sleep(6 * 3600)
        try:
            n = await journal_update_once()
            logger.info(f"journal: закрыто прогнозов {n}")
        except Exception as e:
            logger.error(f"journal loop: {e}")

def stats_text(items: List[dict]) -> str:
    closed = [r for r in items if r.get("status") not in (None, "open")]
    open_n = sum(1 for r in items if r.get("status") == "open")
    if not closed:
        return f"📒 Журнал: проверенных прогнозов пока нет (открытых: {open_n}). Проверка идёт автоматически по дневным барам."
    n = len(closed)
    cnt = {k: sum(1 for r in closed if r["status"] == k) for k in ("t2", "t1", "stop", "expired")}
    rs = [r["R"] for r in closed]
    net = []
    for r in closed:
        risk = abs(r["price"] - r["inv"])
        net.append(r["R"] - (COST_PCT / 100 * r["price"]) / risk if risk > 0 else r["R"])
    hit = (cnt["t1"] + cnt["t2"]) / n * 100
    lines = [f"📒 Журнал прогнозов: проверено {n}, открытых {open_n}",
             f"Цель достигнута: {cnt['t1'] + cnt['t2']} ({hit:.0f}%), отмена: {cnt['stop']}, срок вышел: {cnt['expired']}",
             f"Средний результат: {np.mean(rs):+.2f} R, после издержек ({COST_PCT}%): {np.mean(net):+.2f} R"]
    lines.append("⚠️ Мало данных: до 30 прогнозов выводы ненадёжны." if n < 30 else "Нужны 100+ прогнозов и устойчивый плюс после издержек.")
    return "\n".join(lines)

# ====================== АНАЛИЗ ======================
SYSTEM_PROMPT = """Ты — Товарищ Ганн. Система:
- Веер Ганна с фиксированной шкалой (цена за бар) от подтверждённых пивотов; зона схождения = несколько линий от разных пивотов рядом.
- Обязательные фильтры: дивергенции/экстремумы CCI и облако EMA (приоритет направления).
- Углы Ганна не имеют доказанной прогностической силы и дают ложные пробои; не приписывай им уверенность.

ПРАВИЛА:
1. Используй ТОЛЬКО числа из блока данных. Не выдумывай уровни.
2. Цели и отмену бери из списка ориентиров (пивоты, зоны, EMA, кратные ATR).
3. Если картина неясна или противоречива — direction = "flat".
4. Стиль: коротко, без воды, до 900 символов.

Формат ответа:
#STOCKS #ТИКЕР
Текущая картина: ...
Сценарий: ...
Первая цель: ~число
Более глубокая цель: ~число
Отмена сценария: закрепление выше/ниже ~число
Комментарий: ...

В самом конце ОДНОЙ строкой (без пояснений):
FORECAST_JSON: {"direction":"long|short|flat","target1":число,"target2":число,"invalidation":число,"horizon_days":целое 3-20}
"""

def build_facts(ticker: str, df: pd.DataFrame, df4h: Optional[pd.DataFrame]) -> Tuple[str, dict]:
    price = float(df["close"].iloc[-1])
    atr = float(calc_atr(df, 20).iloc[-1])
    emas = {f"EMA{n}": float(calc_ema(df["close"], n).iloc[-1]) for n in (20, 50, 100, 200)}
    cci = calc_cci(df, 20)
    piv = find_pivots(df)
    unit = atr * GANN_UNIT_ATR
    lines = fan_lines(df, piv, unit)
    zones = confluence_zones(lines, price, atr)
    highs = [p for p in piv if p["type"] == "high"][-3:]
    lows = [p for p in piv if p["type"] == "low"][-3:]
    refs = [f"цена ±1/2/3 ATR: {price - atr:.2f} / {price - 2 * atr:.2f} / {price - 3 * atr:.2f} | {price + atr:.2f} / {price + 2 * atr:.2f} / {price + 3 * atr:.2f}"]
    refs += [f"пивот-{p['type']} {p['price']:.2f} ({p['time'].strftime('%d.%m.%Y')})" for p in highs + lows]
    refs += [f"{k}={v:.2f}" for k, v in emas.items()]
    h4 = "4H: данные недоступны"
    if df4h is not None and len(df4h) > 30:
        p4, e4 = float(df4h["close"].iloc[-1]), float(calc_ema(df4h["close"], 50).iloc[-1])
        h4 = f"4H: цена {p4:.2f}, EMA50(4H) {e4:.2f} ({'выше' if p4 > e4 else 'ниже'})"
    ztxt = "\n".join(f"- {z['price']:.2f} (линий от разных пивотов: {z['strength']}; {', '.join(z['lines'])})" for z in zones) or "значимых зон схождения рядом нет (так бывает в ~3 днях из 4)"
    facts = (f"Тикер: {ticker}\nЦена (D): {price:.2f}\nATR(20): {atr:.2f} ({atr / price * 100:.2f}%)\n"
             f"Облако: {', '.join(f'{k}={v:.2f}' for k, v in emas.items())} → {cloud_state(emas, price)}\n"
             f"CCI(20): {cci.iloc[-1]:.1f} → {cci_state(df, cci)}\n{h4}\n\n"
             f"Зоны схождения веера (шкала: 1x1 = {GANN_UNIT_ATR} ATR за бар):\n{ztxt}\n\nОриентиры для целей/отмены:\n" + "\n".join(refs))
    return facts, {"price": price, "atr": atr, "emas": emas}

def fallback_text(ticker: str, facts: str, why: str) -> str:
    return f"#{ticker}\nИИ сейчас недоступен ({why}). Данные расчёта:\n\n{facts}"

FOOTER = "\n\nℹ️ Не инвестсовет. Углы Ганна не имеют подтверждённой прогностической силы; точность прогнозов бота копится в /stats."

async def analyze_ticker(raw: str, user_id: int) -> str:
    ticker = TICKER_ALIAS.get(raw.upper().strip(), raw.upper().strip())
    if not re.fullmatch(r"[A-Z0-9]{2,6}", ticker):
        return "Нужен тикер российской акции, например: SBER"
    df = await moex_daily(ticker)
    if df is None or len(df) < 120:
        return f"❌ Нет данных по {ticker} на MOEX (или слишком короткая история). Проверьте тикер."
    df4h = await ti_4h(ticker)
    facts, ctx = build_facts(ticker, df, df4h)
    try:
        text, prov = await ai_complete(SYSTEM_PROMPT, f"Сделай разбор по твоей системе:\n\n{facts}")
    except AIError as e:
        logger.error(f"AI: {e}")
        return fallback_text(ticker, facts, "оба провайдера не ответили") + FOOTER
    clean, fc, why = parse_forecast(text, ctx["price"], ctx["atr"])
    note = ""
    if fc:
        rec = {"id": uuid.uuid4().hex[:6], "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "date": datetime.now(MSK).strftime("%Y-%m-%d"), "ticker": ticker, "user": user_id, "price": ctx["price"],
               "atr": ctx["atr"], "provider": prov, "status": "open", **fc}
        await journal_add(rec)
        note = f"\n\n📒 Прогноз записан в журнал (#{rec['id']}, проверка через ≤{fc['horizon']} дн.)"
    else:
        note = f"\n\n📒 В журнал не записано: {why}"
    return f"{clean}{note}\n[ИИ: {prov}]{FOOTER}"

# ====================== ДОСТУП И ЛИМИТЫ ======================
class Limiter:
    def __init__(self):
        self.hits: Dict[int, List[float]] = {}
        self.day, self.day_n = "", 0

    def check(self, uid: int, admin: bool) -> Optional[str]:
        today = datetime.now(MSK).strftime("%Y-%m-%d")
        if today != self.day:
            self.day, self.day_n = today, 0
        if self.day_n >= DAILY_LIMIT:
            return "Дневной лимит бота исчерпан. Попробуйте завтра."
        now = time.time()
        h = [t for t in self.hits.get(uid, []) if now - t < 3600]
        if not admin and len(h) >= USER_LIMIT_PER_HOUR:
            return f"Лимит {USER_LIMIT_PER_HOUR} разборов в час. Подождите немного."
        h.append(now)
        self.hits[uid] = h
        self.day_n += 1
        return None

limiter = Limiter()
ai_sem = asyncio.Semaphore(2)

def is_allowed(uid: int) -> bool:
    return not ALLOWED_USER_IDS or uid in ALLOWED_USER_IDS

def is_admin(uid: int) -> bool:
    return uid in ADMIN_IDS

# ====================== HANDLERS ======================
dp = Dispatcher()

HELP = ("Кидай тикер российской акции (SBER, GAZP, LKOH, YDEX...).\n"
        "Разбор: веер Ганна + зоны схождения + CCI + облако EMA + 4H.\n\n"
        "/stats — статистика точности прошлых прогнозов\n/journal — последние прогнозы\n/id — ваш Telegram ID")

@dp.message(Command("start", "help"))
async def h_start(message: types.Message):
    await message.answer(HELP)

@dp.message(Command("id"))
async def h_id(message: types.Message):
    await message.answer(f"Ваш Telegram ID: {message.from_user.id}\nДобавьте его в ALLOWED_USER_IDS.")

@dp.message(Command("stats"))
async def h_stats(message: types.Message):
    if not is_allowed(message.from_user.id):
        return await message.answer("Доступ закрыт.")
    await message.answer(stats_text(await store.load()))

@dp.message(Command("journal"))
async def h_journal(message: types.Message):
    if not is_allowed(message.from_user.id):
        return await message.answer("Доступ закрыт.")
    items = (await store.load())[-10:][::-1]
    if not items:
        return await message.answer("Журнал пуст.")
    out = []
    for r in items:
        res = f" → {r['status']} {r.get('R', '')}R" if r["status"] != "open" else " (открыт)"
        out.append(f"#{r['id']} {r['ticker']} {r['dir']} {r['price']:.2f} → {r['t1']:.2f} / стоп {r['inv']:.2f}{res}")
    await message.answer("\n".join(out))

@dp.message(F.text)
async def h_ticker(message: types.Message):
    uid = message.from_user.id if message.from_user else 0
    text = (message.text or "").strip()
    if text.startswith("/"):
        return
    if not is_allowed(uid):
        return await message.answer(f"Доступ закрыт. Ваш ID: {uid}")
    if len(text) > 12 or " " in text:
        return await message.answer("Просто тикер, например: SBER")
    err = limiter.check(uid, is_admin(uid))
    if err:
        return await message.answer(err)
    msg = await message.answer(f"Считаю {text.upper()}...")
    try:
        async with ai_sem:
            result = await analyze_ticker(text, uid)
        await msg.edit_text(result[:4000])
    except Exception:
        logger.exception("analyze failed")
        await msg.edit_text("Не удалось сделать разбор. Попробуйте позже.")

# ====================== WEBHOOK ======================
async def health(request):
    return web.json_response({"ok": True})

def run_mode() -> str:
    return "webhook" if WEBHOOK_URL else "polling"

def main():
    if not BOT_TOKEN:
        raise SystemExit("Не задан BOT_TOKEN")
    if not (XAI_API_KEY or GEMINI_API_KEY or OPENROUTER_API_KEY):
        logger.warning("Нет ключей ИИ (XAI_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY): бот будет отдавать только расчёт без текста.")
    if not ALLOWED_USER_IDS:
        logger.warning("ALLOWED_USER_IDS не задан: бот открыт всем (лимиты включены). Узнайте ID командой /id и задайте переменную.")
    logger.info(f"Режим: {run_mode()}; журнал: {store.path}")
    tg_proxy = os.getenv("TELEGRAM_PROXY")             # если с хостинга не открывается api.telegram.org
    if tg_proxy:
        from aiogram.client.session.aiohttp import AiohttpSession
        bot = Bot(token=BOT_TOKEN, session=AiohttpSession(proxy=tg_proxy))
    else:
        bot = Bot(token=BOT_TOKEN)
    tasks: List[asyncio.Task] = []

    if run_mode() == "polling":
        async def run():
            tasks.append(asyncio.create_task(journal_loop()))
            await bot.delete_webhook(drop_pending_updates=True)
            try:
                await dp.start_polling(bot)
            finally:
                for t in tasks:
                    t.cancel()
                await bot.session.close()
        asyncio.run(run())
        return

    if not os.getenv("WEBHOOK_SECRET"):
        logger.warning("WEBHOOK_SECRET не задан: использован случайный на время работы процесса.")

    async def on_startup(app):
        tasks.append(asyncio.create_task(journal_loop()))
        await bot.set_webhook(WEBHOOK_URL, secret_token=WEBHOOK_SECRET)
        logger.info(f"Webhook set → {WEBHOOK_URL}")

    async def on_shutdown(app):
        for t in tasks:
            t.cancel()
        await bot.delete_webhook()
        await bot.session.close()

    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    SimpleRequestHandler(dispatcher=dp, bot=bot, secret_token=WEBHOOK_SECRET).register(app, path=WEBHOOK_PATH)
    setup_application(app, dp, bot=bot)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    web.run_app(app, host="0.0.0.0", port=PORT)

if __name__ == "__main__":
    main()
