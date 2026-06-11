"""
market_feed.py  v1.2
====================
Capa de datos unificada. Tanto el backtest como el bot en vivo
usan esta misma interfaz. El resto del sistema no sabe (ni le importa)
de dónde vienen los datos.

Cambios respecto a v1.1:
  - Descarga histórica corregida: itera hacia atrás en el tiempo
    con múltiples requests (Binance limita 1500 velas por request)
  - download_historical_data() movido a función de módulo (no método estático)
    para poder llamarse desde backtest_v3.py y desde __main__
  - backtest_v3.py llama a download_historical_data() automáticamente
    si no encuentra el CSV
  - Argumento de línea de comandos corregido: acepta --download y download
  - Progreso de descarga más claro

Modos:
  - LiveFeed:     WebSocket + REST contra Binance (testnet o producción)
  - BacktestFeed: Lee CSV o descarga datos históricos, los sirve vela a vela

Contrato de salida: MarketSnapshot (dataclass definida acá)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from config import cfg


# ─────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """
    Todo lo que el sistema necesita saber del mercado en un instante dado.
    indicators.py recibe esto y devuelve los valores calculados.
    """
    symbol:    str
    timestamp: datetime

    # OHLCV por timeframe. Cada DataFrame tiene columnas:
    # ['open', 'high', 'low', 'close', 'volume'] con índice datetime UTC.
    ohlcv_1m:  pd.DataFrame
    ohlcv_5m:  pd.DataFrame
    ohlcv_15m: pd.DataFrame
    ohlcv_1h:  pd.DataFrame
    ohlcv_4h:  pd.DataFrame
    ohlcv_1d:  pd.DataFrame
    ohlcv_1w:  pd.DataFrame

    last_price:          float = 0.0
    funding_rate:        float = 0.0
    funding_rate_next:   float = 0.0
    orderbook_imbalance: float = 0.0
    bid:                 float = 0.0
    ask:                 float = 0.0
    is_backtest:         bool  = False
    feed_latency_ms:     float = 0.0

    @property
    def spread_pct(self) -> float:
        if self.bid > 0:
            return (self.ask - self.bid) / self.bid * 100
        return 0.0

    @property
    def current_close(self) -> float:
        if not self.ohlcv_1m.empty:
            return float(self.ohlcv_1m["close"].iloc[-1])
        return self.last_price


# ─────────────────────────────────────────────
# INTERFAZ BASE
# ─────────────────────────────────────────────

class MarketFeed(ABC):
    @abstractmethod
    def get_snapshot(self) -> MarketSnapshot: ...
    @abstractmethod
    def start(self) -> None: ...
    @abstractmethod
    def stop(self) -> None: ...


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────

TIMEFRAME_MAP = {
    "1m": 1, "5m": 5, "15m": 15,
    "1h": 60, "4h": 240, "1d": 1440, "1w": 10080,
}


def _parse_klines(raw: list) -> pd.DataFrame:
    """Convierte respuesta de API de Binance a DataFrame limpio."""
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def _resample_from_1m(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Construye cualquier timeframe mayor a partir de velas de 1m."""
    rule_map = {
        "5m": "5min", "15m": "15min", "1h": "1h",
        "4h": "4h",   "1d":  "1D",   "1w": "1W",
    }
    rule = rule_map.get(timeframe)
    if rule is None:
        raise ValueError(f"Timeframe no soportado: {timeframe}")
    return df_1m.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max",
        "low":  "min",   "close": "last", "volume": "sum",
    }).dropna()


# ─────────────────────────────────────────────
# DESCARGA HISTÓRICA (función de módulo)
# ─────────────────────────────────────────────

def download_historical_data(
    symbol:    str  = "BTCUSDC",
    days:      int  = 1000,
    testnet:   bool = False,
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Descarga datos históricos de velas 1m desde Binance Futuros.

    Binance limita a 1500 velas por request (~25 horas de datos de 1m).
    Para cubrir 1000 días necesitamos ~960 requests.
    Esta función itera hacia atrás en el tiempo hasta cubrir los días pedidos.

    Parámetros:
        symbol    Par a descargar (default: BTCUSDC)
        days      Días de historia (default: 1000)
        testnet   False = producción (recomendado para datos históricos reales)
        save_path Ruta donde guardar el CSV. Si es None no guarda.

    Devuelve:
        DataFrame con columnas: timestamp, open, high, low, close, volume
    """
    base_url = (
        "https://testnet.binancefuture.com" if testnet
        else "https://fapi.binance.com"
    )
    url = f"{base_url}/fapi/v1/klines"

    total_minutes = days * 24 * 60
    batch_size    = 1500
    all_frames: list[pd.DataFrame] = []
    end_time_ms: Optional[int]     = None
    fetched = 0

    print(f"\n[MarketFeed] Descargando {total_minutes:,} velas de 1m ({days} días)...")
    print(f"[MarketFeed] Símbolo: {symbol} | Fuente: {'TESTNET' if testnet else 'PRODUCCIÓN'}")
    print(f"[MarketFeed] Esto puede tardar 15-30 minutos.\n")

    retries = 0
    max_retries = 5

    while fetched < total_minutes:
        params: dict = {
            "symbol":   symbol.upper(),
            "interval": "1m",
            "limit":    batch_size,
        }
        if end_time_ms is not None:
            params["endTime"] = end_time_ms

        try:
            resp = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            raw = resp.json()
            retries = 0
        except requests.exceptions.HTTPError as e:
            # 429 = rate limit → esperar más
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 60))
                print(f"\n[MarketFeed] Rate limit alcanzado. Esperando {wait}s...")
                time.sleep(wait)
                continue
            print(f"\n[MarketFeed] Error HTTP: {e}. Reintento {retries+1}/{max_retries}...")
            retries += 1
            if retries >= max_retries:
                raise
            time.sleep(5 * retries)
            continue
        except Exception as e:
            print(f"\n[MarketFeed] Error de red: {e}. Reintento {retries+1}/{max_retries}...")
            retries += 1
            if retries >= max_retries:
                raise
            time.sleep(5 * retries)
            continue

        if not raw:
            print("\n[MarketFeed] No hay más datos disponibles.")
            break

        # Parsear batch
        df_batch = pd.DataFrame(raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades",
            "buy_base", "buy_quote", "ignore",
        ])
        df_batch["timestamp"] = pd.to_datetime(
            df_batch["timestamp"], unit="ms", utc=True
        )
        df_batch = df_batch[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df_batch[col] = df_batch[col].astype(float)

        all_frames.append(df_batch)
        fetched     += len(df_batch)
        # La próxima request termina 1ms antes del inicio de este batch
        end_time_ms  = int(raw[0][0]) - 1

        pct = min(fetched / total_minutes * 100, 100)
        print(
            f"  {fetched:,} / {total_minutes:,} velas ({pct:.1f}%) | "
            f"hasta {df_batch['timestamp'].iloc[0].strftime('%Y-%m-%d')}   ",
            end="\r",
        )

        # Respetar rate limits de Binance (max ~2400 requests/min en futuros)
        time.sleep(0.25)

    print(f"\n\n[MarketFeed] Combinando {len(all_frames)} batches...")

    if not all_frames:
        raise RuntimeError("No se descargaron datos. Verificá la conexión y el símbolo.")

    df = pd.concat(all_frames, ignore_index=True)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        df.to_csv(save_path, index=False)
        size_mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"[MarketFeed] Guardado en {save_path} ({size_mb:.1f} MB)")

    first = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
    last  = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    print(f"[MarketFeed] Total: {len(df):,} velas | {first} → {last}")
    return df


def load_csv(path: str) -> pd.DataFrame:
    """
    Carga el CSV histórico. Si no existe, lo descarga automáticamente.
    Llamado desde backtest_v3.py.
    """
    if not os.path.exists(path):
        print(f"[MarketFeed] CSV no encontrado en {path}. Descargando...")
        return download_historical_data(
            symbol    = cfg.network.symbol,
            days      = cfg.data.backtest_days,
            testnet   = False,
            save_path = path,
        )

    print(f"[MarketFeed] Cargando CSV: {path}")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Parsear timestamp (acepta unix ms, unix s, o string datetime)
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    if df[ts_col].dtype in ("int64", "float64"):
        unit = "ms" if df[ts_col].iloc[0] > 1e12 else "s"
        df["timestamp"] = pd.to_datetime(df[ts_col], unit=unit, utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    first = df["timestamp"].iloc[0].strftime("%Y-%m-%d")
    last  = df["timestamp"].iloc[-1].strftime("%Y-%m-%d")
    print(f"[MarketFeed] {len(df):,} velas | {first} → {last}")
    return df


# ─────────────────────────────────────────────
# LIVE FEED
# ─────────────────────────────────────────────

class LiveFeed(MarketFeed):
    """
    Se conecta a Binance Futuros y mantiene un snapshot actualizado.
    """

    CANDLES_TO_LOAD = {
        "1m": 500, "5m": 300, "15m": 200,
        "1h": 200, "4h": 150, "1d":  200, "1w": 100,
    }

    def __init__(self, symbol: str = "BTCUSDC", testnet: bool = True):
        self.symbol   = symbol.upper()
        self.testnet  = testnet
        self.base_url = (
            "https://testnet.binancefuture.com" if testnet
            else "https://fapi.binance.com"
        )
        self._ohlcv:               dict[str, pd.DataFrame] = {}
        self._last_price:          float = 0.0
        self._funding_rate:        float = 0.0
        self._funding_rate_next:   float = 0.0
        self._orderbook_imbalance: float = 0.0
        self._bid:                 float = 0.0
        self._ask:                 float = 0.0
        self._feed_latency_ms:     float = 0.0
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        print(f"[LiveFeed] Iniciando {'TESTNET' if self.testnet else 'PRODUCCIÓN'}...")
        self._preload_all_timeframes()
        self._refresh_funding_rate()
        self._running = True
        self._thread  = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        print(f"[LiveFeed] Listo. Precio: {self._last_price:.2f}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[LiveFeed] Detenido.")

    def get_snapshot(self) -> MarketSnapshot:
        t0 = time.monotonic()
        with self._lock:
            snap = MarketSnapshot(
                symbol              = self.symbol,
                timestamp           = datetime.now(timezone.utc),
                ohlcv_1m            = self._ohlcv.get("1m",  pd.DataFrame()).copy(),
                ohlcv_5m            = self._ohlcv.get("5m",  pd.DataFrame()).copy(),
                ohlcv_15m           = self._ohlcv.get("15m", pd.DataFrame()).copy(),
                ohlcv_1h            = self._ohlcv.get("1h",  pd.DataFrame()).copy(),
                ohlcv_4h            = self._ohlcv.get("4h",  pd.DataFrame()).copy(),
                ohlcv_1d            = self._ohlcv.get("1d",  pd.DataFrame()).copy(),
                ohlcv_1w            = self._ohlcv.get("1w",  pd.DataFrame()).copy(),
                last_price          = self._last_price,
                funding_rate        = self._funding_rate,
                funding_rate_next   = self._funding_rate_next,
                orderbook_imbalance = self._orderbook_imbalance,
                bid                 = self._bid,
                ask                 = self._ask,
                is_backtest         = False,
                feed_latency_ms     = self._feed_latency_ms,
            )
        snap.feed_latency_ms = (time.monotonic() - t0) * 1000
        return snap

    def _preload_all_timeframes(self) -> None:
        for tf, limit in self.CANDLES_TO_LOAD.items():
            try:
                df = self._fetch_klines(tf, limit)
                with self._lock:
                    self._ohlcv[tf] = df
                print(f"  [LiveFeed] {tf}: {len(df)} velas")
            except Exception as e:
                print(f"  [LiveFeed] Error cargando {tf}: {e}")
        with self._lock:
            if "1m" in self._ohlcv and not self._ohlcv["1m"].empty:
                self._last_price = float(self._ohlcv["1m"]["close"].iloc[-1])

    def _fetch_klines(self, interval: str, limit: int) -> pd.DataFrame:
        url  = f"{self.base_url}/fapi/v1/klines"
        resp = requests.get(
            url,
            params={"symbol": self.symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        resp.raise_for_status()
        return _parse_klines(resp.json())

    def _background_loop(self) -> None:
        counters = {tf: 0 for tf in self.CANDLES_TO_LOAD}
        refresh_every = {
            "1m": 15, "5m": 30,  "15m": 60,
            "1h": 120, "4h": 300, "1d": 600, "1w": 3600,
        }
        while self._running:
            for tf in self.CANDLES_TO_LOAD:
                counters[tf] += 1
                if counters[tf] >= refresh_every[tf]:
                    counters[tf] = 0
                    try:
                        df = self._fetch_klines(tf, 10)
                        with self._lock:
                            existing = self._ohlcv.get(tf, pd.DataFrame())
                            if existing.empty:
                                self._ohlcv[tf] = df
                            else:
                                combined = pd.concat([existing, df])
                                combined = combined[~combined.index.duplicated(keep="last")]
                                self._ohlcv[tf] = combined.sort_index()
                    except Exception as e:
                        print(f"[LiveFeed] Error actualizando {tf}: {e}")

            if counters.get("_funding", 0) % 300 == 0:
                try:
                    self._refresh_funding_rate()
                except Exception:
                    pass
            counters["_funding"] = counters.get("_funding", 0) + 1

            if counters.get("_ob", 0) % 5 == 0:
                try:
                    self._refresh_orderbook()
                except Exception:
                    pass
            counters["_ob"] = counters.get("_ob", 0) + 1

            time.sleep(1)

    def _refresh_funding_rate(self) -> None:
        url  = f"{self.base_url}/fapi/v1/premiumIndex"
        resp = requests.get(url, params={"symbol": self.symbol}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        with self._lock:
            self._last_price        = float(data.get("markPrice",       self._last_price))
            self._funding_rate      = float(data.get("lastFundingRate",  0))
            self._funding_rate_next = float(data.get("nextFundingRate",  0))

    def _refresh_orderbook(self) -> None:
        url  = f"{self.base_url}/fapi/v1/depth"
        resp = requests.get(url, params={"symbol": self.symbol, "limit": 20}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        bids_vol = sum(float(b[1]) for b in data.get("bids", []))
        asks_vol = sum(float(a[1]) for a in data.get("asks", []))
        total    = bids_vol + asks_vol
        with self._lock:
            self._orderbook_imbalance = (bids_vol - asks_vol) / total if total > 0 else 0.0
            if data.get("bids"):
                self._bid = float(data["bids"][0][0])
            if data.get("asks"):
                self._ask = float(data["asks"][0][0])


# ─────────────────────────────────────────────
# BACKTEST FEED
# ─────────────────────────────────────────────

class BacktestFeed(MarketFeed):
    """
    Sirve datos históricos vela a vela para el backtest.
    Si no encuentra el CSV, lo descarga automáticamente.
    """

    LOOKBACK = {
        "1m": 500, "5m": 300, "15m": 200,
        "1h": 200, "4h": 150, "1d":  200, "1w": 100,
    }

    def __init__(
        self,
        symbol:     str            = "BTCUSDC",
        csv_path:   Optional[str]  = None,
        start_date: Optional[str]  = None,
        end_date:   Optional[str]  = None,
        testnet:    bool           = False,
    ):
        self.symbol     = symbol.upper()
        self.csv_path   = csv_path or cfg.data.csv_path
        self.start_date = start_date
        self.end_date   = end_date
        self.testnet    = testnet

        self._df_1m_full:         pd.DataFrame          = pd.DataFrame()
        self._cursor:             int                   = 0
        self._current_snapshot:   Optional[MarketSnapshot] = None

    def start(self) -> None:
        # load_csv descarga automáticamente si no existe el archivo
        df = load_csv(self.csv_path)

        # Asegurar que timestamp sea el índice datetime
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index, utc=True)

        if self.start_date:
            df = df[df.index >= self.start_date]
        if self.end_date:
            df = df[df.index <= self.end_date]

        self._df_1m_full = df.sort_index()

        min_lookback  = max(self.LOOKBACK.values())
        self._cursor  = min_lookback
        print(f"[BacktestFeed] {len(self._df_1m_full):,} velas. "
              f"Backtest desde vela {self._cursor}.")

    def stop(self) -> None:
        print("[BacktestFeed] Finalizado.")

    def next(self) -> bool:
        if self._cursor >= len(self._df_1m_full):
            return False
        self._build_snapshot()
        self._cursor += 1
        return True

    @property
    def progress(self) -> float:
        total = len(self._df_1m_full)
        return self._cursor / total if total > 0 else 0.0

    def get_snapshot(self) -> MarketSnapshot:
        if self._current_snapshot is None:
            raise RuntimeError("Llamá a next() antes de get_snapshot()")
        return self._current_snapshot

    def _build_snapshot(self) -> None:
        df_1m = self._df_1m_full.iloc[
            max(0, self._cursor - self.LOOKBACK["1m"]): self._cursor
        ]
        current_time = df_1m.index[-1]
        last_price   = float(df_1m["close"].iloc[-1])

        def _resample_window(minutes: int, tf: str) -> pd.DataFrame:
            rows = max(0, self._cursor - self.LOOKBACK[tf] * minutes)
            return _resample_from_1m(
                self._df_1m_full.iloc[rows: self._cursor], tf
            ).tail(self.LOOKBACK[tf])

        self._current_snapshot = MarketSnapshot(
            symbol              = self.symbol,
            timestamp           = current_time.to_pydatetime(),
            ohlcv_1m            = df_1m,
            ohlcv_5m            = _resample_window(5,    "5m"),
            ohlcv_15m           = _resample_window(15,   "15m"),
            ohlcv_1h            = _resample_window(60,   "1h"),
            ohlcv_4h            = _resample_window(240,  "4h"),
            ohlcv_1d            = _resample_window(1440, "1d"),
            ohlcv_1w            = _resample_window(10080,"1w"),
            last_price          = last_price,
            funding_rate        = 0.0,
            funding_rate_next   = 0.0,
            orderbook_imbalance = 0.0,
            bid                 = last_price,
            ask                 = last_price,
            is_backtest         = True,
        )


# ─────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────

def create_feed(mode: str = "live", **kwargs) -> MarketFeed:
    if mode == "live":
        return LiveFeed(**kwargs)
    elif mode == "backtest":
        return BacktestFeed(**kwargs)
    else:
        raise ValueError(f"Modo desconocido: '{mode}'. Usar 'live' o 'backtest'.")


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from config import cfg

    print(f"market_feed.py v1.2")
    print(f"  Símbolo desde cfg:    {cfg.network.symbol}")
    print(f"  Entorno:              {'TESTNET' if cfg.network.testnet else 'PRODUCCIÓN'}")
    print(f"  Base URL:             {cfg.network.base_url}")
    print(f"  CSV path:             {cfg.data.csv_path}")
    print(f"  Días backtest:        {cfg.data.backtest_days}")
    print(f"  Actualización incr.:  activada")

    args = sys.argv[1:]

    if not args:
        print("\nUso:")
        print("  py -3.12 market_feed.py download        — descarga datos históricos (config.py)")
        print("  py -3.12 market_feed.py download 90     — descarga últimos 90 días")
        print("  py -3.12 market_feed.py live             — prueba conexión en vivo")
        sys.exit(0)

    cmd = args[0].lstrip("-")   # acepta tanto 'download' como '--download'

    if cmd == "download":
        days = int(args[1]) if len(args) > 1 else cfg.data.backtest_days
        download_historical_data(
            symbol    = cfg.network.symbol,
            days      = days,
            testnet   = False,           # siempre producción para datos históricos reales
            save_path = cfg.data.csv_path,
        )

    elif cmd == "live":
        feed = create_feed("live", symbol=cfg.network.symbol, testnet=cfg.network.testnet)
        feed.start()
        try:
            for _ in range(5):
                snap = feed.get_snapshot()
                print(
                    f"[{snap.timestamp:%H:%M:%S}] "
                    f"Precio: {snap.last_price:.2f} | "
                    f"Funding: {snap.funding_rate:.4%} | "
                    f"OB: {snap.orderbook_imbalance:+.3f}"
                )
                time.sleep(5)
        finally:
            feed.stop()

    else:
        print(f"Comando desconocido: '{args[0]}'")
        print("Usar: download | live")
        sys.exit(1)
