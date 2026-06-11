"""
market_feed.py  v1.1
====================
Capa de datos unificada.

Cambios respecto a v1.0:
  - URLs, símbolo, CANDLES_TO_LOAD y refresh_every leídos desde cfg.
  - Lógica de actualización incremental del CSV:
    al arrancar detecta la última fecha guardada y descarga
    solo las velas nuevas, appendeando al archivo existente.
    La descarga completa ocurre solo la primera vez.
  - BacktestFeed expone también daily_closes, daily_highs,
    daily_lows y recent_atrs para alimentar al RegimeDetector.

Contrato de salida: MarketSnapshot (sin cambios de interfaz).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from config import cfg

# ─────────────────────────────────────────────
# ESTRUCTURAS DE DATOS (sin cambios de interfaz)
# ─────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """
    Todo lo que el sistema necesita en un instante dado.
    indicators.py y regime_detector.py reciben esto.
    """
    symbol:    str
    timestamp: datetime

    # OHLCV por timeframe — DataFrame con columnas:
    # ['open','high','low','close','volume'], DatetimeIndex UTC
    # Las filas más recientes son las últimas.
    ohlcv_1m:  pd.DataFrame
    ohlcv_5m:  pd.DataFrame
    ohlcv_15m: pd.DataFrame
    ohlcv_1h:  pd.DataFrame
    ohlcv_4h:  pd.DataFrame
    ohlcv_1d:  pd.DataFrame
    ohlcv_1w:  pd.DataFrame

    # Datos en tiempo real (en backtest se aproximan)
    last_price:          float = 0.0
    funding_rate:        float = 0.0
    orderbook_imbalance: float = 0.0
    bid:                 float = 0.0
    ask:                 float = 0.0

    def get_ohlcv(self, timeframe: str) -> Optional[pd.DataFrame]:
        """Acceso por nombre de timeframe string."""
        return {
            "1m":  self.ohlcv_1m,  "5m":  self.ohlcv_5m,
            "15m": self.ohlcv_15m, "1h":  self.ohlcv_1h,
            "4h":  self.ohlcv_4h,  "1d":  self.ohlcv_1d,
            "1w":  self.ohlcv_1w,
        }.get(timeframe)

    # Propiedades para RegimeDetector (listas de precios diarios)
    @property
    def daily_closes(self) -> list[float]:
        if self.ohlcv_1d is not None and not self.ohlcv_1d.empty:
            return self.ohlcv_1d["close"].tolist()
        return []

    @property
    def daily_highs(self) -> list[float]:
        if self.ohlcv_1d is not None and not self.ohlcv_1d.empty:
            return self.ohlcv_1d["high"].tolist()
        return []

    @property
    def daily_lows(self) -> list[float]:
        if self.ohlcv_1d is not None and not self.ohlcv_1d.empty:
            return self.ohlcv_1d["low"].tolist()
        return []


# ─────────────────────────────────────────────
# INTERFAZ BASE
# ─────────────────────────────────────────────

class BaseFeed(ABC):
    @abstractmethod
    def get_snapshot(self) -> MarketSnapshot:
        """Devuelve el snapshot más reciente del mercado."""
        ...

    @abstractmethod
    def is_ready(self) -> bool:
        """True si el feed tiene datos suficientes para operar."""
        ...


# ─────────────────────────────────────────────
# HELPERS REST
# ─────────────────────────────────────────────

def _fetch_klines_rest(
    symbol:    str,
    interval:  str,
    limit:     int = 1000,
    start_ms:  Optional[int] = None,
    base_url:  Optional[str] = None,
) -> list[list]:
    """
    Descarga velas desde la API REST de Binance Futuros.

    Parámetros:
        symbol    Par de trading (ej: "BTCUSDC")
        interval  Timeframe (ej: "1m", "1h", "1d")
        limit     Máximo de velas por request (máx Binance: 1500)
        start_ms  Timestamp de inicio en milisegundos (opcional)
        base_url  URL base del endpoint. Si None usa cfg.

    Devuelve lista de listas con el formato de Binance klines.
    """
    url = (base_url or cfg.network.base_url) + "/fapi/v1/klines"
    params: dict = {
        "symbol":   symbol,
        "interval": interval,
        "limit":    min(limit, 1500),
    }
    if start_ms:
        params["startTime"] = start_ms

    resp = requests.get(url, params=params, timeout=cfg.network.rest_timeout)
    resp.raise_for_status()
    return resp.json()


def _klines_to_df(raw: list[list]) -> pd.DataFrame:
    """Convierte la respuesta de Binance klines a DataFrame OHLCV."""
    if not raw:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.set_index("open_time", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def _resample_to_tf(df_1m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """
    Resamplea un DataFrame de 1m a un timeframe mayor.

    Timeframes soportados: 5m, 15m, 1h, 4h, 1d, 1w
    """
    rule_map = {
        "5m": "5min", "15m": "15min", "1h": "1h",
        "4h": "4h",   "1d": "1D",     "1w": "1W",
    }
    rule = rule_map.get(tf)
    if not rule:
        return df_1m

    resampled = df_1m.resample(rule).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna()
    return resampled


# ─────────────────────────────────────────────
# LIVE FEED
# ─────────────────────────────────────────────

class LiveFeed(BaseFeed):
    """
    Feed en vivo contra Binance Futuros (testnet o producción).
    Lee todos sus parámetros desde cfg.

    Mantiene un buffer de velas por timeframe actualizado
    en background mediante polling REST.
    """

    def __init__(self):
        self._symbol   = cfg.network.symbol
        self._base_url = cfg.network.base_url
        self._buffers: dict[str, pd.DataFrame] = {}
        self._funding_rate   = 0.0
        self._obi            = 0.0
        self._bid            = 0.0
        self._ask            = 0.0
        self._last_price     = 0.0
        self._ready          = False
        self._lock           = threading.Lock()
        self._refresh_counts = {tf: 0 for tf in cfg.data.candles_live}

    def start(self) -> None:
        """Arranca el background thread de actualización."""
        self._load_initial()
        t = threading.Thread(target=self._background_loop, daemon=True)
        t.start()

    def _load_initial(self) -> None:
        """Carga inicial de todos los timeframes."""
        for tf, n_candles in cfg.data.candles_live.items():
            raw = _fetch_klines_rest(
                self._symbol, tf, limit=n_candles,
                base_url=self._base_url
            )
            df = _klines_to_df(raw)
            with self._lock:
                self._buffers[tf] = df
        self._ready = True

    def _background_loop(self) -> None:
        """Actualiza cada timeframe según su intervalo de refresco."""
        while True:
            for tf, interval_secs in cfg.data.refresh_intervals.items():
                self._refresh_counts[tf] = self._refresh_counts.get(tf, 0) + 1
                if self._refresh_counts[tf] * 15 >= interval_secs:
                    self._refresh_counts[tf] = 0
                    try:
                        self._update_tf(tf)
                    except Exception as e:
                        pass  # log en producción
            self._update_realtime()
            time.sleep(15)

    def _update_tf(self, tf: str) -> None:
        """Descarga las últimas N velas y actualiza el buffer."""
        raw = _fetch_klines_rest(
            self._symbol, tf, limit=5, base_url=self._base_url
        )
        new_df = _klines_to_df(raw)
        with self._lock:
            existing = self._buffers.get(tf, pd.DataFrame())
            combined = pd.concat([existing, new_df])
            combined = combined[~combined.index.duplicated(keep="last")]
            combined.sort_index(inplace=True)
            max_candles = cfg.data.candles_live.get(tf, 200)
            self._buffers[tf] = combined.iloc[-max_candles:]

    def _update_realtime(self) -> None:
        """Actualiza precio, funding rate y order book."""
        try:
            # Precio mark
            url = self._base_url + "/fapi/v1/ticker/price"
            r = requests.get(
                url, params={"symbol": self._symbol},
                timeout=cfg.network.rest_timeout
            )
            if r.ok:
                self._last_price = float(r.json()["price"])

            # Funding rate
            url2 = self._base_url + "/fapi/v1/premiumIndex"
            r2 = requests.get(
                url2, params={"symbol": self._symbol},
                timeout=cfg.network.rest_timeout
            )
            if r2.ok:
                self._funding_rate = float(r2.json().get("lastFundingRate", 0))

            # Order book (top 20 niveles)
            url3 = self._base_url + "/fapi/v1/depth"
            r3 = requests.get(
                url3, params={"symbol": self._symbol, "limit": 20},
                timeout=cfg.network.rest_timeout
            )
            if r3.ok:
                book  = r3.json()
                bids  = sum(float(b[1]) for b in book.get("bids", []))
                asks  = sum(float(a[1]) for a in book.get("asks", []))
                total = bids + asks
                self._obi  = (bids - asks) / total if total > 0 else 0.0
                if book.get("bids"):
                    self._bid = float(book["bids"][0][0])
                if book.get("asks"):
                    self._ask = float(book["asks"][0][0])
        except Exception:
            pass

    def get_snapshot(self) -> MarketSnapshot:
        with self._lock:
            buffers = {tf: df.copy() for tf, df in self._buffers.items()}

        def _get(tf):
            return buffers.get(tf, pd.DataFrame())

        return MarketSnapshot(
            symbol    = self._symbol,
            timestamp = datetime.now(timezone.utc),
            ohlcv_1m  = _get("1m"),
            ohlcv_5m  = _get("5m"),
            ohlcv_15m = _get("15m"),
            ohlcv_1h  = _get("1h"),
            ohlcv_4h  = _get("4h"),
            ohlcv_1d  = _get("1d"),
            ohlcv_1w  = _get("1w"),
            last_price          = self._last_price,
            funding_rate        = self._funding_rate,
            orderbook_imbalance = self._obi,
            bid                 = self._bid,
            ask                 = self._ask,
        )

    def is_ready(self) -> bool:
        return self._ready


# ─────────────────────────────────────────────
# BACKTEST FEED (con descarga incremental)
# ─────────────────────────────────────────────

class BacktestFeed(BaseFeed):
    """
    Feed para backtest. Lee el CSV histórico de 1m y lo sirve
    vela a vela resampleando a los demás timeframes en cada paso.

    Descarga incremental:
    - Si el CSV no existe → descarga completa (cfg.data.backtest_days días)
    - Si el CSV existe → descarga solo las velas nuevas desde la
      última fecha guardada hasta hoy y las appendea

    Lee su configuración desde cfg (símbolo, días, paths).
    """

    def __init__(self, days: Optional[int] = None):
        self._symbol  = cfg.network.symbol
        self._days    = days or cfg.data.backtest_days
        self._csv_path = Path(cfg.data.data_dir) / cfg.data.csv_filename

        self._df_1m: Optional[pd.DataFrame] = None
        self._cursor: int = 0   # índice de la vela actual en el backtest

        # Cuántas velas de lookback usar para cada TF en el snapshot
        self._lookback = cfg.data.candles_backtest_lookback

    # ─────────────────────────────────────────
    # CARGA Y DESCARGA INCREMENTAL
    # ─────────────────────────────────────────

    def load(self) -> None:
        """
        Carga el CSV histórico. Si no existe o está desactualizado,
        descarga lo que falta y actualiza el archivo.
        """
        Path(cfg.data.data_dir).mkdir(parents=True, exist_ok=True)

        if self._csv_path.exists():
            print(f"[BacktestFeed] Cargando CSV existente: {self._csv_path}")
            self._df_1m = pd.read_csv(
                self._csv_path,
                index_col=0,
                parse_dates=True,
            )
            self._df_1m.index = pd.to_datetime(self._df_1m.index, utc=True)
            self._df_1m.sort_index(inplace=True)

            # Detectar cuántas velas faltan desde la última fecha
            last_ts = self._df_1m.index[-1]
            now_utc = datetime.now(timezone.utc)
            gap_minutes = int((now_utc - last_ts).total_seconds() / 60)

            if gap_minutes > 2:
                print(
                    f"[BacktestFeed] CSV desactualizado por {gap_minutes} velas. "
                    "Descargando incremento..."
                )
                self._download_incremental(since_ts=last_ts)
            else:
                print("[BacktestFeed] CSV actualizado.")
        else:
            print(
                f"[BacktestFeed] CSV no encontrado. "
                f"Descargando {self._days} días de historia..."
            )
            self._download_full()

        print(f"[BacktestFeed] Total velas 1m: {len(self._df_1m):,}")

    def _download_full(self) -> None:
        """Descarga completa: cfg.data.backtest_days días de velas 1m."""
        end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
        start_ms = end_ms - self._days * 24 * 60 * 60 * 1000

        all_candles: list[pd.DataFrame] = []
        current_ms  = start_ms
        base_url    = cfg.network.base_url

        print(f"[BacktestFeed] Descargando desde Binance ({self._days} días)...")
        while current_ms < end_ms:
            raw = _fetch_klines_rest(
                self._symbol, "1m",
                limit=1500, start_ms=current_ms,
                base_url=base_url,
            )
            if not raw:
                break
            df_chunk = _klines_to_df(raw)
            all_candles.append(df_chunk)
            current_ms = int(df_chunk.index[-1].timestamp() * 1000) + 60_000
            print(
                f"  Hasta {df_chunk.index[-1].strftime('%Y-%m-%d %H:%M')} UTC "
                f"({len(df_chunk)} velas)...",
                end="\r",
            )
            time.sleep(0.1)   # respetar rate limit

        self._df_1m = pd.concat(all_candles)
        self._df_1m = self._df_1m[~self._df_1m.index.duplicated(keep="last")]
        self._df_1m.sort_index(inplace=True)
        self._df_1m.to_csv(self._csv_path)
        print(f"\n[BacktestFeed] Guardado en {self._csv_path}")

    def _download_incremental(self, since_ts: pd.Timestamp) -> None:
        """Descarga solo las velas nuevas y appendea al CSV."""
        start_ms = int(since_ts.timestamp() * 1000) + 60_000
        end_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
        base_url = cfg.network.base_url

        new_candles: list[pd.DataFrame] = []
        current_ms = start_ms

        while current_ms < end_ms:
            raw = _fetch_klines_rest(
                self._symbol, "1m",
                limit=1500, start_ms=current_ms,
                base_url=base_url,
            )
            if not raw:
                break
            df_chunk = _klines_to_df(raw)
            new_candles.append(df_chunk)
            current_ms = int(df_chunk.index[-1].timestamp() * 1000) + 60_000
            time.sleep(0.05)

        if new_candles:
            df_new = pd.concat(new_candles)
            df_new = df_new[~df_new.index.duplicated(keep="last")]
            self._df_1m = pd.concat([self._df_1m, df_new])
            self._df_1m = self._df_1m[~self._df_1m.index.duplicated(keep="last")]
            self._df_1m.sort_index(inplace=True)
            self._df_1m.to_csv(self._csv_path)
            print(f"[BacktestFeed] +{len(df_new)} velas nuevas guardadas.")

    # ─────────────────────────────────────────
    # ITERACIÓN VELA A VELA
    # ─────────────────────────────────────────

    def reset(self, start_idx: int = 0) -> None:
        """Reinicia el cursor al índice dado."""
        self._cursor = start_idx

    def step(self) -> bool:
        """
        Avanza una vela de 1m.
        Devuelve True si hay más velas, False si el backtest terminó.
        """
        self._cursor += 1
        return self._cursor < len(self._df_1m)

    def get_snapshot(self) -> MarketSnapshot:
        """
        Devuelve el snapshot del momento actual del cursor.
        Construye cada timeframe resampleando desde los datos 1m visibles.
        """
        if self._df_1m is None:
            raise RuntimeError("Llamar load() antes de get_snapshot()")

        # Solo usar datos hasta el cursor (sin lookahead)
        df_visible = self._df_1m.iloc[: self._cursor + 1]
        current_row = df_visible.iloc[-1]

        def _tail(df: pd.DataFrame, tf: str) -> pd.DataFrame:
            n = self._lookback.get(tf, 200)
            return df.iloc[-n:] if len(df) >= n else df

        df_5m  = _resample_to_tf(df_visible, "5m")
        df_15m = _resample_to_tf(df_visible, "15m")
        df_1h  = _resample_to_tf(df_visible, "1h")
        df_4h  = _resample_to_tf(df_visible, "4h")
        df_1d  = _resample_to_tf(df_visible, "1d")
        df_1w  = _resample_to_tf(df_visible, "1w")

        close = float(current_row["close"])

        return MarketSnapshot(
            symbol    = self._symbol,
            timestamp = df_visible.index[-1].to_pydatetime(),
            ohlcv_1m  = _tail(df_visible, "1m"),
            ohlcv_5m  = _tail(df_5m,  "5m"),
            ohlcv_15m = _tail(df_15m, "15m"),
            ohlcv_1h  = _tail(df_1h,  "1h"),
            ohlcv_4h  = _tail(df_4h,  "4h"),
            ohlcv_1d  = _tail(df_1d,  "1d"),
            ohlcv_1w  = _tail(df_1w,  "1w"),
            last_price          = close,
            funding_rate        = 0.0,   # no disponible en backtest
            orderbook_imbalance = 0.0,
            bid                 = close * 0.9999,
            ask                 = close * 1.0001,
        )

    def is_ready(self) -> bool:
        return self._df_1m is not None and len(self._df_1m) > 0

    @property
    def total_candles(self) -> int:
        return len(self._df_1m) if self._df_1m is not None else 0

    @property
    def current_index(self) -> int:
        return self._cursor

    @property
    def progress_pct(self) -> float:
        if self.total_candles == 0:
            return 0.0
        return self._cursor / self.total_candles * 100


# ─────────────────────────────────────────────
# FUNCIÓN DE ACCESO UNIFICADO
# ─────────────────────────────────────────────

def create_feed(mode: str = "live", days: Optional[int] = None) -> BaseFeed:
    """
    Punto de entrada único para crear el feed correcto.

    Parámetros:
        mode   "live" para el bot en vivo, "backtest" para simulación
        days   Solo para backtest: cuántos días de historia usar.
               Si None usa cfg.data.backtest_days

    Uso:
        feed = create_feed("backtest", days=90)
        if isinstance(feed, BacktestFeed):
            feed.load()
        while feed.step():
            snapshot = feed.get_snapshot()
    """
    if mode == "backtest":
        return BacktestFeed(days=days)
    elif mode == "live":
        feed = LiveFeed()
        feed.start()
        return feed
    else:
        raise ValueError(f"mode debe ser 'live' o 'backtest', got '{mode}'")


if __name__ == "__main__":
    print(f"market_feed.py v1.1")
    print(f"  Símbolo desde cfg:   {cfg.network.symbol}")
    print(f"  Entorno:             {'TESTNET' if cfg.network.testnet else 'PRODUCCIÓN'}")
    print(f"  Base URL:            {cfg.network.base_url}")
    print(f"  CSV path:            {cfg.data.csv_path}")
    print(f"  Días backtest:       {cfg.data.backtest_days}")
    print(f"  Actualización incr.: activada")
