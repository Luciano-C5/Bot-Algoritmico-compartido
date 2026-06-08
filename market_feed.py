"""
market_feed.py
==============
Capa de datos unificada. Tanto el backtest como el bot en vivo
usan esta misma interfaz. El resto del sistema no sabe (ni le importa)
de dónde vienen los datos.

Modos:
  - LiveFeed:     WebSocket + REST contra Binance (testnet o producción)
  - BacktestFeed: Lee CSVs o descarga datos históricos, los sirve vela a vela

Contrato de salida: MarketSnapshot (dataclass definida acá)
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

# ─────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────

@dataclass
class MarketSnapshot:
    """
    Todo lo que el sistema necesita saber del mercado en un instante dado.
    indicators.py recibe esto y devuelve los valores calculados.
    """
    symbol: str
    timestamp: datetime

    # OHLCV por timeframe. Cada DataFrame tiene columnas:
    # ['open', 'high', 'low', 'close', 'volume']
    # con DatetimeIndex UTC. Las filas más recientes son las últimas.
    ohlcv_1m:  pd.DataFrame
    ohlcv_5m:  pd.DataFrame
    ohlcv_15m: pd.DataFrame
    ohlcv_1h:  pd.DataFrame
    ohlcv_4h:  pd.DataFrame
    ohlcv_1d:  pd.DataFrame
    ohlcv_1w:  pd.DataFrame

    # Datos de mercado en tiempo real (en backtest se aproximan o se ignoran)
    last_price:          float = 0.0
    funding_rate:        float = 0.0          # actual, no el próximo
    funding_rate_next:   float = 0.0          # próximo cobro
    orderbook_imbalance: float = 0.0          # rango -1.0 a +1.0
    bid:                 float = 0.0
    ask:                 float = 0.0

    # Metadata útil para logging y decisiones
    is_backtest:         bool  = False
    feed_latency_ms:     float = 0.0          # latencia del feed en vivo

    @property
    def spread_pct(self) -> float:
        if self.bid > 0:
            return (self.ask - self.bid) / self.bid * 100
        return 0.0

    @property
    def current_close(self) -> float:
        """Precio de cierre de la última vela de 1m (o last_price si es más reciente)."""
        if not self.ohlcv_1m.empty:
            return float(self.ohlcv_1m['close'].iloc[-1])
        return self.last_price


# ─────────────────────────────────────────────
# INTERFAZ BASE (contrato que deben cumplir Live y Backtest)
# ─────────────────────────────────────────────

class MarketFeed(ABC):
    """
    Interfaz que strategy.py, scoring.py y bot.py usan.
    Nunca llaman a LiveFeed o BacktestFeed directamente.
    """

    @abstractmethod
    def get_snapshot(self) -> MarketSnapshot:
        """Devuelve el estado actual del mercado."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Inicia conexiones, precarga de datos, etc."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Cierra conexiones limpiamente."""
        ...


# ─────────────────────────────────────────────
# FUNCIONES AUXILIARES COMPARTIDAS
# ─────────────────────────────────────────────

TIMEFRAME_MAP = {
    '1m':  1,
    '5m':  5,
    '15m': 15,
    '1h':  60,
    '4h':  240,
    '1d':  1440,
    '1w':  10080,
}

def _parse_klines(raw: list) -> pd.DataFrame:
    """
    Convierte la respuesta de la API de Binance (lista de listas)
    a un DataFrame limpio con DatetimeIndex UTC.
    """
    df = pd.DataFrame(raw, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    df.set_index('open_time', inplace=True)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]


def _resample_from_1m(df_1m: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Construye cualquier timeframe mayor a partir de velas de 1m.
    Útil en backtest para no necesitar archivos separados por timeframe.
    """
    rule_map = {
        '5m': '5min', '15m': '15min', '1h': '1h',
        '4h': '4h',   '1d': '1D',    '1w': '1W',
    }
    rule = rule_map.get(timeframe)
    if rule is None:
        raise ValueError(f"Timeframe no soportado para resample: {timeframe}")

    df = df_1m.resample(rule, label='left', closed='left').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna()
    return df


# ─────────────────────────────────────────────
# LIVE FEED
# ─────────────────────────────────────────────

class LiveFeed(MarketFeed):
    """
    Se conecta a Binance Futuros (testnet o producción) y mantiene
    un snapshot actualizado en memoria. El bot lee ese snapshot
    en cada ciclo sin bloquear.

    Uso:
        feed = LiveFeed(symbol='BTCUSDC', testnet=True)
        feed.start()
        snapshot = feed.get_snapshot()
        feed.stop()
    """

    BASE_URL_TESTNET = 'https://testnet.binancefuture.com'
    BASE_URL_PROD    = 'https://fapi.binance.com'
    WS_URL_TESTNET   = 'wss://stream.binancefuture.com'
    WS_URL_PROD      = 'wss://fstream.binance.com'

    # Cuántas velas históricas cargar por timeframe al arrancar
    CANDLES_TO_LOAD = {
        '1m': 500, '5m': 300, '15m': 200,
        '1h': 200, '4h': 150, '1d': 200, '1w': 100,
    }

    def __init__(self, symbol: str = 'BTCUSDC', testnet: bool = True):
        self.symbol  = symbol.upper()
        self.testnet = testnet
        self.base_url = self.BASE_URL_TESTNET if testnet else self.BASE_URL_PROD

        # Cache interno de DataFrames, actualizado en background
        self._ohlcv: dict[str, pd.DataFrame] = {}
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

    # ── Arranque y parada ──────────────────────────────────────────────

    def start(self) -> None:
        print(f"[LiveFeed] Iniciando en {'TESTNET' if self.testnet else 'PRODUCCIÓN'}...")
        self._preload_all_timeframes()
        self._refresh_funding_rate()
        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True)
        self._thread.start()
        print(f"[LiveFeed] Listo. Precio actual: {self._last_price:.2f}")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        print("[LiveFeed] Detenido.")

    # ── Snapshot público ───────────────────────────────────────────────

    def get_snapshot(self) -> MarketSnapshot:
        t0 = time.monotonic()
        with self._lock:
            snap = MarketSnapshot(
                symbol            = self.symbol,
                timestamp         = datetime.now(timezone.utc),
                ohlcv_1m          = self._ohlcv.get('1m',  pd.DataFrame()).copy(),
                ohlcv_5m          = self._ohlcv.get('5m',  pd.DataFrame()).copy(),
                ohlcv_15m         = self._ohlcv.get('15m', pd.DataFrame()).copy(),
                ohlcv_1h          = self._ohlcv.get('1h',  pd.DataFrame()).copy(),
                ohlcv_4h          = self._ohlcv.get('4h',  pd.DataFrame()).copy(),
                ohlcv_1d          = self._ohlcv.get('1d',  pd.DataFrame()).copy(),
                ohlcv_1w          = self._ohlcv.get('1w',  pd.DataFrame()).copy(),
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

    # ── Carga inicial ──────────────────────────────────────────────────

    def _preload_all_timeframes(self) -> None:
        for tf, limit in self.CANDLES_TO_LOAD.items():
            df = self._fetch_klines(tf, limit)
            with self._lock:
                self._ohlcv[tf] = df
            print(f"  [LiveFeed] {tf}: {len(df)} velas cargadas")

        # Precio inicial desde la vela más reciente de 1m
        with self._lock:
            if not self._ohlcv['1m'].empty:
                self._last_price = float(self._ohlcv['1m']['close'].iloc[-1])

    def _fetch_klines(self, interval: str, limit: int) -> pd.DataFrame:
        """Descarga velas OHLCV de la API REST."""
        url = f"{self.base_url}/fapi/v1/klines"
        params = {'symbol': self.symbol, 'interval': interval, 'limit': limit}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return _parse_klines(resp.json())

    # ── Loop de actualización en background ───────────────────────────

    def _background_loop(self) -> None:
        """
        Hilo daemon que actualiza datos periódicamente.
        En producción esto se reemplazaría por WebSocket para 1m,
        pero polling cada segundo es suficiente para empezar
        y más fácil de debuggear.
        """
        counters = {tf: 0 for tf in self.CANDLES_TO_LOAD}
        # Cada cuántos segundos refrescar cada timeframe
        refresh_every = {
            '1m': 15,   '5m': 30,   '15m': 60,
            '1h': 120,  '4h': 300,  '1d': 600, '1w': 3600,
        }

        while self._running:
            for tf in self.CANDLES_TO_LOAD:
                counters[tf] += 1
                if counters[tf] >= refresh_every[tf]:
                    counters[tf] = 0
                    try:
                        df = self._fetch_klines(tf, 10)  # Solo las últimas 10 velas
                        with self._lock:
                            # Actualiza o agrega las velas nuevas
                            existing = self._ohlcv.get(tf, pd.DataFrame())
                            if existing.empty:
                                self._ohlcv[tf] = df
                            else:
                                combined = pd.concat([existing, df])
                                combined = combined[~combined.index.duplicated(keep='last')]
                                self._ohlcv[tf] = combined.sort_index()
                    except Exception as e:
                        print(f"[LiveFeed] Error actualizando {tf}: {e}")

            # Funding rate cada 5 minutos
            if counters.get('_funding', 0) % 300 == 0:
                try:
                    self._refresh_funding_rate()
                except Exception as e:
                    print(f"[LiveFeed] Error funding rate: {e}")
            counters['_funding'] = counters.get('_funding', 0) + 1

            # Orderbook cada 5 segundos
            if counters.get('_ob', 0) % 5 == 0:
                try:
                    self._refresh_orderbook()
                except Exception as e:
                    print(f"[LiveFeed] Error orderbook: {e}")
            counters['_ob'] = counters.get('_ob', 0) + 1

            time.sleep(1)

    def _refresh_funding_rate(self) -> None:
        url = f"{self.base_url}/fapi/v1/premiumIndex"
        resp = requests.get(url, params={'symbol': self.symbol}, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        with self._lock:
            self._last_price        = float(data.get('markPrice', self._last_price))
            self._funding_rate      = float(data.get('lastFundingRate', 0))
            self._funding_rate_next = float(data.get('nextFundingRate', 0))

    def _refresh_orderbook(self) -> None:
        url = f"{self.base_url}/fapi/v1/depth"
        resp = requests.get(url, params={'symbol': self.symbol, 'limit': 20}, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        bids_vol = sum(float(b[1]) for b in data.get('bids', []))
        asks_vol = sum(float(a[1]) for a in data.get('asks', []))
        total = bids_vol + asks_vol

        with self._lock:
            if total > 0:
                # +1.0 = solo compras, -1.0 = solo ventas, 0 = equilibrado
                self._orderbook_imbalance = (bids_vol - asks_vol) / total
            if data.get('bids'):
                self._bid = float(data['bids'][0][0])
            if data.get('asks'):
                self._ask = float(data['asks'][0][0])


# ─────────────────────────────────────────────
# BACKTEST FEED
# ─────────────────────────────────────────────

class BacktestFeed(MarketFeed):
    """
    Sirve datos históricos vela a vela, exactamente igual que LiveFeed.
    El backtest avanza el tiempo llamando next() en cada iteración.

    Uso:
        feed = BacktestFeed(symbol='BTCUSDC', csv_path='data/BTCUSDC_1m.csv')
        feed.start()
        while feed.next():
            snapshot = feed.get_snapshot()
            # ... misma lógica que en vivo
        feed.stop()

    Si no tenés el CSV, usa download_historical_data() para descargarlo.
    """

    # Cuántas velas de contexto hacia atrás incluir en el snapshot
    # (suficiente para calcular todos los indicadores)
    LOOKBACK = {
        '1m': 500, '5m': 300, '15m': 200,
        '1h': 200, '4h': 150, '1d': 200, '1w': 100,
    }

    def __init__(
        self,
        symbol:     str = 'BTCUSDC',
        csv_path:   Optional[str] = None,
        start_date: Optional[str] = None,   # 'YYYY-MM-DD'
        end_date:   Optional[str] = None,
        testnet:    bool = False,            # para descarga si no hay CSV
    ):
        self.symbol     = symbol.upper()
        self.csv_path   = csv_path
        self.start_date = start_date
        self.end_date   = end_date
        self.testnet    = testnet

        self._df_1m_full: pd.DataFrame = pd.DataFrame()  # todos los datos de 1m
        self._cursor:     int = 0                        # posición actual
        self._current_snapshot: Optional[MarketSnapshot] = None

    # ── Arranque ───────────────────────────────────────────────────────

    def start(self) -> None:
        print(f"[BacktestFeed] Cargando datos para {self.symbol}...")

        if self.csv_path:
            self._df_1m_full = self._load_csv(self.csv_path)
        else:
            print("[BacktestFeed] No se proveyó CSV, descargando datos históricos...")
            self._df_1m_full = self._download_1m(testnet=self.testnet)

        # Filtrar por rango de fechas si se especificó
        if self.start_date:
            self._df_1m_full = self._df_1m_full[self._df_1m_full.index >= self.start_date]
        if self.end_date:
            self._df_1m_full = self._df_1m_full[self._df_1m_full.index <= self.end_date]

        # El cursor empieza donde hay suficientes velas para el lookback más largo
        min_lookback = max(self.LOOKBACK.values())
        self._cursor = min_lookback
        total = len(self._df_1m_full)
        print(f"[BacktestFeed] {total:,} velas de 1m cargadas. "
              f"Backtest desde la vela {self._cursor} hasta {total}.")

    def stop(self) -> None:
        print("[BacktestFeed] Finalizado.")

    # ── Iteración ──────────────────────────────────────────────────────

    def next(self) -> bool:
        """
        Avanza una vela. Devuelve False cuando se terminaron los datos.
        El backtest llama esto en cada iteración del loop principal.
        """
        if self._cursor >= len(self._df_1m_full):
            return False
        self._build_snapshot()
        self._cursor += 1
        return True

    @property
    def progress(self) -> float:
        """Porcentaje de progreso del backtest (0.0 a 1.0)."""
        total = len(self._df_1m_full)
        if total == 0:
            return 0.0
        return self._cursor / total

    @property
    def total_candles(self) -> int:
        return max(0, len(self._df_1m_full) - max(self.LOOKBACK.values()))

    # ── Snapshot ───────────────────────────────────────────────────────

    def get_snapshot(self) -> MarketSnapshot:
        if self._current_snapshot is None:
            raise RuntimeError("Llamá a next() antes de get_snapshot()")
        return self._current_snapshot

    def _build_snapshot(self) -> None:
        """Construye el snapshot con los datos hasta el cursor actual."""
        df_1m = self._df_1m_full.iloc[max(0, self._cursor - self.LOOKBACK['1m']):self._cursor]
        current_time = df_1m.index[-1]
        last_price   = float(df_1m['close'].iloc[-1])

        self._current_snapshot = MarketSnapshot(
            symbol    = self.symbol,
            timestamp = current_time.to_pydatetime(),

            ohlcv_1m  = df_1m,
            ohlcv_5m  = _resample_from_1m(df_1m, '5m').tail(self.LOOKBACK['5m']),
            ohlcv_15m = _resample_from_1m(df_1m, '15m').tail(self.LOOKBACK['15m']),
            ohlcv_1h  = _resample_from_1m(
                self._df_1m_full.iloc[max(0, self._cursor - self.LOOKBACK['1h']*60):self._cursor],
                '1h'
            ).tail(self.LOOKBACK['1h']),
            ohlcv_4h  = _resample_from_1m(
                self._df_1m_full.iloc[max(0, self._cursor - self.LOOKBACK['4h']*240):self._cursor],
                '4h'
            ).tail(self.LOOKBACK['4h']),
            ohlcv_1d  = _resample_from_1m(
                self._df_1m_full.iloc[max(0, self._cursor - self.LOOKBACK['1d']*1440):self._cursor],
                '1d'
            ).tail(self.LOOKBACK['1d']),
            ohlcv_1w  = _resample_from_1m(
                self._df_1m_full.iloc[max(0, self._cursor - self.LOOKBACK['1w']*10080):self._cursor],
                '1w'
            ).tail(self.LOOKBACK['1w']),

            last_price          = last_price,
            funding_rate        = 0.0,   # no disponible en backtest histórico
            funding_rate_next   = 0.0,
            orderbook_imbalance = 0.0,   # no disponible en backtest histórico
            bid                 = last_price,
            ask                 = last_price,
            is_backtest         = True,
        )

    # ── Carga y descarga de datos ─────────────────────────────────────

    @staticmethod
    def _load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        return df[['open', 'high', 'low', 'close', 'volume']]

    @staticmethod
    def download_historical_data(
        symbol:    str = 'BTCUSDC',
        days:      int = 1000,
        testnet:   bool = False,
        save_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Descarga datos históricos de 1m desde Binance.
        Binance limita a 1500 velas por request, así que itera
        hacia atrás en el tiempo hasta cubrir los días pedidos.

        Ejemplo:
            df = BacktestFeed.download_historical_data(days=365, save_path='data/BTCUSDC_1m.csv')
        """
        base_url = (BacktestFeed.BASE_URL_TESTNET if testnet
                    else 'https://fapi.binance.com')
        url = f"{base_url}/fapi/v1/klines"

        total_minutes = days * 24 * 60
        batch_size    = 1500
        all_frames    = []
        end_time_ms   = None   # None = desde ahora hacia atrás

        print(f"[BacktestFeed] Descargando ~{total_minutes:,} velas de 1m ({days} días)...")

        fetched = 0
        while fetched < total_minutes:
            params = {
                'symbol':   symbol.upper(),
                'interval': '1m',
                'limit':    batch_size,
            }
            if end_time_ms:
                params['endTime'] = end_time_ms

            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                break

            df_batch = _parse_klines(raw)
            all_frames.append(df_batch)
            fetched     += len(df_batch)
            end_time_ms  = int(df_batch.index[0].timestamp() * 1000) - 1

            print(f"  {fetched:,} / {total_minutes:,} velas descargadas...", end='\r')
            time.sleep(0.2)   # respetar rate limits

        print()
        if not all_frames:
            raise RuntimeError("No se pudieron descargar datos históricos.")

        df = pd.concat(all_frames).sort_index()
        df = df[~df.index.duplicated(keep='last')]

        if save_path:
            df.to_csv(save_path)
            print(f"[BacktestFeed] Guardado en {save_path}")

        print(f"[BacktestFeed] Total: {len(df):,} velas descargadas.")
        return df

    # Referencia estática para la URL de testnet (usada en download)
    BASE_URL_TESTNET = 'https://testnet.binancefuture.com'


# ─────────────────────────────────────────────
# FACTORY — el resto del sistema usa esto
# ─────────────────────────────────────────────

def create_feed(mode: str = 'live', **kwargs) -> MarketFeed:
    """
    Punto de entrada único para crear feeds.

    Ejemplos:
        # Bot en vivo contra testnet
        feed = create_feed('live', symbol='BTCUSDC', testnet=True)

        # Backtest desde CSV
        feed = create_feed('backtest', csv_path='data/BTCUSDC_1m.csv',
                           start_date='2023-01-01', end_date='2024-01-01')

        # Backtest descargando datos
        feed = create_feed('backtest', days=365)
    """
    if mode == 'live':
        return LiveFeed(**kwargs)
    elif mode == 'backtest':
        return BacktestFeed(**kwargs)
    else:
        raise ValueError(f"Modo de feed desconocido: '{mode}'. Usar 'live' o 'backtest'.")


# ─────────────────────────────────────────────
# USO DE EJEMPLO (correr este archivo directamente para testear)
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'download':
        # python market_feed.py download
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        BacktestFeed.download_historical_data(
            symbol='BTCUSDC',
            days=days,
            testnet=False,
            save_path=f'data/BTCUSDC_1m_{days}d.csv'
        )

    elif len(sys.argv) > 1 and sys.argv[1] == 'live':
        # python market_feed.py live
        feed = create_feed('live', symbol='BTCUSDC', testnet=True)
        feed.start()
        try:
            for _ in range(5):
                snap = feed.get_snapshot()
                print(f"[{snap.timestamp:%H:%M:%S}] Precio: {snap.last_price:.2f} | "
                      f"Funding: {snap.funding_rate:.4%} | "
                      f"OB Imbalance: {snap.orderbook_imbalance:+.3f} | "
                      f"Velas 1m: {len(snap.ohlcv_1m)}")
                time.sleep(5)
        finally:
            feed.stop()

    else:
        print("Uso:")
        print("  python market_feed.py download [días]  — descarga datos históricos")
        print("  python market_feed.py live             — prueba conexión en testnet")
