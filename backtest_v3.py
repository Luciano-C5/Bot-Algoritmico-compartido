"""
backtest_v3.py
==============
Backtest de alta fidelidad del sistema completo.

Características:
  - Usa velas de 1 minuto para simular ejecución exacta de TP y SL
    dentro de velas mayores (sin lookahead bias)
  - Reconstruye todos los timeframes desde el CSV de 1m
  - Corre el sistema completo: indicators → regime → scoring → strategy → validate_rr
  - Split train/val/test según config (70/15/15)
  - cfg.risk.rr_check_enabled se puede desactivar para optimización
  - Métricas completas: retorno, winrate, sharpe, profit factor, drawdown,
    desglose por modo, nivel, régimen y duración

Uso:
    # Prueba rápida (90 días)
    py -3.12 backtest_v3.py --days 90

    # Backtest completo train (1000 días)
    py -3.12 backtest_v3.py

    # Solo set de validación
    py -3.12 backtest_v3.py --split val

    # Set de test (¡usar una sola vez al final!)
    py -3.12 backtest_v3.py --split test

    # Deshabilitar check R/R (para optimizer)
    py -3.12 backtest_v3.py --no-rr-check

Prerequisito:
    CSV en data/BTCUSDC_1m.csv con columnas:
    timestamp, open, high, low, close, volume
    (timestamp en ms unix o datetime string)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

# ── Imports del sistema ──────────────────────────────────────
from config import cfg
from indicators import IndicatorCalculator, MacroTrend, analyze_macro_trend, IndicatorValues
from regime_detector import RegimeDetector, RegimeResult, get_scoring_weights
from scoring import StrategyEvaluator, ScoreResult
from strategy import validate_rr, RRResult

# ─────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────

TIMEFRAMES = {
    "1m":  1,
    "5m":  5,
    "15m": 15,
    "1h":  60,
    "4h":  240,
    "1d":  1440,
    "1w":  10080,
}

# Velas de lookback necesarias por timeframe para que los indicadores
# tengan suficientes datos al arrancar
LOOKBACK_CANDLES = cfg.data.candles_backtest_lookback  # desde config


# ─────────────────────────────────────────────────────────────
# ESTRUCTURAS DE DATOS
# ─────────────────────────────────────────────────────────────

@dataclass
class Trade:
    """Registro completo de una operación."""
    entry_time:  datetime
    exit_time:   datetime
    mode:        str
    direction:   str
    level:       int
    leverage:    int
    regime:      str

    entry_price: float
    exit_price:  float
    exit_reason: str     # "tp1", "tp2", "tp3", "sl", "trail", "counter_signal", "end_of_data"

    capital_before: float
    capital_after:  float

    score_pct:   float   # normalized score al momento de entrada
    rr_ratio:    float

    # Desglose del score al momento de entrada
    score_breakdown: dict = field(default_factory=dict)

    @property
    def pnl_pct(self) -> float:
        return (self.capital_after - self.capital_before) / self.capital_before * 100

    @property
    def pnl_usdc(self) -> float:
        return self.capital_after - self.capital_before

    @property
    def won(self) -> bool:
        return self.capital_after >= self.capital_before

    @property
    def duration_minutes(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 60


@dataclass
class OpenPosition:
    """Posición abierta durante el backtest."""
    entry_time:  datetime
    mode:        str
    direction:   str
    level:       int
    leverage:    int
    regime:      str

    entry_price: float
    sl_price:    float
    tp_prices:   list[float]    # [tp1, tp2, tp3]
    tp_sizes:    list[float]    # [0.40, 0.35, 0.25]

    capital_at_entry: float
    score:            ScoreResult

    # Estado de TPs ejecutados
    tps_hit:     list[bool] = field(default_factory=lambda: [False, False, False])
    # Precio promedio de salida ponderado
    exit_prices: list[float] = field(default_factory=list)
    size_exited: float = 0.0   # fracción ya cerrada (0.0 a 1.0)

    # Trailing stop
    trailing_active:   bool  = False
    trailing_stop:     float = 0.0
    trailing_trigger:  float = 0.0
    trailing_distance: float = 0.0
    highest_price:     float = 0.0
    lowest_price:      float = 0.0
    breakeven_set:     bool  = False

    rr_ratio: float = 0.0


# ─────────────────────────────────────────────────────────────
# ADAPTADOR MARKET SNAPSHOT → BACKTEST
# ─────────────────────────────────────────────────────────────

class BacktestSnapshot:
    """
    Simula un MarketSnapshot usando slices del CSV histórico.
    No llama a Binance — todo desde pandas.
    """

    def __init__(self, df_1m: pd.DataFrame, idx: int):
        """
        df_1m: DataFrame completo de velas 1m
        idx:   índice de la vela 1m actual (0-based)
        """
        self.current_close    = float(df_1m["close"].iloc[idx])
        self.feed_latency_ms  = 0.0
        self.funding_rate     = 0.0
        self.orderbook_imbalance = 0.0
        self.spread_pct       = 0.001   # 0.1% estimado en backtest

        # Construir OHLCV para cada timeframe resampling
        self._df_1m = df_1m.iloc[:idx + 1].copy()

    def _resample(self, minutes: int) -> Optional[pd.DataFrame]:
        """Resamplea velas de 1m a un timeframe mayor."""
        n_candles = LOOKBACK_CANDLES.get(self._minutes_to_tf(minutes), 200)
        # Solo los últimos n_candles × minutes de datos para eficiencia
        rows_needed = n_candles * minutes + minutes
        df = self._df_1m.iloc[-rows_needed:].copy()

        if len(df) < minutes:
            return None

        df = df.set_index("timestamp")
        rule = f"{minutes}min"

        resampled = df.resample(rule, closed="left", label="left").agg({
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }).dropna()

        if len(resampled) < 30:
            return None

        return resampled.reset_index().tail(n_candles)

    @staticmethod
    def _minutes_to_tf(minutes: int) -> str:
        for tf, m in TIMEFRAMES.items():
            if m == minutes:
                return tf
        return "15m"

    # Propiedades que IndicatorCalculator espera en un MarketSnapshot
    @property
    def ohlcv_1m(self)  -> Optional[pd.DataFrame]: return self._resample(1)
    @property
    def ohlcv_5m(self)  -> Optional[pd.DataFrame]: return self._resample(5)
    @property
    def ohlcv_15m(self) -> Optional[pd.DataFrame]: return self._resample(15)
    @property
    def ohlcv_1h(self)  -> Optional[pd.DataFrame]: return self._resample(60)
    @property
    def ohlcv_4h(self)  -> Optional[pd.DataFrame]: return self._resample(240)
    @property
    def ohlcv_1d(self)  -> Optional[pd.DataFrame]: return self._resample(1440)
    @property
    def ohlcv_1w(self)  -> Optional[pd.DataFrame]: return self._resample(10080)


# ─────────────────────────────────────────────────────────────
# ADAPTADOR REGIME DETECTOR
# ─────────────────────────────────────────────────────────────

def build_regime_result_from_ivs(
    indicators: dict[str, IndicatorValues],
    detector:   RegimeDetector,
    df_1m:      pd.DataFrame,
    idx:        int,
) -> RegimeResult:
    """
    Construye un RegimeResult usando los datos ya calculados en IndicatorValues.

    El RegimeDetector.detect() espera listas de precios raw. En el backtest
    tenemos IndicatorValues que ya tienen adx, hurst, volatility_ratio calculados.
    Esta función extrae esos valores y llama al detector con datos mínimos para
    que calcule la clasificación final (votación ponderada) sin recalcular los
    indicadores desde cero.
    """
    iv_1d = indicators.get("1d")

    # Construir listas diarias mínimas para el detector
    # (solo necesita n >= ema_long_period + adx_period*3 + 5 ≈ 247)
    rows_needed = 260
    df_daily_slice = df_1m.iloc[max(0, idx - rows_needed * 1440): idx + 1]

    if len(df_daily_slice) >= 1440:
        df_daily = (
            df_daily_slice
            .set_index("timestamp")
            .resample("1440min", closed="left", label="left")
            .agg({"open": "first", "high": "max", "low": "min",
                  "close": "last", "volume": "sum"})
            .dropna()
            .reset_index()
        )
        daily_closes = df_daily["close"].tolist()
        daily_highs  = df_daily["high"].tolist()
        daily_lows   = df_daily["low"].tolist()
    else:
        # Datos insuficientes → régimen por defecto
        result = RegimeResult()
        result.regime = "volatile"
        result.confidence = 0.0
        return result

    # ATR actual y recientes del timeframe principal (15m)
    iv_15m = indicators.get("15m")
    current_atr = iv_15m.atr if iv_15m else 500.0
    recent_atrs = [current_atr] * 20

    current_price = float(df_1m["close"].iloc[idx])

    result = detector.detect(
        daily_closes   = daily_closes,
        daily_highs    = daily_highs,
        daily_lows     = daily_lows,
        current_atr    = current_atr,
        recent_atrs    = recent_atrs,
        bid            = current_price * 0.9999,
        ask            = current_price * 1.0001,
        use_fear_greed = False,   # siempre False en backtest
    )

    return result


# ─────────────────────────────────────────────────────────────
# SIMULACIÓN DE EJECUCIÓN CON VELAS DE 1M
# ─────────────────────────────────────────────────────────────

def simulate_execution_1m(
    pos:         OpenPosition,
    df_1m:       pd.DataFrame,
    start_idx:   int,
    end_idx:     int,
) -> tuple[float, str, datetime]:
    """
    Avanza vela a vela (1m) con la posición abierta.
    Verifica en qué orden se alcanzan SL, TPs y trailing stop.

    Devuelve:
        (exit_price, exit_reason, exit_time)

    La lógica de "dentro de la vela" es:
      - Si el rango (low, high) de una vela toca TANTO el SL como un TP,
        se asume que se ejecutó primero el que estaba más cerca del open de esa vela.
      - Esto evita el sesgo de siempre asumir que el TP se ejecuta primero.
    """
    mode_params    = cfg.modes.get(pos.mode)
    fee            = cfg.capital.fee_worst_case
    direction      = pos.direction
    remaining      = 1.0 - pos.size_exited
    weighted_exit  = sum(p * s for p, s in zip(pos.exit_prices,
                         [cfg.modes.tp1_size_pct / 100,
                          cfg.modes.tp2_size_pct / 100,
                          cfg.modes.tp3_size_pct / 100][:len(pos.exit_prices)]))

    pos.highest_price = pos.entry_price
    pos.lowest_price  = pos.entry_price

    for i in range(start_idx, min(end_idx, len(df_1m))):
        row   = df_1m.iloc[i]
        candle_open  = float(row["open"])
        candle_high  = float(row["high"])
        candle_low   = float(row["low"])
        candle_close = float(row["close"])
        ts           = row["timestamp"]

        if direction == "long":
            pos.highest_price = max(pos.highest_price, candle_high)
        else:
            pos.lowest_price  = min(pos.lowest_price, candle_low)

        # ── Actualizar trailing stop ──────────────────────────────
        if direction == "long":
            advance = (pos.highest_price - pos.entry_price) / pos.entry_price
            if advance >= pos.trailing_trigger:
                pos.trailing_active = True
            if pos.trailing_active:
                new_stop = pos.highest_price * (1 - pos.trailing_distance)
                if new_stop > pos.trailing_stop:
                    pos.trailing_stop = new_stop
                    if new_stop > pos.sl_price:
                        pos.sl_price = new_stop
        else:
            retreat = (pos.entry_price - pos.lowest_price) / pos.entry_price
            if retreat >= pos.trailing_trigger:
                pos.trailing_active = True
            if pos.trailing_active:
                new_stop = pos.lowest_price * (1 + pos.trailing_distance)
                if pos.trailing_stop == 0 or new_stop < pos.trailing_stop:
                    pos.trailing_stop = new_stop
                    if new_stop < pos.sl_price or pos.sl_price == 0:
                        pos.sl_price = new_stop

        # ── Determinar si SL o TP fueron tocados en esta vela ────
        sl_touched = (
            (direction == "long"  and candle_low  <= pos.sl_price) or
            (direction == "short" and candle_high >= pos.sl_price)
        )

        tp_touched = []
        for j, (tp_p, tp_s) in enumerate(zip(pos.tp_prices, pos.tp_sizes)):
            if pos.tps_hit[j]:
                continue
            hit = (
                (direction == "long"  and candle_high >= tp_p) or
                (direction == "short" and candle_low  <= tp_p)
            )
            tp_touched.append((j, tp_p, tp_s, hit))

        # ── Lógica de qué se ejecutó primero ─────────────────────
        # Distancia del open de la vela a cada nivel → menor = ocurrió antes
        def dist_to(level: float) -> float:
            return abs(candle_open - level)

        next_tp_hit = next((t for t in tp_touched if t[3]), None)

        if sl_touched and next_tp_hit:
            sl_dist = dist_to(pos.sl_price)
            tp_dist = dist_to(next_tp_hit[1])
            if sl_dist <= tp_dist:
                # SL primero
                exit_price = pos.sl_price
                reason = "trail" if pos.trailing_active else "sl"
                return _apply_fees(exit_price, direction, fee), reason, _to_dt(ts)
            else:
                # TP primero → procesar ese TP
                j, tp_p, tp_s, _ = next_tp_hit
                pos.tps_hit[j]   = True
                pos.exit_prices.append(tp_p)
                pos.size_exited += tp_s

                # Si era el último TP pendiente → posición completamente cerrada
                if all(pos.tps_hit):
                    blended = _blend_exit_price(pos)
                    return _apply_fees(blended, direction, fee), "tp3", _to_dt(ts)

                # Mover SL a breakeven al cerrar primer TP
                if j == 0 and not pos.breakeven_set:
                    be = pos.entry_price * (1 + fee) if direction == "long" \
                         else pos.entry_price * (1 - fee)
                    if direction == "long" and be > pos.sl_price:
                        pos.sl_price = be
                        pos.breakeven_set = True
                    elif direction == "short" and be < pos.sl_price:
                        pos.sl_price = be
                        pos.breakeven_set = True

        elif sl_touched:
            exit_price = pos.sl_price
            reason = "trail" if pos.trailing_active else "sl"
            return _apply_fees(exit_price, direction, fee), reason, _to_dt(ts)

        elif next_tp_hit:
            j, tp_p, tp_s, _ = next_tp_hit
            pos.tps_hit[j]   = True
            pos.exit_prices.append(tp_p)
            pos.size_exited += tp_s

            if all(pos.tps_hit):
                blended = _blend_exit_price(pos)
                return _apply_fees(blended, direction, fee), "tp3", _to_dt(ts)

            if j == 0 and not pos.breakeven_set:
                be = pos.entry_price * (1 + fee) if direction == "long" \
                     else pos.entry_price * (1 - fee)
                if direction == "long" and be > pos.sl_price:
                    pos.sl_price = be
                    pos.breakeven_set = True
                elif direction == "short" and be < pos.sl_price:
                    pos.sl_price = be
                    pos.breakeven_set = True

    # Sin SL ni TP alcanzado al final del período → cerrar al último precio
    last_close = float(df_1m["close"].iloc[min(end_idx, len(df_1m) - 1)])
    ts_last    = df_1m["timestamp"].iloc[min(end_idx, len(df_1m) - 1)]

    if pos.exit_prices:
        # TPs parciales ya ejecutados → blendear con el cierre final
        remaining_fraction = 1.0 - pos.size_exited
        blended = _blend_exit_price(pos, last_close, remaining_fraction)
        return _apply_fees(blended, direction, fee), "end_of_data", _to_dt(ts_last)
    else:
        return _apply_fees(last_close, direction, fee), "end_of_data", _to_dt(ts_last)


def _blend_exit_price(
    pos:               OpenPosition,
    extra_price:       float = 0.0,
    extra_fraction:    float = 0.0,
) -> float:
    """Precio promedio ponderado de salida considerando TPs parciales."""
    sizes = [cfg.modes.tp1_size_pct / 100,
             cfg.modes.tp2_size_pct / 100,
             cfg.modes.tp3_size_pct / 100]
    total_weight = 0.0
    weighted_sum = 0.0
    for price, size in zip(pos.exit_prices, sizes[:len(pos.exit_prices)]):
        weighted_sum  += price * size
        total_weight  += size
    if extra_fraction > 0 and extra_price > 0:
        weighted_sum += extra_price * extra_fraction
        total_weight += extra_fraction
    return weighted_sum / total_weight if total_weight > 0 else pos.entry_price


def _apply_fees(price: float, direction: str, fee: float) -> float:
    """Ajusta el precio de salida por las comisiones (ya incluidas en el PNL)."""
    return price   # Las comisiones se aplican en el cálculo del PNL


def _to_dt(ts) -> datetime:
    """Convierte timestamp a datetime UTC."""
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts, tz=timezone.utc)
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────
# CÁLCULO DE PNL
# ─────────────────────────────────────────────────────────────

def calc_pnl(
    capital:      float,
    entry_price:  float,
    exit_price:   float,
    direction:    str,
    leverage:     int,
    fee:          float,
) -> float:
    """
    PNL neto = capital + ganancia/pérdida - comisiones

    Fórmula:
      position_size = capital × leverage
      raw_return    = (exit - entry) / entry × position_size × signo
      comisiones    = position_size × fee_worst_case
      pnl_neto      = raw_return - comisiones
    """
    position_size = capital * leverage
    sign          = 1.0 if direction == "long" else -1.0
    raw_return    = (exit_price - entry_price) / entry_price * position_size * sign
    commissions   = position_size * fee
    return capital + raw_return - commissions


# ─────────────────────────────────────────────────────────────
# CARGA Y PREPARACIÓN DE DATOS
# ─────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """
    Carga el CSV de 1m con manejo robusto de formatos de timestamp.
    Columnas esperadas: timestamp, open, high, low, close, volume
    """
    print(f"[Backtest] Cargando CSV: {path}")
    df = pd.read_csv(path)

    # Normalizar nombres de columnas
    df.columns = [c.lower().strip() for c in df.columns]

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV debe tener columnas: timestamp, open, high, low, close, volume")

    # Parsear timestamp
    ts_col = df.columns[0]   # primera columna = timestamp
    if df[ts_col].dtype in (np.int64, np.float64):
        # Unix ms o Unix s
        if df[ts_col].iloc[0] > 1e12:
            df["timestamp"] = pd.to_datetime(df[ts_col], unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df[ts_col], unit="s", utc=True)
    else:
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    print(f"[Backtest] {len(df):,} velas cargadas | "
          f"{df['timestamp'].iloc[0].date()} → {df['timestamp'].iloc[-1].date()}")
    return df


def split_data(
    df: pd.DataFrame,
    days: Optional[int] = None,
    split: str = "train",
) -> pd.DataFrame:
    """
    Aplica el split train/val/test según config.

    Si days está definido, toma solo los últimos N días del set train.
    """
    total = len(df)

    train_end = int(total * cfg.data.train_pct)
    val_end   = int(total * (cfg.data.train_pct + cfg.data.val_pct))

    if split == "train":
        df_split = df.iloc[:train_end]
    elif split == "val":
        df_split = df.iloc[train_end:val_end]
    elif split == "test":
        df_split = df.iloc[val_end:]
    else:
        raise ValueError(f"split debe ser 'train', 'val' o 'test'")

    if days is not None and split == "train":
        rows = days * 1440
        df_split = df_split.iloc[-rows:] if len(df_split) > rows else df_split

    print(f"[Backtest] Split '{split}': {len(df_split):,} velas "
          f"({df_split['timestamp'].iloc[0].date()} → "
          f"{df_split['timestamp'].iloc[-1].date()})")
    return df_split.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────

@dataclass
class BacktestMetrics:
    """Métricas completas del backtest."""

    # Generales
    total_trades:   int   = 0
    winning_trades: int   = 0
    losing_trades:  int   = 0
    winrate:        float = 0.0

    initial_capital: float = 0.0
    final_capital:   float = 0.0
    total_return_pct: float = 0.0

    profit_factor: float = 0.0
    sharpe_ratio:  float = 0.0
    max_drawdown_pct: float = 0.0

    avg_trade_duration_min: float = 0.0
    trades_per_day: float = 0.0
    total_days: float = 0.0

    # Desglose por modo
    by_mode: dict = field(default_factory=dict)
    # Desglose por nivel de señal
    by_level: dict = field(default_factory=dict)
    # Desglose por régimen
    by_regime: dict = field(default_factory=dict)
    # Desglose por exit reason
    by_exit: dict = field(default_factory=dict)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print("  RESULTADOS DEL BACKTEST")
        print("=" * 60)
        print(f"  Capital inicial:  ${self.initial_capital:,.2f} USDC")
        print(f"  Capital final:    ${self.final_capital:,.2f} USDC")
        print(f"  Retorno total:    {self.total_return_pct:+.2f}%")
        print(f"  Sharpe ratio:     {self.sharpe_ratio:.3f}")
        print(f"  Profit factor:    {self.profit_factor:.3f}")
        print(f"  Max drawdown:     {self.max_drawdown_pct:.2f}%")
        print()
        print(f"  Operaciones:      {self.total_trades}")
        print(f"  Winrate:          {self.winrate*100:.1f}%")
        print(f"  Por día:          {self.trades_per_day:.2f}")
        print(f"  Duración media:   {self.avg_trade_duration_min:.0f} min")
        print()

        if self.by_mode:
            print("  ── Por modo ──────────────────────────────────")
            for mode, stats in self.by_mode.items():
                print(f"  {mode:<9} n={stats['n']:<4} wr={stats['winrate']*100:.0f}%  "
                      f"pnl={stats['total_pnl']:+.2f} USDC")

        if self.by_level:
            print("  ── Por nivel de señal ────────────────────────")
            for level, stats in sorted(self.by_level.items()):
                print(f"  N{level}        n={stats['n']:<4} wr={stats['winrate']*100:.0f}%  "
                      f"pnl={stats['total_pnl']:+.2f} USDC")

        if self.by_regime:
            print("  ── Por régimen ───────────────────────────────")
            for regime, stats in self.by_regime.items():
                print(f"  {regime:<12} n={stats['n']:<4} wr={stats['winrate']*100:.0f}%  "
                      f"pnl={stats['total_pnl']:+.2f} USDC")

        if self.by_exit:
            print("  ── Por motivo de salida ──────────────────────")
            for reason, count in sorted(self.by_exit.items(), key=lambda x: -x[1]):
                print(f"  {reason:<20} {count}")

        print("=" * 60)


def calc_metrics(trades: list[Trade], initial_capital: float) -> BacktestMetrics:
    m = BacktestMetrics(
        initial_capital = initial_capital,
        total_trades    = len(trades),
    )

    if not trades:
        return m

    m.final_capital  = trades[-1].capital_after
    m.total_return_pct = (m.final_capital - initial_capital) / initial_capital * 100
    m.winning_trades   = sum(1 for t in trades if t.won)
    m.losing_trades    = m.total_trades - m.winning_trades
    m.winrate          = m.winning_trades / m.total_trades

    # Profit factor
    gross_profit = sum(t.pnl_usdc for t in trades if t.won)
    gross_loss   = abs(sum(t.pnl_usdc for t in trades if not t.won))
    m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Sharpe ratio (sobre retornos por operación)
    returns = [t.pnl_pct / 100 for t in trades]
    if len(returns) > 1:
        mean_r = sum(returns) / len(returns)
        std_r  = (sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
        m.sharpe_ratio = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0

    # Max drawdown
    peak    = initial_capital
    max_dd  = 0.0
    capital = initial_capital
    for t in trades:
        capital = t.capital_after
        if capital > peak:
            peak = capital
        dd = (peak - capital) / peak * 100
        if dd > max_dd:
            max_dd = dd
    m.max_drawdown_pct = max_dd

    # Duración y frecuencia
    durations = [t.duration_minutes for t in trades]
    m.avg_trade_duration_min = sum(durations) / len(durations)
    if trades:
        span_days = (trades[-1].exit_time - trades[0].entry_time).total_seconds() / 86400
        m.total_days    = span_days
        m.trades_per_day = m.total_trades / span_days if span_days > 0 else 0

    # Desgloses
    for key, group in [
        ("by_mode",   lambda t: t.mode),
        ("by_level",  lambda t: t.level),
        ("by_regime", lambda t: t.regime),
    ]:
        d = {}
        for t in trades:
            k = group(t)
            if k not in d:
                d[k] = {"n": 0, "won": 0, "total_pnl": 0.0}
            d[k]["n"]         += 1
            d[k]["won"]       += int(t.won)
            d[k]["total_pnl"] += t.pnl_usdc
        for v in d.values():
            v["winrate"] = v["won"] / v["n"] if v["n"] > 0 else 0.0
        setattr(m, key, d)

    by_exit: dict[str, int] = {}
    for t in trades:
        by_exit[t.exit_reason] = by_exit.get(t.exit_reason, 0) + 1
    m.by_exit = by_exit

    return m


# ─────────────────────────────────────────────────────────────
# ESTADO DE RIESGO (simplificado para backtest)
# ─────────────────────────────────────────────────────────────

class BacktestRiskState:
    """
    Gestiona pausas y modo revisión durante el backtest.
    Replica la lógica de risk_manager.py sin estado persistente.
    """

    def __init__(self):
        self.consecutive_losses: dict[str, int] = {"scalp": 0, "mediano": 0, "swing": 0}
        self.daily_losses: int = 0
        self.paused_modes: dict[str, int] = {}   # mode → velas restantes de pausa
        self.global_pause_until: Optional[datetime] = None
        self.n3_fails: int = 0
        self.n3_disabled: bool = False
        self.recent_trades: list[bool] = []

    def is_blocked(self, mode: str, level: int, current_time: datetime) -> bool:
        # Modo revisión
        if len(self.recent_trades) >= cfg.risk.winrate_window:
            wr = sum(self.recent_trades[-cfg.risk.winrate_window:]) / cfg.risk.winrate_window
            if wr < cfg.risk.review_mode_winrate_threshold:
                return True

        # Pausa global diaria
        if self.global_pause_until and current_time < self.global_pause_until:
            if level < cfg.risk.override_pause_level:
                return True

        # Pausa por modo
        if self.paused_modes.get(mode, 0) > 0:
            if level < cfg.risk.override_pause_level:
                return True

        # N3 desactivado
        if level == 3 and self.n3_disabled:
            return True

        return False

    def record(
        self,
        trade:     Trade,
        mode:      str,
        tf_minutes: int,
    ) -> None:
        won = trade.won
        self.recent_trades.append(won)
        if len(self.recent_trades) > 50:
            self.recent_trades.pop(0)

        if not won:
            self.consecutive_losses[mode] = self.consecutive_losses.get(mode, 0) + 1
            self.daily_losses += 1

            if self.consecutive_losses[mode] >= cfg.risk.max_consecutive_losses_per_mode:
                pause_velas = cfg.modes.get(mode).pause_candles
                self.paused_modes[mode] = pause_velas * tf_minutes  # en minutos

            if self.daily_losses >= cfg.risk.max_daily_losses:
                # Pausa hasta medianoche UTC
                day_end = trade.exit_time.replace(hour=0, minute=0, second=0, microsecond=0)
                import datetime as dt_module
                day_end = day_end + dt_module.timedelta(days=1)
                self.global_pause_until = day_end

            if trade.level == 3:
                self.n3_fails += 1
                if self.n3_fails >= 2:
                    self.n3_disabled = True
        else:
            self.consecutive_losses[mode] = 0

    def tick_minute(self, mode: str) -> None:
        if mode in self.paused_modes and self.paused_modes[mode] > 0:
            self.paused_modes[mode] -= 1

    def reset_daily(self) -> None:
        self.daily_losses   = 0
        self.n3_fails       = 0
        self.n3_disabled    = False
        self.global_pause_until = None


# ─────────────────────────────────────────────────────────────
# LOOP PRINCIPAL DEL BACKTEST
# ─────────────────────────────────────────────────────────────

class Backtester:

    def __init__(self, rr_check_enabled: bool = True):
        self.calc      = IndicatorCalculator()
        self.evaluator = StrategyEvaluator()
        self.detector  = RegimeDetector()
        cfg.risk.rr_check_enabled = rr_check_enabled

    def run(
        self,
        df_1m:      pd.DataFrame,
        capital:    float = None,
        progress:   bool  = True,
    ) -> tuple[list[Trade], BacktestMetrics]:

        if capital is None:
            capital = cfg.capital.initial_capital

        trades:   list[Trade] = []
        position: Optional[OpenPosition] = None
        risk      = BacktestRiskState()

        # Índice de inicio: necesitamos suficientes velas para los indicadores
        # EMA200 diaria necesita 200 días ≈ 200 × 1440 = 288000 velas de 1m
        # Usamos un lookback mínimo más práctico: 60 días ≈ 86400 velas
        lookback_min = 60 * 1440
        start_idx    = min(lookback_min, len(df_1m) // 3)

        # Ciclos del bot: evaluar cada N minutos según el modo prioritario
        # En backtest evaluamos cada 15 minutos (= una vela 15m) para scalp
        eval_interval = 15   # minutos
        total_steps   = (len(df_1m) - start_idx) // eval_interval

        last_daily_reset = _to_dt(df_1m["timestamp"].iloc[start_idx])
        t0 = time.monotonic()

        for step, idx in enumerate(range(start_idx, len(df_1m) - eval_interval, eval_interval)):

            current_time = _to_dt(df_1m["timestamp"].iloc[idx])

            # Reset diario a medianoche UTC
            if current_time.date() > last_daily_reset.date():
                risk.reset_daily()
                last_daily_reset = current_time

            # Tick de pausas
            for mode in ("scalp", "mediano", "swing"):
                for _ in range(eval_interval):
                    risk.tick_minute(mode)

            # Progreso
            if progress and step % 500 == 0:
                elapsed  = time.monotonic() - t0
                pct_done = step / total_steps * 100
                eta_s    = (elapsed / max(step, 1)) * (total_steps - step)
                print(f"\r  {pct_done:.1f}% | trades={len(trades)} | "
                      f"capital=${capital:.2f} | ETA={eta_s:.0f}s  ", end="", flush=True)

            # ── Si hay posición abierta: simular 1 a 1 ───────────────
            if position is not None:
                exit_price, reason, exit_time = simulate_execution_1m(
                    position, df_1m, idx, idx + eval_interval
                )

                if reason != "__continue__":
                    new_capital = calc_pnl(
                        capital      = position.capital_at_entry,
                        entry_price  = position.entry_price,
                        exit_price   = exit_price,
                        direction    = position.direction,
                        leverage     = position.leverage,
                        fee          = cfg.capital.fee_worst_case,
                    )
                    trade = Trade(
                        entry_time      = position.entry_time,
                        exit_time       = exit_time,
                        mode            = position.mode,
                        direction       = position.direction,
                        level           = position.level,
                        leverage        = position.leverage,
                        regime          = position.regime,
                        entry_price     = position.entry_price,
                        exit_price      = exit_price,
                        exit_reason     = reason,
                        capital_before  = position.capital_at_entry,
                        capital_after   = max(0.01, new_capital),
                        score_pct       = position.score.normalized,
                        rr_ratio        = position.rr_ratio,
                        score_breakdown = position.score.breakdown,
                    )
                    trades.append(trade)
                    risk.record(trade, position.mode, TIMEFRAMES[cfg.modes.get(position.mode).timeframe_main])
                    capital  = trade.capital_after
                    position = None
                    continue

            # ── Sin posición: evaluar señales ────────────────────────
            # Construir snapshot y calcular indicadores
            snap = BacktestSnapshot(df_1m, idx)
            try:
                ivs   = self.calc.calculate(snap)
                macro = analyze_macro_trend(ivs)
            except Exception as e:
                continue

            # Régimen de mercado
            try:
                regime = build_regime_result_from_ivs(ivs, self.detector, df_1m, idx)
            except Exception:
                regime = RegimeResult()

            # Modos activos (los no pausados)
            active_modes = [
                m for m in ("scalp", "mediano", "swing")
                if not risk.is_blocked(m, 1, current_time)
            ]
            if not active_modes:
                continue

            # Evaluar scoring
            try:
                best = self.evaluator.evaluate(
                    indicators          = ivs,
                    macro               = macro,
                    active_modes        = active_modes,
                    regime              = regime.regime,
                    regime_confidence   = regime.confidence,
                    threshold_increment = regime.threshold_increment,
                )
            except Exception:
                continue

            if best is None:
                continue

            # Verificar riesgo por nivel
            if risk.is_blocked(best.mode, best.signal_level, current_time):
                continue

            # Calcular precios de entrada, SL y TPs
            iv_main = ivs.get(cfg.modes.get(best.mode).timeframe_main)
            if iv_main is None:
                continue

            entry_price = iv_main.current_price
            if entry_price <= 0:
                continue

            mode_params = cfg.modes.get(best.mode)
            sl_pct      = mode_params.sl_base_pct
            # Ajuste por ATR si disponible
            if iv_main.atr > 0:
                atr_pct = iv_main.atr / entry_price
                sl_pct  = max(sl_pct, min(sl_pct * 1.5, atr_pct * 1.5))

            if best.direction == "long":
                sl_price = entry_price * (1 - sl_pct)
            else:
                sl_price = entry_price * (1 + sl_pct)

            tp_base = best.approx_tp_pct
            tp_prices = []
            for factor in [cfg.modes.tp1_distance_factor,
                           cfg.modes.tp2_distance_factor,
                           cfg.modes.tp3_distance_factor]:
                dist = tp_base * factor
                if best.direction == "long":
                    tp_prices.append(entry_price * (1 + dist))
                else:
                    tp_prices.append(entry_price * (1 - dist))

            tp_sizes = [
                cfg.modes.tp1_size_pct / 100,
                cfg.modes.tp2_size_pct / 100,
                cfg.modes.tp3_size_pct / 100,
            ]

            # Validación R/R
            rr = validate_rr(entry_price, sl_price, tp_prices[-1], best.leverage)
            if not rr.valid:
                continue

            position = OpenPosition(
                entry_time       = current_time,
                mode             = best.mode,
                direction        = best.direction,
                level            = best.signal_level,
                leverage         = best.leverage,
                regime           = regime.regime,
                entry_price      = entry_price,
                sl_price         = sl_price,
                tp_prices        = tp_prices,
                tp_sizes         = tp_sizes,
                capital_at_entry = capital,
                score            = best,
                trailing_trigger  = mode_params.trailing_trigger,
                trailing_distance = mode_params.trailing_distance,
                highest_price    = entry_price,
                lowest_price     = entry_price,
                rr_ratio         = rr.rr_ratio,
            )

        # Cerrar posición abierta al final de los datos
        if position is not None:
            last_price = float(df_1m["close"].iloc[-1])
            last_time  = _to_dt(df_1m["timestamp"].iloc[-1])
            new_capital = calc_pnl(
                capital      = position.capital_at_entry,
                entry_price  = position.entry_price,
                exit_price   = last_price,
                direction    = position.direction,
                leverage     = position.leverage,
                fee          = cfg.capital.fee_worst_case,
            )
            trade = Trade(
                entry_time      = position.entry_time,
                exit_time       = last_time,
                mode            = position.mode,
                direction       = position.direction,
                level           = position.level,
                leverage        = position.leverage,
                regime          = position.regime,
                entry_price     = position.entry_price,
                exit_price      = last_price,
                exit_reason     = "end_of_data",
                capital_before  = position.capital_at_entry,
                capital_after   = max(0.01, new_capital),
                score_pct       = position.score.normalized,
                rr_ratio        = position.rr_ratio,
                score_breakdown = position.score.breakdown,
            )
            trades.append(trade)
            capital = trade.capital_after

        if progress:
            print()   # nueva línea tras la barra de progreso

        metrics = calc_metrics(trades, cfg.capital.initial_capital)
        return trades, metrics


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest del bot algorítmico BTC/USDC")
    parser.add_argument("--days",         type=int,   default=None,
                        help="Días a testear (default: todos los del split)")
    parser.add_argument("--split",        type=str,   default="train",
                        choices=["train", "val", "test"],
                        help="Split a usar (default: train)")
    parser.add_argument("--capital",      type=float, default=None,
                        help="Capital inicial en USDC (default: desde config.py)")
    parser.add_argument("--no-rr-check",  action="store_true",
                        help="Desactivar validación R/R (útil para optimizer)")
    parser.add_argument("--csv",          type=str,
                        default=cfg.data.csv_path,
                        help=f"Ruta al CSV (default: {cfg.data.csv_path})")
    args = parser.parse_args()

    print("=" * 60)
    print("  BOT ALGORÍTMICO — BACKTEST v3")
    print("=" * 60)
    print(f"  CSV:       {args.csv}")
    print(f"  Split:     {args.split}")
    print(f"  Días:      {args.days or 'todos'}")
    print(f"  Capital:   ${args.capital or cfg.capital.initial_capital:.2f} USDC")
    print(f"  R/R check: {'DESACTIVADO' if args.no_rr_check else 'activado'}")
    print()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV no encontrado: {args.csv}")
        print("  Descargalo con market_feed.py primero:")
        print("  py -3.12 market_feed.py --download")
        sys.exit(1)

    df = load_csv(args.csv)
    df = split_data(df, days=args.days, split=args.split)

    if len(df) < 30000:
        print(f"[ADVERTENCIA] Pocos datos ({len(df):,} velas). "
              "Los resultados pueden no ser representativos.")

    backtester = Backtester(rr_check_enabled=not args.no_rr_check)

    t0 = time.monotonic()
    trades, metrics = backtester.run(df, capital=args.capital)
    elapsed = time.monotonic() - t0

    metrics.print_summary()
    print(f"\n  Tiempo de ejecución: {elapsed:.1f}s")

    if metrics.total_trades < cfg.optimizer.min_trades:
        print(f"\n  [AVISO] Solo {metrics.total_trades} operaciones — "
              f"mínimo recomendado: {cfg.optimizer.min_trades}. "
              "Resultados estadísticamente poco confiables.")

    return trades, metrics


if __name__ == "__main__":
    main()
