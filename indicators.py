"""
indicators.py  v1.1
===================
Cálculo de todos los indicadores técnicos del sistema.

Cambios respecto a v1.0:
  - Import desde config (cfg)
  - Cuatro nuevos campos en IndicatorValues para regime_detector:
    adx, volatility_ratio, hurst, microstructure_ok, recent_atrs
  - Cuatro nuevas funciones de cálculo al final: calc_adx,
    calc_hurst, calc_volatility_ratio, check_microstructure
  - En _calculate_for_df: se calculan y asignan los nuevos campos
  - snap.current_close → snap.last_price (nombre correcto en MarketSnapshot v1.1)

Regla fundamental: funciones puras, sin estado, sin efectos secundarios.

Dependencias: pandas, numpy, pandas-ta
    pip install pandas numpy pandas-ta
"""

from __future__ import annotations

import math
import statistics as _stats
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("Instalá pandas-ta:  pip install pandas-ta")

from market_feed import MarketSnapshot
from config import cfg


# ─────────────────────────────────────────────
# ESTRUCTURA DE SALIDA
# ─────────────────────────────────────────────

@dataclass
class IndicatorValues:
    """
    Todos los valores calculados para UN timeframe + datos globales.
    scoring.py recibe esto y asigna puntajes.
    """

    # ── RSI ───────────────────────────────────────────────────────────
    rsi: float = 50.0
    rsi_prev: float = 50.0
    rsi_momentum_bearish: bool = False
    rsi_momentum_bullish: bool = False

    # ── EMAs ──────────────────────────────────────────────────────────
    ema7:   float = 0.0
    ema25:  float = 0.0
    ema50:  float = 0.0
    ema99:  float = 0.0
    ema200: float = 0.0
    emas_aligned_bullish: int = 0
    emas_aligned_bearish: int = 0
    ema_separation_growing: bool = False
    ema_compression: bool = False
    ema_compression_pct: float = 0.0

    # ── MACD ──────────────────────────────────────────────────────────
    macd_line:      float = 0.0
    macd_signal:    float = 0.0
    macd_histogram: float = 0.0
    macd_histogram_prev: float = 0.0
    macd_cross_bullish: bool = False
    macd_cross_bearish: bool = False
    macd_histogram_growing: bool = False
    macd_histogram_shrinking: bool = False
    macd_divergence_bullish: bool = False
    macd_divergence_bearish: bool = False

    # ── UT Bot ────────────────────────────────────────────────────────
    ut_bot_signal: str = "none"
    ut_bot_trailing_stop: float = 0.0
    ut_bot_price_near_stop: bool = False

    # ── Squeeze Momentum ──────────────────────────────────────────────
    sqz_on: bool = False
    sqz_off: bool = False
    sqz_histogram: float = 0.0
    sqz_histogram_prev: float = 0.0
    sqz_histogram_color: str = "none"
    sqz_histogram_color_prev: str = "none"
    sqz_color_change: bool = False

    # ── Ichimoku ──────────────────────────────────────────────────────
    ichi_tenkan:   float = 0.0
    ichi_kijun:    float = 0.0
    ichi_span_a:   float = 0.0
    ichi_span_b:   float = 0.0
    ichi_cloud_top:    float = 0.0
    ichi_cloud_bottom: float = 0.0
    price_above_cloud: bool = False
    price_below_cloud: bool = False
    price_in_cloud:    bool = False
    ichi_tk_cross_bullish: bool = False
    ichi_tk_cross_bearish: bool = False

    # ── VWAP ──────────────────────────────────────────────────────────
    vwap: float = 0.0
    price_above_vwap: bool = False
    vwap_cross_bullish: bool = False
    vwap_cross_bearish: bool = False

    # ── Volumen ───────────────────────────────────────────────────────
    volume_current: float = 0.0
    volume_avg_20:  float = 0.0
    volume_ratio:   float = 1.0

    # ── Bollinger Bands ───────────────────────────────────────────────
    bb_upper:  float = 0.0
    bb_middle: float = 0.0
    bb_lower:  float = 0.0
    bb_width:  float = 0.0
    price_at_lower_band:   bool = False
    price_at_upper_band:   bool = False
    price_in_lower_third:  bool = False
    price_in_upper_third:  bool = False
    price_above_upper_band: bool = False
    price_below_lower_band: bool = False

    # ── ATR ───────────────────────────────────────────────────────────
    atr: float = 0.0
    atr_pct: float = 0.0

    # ── CCI ───────────────────────────────────────────────────────────
    cci: float = 0.0
    cci_prev: float = 0.0
    cci_cross_up_100:   bool = False
    cci_cross_down_100: bool = False
    cci_extreme_bullish: bool = False
    cci_extreme_bearish: bool = False

    # ── Estocástico ───────────────────────────────────────────────────
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_k_prev: float = 50.0
    stoch_d_prev: float = 50.0
    stoch_cross_bullish: bool = False
    stoch_cross_bearish: bool = False
    stoch_oversold:  bool = False
    stoch_overbought: bool = False

    # ── Pivotes ───────────────────────────────────────────────────────
    pivot:  float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    pivot_zone: str = "none"

    # ── Lateralización ────────────────────────────────────────────────
    lateralization_score: float = 0.0
    doji_count_recent: int = 0
    long_wick_count: int = 0
    highs_lows_converging: bool = False

    # ── Patrones de velas ─────────────────────────────────────────────
    candle_pattern: str = "none"
    pattern_at_support: bool = False
    pattern_strength: float = 0.0

    # ── Datos de mercado ──────────────────────────────────────────────
    current_price: float = 0.0
    funding_rate:  float = 0.0
    orderbook_imbalance: float = 0.0

    # ── Timeframe ─────────────────────────────────────────────────────
    timeframe: str = "15m"

    # ── NUEVOS v1.1: Para regime_detector ─────────────────────────────
    adx:               float = 20.0   # 0-100, fuerza de tendencia
    volatility_ratio:  float = 1.0    # ATR_actual / ATR_promedio_20
    hurst:             float = 0.5    # >0.5 tendencia, <0.5 reversión
    microstructure_ok: bool  = True   # False = spread anormal → no operar
    recent_atrs:       list  = field(default_factory=list)  # últimas 20 ATRs


# ─────────────────────────────────────────────
# MACRO TREND
# ─────────────────────────────────────────────

@dataclass
class MacroTrend:
    trend_1w: str = "neutral"
    trend_1d: str = "neutral"
    trend_4h: str = "neutral"
    macro_aligned: bool = False
    weekly_daily_aligned: bool = False
    daily_aligned: bool = False
    divergence_daily_weekly: bool = False
    weekly_trend: str = "neutral"
    daily_trend:  str = "neutral"


def analyze_macro_trend(ivs: dict) -> MacroTrend:
    macro = MacroTrend()
    def _trend(tf):
        iv = ivs.get(tf)
        if iv is None:
            return "neutral"
        if iv.emas_aligned_bullish >= 3 and iv.rsi > 50:
            return "bullish"
        if iv.emas_aligned_bearish >= 3 and iv.rsi < 50:
            return "bearish"
        return "neutral"

    macro.trend_1w = _trend("1w")
    macro.trend_1d = _trend("1d")
    macro.trend_4h = _trend("4h")
    macro.weekly_trend = macro.trend_1w
    macro.daily_trend  = macro.trend_1d

    macro.macro_aligned = (
        macro.trend_1w == macro.trend_1d == macro.trend_4h
        and macro.trend_1w != "neutral"
    )
    macro.weekly_daily_aligned = (
        macro.trend_1w == macro.trend_1d and macro.trend_1w != "neutral"
    )
    macro.daily_aligned = macro.trend_1d != "neutral"
    macro.divergence_daily_weekly = (
        macro.trend_1w != "neutral"
        and macro.trend_1d != "neutral"
        and macro.trend_1w != macro.trend_1d
    )
    return macro


# ─────────────────────────────────────────────
# CALCULADOR PRINCIPAL
# ─────────────────────────────────────────────

class IndicatorCalculator:

    def calculate(self, snap: MarketSnapshot) -> dict[str, IndicatorValues]:
        results = {}
        tf_map = {
            '1m':  snap.ohlcv_1m,
            '5m':  snap.ohlcv_5m,
            '15m': snap.ohlcv_15m,
            '1h':  snap.ohlcv_1h,
            '4h':  snap.ohlcv_4h,
            '1d':  snap.ohlcv_1d,
            '1w':  snap.ohlcv_1w,
        }
        for tf, df in tf_map.items():
            if df is None or len(df) < 30:
                continue
            try:
                iv = self._calculate_for_df(df, tf, snap)
                results[tf] = iv
            except Exception as e:
                print(f"[Indicators] Error calculando {tf}: {e}")
        return results

    def _calculate_for_df(
        self, df: pd.DataFrame, timeframe: str, snap: MarketSnapshot
    ) -> IndicatorValues:
        iv    = IndicatorValues(timeframe=timeframe)
        close  = df['close']
        high   = df['high']
        low    = df['low']
        volume = df['volume']
        price  = float(close.iloc[-1])

        # ── Indicadores existentes (sin cambios) ─────────────────────
        self._calc_rsi(iv, close)
        self._calc_emas(iv, close, price)
        self._calc_macd(iv, close)
        self._calc_ut_bot(iv, close, high, low, price, atr_period=10, key_value=1.0)
        self._calc_squeeze(iv, close, high, low)
        self._calc_ichimoku(iv, close, high, low, price)
        self._calc_vwap(iv, df, price)
        self._calc_volume(iv, volume)
        self._calc_bollinger(iv, close, price)
        self._calc_atr(iv, close, high, low, price)
        self._calc_cci(iv, close, high, low)
        self._calc_stochastic(iv, close, high, low)
        self._calc_pivots(iv, df, price)
        self._calc_lateralization(iv, close, high, low)
        self._calc_candle_patterns(iv, df, price)

        # ── Datos de mercado ──────────────────────────────────────────
        iv.current_price       = snap.last_price
        iv.funding_rate        = snap.funding_rate
        iv.orderbook_imbalance = snap.orderbook_imbalance

        # ── NUEVOS v1.1: Para regime_detector ────────────────────────
        iv.recent_atrs    = _calc_recent_atrs(df, period=14, n=20)
        iv.adx            = _calc_adx(df, period=14)
        iv.hurst          = _calc_hurst(close.tolist(), window=100)
        iv.volatility_ratio = _calc_volatility_ratio(iv.atr, iv.recent_atrs)
        iv.microstructure_ok = _check_microstructure(snap.bid, snap.ask, iv.atr)

        return iv

    # ── Indicadores individuales (sin cambios respecto a v1.0) ────────

    def _calc_rsi(self, iv, close):
        rsi_series = ta.rsi(close, length=14)
        if rsi_series is None or rsi_series.dropna().empty:
            return
        rsi_clean = rsi_series.dropna()
        iv.rsi      = float(rsi_clean.iloc[-1])
        iv.rsi_prev = float(rsi_clean.iloc[-2]) if len(rsi_clean) >= 2 else iv.rsi
        if len(rsi_clean) >= 4:
            last4 = rsi_clean.iloc[-4:]
            iv.rsi_momentum_bearish = (
                float(last4.iloc[0]) > 50 and
                all(last4.iloc[i] > last4.iloc[i+1] for i in range(3))
            )
            iv.rsi_momentum_bullish = (
                float(last4.iloc[0]) < 50 and
                all(last4.iloc[i] < last4.iloc[i+1] for i in range(3))
            )

    def _calc_emas(self, iv, close, price):
        for period, attr in [(7,'ema7'),(25,'ema25'),(50,'ema50'),(99,'ema99'),(200,'ema200')]:
            s = ta.ema(close, length=period)
            if s is not None and not s.dropna().empty:
                setattr(iv, attr, float(s.iloc[-1]))
        emas = [iv.ema7, iv.ema25, iv.ema50, iv.ema99, iv.ema200]
        bullish = sum(1 for i in range(len(emas)-1) if emas[i] > emas[i+1] > 0)
        bearish = sum(1 for i in range(len(emas)-1) if emas[i] < emas[i+1] > 0)
        iv.emas_aligned_bullish = bullish + 1 if bullish == 4 else bullish
        iv.emas_aligned_bearish = bearish + 1 if bearish == 4 else bearish
        ema7_s   = ta.ema(close, length=7)
        ema200_s = ta.ema(close, length=200)
        if (ema7_s is not None and ema200_s is not None and
                len(ema7_s.dropna()) >= 2 and len(ema200_s.dropna()) >= 2):
            sep_now  = abs(float(ema7_s.iloc[-1])  - float(ema200_s.iloc[-1]))
            sep_prev = abs(float(ema7_s.iloc[-2])  - float(ema200_s.iloc[-2]))
            iv.ema_separation_growing = sep_now > sep_prev * 1.001
        if iv.ema7 > 0 and iv.ema25 > 0:
            iv.ema_compression_pct = abs(iv.ema7 - iv.ema25) / iv.ema25 * 100
            iv.ema_compression     = iv.ema_compression_pct < 0.1

    def _calc_macd(self, iv, close):
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty:
            return
        cols     = macd_df.columns.tolist()
        macd_col = [c for c in cols if c.startswith('MACD_')]
        hist_col = [c for c in cols if c.startswith('MACDh_')]
        sig_col  = [c for c in cols if c.startswith('MACDs_')]
        if not (macd_col and hist_col and sig_col):
            return
        macd_s = macd_df[macd_col[0]].dropna()
        hist_s = macd_df[hist_col[0]].dropna()
        sig_s  = macd_df[sig_col[0]].dropna()
        if len(hist_s) < 2:
            return
        iv.macd_line      = float(macd_s.iloc[-1])
        iv.macd_signal    = float(sig_s.iloc[-1])
        iv.macd_histogram = float(hist_s.iloc[-1])
        iv.macd_histogram_prev = float(hist_s.iloc[-2])
        iv.macd_cross_bullish = (
            float(macd_s.iloc[-2]) < float(sig_s.iloc[-2]) and
            iv.macd_line >= iv.macd_signal
        )
        iv.macd_cross_bearish = (
            float(macd_s.iloc[-2]) > float(sig_s.iloc[-2]) and
            iv.macd_line <= iv.macd_signal
        )
        iv.macd_histogram_growing   = iv.macd_histogram > 0 and iv.macd_histogram > iv.macd_histogram_prev
        iv.macd_histogram_shrinking = iv.macd_histogram > 0 and iv.macd_histogram < iv.macd_histogram_prev
        # Divergencia: mirar 10 velas atrás
        if len(close) >= 12 and len(macd_s) >= 12:
            prices_10 = close.iloc[-10:].tolist()
            macd_10   = macd_s.iloc[-10:].tolist()
            iv.macd_divergence_bullish = (
                prices_10[-1] < prices_10[0] and macd_10[-1] > macd_10[0]
            )
            iv.macd_divergence_bearish = (
                prices_10[-1] > prices_10[0] and macd_10[-1] < macd_10[0]
            )

    def _calc_ut_bot(self, iv, close, high, low, price, atr_period=10, key_value=1.0):
        atr_s = ta.atr(high, low, close, length=atr_period)
        if atr_s is None or atr_s.dropna().empty:
            return
        atr_val = float(atr_s.dropna().iloc[-1])
        stop    = price - key_value * atr_val
        iv.ut_bot_trailing_stop  = stop
        iv.ut_bot_price_near_stop = abs(price - stop) / price < 0.002
        # Señal simple: precio cruza el trailing stop
        if len(close) >= 2:
            prev_price = float(close.iloc[-2])
            if prev_price < stop and price > stop:
                iv.ut_bot_signal = "buy"
            elif prev_price > stop and price < stop:
                iv.ut_bot_signal = "sell"

    def _calc_squeeze(self, iv, close, high, low):
        # Bollinger Bands (20, 2)
        bb = ta.bbands(close, length=20, std=2)
        # Keltner Channels (20, 1.5 × ATR)
        atr_s = ta.atr(high, low, close, length=14)
        ema20 = ta.ema(close, length=20)
        if bb is None or atr_s is None or ema20 is None:
            return
        if bb.empty or atr_s.dropna().empty or ema20.dropna().empty:
            return
        bb_upper_col = [c for c in bb.columns if 'BBU' in c]
        bb_lower_col = [c for c in bb.columns if 'BBL' in c]
        if not bb_upper_col or not bb_lower_col:
            return
        bb_upper = float(bb[bb_upper_col[0]].iloc[-1])
        bb_lower = float(bb[bb_lower_col[0]].iloc[-1])
        atr_val  = float(atr_s.dropna().iloc[-1])
        ema_val  = float(ema20.dropna().iloc[-1])
        kc_upper = ema_val + 1.5 * atr_val
        kc_lower = ema_val - 1.5 * atr_val
        iv.sqz_on  = bb_upper < kc_upper and bb_lower > kc_lower
        iv.sqz_off = not iv.sqz_on
        # Histograma del Squeeze (momentum)
        delta = close - (high.rolling(20).max() + low.rolling(20).min()) / 2 - ema20
        if len(delta.dropna()) >= 2:
            iv.sqz_histogram      = float(delta.iloc[-1])
            iv.sqz_histogram_prev = float(delta.iloc[-2])
            h, hp = iv.sqz_histogram, iv.sqz_histogram_prev
            if h > 0:
                iv.sqz_histogram_color = "dark_green" if h > hp else "light_green"
            else:
                iv.sqz_histogram_color = "dark_red" if h < hp else "light_red"
            iv.sqz_color_change = (
                iv.sqz_histogram_color != iv.sqz_histogram_color_prev and
                iv.sqz_histogram_color_prev != "none"
            )

    def _calc_ichimoku(self, iv, close, high, low, price):
        ich = ta.ichimoku(high, low, close)
        if ich is None:
            return
        df_ich = ich[0] if isinstance(ich, tuple) else ich
        if df_ich is None or df_ich.empty:
            return
        cols = df_ich.columns.tolist()
        def _get(prefix):
            c = [x for x in cols if x.startswith(prefix)]
            if c and not df_ich[c[0]].dropna().empty:
                return float(df_ich[c[0]].dropna().iloc[-1])
            return 0.0
        iv.ichi_tenkan = _get('ITS')
        iv.ichi_kijun  = _get('IKS')
        iv.ichi_span_a = _get('ISA')
        iv.ichi_span_b = _get('ISB')
        iv.ichi_cloud_top    = max(iv.ichi_span_a, iv.ichi_span_b)
        iv.ichi_cloud_bottom = min(iv.ichi_span_a, iv.ichi_span_b)
        iv.price_above_cloud = price > iv.ichi_cloud_top    > 0
        iv.price_below_cloud = price < iv.ichi_cloud_bottom > 0
        iv.price_in_cloud    = iv.ichi_cloud_bottom < price < iv.ichi_cloud_top
        if iv.ichi_tenkan > 0 and iv.ichi_kijun > 0:
            iv.ichi_tk_cross_bullish = iv.ichi_tenkan >= iv.ichi_kijun
            iv.ichi_tk_cross_bearish = iv.ichi_tenkan <= iv.ichi_kijun

    def _calc_vwap(self, iv, df, price):
        try:
            vwap_s = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
            if vwap_s is not None and not vwap_s.dropna().empty:
                iv.vwap = float(vwap_s.dropna().iloc[-1])
                prev_vwap = float(vwap_s.dropna().iloc[-2]) if len(vwap_s.dropna()) >= 2 else iv.vwap
                prev_close = float(df['close'].iloc[-2]) if len(df) >= 2 else price
                iv.price_above_vwap  = price > iv.vwap
                iv.vwap_cross_bullish = prev_close < prev_vwap and price >= iv.vwap
                iv.vwap_cross_bearish = prev_close > prev_vwap and price <= iv.vwap
        except Exception:
            pass

    def _calc_volume(self, iv, volume):
        if len(volume) < 2:
            return
        iv.volume_current = float(volume.iloc[-1])
        iv.volume_avg_20  = float(volume.iloc[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
        iv.volume_ratio   = iv.volume_current / iv.volume_avg_20 if iv.volume_avg_20 > 0 else 1.0

    def _calc_bollinger(self, iv, close, price):
        bb = ta.bbands(close, length=20, std=2)
        if bb is None or bb.empty:
            return
        cols = bb.columns.tolist()
        upper_c = [c for c in cols if 'BBU' in c]
        mid_c   = [c for c in cols if 'BBM' in c]
        lower_c = [c for c in cols if 'BBL' in c]
        if not (upper_c and mid_c and lower_c):
            return
        iv.bb_upper  = float(bb[upper_c[0]].iloc[-1])
        iv.bb_middle = float(bb[mid_c[0]].iloc[-1])
        iv.bb_lower  = float(bb[lower_c[0]].iloc[-1])
        if iv.bb_middle > 0:
            iv.bb_width = (iv.bb_upper - iv.bb_lower) / iv.bb_middle
        band_range = iv.bb_upper - iv.bb_lower
        if band_range > 0:
            third = band_range / 3
            iv.price_at_lower_band    = price <= iv.bb_lower * 1.002
            iv.price_at_upper_band    = price >= iv.bb_upper * 0.998
            iv.price_in_lower_third   = price < iv.bb_lower + third
            iv.price_in_upper_third   = price > iv.bb_upper - third
            iv.price_above_upper_band = price > iv.bb_upper
            iv.price_below_lower_band = price < iv.bb_lower

    def _calc_atr(self, iv, close, high, low, price):
        atr_s = ta.atr(high, low, close, length=14)
        if atr_s is not None and not atr_s.dropna().empty:
            iv.atr     = float(atr_s.dropna().iloc[-1])
            iv.atr_pct = iv.atr / price if price > 0 else 0.0

    def _calc_cci(self, iv, close, high, low):
        cci_s = ta.cci(high, low, close, length=14)
        if cci_s is None or cci_s.dropna().empty:
            return
        cci_clean = cci_s.dropna()
        iv.cci      = float(cci_clean.iloc[-1])
        iv.cci_prev = float(cci_clean.iloc[-2]) if len(cci_clean) >= 2 else iv.cci
        iv.cci_cross_up_100    = iv.cci_prev < -100 and iv.cci >= -100
        iv.cci_cross_down_100  = iv.cci_prev > 100  and iv.cci <= 100
        iv.cci_extreme_bullish = iv.cci > 150
        iv.cci_extreme_bearish = iv.cci < -150

    def _calc_stochastic(self, iv, close, high, low):
        stoch = ta.stoch(high, low, close, k=14, d=3)
        if stoch is None or stoch.empty:
            return
        k_col = [c for c in stoch.columns if 'STOCHk' in c]
        d_col = [c for c in stoch.columns if 'STOCHd' in c]
        if not (k_col and d_col):
            return
        k_s = stoch[k_col[0]].dropna()
        d_s = stoch[d_col[0]].dropna()
        if len(k_s) < 2:
            return
        iv.stoch_k      = float(k_s.iloc[-1])
        iv.stoch_d      = float(d_s.iloc[-1])
        iv.stoch_k_prev = float(k_s.iloc[-2])
        iv.stoch_d_prev = float(d_s.iloc[-2])
        iv.stoch_cross_bullish = (
            iv.stoch_k_prev < iv.stoch_d_prev and
            iv.stoch_k >= iv.stoch_d and iv.stoch_k < 20
        )
        iv.stoch_cross_bearish = (
            iv.stoch_k_prev > iv.stoch_d_prev and
            iv.stoch_k <= iv.stoch_d and iv.stoch_k > 80
        )
        iv.stoch_oversold  = iv.stoch_k < 20
        iv.stoch_overbought = iv.stoch_k > 80

    def _calc_pivots(self, iv, df, price):
        if len(df) < 2:
            return
        prev = df.iloc[-2]
        h, l, c = float(prev['high']), float(prev['low']), float(prev['close'])
        iv.pivot = (h + l + c) / 3
        iv.r1 = 2 * iv.pivot - l
        iv.r2 = iv.pivot + (h - l)
        iv.r3 = h + 2 * (iv.pivot - l)
        iv.s1 = 2 * iv.pivot - h
        iv.s2 = iv.pivot - (h - l)
        iv.s3 = l - 2 * (h - iv.pivot)
        tol = price * 0.002
        if   price >= iv.r1 - tol:           iv.pivot_zone = "above_r1"
        elif abs(price - iv.r1) < tol:       iv.pivot_zone = "near_r1"
        elif iv.pivot < price < iv.r1:       iv.pivot_zone = "between_p_r1"
        elif abs(price - iv.pivot) < tol:    iv.pivot_zone = "at_pivot"
        elif iv.s1 < price < iv.pivot:       iv.pivot_zone = "between_s1_p"
        elif abs(price - iv.s1) < tol:       iv.pivot_zone = "near_s1"
        elif abs(price - iv.s2) < tol:       iv.pivot_zone = "near_s2"
        else:                                iv.pivot_zone = "below_s2"

    def _calc_lateralization(self, iv, close, high, low):
        score = 0.0
        recent_close = close.iloc[-5:]
        recent_high  = high.iloc[-5:]
        recent_low   = low.iloc[-5:]
        # Dojis
        doji_count = 0
        for i in range(-5, 0):
            try:
                o = float(close.iloc[i-1])
                c = float(close.iloc[i])
                h = float(high.iloc[i])
                l = float(low.iloc[i])
                body = abs(c - o)
                wick = h - l
                if wick > 0 and body / wick < 0.1:
                    doji_count += 1
            except Exception:
                pass
        iv.doji_count_recent = doji_count
        score += doji_count * 0.25
        # RSI neutro
        if 45 <= iv.rsi <= 55:
            score += 0.30
        # Máximos y mínimos convergentes
        highs_list = recent_high.tolist()
        lows_list  = recent_low.tolist()
        highs_conv = highs_list[-1] < highs_list[0]
        lows_conv  = lows_list[-1]  > lows_list[0]
        iv.highs_lows_converging = highs_conv and lows_conv
        if iv.highs_lows_converging:
            score += 0.35
        iv.lateralization_score = min(1.0, score)

    def _calc_candle_patterns(self, iv, df, price):
        if len(df) < 3:
            return
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        o, h, l, c = float(curr['open']), float(curr['high']), float(curr['low']), float(curr['close'])
        po, ph, pl, pc = float(prev['open']), float(prev['high']), float(prev['low']), float(prev['close'])
        body     = abs(c - o)
        wick_tot = h - l
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        # Doji
        if wick_tot > 0 and body / wick_tot < 0.1:
            iv.candle_pattern = "doji"
        # Envolvente alcista
        elif c > o and c > ph and o < pl:
            iv.candle_pattern = "engulfing_bull"
        # Envolvente bajista
        elif c < o and c < pl and o > ph:
            iv.candle_pattern = "engulfing_bear"
        # Martillo
        elif lower_wick > body * 2 and upper_wick < body * 0.5:
            iv.candle_pattern = "hammer"
        # Estrella fugaz
        elif upper_wick > body * 2 and lower_wick < body * 0.5:
            iv.candle_pattern = "shooting_star"
        # Coincidencia con soporte/resistencia
        tol = price * 0.002
        near_level = (
            abs(price - iv.s1) < tol or abs(price - iv.s2) < tol or
            abs(price - iv.r1) < tol or abs(price - iv.r2) < tol or
            abs(price - iv.pivot) < tol
        )
        iv.pattern_at_support = near_level and iv.candle_pattern != "none"
        iv.pattern_strength   = 1.0 if iv.pattern_at_support else 0.5


# ─────────────────────────────────────────────
# FUNCIONES NUEVAS v1.1 (para regime_detector)
# ─────────────────────────────────────────────

def _calc_adx(df: pd.DataFrame, period: int = 14) -> float:
    """ADX de Wilder. Rango 0-100. >25 = tendencia, <20 = rango."""
    n = len(df)
    if n < period * 3:
        return 20.0
    highs  = df["high"].tolist()
    lows   = df["low"].tolist()
    closes = df["close"].tolist()
    tr_list, plus_dm, minus_dm = [], [], []
    for i in range(1, n):
        h, l, cp   = highs[i], lows[i], closes[i-1]
        ph, pl     = highs[i-1], lows[i-1]
        tr = max(h-l, abs(h-cp), abs(l-cp))
        up   = h  - ph
        down = pl - l
        tr_list.append(tr)
        plus_dm.append(up   if up > down   and up   > 0 else 0.0)
        minus_dm.append(down if down > up  and down > 0 else 0.0)

    def wilder(data, p):
        out = [sum(data[:p])]
        for v in data[p:]:
            out.append(out[-1] - out[-1]/p + v)
        return out

    tr_s  = wilder(tr_list,  period)
    pdm_s = wilder(plus_dm,  period)
    mdm_s = wilder(minus_dm, period)
    dx_list = []
    for tr_v, p_v, m_v in zip(tr_s, pdm_s, mdm_s):
        if tr_v == 0:
            dx_list.append(0.0)
            continue
        pdi = 100.0 * p_v / tr_v
        mdi = 100.0 * m_v / tr_v
        d   = pdi + mdi
        dx_list.append(100.0 * abs(pdi-mdi)/d if d > 0 else 0.0)
    if len(dx_list) < period:
        return 20.0
    adx_val = sum(dx_list[:period]) / period
    for v in dx_list[period:]:
        adx_val = adx_val - adx_val/period + v/period
    return round(min(100.0, max(0.0, adx_val)), 2)


def _calc_hurst(closes: list, window: int = 100) -> float:
    """
    Hurst Exponent método R/S.
    >0.5 = tendencia, =0.5 = aleatorio, <0.5 = reversión a la media.
    """
    series = closes[-window:] if len(closes) >= window else closes
    n = len(series)
    if n < 20:
        return 0.5
    log_returns = []
    for i in range(1, n):
        if series[i-1] > 0 and series[i] > 0:
            log_returns.append(math.log(series[i] / series[i-1]))
    if len(log_returns) < 10:
        return 0.5
    m = sum(log_returns) / len(log_returns)
    cumdev, acc = [], 0.0
    for r in log_returns:
        acc += r - m
        cumdev.append(acc)
    R = max(cumdev) - min(cumdev)
    try:
        S = _stats.stdev(log_returns)
    except Exception:
        return 0.5
    if S == 0 or R == 0:
        return 0.5
    try:
        hurst = math.log(R/S) / math.log(len(log_returns)/2)
    except Exception:
        return 0.5
    return round(max(0.1, min(0.9, hurst)), 3)


def _calc_volatility_ratio(current_atr: float, recent_atrs: list, window: int = 20) -> float:
    """ATR_actual / ATR_promedio_últimas_N. >1.5 expansión, <0.7 compresión."""
    if not recent_atrs or current_atr <= 0:
        return 1.0
    recent = recent_atrs[-window:]
    avg    = sum(recent) / len(recent) if recent else current_atr
    return round(current_atr / avg, 3) if avg > 0 else 1.0


def _check_microstructure(bid: float, ask: float, atr: float) -> bool:
    """False si spread > 20% del ATR (mercado ilíquido)."""
    if bid <= 0 or ask <= 0 or atr <= 0:
        return True
    return (ask - bid) <= (atr * 0.20)


def _calc_recent_atrs(df: pd.DataFrame, period: int = 14, n: int = 20) -> list:
    """Devuelve las últimas N ATRs calculadas sobre el DataFrame."""
    if len(df) < period + n:
        return []
    closes = df["close"].tolist()
    highs  = df["high"].tolist()
    lows   = df["low"].tolist()
    trs = []
    for i in range(1, len(df)):
        cp = closes[i-1]
        tr = max(highs[i]-lows[i], abs(highs[i]-cp), abs(lows[i]-cp))
        trs.append(tr)
    atrs = []
    for i in range(period-1, len(trs)):
        atrs.append(sum(trs[i-period+1: i+1]) / period)
    return atrs[-n:] if len(atrs) >= n else atrs


if __name__ == "__main__":
    print("indicators.py v1.1 — importado correctamente.")
    print(f"  Config desde cfg: símbolo={cfg.network.symbol}")
    print(f"  Nuevos campos: adx, hurst, volatility_ratio, microstructure_ok, recent_atrs")
