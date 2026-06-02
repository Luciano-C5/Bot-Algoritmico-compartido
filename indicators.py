"""
indicators.py
=============
Cálculo de todos los indicadores técnicos del sistema.

Regla fundamental: este archivo no sabe nada de órdenes, puntajes,
ni decisiones. Solo recibe un MarketSnapshot y devuelve IndicatorValues.
Funciones puras, sin estado, sin efectos secundarios.

Dependencias: pandas, numpy, pandas-ta
    pip install pandas numpy pandas-ta
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("Instalá pandas-ta:  pip install pandas-ta")

from market_feed import MarketSnapshot


# ─────────────────────────────────────────────
# ESTRUCTURA DE SALIDA
# ─────────────────────────────────────────────

@dataclass
class IndicatorValues:
    """
    Todos los valores calculados para UN timeframe + datos globales.
    scoring.py recibe esto y asigna puntajes.
    """

    # ── RSI ────────────────────────────────────────────────────────────
    rsi: float = 50.0                    # valor actual
    rsi_prev: float = 50.0              # vela anterior
    rsi_momentum_bearish: bool = False   # bajando desde >50 por 3 velas consecutivas
    rsi_momentum_bullish: bool = False   # subiendo desde <50 por 3 velas consecutivas

    # ── EMAs ───────────────────────────────────────────────────────────
    ema7:   float = 0.0
    ema25:  float = 0.0
    ema50:  float = 0.0
    ema99:  float = 0.0
    ema200: float = 0.0

    emas_aligned_bullish: int = 0        # cuántas de 5 están alineadas alcistas (7>25>50>99>200)
    emas_aligned_bearish: int = 0        # cuántas de 5 están alineadas bajistas
    ema_separation_growing: bool = False # separación entre EMAs creciendo (tendencia acelerando)
    ema_compression: bool = False        # EMA7 y EMA25 separadas menos de 0.1%
    ema_compression_pct: float = 0.0     # separación actual entre EMA7 y EMA25 en %

    # ── MACD ───────────────────────────────────────────────────────────
    macd_line:      float = 0.0
    macd_signal:    float = 0.0
    macd_histogram: float = 0.0
    macd_histogram_prev: float = 0.0
    macd_cross_bullish: bool = False     # cruce alcista activo (esta o última vela)
    macd_cross_bearish: bool = False
    macd_histogram_growing: bool = False # histograma positivo y creciendo
    macd_histogram_shrinking: bool = False
    macd_divergence_bullish: bool = False  # precio mínimos más bajos, MACD mínimos más altos
    macd_divergence_bearish: bool = False  # precio máximos más altos, MACD máximos más bajos

    # ── UT Bot (trailing stop dinámico basado en ATR) ──────────────────
    ut_bot_signal: str = "none"          # "buy", "sell", "none"
    ut_bot_trailing_stop: float = 0.0
    ut_bot_price_near_stop: bool = False # precio a menos de 0.2% del trailing stop

    # ── Squeeze Momentum (LazyBear) ────────────────────────────────────
    sqz_on: bool = False                 # squeeze activo (BB dentro de KC)
    sqz_off: bool = False                # squeeze terminando (BB saliendo de KC)
    sqz_histogram: float = 0.0
    sqz_histogram_prev: float = 0.0
    sqz_histogram_color: str = "none"    # "dark_green","light_green","dark_red","light_red"
    sqz_histogram_color_prev: str = "none"
    sqz_color_change: bool = False       # cambió de intensidad esta vela

    # ── Ichimoku ───────────────────────────────────────────────────────
    ichi_tenkan:   float = 0.0           # línea de conversión (9)
    ichi_kijun:    float = 0.0           # línea base (26)
    ichi_span_a:   float = 0.0           # borde superior de la nube
    ichi_span_b:   float = 0.0           # borde inferior de la nube
    ichi_cloud_top:    float = 0.0       # max(span_a, span_b)
    ichi_cloud_bottom: float = 0.0       # min(span_a, span_b)
    price_above_cloud: bool = False
    price_below_cloud: bool = False
    price_in_cloud:    bool = False
    ichi_tk_cross_bullish: bool = False  # tenkan cruza kijun hacia arriba
    ichi_tk_cross_bearish: bool = False

    # ── VWAP ───────────────────────────────────────────────────────────
    vwap: float = 0.0
    price_above_vwap: bool = False
    vwap_cross_bullish: bool = False     # cruza al alza en esta vela
    vwap_cross_bearish: bool = False

    # ── Volumen ────────────────────────────────────────────────────────
    volume_current: float = 0.0
    volume_avg_20:  float = 0.0
    volume_ratio:   float = 1.0          # current / avg_20

    # ── Bollinger Bands ────────────────────────────────────────────────
    bb_upper:  float = 0.0
    bb_middle: float = 0.0
    bb_lower:  float = 0.0
    bb_width:  float = 0.0              # (upper - lower) / middle, medida de volatilidad
    price_at_lower_band:   bool = False  # tocando banda inferior (long)
    price_at_upper_band:   bool = False  # tocando banda superior (short)
    price_in_lower_third:  bool = False
    price_in_upper_third:  bool = False
    price_above_upper_band: bool = False # sobreextendido (penalización en long)
    price_below_lower_band: bool = False # sobreextendido (penalización en short)

    # ── ATR ────────────────────────────────────────────────────────────
    atr: float = 0.0
    atr_pct: float = 0.0               # ATR como % del precio

    # ── CCI ────────────────────────────────────────────────────────────
    cci: float = 0.0
    cci_prev: float = 0.0
    cci_cross_up_100:   bool = False    # salió de sobreventa (cruzó +100 hacia arriba... espera, CCI usa -100/+100)
    cci_cross_down_100: bool = False    # salió de sobrecompra
    cci_extreme_bullish: bool = False   # > 150
    cci_extreme_bearish: bool = False   # < -150

    # ── Estocástico ────────────────────────────────────────────────────
    stoch_k: float = 50.0
    stoch_d: float = 50.0
    stoch_k_prev: float = 50.0
    stoch_d_prev: float = 50.0
    stoch_cross_bullish: bool = False   # K cruza D hacia arriba en zona < 20
    stoch_cross_bearish: bool = False   # K cruza D hacia abajo en zona > 80
    stoch_oversold:  bool = False       # K < 20 sin cruce aún
    stoch_overbought: bool = False      # K > 80 sin cruce aún

    # ── Pivotes ────────────────────────────────────────────────────────
    pivot:  float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    s1: float = 0.0
    s2: float = 0.0
    s3: float = 0.0
    pivot_zone: str = "none"           # "above_r1","near_r1","between_p_r1",
                                       # "at_pivot","between_s1_p","near_s1",
                                       # "near_s2","below_s2"

    # ── Lateralización ────────────────────────────────────────────────
    lateralization_score: float = 0.0  # 0.0 a 1.0, >0.65 = no operar
    doji_count_recent: int = 0         # dojis en las últimas 5 velas
    long_wick_count: int = 0           # mechas largas en ambos lados últimas 5 velas
    highs_lows_converging: bool = False # máximos y mínimos convergentes últimas 5 velas

    # ── Patrones de velas ─────────────────────────────────────────────
    candle_pattern: str = "none"       # "engulfing_bull","engulfing_bear",
                                       # "hammer","shooting_star","doji","none"
    pattern_at_support: bool = False   # patrón coincide con soporte/resistencia
    pattern_strength: float = 0.0      # 0.0 a 1.0

    # ── Datos de mercado (del snapshot) ───────────────────────────────
    current_price: float = 0.0
    funding_rate:  float = 0.0
    orderbook_imbalance: float = 0.0   # -1.0 a +1.0

    # ── Timeframe de estos indicadores ────────────────────────────────
    timeframe: str = "15m"


# ─────────────────────────────────────────────
# CALCULADOR PRINCIPAL
# ─────────────────────────────────────────────

class IndicatorCalculator:
    """
    Calcula todos los indicadores para todos los timeframes relevantes.

    Uso:
        calc = IndicatorCalculator()
        result = calc.calculate(snapshot)
        # result es un dict: {'15m': IndicatorValues, '1h': IndicatorValues, ...}
        # más result['macro'] con el análisis de tendencia macro
    """

    def calculate(self, snap: MarketSnapshot) -> dict[str, IndicatorValues]:
        """
        Punto de entrada principal.
        Devuelve indicadores calculados para cada timeframe relevante.
        """
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
                iv = self._calculate_for_df(df, tf)
                iv.current_price        = snap.current_close
                iv.funding_rate         = snap.funding_rate
                iv.orderbook_imbalance  = snap.orderbook_imbalance
                results[tf] = iv
            except Exception as e:
                print(f"[Indicators] Error calculando {tf}: {e}")

        return results

    # ── Cálculo por DataFrame ─────────────────────────────────────────

    def _calculate_for_df(self, df: pd.DataFrame, timeframe: str) -> IndicatorValues:
        iv = IndicatorValues(timeframe=timeframe)
        close  = df['close']
        high   = df['high']
        low    = df['low']
        volume = df['volume']
        price  = float(close.iloc[-1])

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

        return iv

    # ─────────────────────────────────────────────────────────────────
    # INDICADORES INDIVIDUALES
    # ─────────────────────────────────────────────────────────────────

    def _calc_rsi(self, iv: IndicatorValues, close: pd.Series) -> None:
        rsi_series = ta.rsi(close, length=14)
        if rsi_series is None or rsi_series.dropna().empty:
            return
        rsi_clean = rsi_series.dropna()
        iv.rsi      = float(rsi_clean.iloc[-1])
        iv.rsi_prev = float(rsi_clean.iloc[-2]) if len(rsi_clean) >= 2 else iv.rsi

        # Momentum: bajando desde >50 por 3 velas consecutivas
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

    def _calc_emas(self, iv: IndicatorValues, close: pd.Series, price: float) -> None:
        for period, attr in [(7,'ema7'),(25,'ema25'),(50,'ema50'),(99,'ema99'),(200,'ema200')]:
            s = ta.ema(close, length=period)
            if s is not None and not s.dropna().empty:
                setattr(iv, attr, float(s.iloc[-1]))

        emas = [iv.ema7, iv.ema25, iv.ema50, iv.ema99, iv.ema200]

        # Contar cuántas están alineadas (cada una mayor que la siguiente)
        bullish = sum(1 for i in range(len(emas)-1) if emas[i] > emas[i+1] > 0)
        bearish = sum(1 for i in range(len(emas)-1) if emas[i] < emas[i+1] > 0)
        iv.emas_aligned_bullish = bullish + 1 if bullish == 4 else bullish
        iv.emas_aligned_bearish = bearish + 1 if bearish == 4 else bearish

        # Separación creciendo: comparar distancia EMA7-EMA200 vs vela anterior
        ema7_s   = ta.ema(close, length=7)
        ema200_s = ta.ema(close, length=200)
        if (ema7_s is not None and ema200_s is not None and
                len(ema7_s.dropna()) >= 2 and len(ema200_s.dropna()) >= 2):
            sep_now  = abs(float(ema7_s.iloc[-1])   - float(ema200_s.iloc[-1]))
            sep_prev = abs(float(ema7_s.iloc[-2])   - float(ema200_s.iloc[-2]))
            iv.ema_separation_growing = sep_now > sep_prev * 1.001

        # Compresión EMA7-EMA25
        if iv.ema7 > 0 and iv.ema25 > 0:
            iv.ema_compression_pct = abs(iv.ema7 - iv.ema25) / iv.ema25 * 100
            iv.ema_compression     = iv.ema_compression_pct < 0.1

    def _calc_macd(self, iv: IndicatorValues, close: pd.Series) -> None:
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is None or macd_df.empty:
            return
        # pandas-ta nombra las columnas: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        cols = macd_df.columns.tolist()
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

        # Cruces
        iv.macd_cross_bullish = (float(macd_s.iloc[-2]) < float(sig_s.iloc[-2]) and
                                  iv.macd_line >= iv.macd_signal)
        iv.macd_cross_bearish = (float(macd_s.iloc[-2]) > float(sig_s.iloc[-2]) and
                                  iv.macd_line <= iv.macd_signal)

        iv.macd_histogram_growing   = iv.macd_histogram > 0 and iv.macd_histogram > iv.macd_histogram_prev
        iv.macd_histogram_shrinking = iv.macd_histogram > 0 and iv.macd_histogram < iv.macd_histogram_prev

        # Divergencia: mirar 10 velas atrás
        self._calc_macd_divergence(iv, close, hist_s)

    def _calc_macd_divergence(self, iv: IndicatorValues,
                               close: pd.Series, hist_s: pd.Series) -> None:
        if len(close) < 12 or len(hist_s) < 12:
            return

        lookback = 10
        prices = close.iloc[-lookback:].values
        hists  = hist_s.iloc[-lookback:].values

        # Divergencia alcista: precio hace mínimos más bajos, MACD hace mínimos más altos
        price_min_now  = min(prices[-3:])
        price_min_prev = min(prices[:5])
        hist_min_now   = min(hists[-3:])
        hist_min_prev  = min(hists[:5])

        iv.macd_divergence_bullish = (price_min_now  < price_min_prev and
                                       hist_min_now   > hist_min_prev)

        # Divergencia bajista: precio hace máximos más altos, MACD hace máximos más bajos
        price_max_now  = max(prices[-3:])
        price_max_prev = max(prices[:5])
        hist_max_now   = max(hists[-3:])
        hist_max_prev  = max(hists[:5])

        iv.macd_divergence_bearish = (price_max_now  > price_max_prev and
                                       hist_max_now   < hist_max_prev)

    def _calc_ut_bot(self, iv: IndicatorValues,
                     close: pd.Series, high: pd.Series, low: pd.Series,
                     price: float, atr_period: int = 10, key_value: float = 1.0) -> None:
        """
        UT Bot: trailing stop dinámico basado en ATR.
        Genera señal cuando el precio cruza el trailing stop.
        key_value=1, atr_period=10 según el spec.
        """
        atr_s = ta.atr(high, low, close, length=atr_period)
        if atr_s is None or len(atr_s.dropna()) < 5:
            return

        atr_arr = atr_s.ffill().values
        close_arr = close.values
        n         = len(close_arr)

        trail_arr = np.zeros(n)
        trail_arr[0] = close_arr[0]

        for i in range(1, n):
            atr_val  = atr_arr[i] if not np.isnan(atr_arr[i]) else atr_arr[i-1]
            stop_dist = key_value * atr_val

            if close_arr[i] > trail_arr[i-1]:
                trail_arr[i] = max(trail_arr[i-1], close_arr[i] - stop_dist)
            else:
                trail_arr[i] = min(trail_arr[i-1], close_arr[i] + stop_dist)

        iv.ut_bot_trailing_stop = float(trail_arr[-1])

        # Señal de cruce
        if n >= 2:
            prev_above = close_arr[-2] > trail_arr[-2]
            curr_above = close_arr[-1] > trail_arr[-1]
            if not prev_above and curr_above:
                iv.ut_bot_signal = "buy"
            elif prev_above and not curr_above:
                iv.ut_bot_signal = "sell"
            else:
                iv.ut_bot_signal = "none"

        # Precio cerca del trailing stop (menos de 0.2%)
        if iv.ut_bot_trailing_stop > 0:
            dist_pct = abs(price - iv.ut_bot_trailing_stop) / iv.ut_bot_trailing_stop * 100
            iv.ut_bot_price_near_stop = dist_pct < 0.2

    def _calc_squeeze(self, iv: IndicatorValues,
                      close: pd.Series, high: pd.Series, low: pd.Series) -> None:
        """
        Squeeze Momentum de LazyBear.
        Compara Bollinger Bands con Keltner Channels.
        sqz_on  = BB dentro de KC (mercado comprimido)
        sqz_off = BB saliendo de KC (energía liberándose)
        """
        # Bollinger Bands (20, 2.0)
        bb = ta.bbands(close, length=20, std=2.0)
        # Keltner Channels (20, 1.5 ATR)
        kc = ta.kc(high, low, close, length=20, scalar=1.5)

        if bb is None or kc is None:
            return

        bb_cols = bb.columns.tolist()
        kc_cols = kc.columns.tolist()

        bb_lower_col = [c for c in bb_cols if 'BBL' in c]
        bb_upper_col = [c for c in bb_cols if 'BBU' in c]
        kc_lower_col = [c for c in kc_cols if 'KCL' in c]
        kc_upper_col = [c for c in kc_cols if 'KCU' in c]

        if not (bb_lower_col and bb_upper_col and kc_lower_col and kc_upper_col):
            return

        bb_l = float(bb[bb_lower_col[0]].iloc[-1])
        bb_u = float(bb[bb_upper_col[0]].iloc[-1])
        kc_l = float(kc[kc_lower_col[0]].iloc[-1])
        kc_u = float(kc[kc_upper_col[0]].iloc[-1])

        if any(np.isnan([bb_l, bb_u, kc_l, kc_u])):
            return

        iv.sqz_on  = bb_l > kc_l and bb_u < kc_u   # BB dentro de KC
        iv.sqz_off = bb_l < kc_l and bb_u > kc_u   # BB fuera de KC

        # Histograma del momentum (momentum lineal simple)
        highest_high = high.rolling(20).max()
        lowest_low   = low.rolling(20).min()
        mid_hl = (highest_high + lowest_low) / 2
        mid_ema = ta.ema(close, length=20)

        if mid_hl is None or mid_ema is None:
            return

        delta = close - (mid_hl + mid_ema) / 2
        hist  = ta.linreg(delta, length=20)

        if hist is None or len(hist.dropna()) < 2:
            return

        iv.sqz_histogram      = float(hist.iloc[-1])
        iv.sqz_histogram_prev = float(hist.iloc[-2])

        # Color del histograma
        def _sqz_color(val: float, prev: float) -> str:
            if val >= 0:
                return "dark_green" if val >= prev else "light_green"
            else:
                return "dark_red" if val <= prev else "light_red"

        iv.sqz_histogram_color      = _sqz_color(iv.sqz_histogram, iv.sqz_histogram_prev)
        iv.sqz_histogram_color_prev = _sqz_color(
            iv.sqz_histogram_prev,
            float(hist.iloc[-3]) if len(hist.dropna()) >= 3 else iv.sqz_histogram_prev
        )
        iv.sqz_color_change = iv.sqz_histogram_color != iv.sqz_histogram_color_prev

    def _calc_ichimoku(self, iv: IndicatorValues,
                       close: pd.Series, high: pd.Series, low: pd.Series,
                       price: float) -> None:
        ichi = ta.ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52, lookahead=False)
        if ichi is None:
            return

        # pandas-ta devuelve una tupla (df_actual, df_futuro)
        df_ichi = ichi[0] if isinstance(ichi, tuple) else ichi
        if df_ichi is None or df_ichi.empty:
            return

        cols = df_ichi.columns.tolist()

        def _get(prefix: str) -> float:
            c = [x for x in cols if x.startswith(prefix)]
            if c:
                val = df_ichi[c[0]].dropna()
                if not val.empty:
                    return float(val.iloc[-1])
            return 0.0

        def _get_prev(prefix: str) -> float:
            c = [x for x in cols if x.startswith(prefix)]
            if c:
                val = df_ichi[c[0]].dropna()
                if len(val) >= 2:
                    return float(val.iloc[-2])
            return 0.0

        iv.ichi_tenkan = _get('ITS')
        iv.ichi_kijun  = _get('IKS')
        iv.ichi_span_a = _get('ISA')
        iv.ichi_span_b = _get('ISB')

        if iv.ichi_span_a > 0 and iv.ichi_span_b > 0:
            iv.ichi_cloud_top    = max(iv.ichi_span_a, iv.ichi_span_b)
            iv.ichi_cloud_bottom = min(iv.ichi_span_a, iv.ichi_span_b)
            iv.price_above_cloud = price > iv.ichi_cloud_top
            iv.price_below_cloud = price < iv.ichi_cloud_bottom
            iv.price_in_cloud    = iv.ichi_cloud_bottom <= price <= iv.ichi_cloud_top

        # Cruce tenkan/kijun
        if iv.ichi_tenkan > 0 and iv.ichi_kijun > 0:
            tenkan_prev = _get_prev('ITS')
            kijun_prev  = _get_prev('IKS')
            iv.ichi_tk_cross_bullish = (tenkan_prev < kijun_prev and
                                         iv.ichi_tenkan >= iv.ichi_kijun)
            iv.ichi_tk_cross_bearish = (tenkan_prev > kijun_prev and
                                         iv.ichi_tenkan <= iv.ichi_kijun)

    def _calc_vwap(self, iv: IndicatorValues, df: pd.DataFrame, price: float) -> None:
        """VWAP diario (se resetea en cada sesión)."""
        vwap_s = ta.vwap(df['high'], df['low'], df['close'], df['volume'])
        if vwap_s is None or vwap_s.dropna().empty:
            return

        vwap_clean = vwap_s.dropna()
        iv.vwap = float(vwap_clean.iloc[-1])

        if iv.vwap > 0:
            iv.price_above_vwap = price > iv.vwap
            if len(vwap_clean) >= 2:
                close_s = df['close'].iloc[-len(vwap_clean):]
                prev_above = float(close_s.iloc[-2]) > float(vwap_clean.iloc[-2])
                curr_above = price > iv.vwap
                iv.vwap_cross_bullish = not prev_above and curr_above
                iv.vwap_cross_bearish = prev_above and not curr_above

    def _calc_volume(self, iv: IndicatorValues, volume: pd.Series) -> None:
        iv.volume_current = float(volume.iloc[-1])
        iv.volume_avg_20  = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        if iv.volume_avg_20 > 0:
            iv.volume_ratio = iv.volume_current / iv.volume_avg_20

    def _calc_bollinger(self, iv: IndicatorValues,
                        close: pd.Series, price: float) -> None:
        bb = ta.bbands(close, length=20, std=2.0)
        if bb is None or bb.empty:
            return

        cols = bb.columns.tolist()
        lower_col  = [c for c in cols if 'BBL' in c]
        middle_col = [c for c in cols if 'BBM' in c]
        upper_col  = [c for c in cols if 'BBU' in c]

        if not (lower_col and middle_col and upper_col):
            return

        iv.bb_lower  = float(bb[lower_col[0]].iloc[-1])
        iv.bb_middle = float(bb[middle_col[0]].iloc[-1])
        iv.bb_upper  = float(bb[upper_col[0]].iloc[-1])

        if iv.bb_middle > 0 and iv.bb_upper > iv.bb_lower:
            iv.bb_width = (iv.bb_upper - iv.bb_lower) / iv.bb_middle
            band_range  = iv.bb_upper - iv.bb_lower
            third       = band_range / 3

            iv.price_at_lower_band   = price <= iv.bb_lower + third * 0.3
            iv.price_at_upper_band   = price >= iv.bb_upper - third * 0.3
            iv.price_in_lower_third  = price <= iv.bb_lower + third
            iv.price_in_upper_third  = price >= iv.bb_upper - third
            iv.price_above_upper_band = price > iv.bb_upper
            iv.price_below_lower_band = price < iv.bb_lower

    def _calc_atr(self, iv: IndicatorValues,
                  close: pd.Series, high: pd.Series, low: pd.Series,
                  price: float) -> None:
        atr_s = ta.atr(high, low, close, length=14)
        if atr_s is None or atr_s.dropna().empty:
            return
        iv.atr = float(atr_s.iloc[-1])
        if price > 0:
            iv.atr_pct = iv.atr / price * 100

    def _calc_cci(self, iv: IndicatorValues,
                  close: pd.Series, high: pd.Series, low: pd.Series) -> None:
        cci_s = ta.cci(high, low, close, length=20)
        if cci_s is None or len(cci_s.dropna()) < 2:
            return

        cci_clean = cci_s.dropna()
        iv.cci      = float(cci_clean.iloc[-1])
        iv.cci_prev = float(cci_clean.iloc[-2])

        iv.cci_cross_up_100   = iv.cci_prev < -100 and iv.cci >= -100   # salió de sobreventa
        iv.cci_cross_down_100 = iv.cci_prev > 100  and iv.cci <= 100    # salió de sobrecompra
        iv.cci_extreme_bullish = iv.cci > 150
        iv.cci_extreme_bearish = iv.cci < -150

    def _calc_stochastic(self, iv: IndicatorValues,
                          close: pd.Series, high: pd.Series, low: pd.Series) -> None:
        stoch = ta.stoch(high, low, close, k=14, d=3, smooth_k=3)
        if stoch is None or stoch.empty:
            return

        cols   = stoch.columns.tolist()
        k_col  = [c for c in cols if 'STOCHk' in c]
        d_col  = [c for c in cols if 'STOCHd' in c]

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

        iv.stoch_cross_bullish = (iv.stoch_k_prev < iv.stoch_d_prev and
                                   iv.stoch_k >= iv.stoch_d and
                                   iv.stoch_k < 20)
        iv.stoch_cross_bearish = (iv.stoch_k_prev > iv.stoch_d_prev and
                                   iv.stoch_k <= iv.stoch_d and
                                   iv.stoch_k > 80)
        iv.stoch_oversold   = iv.stoch_k < 20 and not iv.stoch_cross_bullish
        iv.stoch_overbought = iv.stoch_k > 80 and not iv.stoch_cross_bearish

    def _calc_pivots(self, iv: IndicatorValues, df: pd.DataFrame, price: float) -> None:
        """
        Pivotes estándar calculados con el OHLC de la vela anterior
        (o la penúltima vela completa del DataFrame).
        """
        if len(df) < 2:
            return

        prev = df.iloc[-2]
        H = float(prev['high'])
        L = float(prev['low'])
        C = float(prev['close'])

        iv.pivot = (H + L + C) / 3
        iv.r1    = 2 * iv.pivot - L
        iv.r2    = iv.pivot + (H - L)
        iv.r3    = H + 2 * (iv.pivot - L)
        iv.s1    = 2 * iv.pivot - H
        iv.s2    = iv.pivot - (H - L)
        iv.s3    = L - 2 * (H - iv.pivot)

        # Zona del precio respecto a los pivotes
        tolerance = (H - L) * 0.05   # 5% del rango como tolerancia para "cerca"
        if price > iv.r1 + tolerance:
            iv.pivot_zone = "above_r1"
        elif abs(price - iv.r1) <= tolerance:
            iv.pivot_zone = "near_r1"
        elif price > iv.pivot:
            iv.pivot_zone = "between_p_r1"
        elif abs(price - iv.pivot) <= tolerance:
            iv.pivot_zone = "at_pivot"
        elif abs(price - iv.s1) <= tolerance:
            iv.pivot_zone = "near_s1"
        elif abs(price - iv.s2) <= tolerance:
            iv.pivot_zone = "near_s2"
        elif price < iv.s2 - tolerance:
            iv.pivot_zone = "below_s2"
        else:
            iv.pivot_zone = "between_s1_p"

    def _calc_lateralization(self, iv: IndicatorValues,
                              close: pd.Series, high: pd.Series,
                              low: pd.Series) -> None:
        if len(close) < 5:
            return

        last5_close = close.iloc[-5:]
        last5_high  = high.iloc[-5:]
        last5_low   = low.iloc[-5:]
        score = 0.0

        # Dojis: cuerpo < 10% del rango de la vela
        doji_count = 0
        for i in range(-5, 0):
            body  = abs(float(close.iloc[i]) - float(close.iloc[i-1]))
            range_ = float(high.iloc[i]) - float(low.iloc[i])
            if range_ > 0 and body / range_ < 0.1:
                doji_count += 1
        iv.doji_count_recent = doji_count
        score += doji_count * 0.25

        # Mechas largas en ambos lados
        wick_count = 0
        for i in range(-5, 0):
            o = float(close.iloc[i-1])
            c = float(close.iloc[i])
            h = float(high.iloc[i])
            l = float(low.iloc[i])
            body    = abs(c - o)
            range_  = h - l
            if range_ > 0:
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l
                if upper_wick > body * 0.5 and lower_wick > body * 0.5:
                    wick_count += 1
        iv.long_wick_count = wick_count
        score += wick_count * 0.20

        # RSI neutro
        if 45 <= iv.rsi <= 55:
            score += 0.30

        # Máximos y mínimos convergentes (rango estrechándose)
        highs = last5_high.values
        lows  = last5_low.values
        high_range = max(highs) - min(highs)
        low_range  = max(lows)  - min(lows)
        price_range = float(close.iloc[-1])
        if price_range > 0:
            convergence = (high_range + low_range) / 2 / price_range
            if convergence < 0.005:   # rango < 0.5% del precio
                iv.highs_lows_converging = True
                score += 0.35

        iv.lateralization_score = min(score, 1.0)

    def _calc_candle_patterns(self, iv: IndicatorValues,
                               df: pd.DataFrame, price: float) -> None:
        if len(df) < 3:
            return

        o1 = float(df['open'].iloc[-2])
        c1 = float(df['close'].iloc[-2])
        h1 = float(df['high'].iloc[-2])
        l1 = float(df['low'].iloc[-2])
        o2 = float(df['open'].iloc[-1])
        c2 = float(df['close'].iloc[-1])
        h2 = float(df['high'].iloc[-1])
        l2 = float(df['low'].iloc[-1])

        body1  = abs(c1 - o1)
        body2  = abs(c2 - o2)
        range2 = h2 - l2

        # Envolvente alcista
        if c1 < o1 and c2 > o2 and c2 > o1 and o2 < c1:
            iv.candle_pattern = "engulfing_bull"
            iv.pattern_strength = min(body2 / body1, 2.0) / 2 if body1 > 0 else 0.5

        # Envolvente bajista
        elif c1 > o1 and c2 < o2 and c2 < o1 and o2 > c1:
            iv.candle_pattern = "engulfing_bear"
            iv.pattern_strength = min(body2 / body1, 2.0) / 2 if body1 > 0 else 0.5

        # Martillo (hammer): mecha inferior larga, cuerpo pequeño arriba
        elif range2 > 0:
            lower_wick = min(o2, c2) - l2
            upper_wick = h2 - max(o2, c2)
            if lower_wick > body2 * 2 and upper_wick < body2 * 0.5 and body2 < range2 * 0.4:
                iv.candle_pattern = "hammer"
                iv.pattern_strength = lower_wick / range2

            # Estrella fugaz (shooting star): mecha superior larga
            elif upper_wick > body2 * 2 and lower_wick < body2 * 0.5 and body2 < range2 * 0.4:
                iv.candle_pattern = "shooting_star"
                iv.pattern_strength = upper_wick / range2

            # Doji
            elif body2 < range2 * 0.1:
                iv.candle_pattern = "doji"
                iv.pattern_strength = 0.5

        # ¿El patrón está en soporte o resistencia?
        if iv.candle_pattern != "none" and iv.pivot > 0:
            tolerance = iv.atr * 0.5 if iv.atr > 0 else price * 0.005
            near_levels = [iv.s1, iv.s2, iv.s3, iv.r1, iv.r2, iv.r3, iv.pivot]
            iv.pattern_at_support = any(abs(price - lvl) <= tolerance
                                         for lvl in near_levels if lvl > 0)


# ─────────────────────────────────────────────
# ANÁLISIS MACRO (tendencia por timeframe alto)
# ─────────────────────────────────────────────

@dataclass
class MacroTrend:
    """Tendencia resumida de los timeframes altos."""
    trend_1w: str = "neutral"    # "bullish", "bearish", "neutral"
    trend_1d: str = "neutral"
    trend_4h: str = "neutral"
    aligned_count: int = 0       # cuántos de los 3 están alineados
    divergence_daily_weekly: bool = False  # 1D y 1W en direcciones opuestas


def analyze_macro_trend(indicators: dict[str, IndicatorValues]) -> MacroTrend:
    """
    Determina la tendencia macro mirando EMAs y precio relativo
    en los timeframes altos.
    """
    macro = MacroTrend()

    def _trend_from_iv(iv: IndicatorValues) -> str:
        price = iv.current_price
        if price <= 0 or iv.ema50 <= 0:
            return "neutral"
        # Tendencia alcista: precio > EMA50 y EMA50 > EMA200 (si disponible)
        bullish = (price > iv.ema50 and
                   (iv.ema200 <= 0 or iv.ema50 > iv.ema200) and
                   iv.emas_aligned_bullish >= 3)
        bearish = (price < iv.ema50 and
                   (iv.ema200 <= 0 or iv.ema50 < iv.ema200) and
                   iv.emas_aligned_bearish >= 3)
        if bullish:
            return "bullish"
        elif bearish:
            return "bearish"
        return "neutral"

    if '1w' in indicators:
        macro.trend_1w = _trend_from_iv(indicators['1w'])
    if '1d' in indicators:
        macro.trend_1d = _trend_from_iv(indicators['1d'])
    if '4h' in indicators:
        macro.trend_4h = _trend_from_iv(indicators['4h'])

    trends = [macro.trend_1w, macro.trend_1d, macro.trend_4h]
    macro.aligned_count = max(
        sum(1 for t in trends if t == "bullish"),
        sum(1 for t in trends if t == "bearish"),
    )

    macro.divergence_daily_weekly = (
        macro.trend_1w != "neutral" and
        macro.trend_1d != "neutral" and
        macro.trend_1w != macro.trend_1d
    )

    return macro


# ─────────────────────────────────────────────
# TEST RÁPIDO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from market_feed import create_feed

    print("Probando indicators.py contra testnet...")
    feed = create_feed('live', symbol='BTCUSDC', testnet=True)
    feed.start()
    time.sleep(2)

    snap = feed.get_snapshot()
    calc = IndicatorCalculator()
    t0   = time.monotonic()
    ivs  = calc.calculate(snap)
    elapsed = (time.monotonic() - t0) * 1000

    print(f"\nCalculados en {elapsed:.1f}ms | Timeframes: {list(ivs.keys())}")

    if '15m' in ivs:
        iv = ivs['15m']
        print(f"\n--- 15m ---")
        print(f"Precio:       {iv.current_price:.2f}")
        print(f"RSI:          {iv.rsi:.1f}")
        print(f"EMAs bull:    {iv.emas_aligned_bullish}/5")
        print(f"EMAs bear:    {iv.emas_aligned_bearish}/5")
        print(f"MACD hist:    {iv.macd_histogram:.4f}")
        print(f"UT Bot:       {iv.ut_bot_signal}")
        print(f"Squeeze:      on={iv.sqz_on} off={iv.sqz_off} color={iv.sqz_histogram_color}")
        print(f"BB lower:     {iv.bb_lower:.2f} | upper: {iv.bb_upper:.2f}")
        print(f"ATR:          {iv.atr:.2f} ({iv.atr_pct:.3f}%)")
        print(f"Volumen x:    {iv.volume_ratio:.2f}")
        print(f"Lateraliz:    {iv.lateralization_score:.2f}")
        print(f"Patrón vela:  {iv.candle_pattern}")
        print(f"Pivot zone:   {iv.pivot_zone}")

    macro = analyze_macro_trend(ivs)
    print(f"\n--- Macro ---")
    print(f"1W: {macro.trend_1w} | 1D: {macro.trend_1d} | 4H: {macro.trend_4h}")
    print(f"Alineados: {macro.aligned_count}/3 | Divergencia D/W: {macro.divergence_daily_weekly}")

    feed.stop()
