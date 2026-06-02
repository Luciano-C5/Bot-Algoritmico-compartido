"""
scoring.py
==========
Sistema de puntaje ponderado continuo.

Recibe el dict[str, IndicatorValues] que devuelve indicators.py
y devuelve un ScoreResult con el puntaje total, el desglose
por indicador, el nivel de señal y la dirección recomendada.

Regla fundamental: este archivo no sabe nada de órdenes ni de
ejecución. Solo puntúa. Sin estado, sin efectos secundarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from indicators import IndicatorValues, MacroTrend, analyze_macro_trend


# ─────────────────────────────────────────────
# ESTRUCTURA DE SALIDA
# ─────────────────────────────────────────────

@dataclass
class ScoreResult:
    """
    Resultado del scoring para una dirección y modo dados.
    strategy.py recibe esto y toma la decisión de operar o no.
    """
    direction: str          # "long" o "short"
    mode: str               # "scalp", "mediano", "swing"
    timeframe: str          # timeframe principal del modo

    total: float = 0.0
    maximum_possible: float = 0.0
    normalized: float = 0.0         # total / maximum_possible (0.0 a 1.0)

    signal_level: int = 0           # 0=no operar, 1=fuerte, 2=moderado, 3=débil
    should_trade: bool = False

    # Desglose por componente (para logs y debugging)
    breakdown: dict[str, float] = field(default_factory=dict)

    # Condiciones de bloqueo activas
    blocked_reasons: list[str] = field(default_factory=list)

    # Contexto macro
    macro_aligned: bool = False
    macro_divergence: bool = False  # 1D y 1W en direcciones opuestas
    leverage_multiplier: float = 1.0  # se reduce a 0.6 si hay divergencia macro

    # TP/SL aproximados ajustados al modo y nivel
    approx_tp_pct: float = 0.0      # % sobre capital
    approx_sl_pct: float = 0.0
    leverage: int = 1

    def __str__(self) -> str:
        lines = [
            f"[ScoreResult] {self.direction.upper()} | {self.mode} | {self.timeframe}",
            f"  Puntaje:  {self.total:.1f} / {self.maximum_possible:.1f} "
            f"({self.normalized*100:.1f}%)",
            f"  Nivel:    {self.signal_level} | Operar: {self.should_trade}",
            f"  Leverage: x{self.leverage} (mult: {self.leverage_multiplier})",
        ]
        if self.blocked_reasons:
            lines.append(f"  BLOQUEADO: {', '.join(self.blocked_reasons)}")
        if self.breakdown:
            lines.append("  Desglose:")
            for k, v in sorted(self.breakdown.items(), key=lambda x: -abs(x[1])):
                bar = "█" * int(abs(v) / 2) if v > 0 else "▒" * int(abs(v) / 2)
                sign = "+" if v >= 0 else ""
                lines.append(f"    {k:<28} {sign}{v:>5.1f}  {bar}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# UMBRALES Y CONFIGURACIÓN
# ─────────────────────────────────────────────

# Umbral como fracción del puntaje máximo posible
THRESHOLDS = {
    1: 0.57,   # Nivel 1: ~57% del máximo → señal fuerte, x5
    2: 0.45,   # Nivel 2: ~45% del máximo → señal moderada, x3
    3: 0.35,   # Nivel 3: ~35% del máximo → señal débil, x2
}

# Apalancamiento por nivel
LEVERAGE_BY_LEVEL = {1: 5, 2: 3, 3: 2}

# TP y SL aproximados por modo y nivel (% sobre capital)
TP_SL_BY_MODE = {
    "scalp":   {"tp": 0.6, "sl": 0.4},
    "mediano": {"tp": 1.0, "sl": 0.5},
    "swing":   {"tp": 2.5, "sl": 0.8},
}

# Incremento de umbral por condición especial (suma a THRESHOLDS)
THRESHOLD_INCREMENTS = {
    "macro_divergence":    0.15,   # 1D y 1W en direcciones opuestas
    "low_volume_hour":     0.05,   # horario sin volumen
    "wall_street_open":    0.10,   # apertura Wall Street 13:30 UTC
    "week_close":          0.10,   # cierre de semana viernes 21:00 UTC
    "day_close":           0.05,   # cierre de día 23:00 UTC
    "news_moderate":       0.15,   # noticias de impacto moderado
}

# Timeframe principal por modo
MODE_TIMEFRAMES = {
    "scalp":   "15m",
    "mediano": "1h",
    "swing":   "4h",
}

# Timeframe de confirmación por modo
CONFIRM_TIMEFRAMES = {
    "scalp":   "5m",
    "mediano": "15m",
    "swing":   "1h",
}


# ─────────────────────────────────────────────
# SCORER PRINCIPAL
# ─────────────────────────────────────────────

class Scorer:
    """
    Calcula el puntaje para una dirección y modo dados.

    Uso:
        scorer = Scorer()
        indicators = IndicatorCalculator().calculate(snapshot)
        macro = analyze_macro_trend(indicators)

        result_long  = scorer.score(indicators, macro, "long",  "scalp")
        result_short = scorer.score(indicators, macro, "short", "scalp")
    """

    def score(
        self,
        indicators: dict[str, IndicatorValues],
        macro: MacroTrend,
        direction: str,          # "long" o "short"
        mode: str,               # "scalp", "mediano", "swing"
        threshold_increment: float = 0.0,   # incrementos por horario/noticias
    ) -> ScoreResult:

        tf_main    = MODE_TIMEFRAMES[mode]
        tf_confirm = CONFIRM_TIMEFRAMES[mode]

        iv_main    = indicators.get(tf_main)
        iv_confirm = indicators.get(tf_confirm)

        result = ScoreResult(
            direction = direction,
            mode      = mode,
            timeframe = tf_main,
        )

        if iv_main is None:
            result.blocked_reasons.append(f"Sin datos para {tf_main}")
            return result

        # ── Verificar condiciones de bloqueo ──────────────────────────
        if self._check_blocks(iv_main, direction, result):
            return result   # bloqueado, puntaje 0

        # ── Calcular puntaje por componente ───────────────────────────
        scores = {}
        max_scores = {}

        self._score_rsi(iv_main, direction, scores, max_scores)
        self._score_emas(iv_main, direction, scores, max_scores)
        self._score_macd(iv_main, direction, scores, max_scores)
        self._score_ut_bot(iv_main, direction, scores, max_scores, macro)
        self._score_squeeze(iv_main, direction, scores, max_scores)
        self._score_ichimoku(iv_main, direction, scores, max_scores)
        self._score_vwap(iv_main, direction, scores, max_scores)
        self._score_volume(iv_main, scores, max_scores)
        self._score_bollinger(iv_main, direction, scores, max_scores)
        self._score_macro(macro, direction, scores, max_scores)
        self._score_pivots(iv_main, direction, scores, max_scores)
        self._score_funding(iv_main, direction, scores, max_scores)
        self._score_orderbook(iv_main, direction, scores, max_scores)
        self._score_candle_patterns(iv_main, direction, mode, scores, max_scores)
        self._score_cci(iv_main, direction, scores, max_scores)
        self._score_stochastic(iv_main, direction, scores, max_scores)
        self._score_macd_divergence(iv_main, direction, scores, max_scores)
        self._score_lateralization(iv_main, macro, scores, max_scores)

        # Confirmación del timeframe secundario (bonus/penalización)
        if iv_confirm is not None:
            self._score_confirmation(iv_confirm, direction, scores, max_scores)

        # ── Totales ───────────────────────────────────────────────────
        result.breakdown        = scores
        result.total            = sum(scores.values())
        result.maximum_possible = sum(max_scores.values())

        if result.maximum_possible > 0:
            result.normalized = result.total / result.maximum_possible
        else:
            result.normalized = 0.0

        # ── Contexto macro ────────────────────────────────────────────
        result.macro_aligned    = macro.aligned_count >= 2
        result.macro_divergence = macro.divergence_daily_weekly

        if result.macro_divergence:
            threshold_increment += THRESHOLD_INCREMENTS["macro_divergence"]
            result.leverage_multiplier = 0.6
            # Con divergencia macro solo se permiten scalps
            if mode != "scalp":
                result.blocked_reasons.append(
                    "Divergencia macro 1D/1W: solo scalps permitidos"
                )
                return result

        # ── Determinar nivel de señal ─────────────────────────────────
        result.signal_level = self._determine_level(
            result.normalized,
            macro,
            direction,
            mode,
            threshold_increment,
        )

        # Nivel 3 requiere macro muy alineada
        if result.signal_level == 3:
            if macro.aligned_count < 2:
                result.signal_level = 0
                result.blocked_reasons.append(
                    "Nivel 3 requiere al menos 2 timeframes macro alineados"
                )

        result.should_trade = result.signal_level > 0

        # ── Parámetros operativos ─────────────────────────────────────
        if result.should_trade:
            base_lev = LEVERAGE_BY_LEVEL[result.signal_level]
            result.leverage = max(1, int(base_lev * result.leverage_multiplier))
            tp_sl = TP_SL_BY_MODE[mode]
            result.approx_tp_pct = tp_sl["tp"]
            result.approx_sl_pct = tp_sl["sl"]
            # Nivel 3: TP reducido
            if result.signal_level == 3:
                result.approx_tp_pct = 0.15

        return result

    # ─────────────────────────────────────────────────────────────────
    # CONDICIONES DE BLOQUEO
    # ─────────────────────────────────────────────────────────────────

    def _check_blocks(
        self, iv: IndicatorValues, direction: str, result: ScoreResult
    ) -> bool:
        """
        Condiciones que bloquean completamente la entrada.
        Devuelve True si hay bloqueo.
        """
        blocked = False

        # RSI saturado + EMAs completamente alineadas = probable lateralización
        if direction == "long":
            rsi_saturated = iv.rsi > 70
            emas_aligned  = iv.emas_aligned_bullish >= 5
        else:
            rsi_saturated = iv.rsi < 30
            emas_aligned  = iv.emas_aligned_bearish >= 5

        if rsi_saturated and emas_aligned:
            result.blocked_reasons.append(
                f"RSI saturado ({iv.rsi:.1f}) + EMAs completamente alineadas: "
                f"probable lateralización"
            )
            blocked = True

        # Squeeze activo fuerte: mercado sin dirección
        if iv.sqz_on and not iv.sqz_off:
            # No bloqueamos completamente pero sí anotamos
            # (se penaliza en el scoring)
            pass

        # Score de lateralización muy alto sin confluencia macro
        if iv.lateralization_score >= 0.65:
            result.blocked_reasons.append(
                f"Score de lateralización alto ({iv.lateralization_score:.2f})"
            )
            blocked = True

        return blocked

    # ─────────────────────────────────────────────────────────────────
    # COMPONENTES DE PUNTAJE
    # ─────────────────────────────────────────────────────────────────

    def _score_rsi(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["rsi"] = 12.0
        s = 0.0

        if direction == "long":
            if iv.rsi < 15:
                s = 12.0
            elif iv.rsi < 20:
                s = 10.0
            elif iv.rsi < 30:
                s = 8.0
            elif iv.rsi < 40:
                s = 4.0
            elif iv.rsi < 50:
                s = 1.0

            if iv.rsi_momentum_bearish:
                s += 3.0    # presión vendedora agotándose
                max_scores["rsi"] += 3.0

            if iv.rsi > 70:
                s -= 4.0    # puede lateralizar

        else:  # short
            if iv.rsi > 85:
                s = 12.0
            elif iv.rsi > 80:
                s = 10.0
            elif iv.rsi > 70:
                s = 8.0
            elif iv.rsi > 60:
                s = 4.0
            elif iv.rsi > 50:
                s = 1.0

            if iv.rsi_momentum_bullish:
                s += 3.0
                max_scores["rsi"] += 3.0

            if iv.rsi < 30:
                s -= 4.0

        scores["rsi"] = s

    def _score_emas(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["emas"] = 13.0
        s = 0.0

        aligned = iv.emas_aligned_bullish if direction == "long" else iv.emas_aligned_bearish

        if aligned >= 5:
            s += 10.0
        elif aligned >= 4:
            s += 6.0
        elif aligned >= 3:
            s += 3.0

        if iv.ema_separation_growing:
            s += 3.0

        # Compresión: solo suma si hay squeeze terminando o volumen creciendo
        if iv.ema_compression:
            if iv.sqz_off or iv.volume_ratio > 1.2:
                s += 4.0
            # Sola no suma nada (puede explotar en cualquier dirección)

        scores["emas"] = s

    def _score_macd(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["macd"] = 14.0
        s = 0.0

        if direction == "long":
            if iv.macd_cross_bullish:
                s += 8.0
            if iv.macd_histogram_growing:
                s += 5.0
            elif iv.macd_histogram_shrinking:
                s += 2.0
        else:  # short
            if iv.macd_cross_bearish:
                s += 8.0
            if iv.macd_histogram < 0 and iv.macd_histogram < iv.macd_histogram_prev:
                s += 5.0   # histograma negativo y más negativo
            elif iv.macd_histogram < 0 and iv.macd_histogram > iv.macd_histogram_prev:
                s += 2.0   # histograma negativo pero recuperándose

        scores["macd"] = s

    def _score_ut_bot(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict, macro: MacroTrend
    ) -> None:
        max_scores["ut_bot"] = 8.0
        s = 0.0

        expected_signal = "buy" if direction == "long" else "sell"

        if iv.ut_bot_signal == expected_signal:
            s += 8.0
        elif iv.ut_bot_price_near_stop:
            s += 3.0

        # Si la señal va contra tendencia macro, reducir a la mitad
        macro_direction = "bullish" if direction == "long" else "bearish"
        macro_trend = macro.trend_1d  # usamos diario como referencia
        if macro_trend != "neutral" and macro_trend != macro_direction:
            s *= 0.5

        scores["ut_bot"] = s

    def _score_squeeze(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["squeeze"] = 10.0
        s = 0.0

        if iv.sqz_off:
            s += 8.0   # mejor momento: energía liberándose

        if direction == "long":
            if iv.sqz_histogram > 0 and iv.sqz_histogram > iv.sqz_histogram_prev:
                s += 5.0
            # Cambio de color que indica agotamiento o cambio inminente
            if iv.sqz_color_change:
                if iv.sqz_histogram_color in ("dark_green", "light_green"):
                    s += 3.0
        else:  # short
            if iv.sqz_histogram < 0 and iv.sqz_histogram < iv.sqz_histogram_prev:
                s += 5.0
            if iv.sqz_color_change:
                if iv.sqz_histogram_color in ("dark_red", "light_red"):
                    s += 3.0

        # Squeeze activo: mercado sin dirección, penalizar
        if iv.sqz_on and not iv.sqz_off:
            s -= 3.0

        scores["squeeze"] = max(min(s, 10.0), -3.0) # no bajar más que la penalización máxima

    def _score_ichimoku(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["ichimoku"] = 9.0
        s = 0.0

        if direction == "long":
            if iv.price_above_cloud:
                s += 5.0
            elif iv.price_in_cloud:
                s -= 2.0
            if iv.ichi_tk_cross_bullish:
                s += 4.0
        else:  # short
            if iv.price_below_cloud:
                s += 5.0
            elif iv.price_in_cloud:
                s -= 2.0
            if iv.ichi_tk_cross_bearish:
                s += 4.0

        scores["ichimoku"] = s

    def _score_vwap(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["vwap"] = 5.0
        s = 0.0

        if direction == "long":
            if iv.vwap_cross_bullish:
                s = 5.0   # cruce fresco al alza
            elif iv.price_above_vwap:
                s = 3.0
        else:  # short
            if iv.vwap_cross_bearish:
                s = 5.0
            elif not iv.price_above_vwap:
                s = 3.0

        scores["vwap"] = s

    def _score_volume(
        self, iv: IndicatorValues,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["volume"] = 8.0
        s = 0.0

        if iv.volume_ratio >= 2.0:
            s = 8.0
        elif iv.volume_ratio >= 1.5:
            s = 5.0
        elif iv.volume_ratio >= 1.2:
            s = 3.0
        elif iv.volume_ratio < 1.0:
            s = -2.0

        scores["volume"] = s

    def _score_bollinger(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["bollinger"] = 6.0
        s = 0.0

        if direction == "long":
            if iv.price_at_lower_band:
                s = 6.0
            elif iv.price_in_lower_third:
                s = 3.0
            if iv.price_above_upper_band:
                s -= 4.0   # sobreextendido, arriesgado entrar long
        else:  # short
            if iv.price_at_upper_band:
                s = 6.0
            elif iv.price_in_upper_third:
                s = 3.0
            if iv.price_below_lower_band:
                s -= 4.0

        scores["bollinger"] = s

    def _score_macro(
        self, macro: MacroTrend, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["macro"] = 15.0
        s = 0.0

        expected = "bullish" if direction == "long" else "bearish"
        opposite = "bearish" if direction == "long" else "bullish"

        trends = [macro.trend_1w, macro.trend_1d, macro.trend_4h]
        aligned = sum(1 for t in trends if t == expected)
        against = sum(1 for t in trends if t == opposite)

        if aligned == 3:
            s = 15.0
        elif macro.trend_1w == expected and macro.trend_1d == expected:
            s = 10.0
        elif macro.trend_1d == expected:
            s = 4.0

        if against == 3:
            s -= 8.0

        scores["macro"] = s

    def _score_pivots(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["pivots"] = 5.0
        s = 0.0

        if direction == "long":
            if iv.pivot_zone == "near_s1":
                s = 5.0
            elif iv.pivot_zone == "near_s2":
                s = 4.0
            elif iv.pivot_zone == "between_s1_p":
                s = 2.0
        else:  # short
            if iv.pivot_zone == "near_r1":
                s = 5.0
            elif iv.pivot_zone == "above_r1":
                s = 4.0
            elif iv.pivot_zone == "between_p_r1":
                s = 2.0

        scores["pivots"] = s

    def _score_funding(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["funding"] = 4.0
        s = 0.0
        fr = iv.funding_rate

        if direction == "long":
            if fr < -0.0001:     # -0.01%
                s = 4.0
            elif fr < -0.00005:  # -0.005%
                s = 2.0
            elif fr > 0.0001:    # +0.01%
                s = -3.0
        else:  # short
            if fr > 0.0001:
                s = 4.0
            elif fr > 0.00005:
                s = 2.0
            elif fr < -0.0001:
                s = -3.0

        scores["funding"] = s

    def _score_orderbook(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["orderbook"] = 4.0
        s = 0.0
        imb = iv.orderbook_imbalance  # -1.0 a +1.0

        if direction == "long":
            if imb > 0.30:
                s = 4.0
            elif imb > 0.15:
                s = 2.0
        else:  # short
            if imb < -0.30:
                s = 4.0
            elif imb < -0.15:
                s = 2.0

        scores["orderbook"] = s

    def _score_candle_patterns(
        self, iv: IndicatorValues, direction: str, mode: str,
        scores: dict, max_scores: dict
    ) -> None:
        # Peso máximo varía por modo
        base_max = 8.0 if mode == "scalp" else 12.0
        max_scores["candles"] = base_max
        s = 0.0

        # Peso por timeframe (patrones más confiables en TF altos)
        tf_weight = 1.0 if iv.timeframe in ("5m", "15m") else 1.5

        if direction == "long":
            if iv.candle_pattern == "engulfing_bull":
                s = 6.0 * tf_weight
            elif iv.candle_pattern == "hammer":
                s = 5.0 * tf_weight
            elif iv.candle_pattern == "doji" and iv.pattern_at_support:
                s = 4.0 * tf_weight
        else:  # short
            if iv.candle_pattern == "engulfing_bear":
                s = 6.0 * tf_weight
            elif iv.candle_pattern == "shooting_star":
                s = 5.0 * tf_weight
            elif iv.candle_pattern == "doji" and iv.pattern_at_support:
                s = 4.0 * tf_weight

        # Se duplica si coincide con soporte/resistencia
        if iv.pattern_at_support and s > 0:
            s = min(s * 2, base_max)

        scores["candles"] = min(s, base_max)

    def _score_cci(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["cci"] = 4.0
        s = 0.0

        if direction == "long":
            if iv.cci_cross_up_100:   # salió de sobreventa
                s = 3.0
            elif iv.cci_extreme_bearish:  # CCI < -150
                s = 2.0
            if iv.cci_extreme_bullish:
                s -= 2.0
        else:  # short
            if iv.cci_cross_down_100:
                s = 3.0
            elif iv.cci_extreme_bullish:
                s = 2.0
            if iv.cci_extreme_bearish:
                s -= 2.0

        scores["cci"] = s

    def _score_stochastic(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["stochastic"] = 4.0
        s = 0.0

        if direction == "long":
            if iv.stoch_cross_bullish:
                s = 3.0
            elif iv.stoch_oversold:
                s = 2.0
        else:  # short
            if iv.stoch_cross_bearish:
                s = 3.0
            elif iv.stoch_overbought:
                s = 2.0

        scores["stochastic"] = s

    def _score_macd_divergence(
        self, iv: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        max_scores["macd_divergence"] = 6.0
        s = 0.0

        if direction == "long" and iv.macd_divergence_bullish:
            s = 6.0
        elif direction == "short" and iv.macd_divergence_bearish:
            s = 6.0

        scores["macd_divergence"] = s

    def _score_lateralization(
        self, iv: IndicatorValues, macro: MacroTrend,
        scores: dict, max_scores: dict
    ) -> None:
        """
        Penalización por lateralización probable.
        No suma puntos, solo resta si el score es alto
        Y no hay confluencia macro muy fuerte.
        """
        max_scores["lateralization"] = 0.0  # no suma al máximo
        lat = iv.lateralization_score

        if lat >= 0.50:
            # Penalizar, pero menos si la macro es muy fuerte
            macro_strength = macro.aligned_count / 3.0
            penalty = lat * 8.0 * (1.0 - macro_strength * 0.5)
            scores["lateralization"] = -penalty
        else:
            scores["lateralization"] = 0.0

    def _score_confirmation(
        self, iv_confirm: IndicatorValues, direction: str,
        scores: dict, max_scores: dict
    ) -> None:
        """
        Bonus/penalización basado en el timeframe de confirmación.
        Máximo 8 puntos bonus, máximo -4 penalización.
        """
        max_scores["confirmation"] = 8.0
        s = 0.0

        # RSI alineado en timeframe de confirmación
        if direction == "long":
            if iv_confirm.rsi < 50:
                s += 2.0
            if iv_confirm.emas_aligned_bullish >= 3:
                s += 2.0
            if iv_confirm.macd_histogram > 0:
                s += 2.0
            if iv_confirm.price_above_vwap:
                s += 2.0
            # Penalización si confirma va en contra
            if iv_confirm.emas_aligned_bearish >= 4:
                s -= 4.0
        else:
            if iv_confirm.rsi > 50:
                s += 2.0
            if iv_confirm.emas_aligned_bearish >= 3:
                s += 2.0
            if iv_confirm.macd_histogram < 0:
                s += 2.0
            if not iv_confirm.price_above_vwap:
                s += 2.0
            if iv_confirm.emas_aligned_bullish >= 4:
                s -= 4.0

        scores["confirmation"] = s

    # ─────────────────────────────────────────────────────────────────
    # DETERMINACIÓN DE NIVEL
    # ─────────────────────────────────────────────────────────────────

    def _determine_level(
        self,
        normalized: float,
        macro: MacroTrend,
        direction: str,
        mode: str,
        threshold_increment: float,
    ) -> int:
        """
        Determina el nivel de señal (1, 2, 3 o 0) según el puntaje
        normalizado y las condiciones del mercado.
        """
        expected = "bullish" if direction == "long" else "bearish"

        # Nivel 1: señal fuerte, siempre opera si supera umbral
        threshold_1 = THRESHOLDS[1] + threshold_increment
        if normalized >= threshold_1:
            return 1

        # Nivel 2: señal moderada + confirmación macro
        threshold_2 = THRESHOLDS[2] + threshold_increment
        macro_confirms = (macro.trend_1d == expected or macro.trend_1w == expected)
        if normalized >= threshold_2 and macro_confirms:
            return 2

        # Nivel 3: señal débil + macro muy alineada + solo ciertos modos
        threshold_3 = THRESHOLDS[3] + threshold_increment
        macro_strong = macro.aligned_count >= 2
        if normalized >= threshold_3 and macro_strong:
            return 3

        return 0


# ─────────────────────────────────────────────
# EVALUADOR MULTI-MODO
# ─────────────────────────────────────────────

class StrategyEvaluator:
    """
    Evalúa todos los modos y direcciones y devuelve
    la mejor oportunidad disponible.

    Uso:
        evaluator = StrategyEvaluator()
        best = evaluator.evaluate(indicators, macro, active_modes=['scalp','mediano','swing'])
        if best and best.should_trade:
            # ejecutar
    """

    def __init__(self):
        self.scorer = Scorer()

    def evaluate(
        self,
        indicators: dict[str, IndicatorValues],
        macro: MacroTrend,
        active_modes: list[str] = None,
        threshold_increment: float = 0.0,
    ) -> Optional[ScoreResult]:

        if active_modes is None:
            active_modes = ["scalp", "mediano", "swing"]

        candidates = []

        for mode in active_modes:
            for direction in ["long", "short"]:
                result = self.scorer.score(
                    indicators, macro, direction, mode, threshold_increment
                )
                if result.should_trade:
                    candidates.append(result)

        if not candidates:
            return None

        # Priorizar por: nivel primero, luego puntaje normalizado
        candidates.sort(key=lambda r: (r.signal_level == 1,
                                        r.normalized), reverse=True)
        return candidates[0]

    def evaluate_all(
        self,
        indicators: dict[str, IndicatorValues],
        macro: MacroTrend,
        active_modes: list[str] = None,
        threshold_increment: float = 0.0,
    ) -> list[ScoreResult]:
        """Devuelve todos los resultados, no solo el mejor."""
        if active_modes is None:
            active_modes = ["scalp", "mediano", "swing"]

        results = []
        for mode in active_modes:
            for direction in ["long", "short"]:
                result = self.scorer.score(
                    indicators, macro, direction, mode, threshold_increment
                )
                results.append(result)

        results.sort(key=lambda r: r.normalized, reverse=True)
        return results


# ─────────────────────────────────────────────
# TEST RÁPIDO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from market_feed import create_feed
    from indicators import IndicatorCalculator

    print("Probando scoring.py contra testnet...\n")
    feed = create_feed('live', symbol='BTCUSDC', testnet=True)
    feed.start()
    time.sleep(1)

    snap  = feed.get_snapshot()
    calc  = IndicatorCalculator()
    ivs   = calc.calculate(snap)
    macro = analyze_macro_trend(ivs)

    evaluator = StrategyEvaluator()

    print(f"Macro: 1W={macro.trend_1w} | 1D={macro.trend_1d} | "
          f"4H={macro.trend_4h} | Alineados={macro.aligned_count}/3\n")

    # Mostrar todos los resultados ordenados por puntaje
    all_results = evaluator.evaluate_all(ivs, macro)

    print("=" * 60)
    print("TODOS LOS MODOS Y DIRECCIONES:")
    print("=" * 60)
    for r in all_results:
        status = "✓ OPERA" if r.should_trade else "✗"
        block  = f" [{r.blocked_reasons[0]}]" if r.blocked_reasons else ""
        print(f"  {r.direction:<5} {r.mode:<8} {r.timeframe:<4} "
              f"| {r.normalized*100:>5.1f}% "
              f"| nivel={r.signal_level} "
              f"| {status}{block}")

    print()
    best = evaluator.evaluate(ivs, macro)
    if best:
        print("MEJOR OPORTUNIDAD:")
        print(best)
    else:
        print("Sin señales operables en este momento.")
        # Mostrar el más cercano al umbral
        if all_results:
            closest = all_results[0]
            needed = THRESHOLDS[3] * closest.maximum_possible
            print(f"\nMás cercano: {closest.direction} {closest.mode} "
                  f"con {closest.normalized*100:.1f}% "
                  f"(necesita ~{THRESHOLDS[3]*100:.0f}% para nivel 3)")

    feed.stop()
