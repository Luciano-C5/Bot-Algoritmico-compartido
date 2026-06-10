"""
scoring.py  v1.2
================
Sistema de puntaje ponderado continuo.

Cambios respecto a v1.1:
  - Resueltos TODOS los mismatches de nombres entre scoring.py e indicators.py:
      * iv.ema_direction, iv.emas_aligned_count  → ya definidos en IndicatorValues v1.2
      * iv.rsi_momentum_down / iv.rsi_momentum_up → aliases correctos
      * iv.squeeze_off, iv.squeeze_active, iv.squeeze_histogram_positive,
        iv.squeeze_histogram_growing → aliases correctos
      * iv.price_near_ut_stop_above / iv.price_near_ut_stop_below → nuevos campos
      * iv.tenkan_above_kijun / iv.tenkan_below_kijun → nuevos campos
      * iv.vwap_cross_up_this_candle / iv.vwap_cross_down_this_candle → nuevos campos
      * iv.price_below_vwap → nuevo campo
      * macro.macro_aligned, macro.daily_aligned, macro.weekly_trend,
        macro.daily_vs_weekly_divergence → campos nuevos en MacroTrend
  - Reemplazados campos inexistentes en _score_candle_patterns,
    _score_volatility_1m, _score_cci, _score_stoch, _score_macd_divergence,
    _score_pivots, _score_macro con los nombres reales de IndicatorValues
  - _score_confirmation_tf completado

Regla fundamental: este archivo no sabe nada de órdenes ni de
ejecución. Solo puntúa. Sin estado, sin efectos secundarios.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from indicators import IndicatorValues, MacroTrend, analyze_macro_trend
from config import cfg
from regime_detector import get_scoring_weights


# ─────────────────────────────────────────────
# ESTRUCTURA DE SALIDA
# ─────────────────────────────────────────────

@dataclass
class ScoreResult:
    """
    Resultado del scoring para una dirección y modo dados.
    strategy.py recibe esto y toma la decisión de operar o no.
    """
    direction: str
    mode:      str
    timeframe: str

    total:            float = 0.0
    maximum_possible: float = 0.0
    normalized:       float = 0.0   # total / maximum_possible (0.0 a 1.0)

    signal_level: int  = 0
    should_trade: bool = False

    breakdown:       dict[str, float] = field(default_factory=dict)
    blocked_reasons: list[str]        = field(default_factory=list)

    macro_aligned:       bool  = False
    macro_divergence:    bool  = False
    leverage_multiplier: float = 1.0

    regime:   str = "volatile"

    approx_tp_pct: float = 0.0
    approx_sl_pct: float = 0.0
    leverage:      int   = 1

    def __str__(self) -> str:
        lines = [
            f"[ScoreResult] {self.direction.upper()} | {self.mode} | "
            f"{self.timeframe} | régimen={self.regime}",
            f"  Puntaje: {self.total:.1f} / {self.maximum_possible:.1f} "
            f"({self.normalized*100:.1f}%)",
            f"  Nivel: {self.signal_level} | Operar: {self.should_trade}",
            f"  Leverage: x{self.leverage} (mult: {self.leverage_multiplier:.2f})",
        ]
        if self.blocked_reasons:
            lines.append(f"  BLOQUEADO: {', '.join(self.blocked_reasons)}")
        if self.breakdown:
            lines.append("  Desglose:")
            for k, v in sorted(self.breakdown.items(), key=lambda x: -abs(x[1])):
                bar  = "█" * int(abs(v) / 2) if v > 0 else "▒" * int(abs(v) / 2)
                sign = "+" if v >= 0 else ""
                lines.append(f"    {k:<28} {sign}{v:>5.1f}  {bar}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# CONFIGURACIÓN — leída desde config.cfg
# ─────────────────────────────────────────────

def _thresholds() -> dict[int, float]:
    return {
        1: cfg.threshold.level_1,
        2: cfg.threshold.level_2,
        3: cfg.threshold.level_3,
    }

def _leverage_by_level() -> dict[int, int]:
    return {
        1: cfg.leverage.level_1,
        2: cfg.leverage.level_2,
        3: cfg.leverage.level_3,
    }

def _tp_sl_by_mode() -> dict[str, dict[str, float]]:
    return {
        "scalp":   {"tp": cfg.modes.scalp.tp_base_pct,   "sl": cfg.modes.scalp.sl_base_pct},
        "mediano": {"tp": cfg.modes.mediano.tp_base_pct, "sl": cfg.modes.mediano.sl_base_pct},
        "swing":   {"tp": cfg.modes.swing.tp_base_pct,   "sl": cfg.modes.swing.sl_base_pct},
    }

def _threshold_increments() -> dict[str, float]:
    return {
        "macro_divergence":  cfg.threshold.increment_macro_divergence,
        "low_volume_hour":   cfg.threshold.increment_low_volume_hour,
        "wall_street_open":  cfg.threshold.increment_wall_street_open,
        "week_close":        cfg.threshold.increment_week_close,
        "day_close":         cfg.threshold.increment_day_close,
        "news_moderate":     cfg.threshold.increment_news_moderate,
    }

def _mode_timeframes() -> dict[str, str]:
    return {
        "scalp":   cfg.modes.scalp.timeframe_main,
        "mediano": cfg.modes.mediano.timeframe_main,
        "swing":   cfg.modes.swing.timeframe_main,
    }

def _confirm_timeframes() -> dict[str, str]:
    return {
        "scalp":   cfg.modes.scalp.timeframe_confirm,
        "mediano": cfg.modes.mediano.timeframe_confirm,
        "swing":   cfg.modes.swing.timeframe_confirm,
    }


# ─────────────────────────────────────────────
# SCORER PRINCIPAL
# ─────────────────────────────────────────────

class Scorer:
    """
    Calcula el puntaje para una dirección y modo dados.

    Uso:
        scorer = Scorer()
        result = scorer.score(
            indicators=indicators,
            macro=macro,
            direction="long",
            mode="scalp",
            regime="bull_trend",
            regime_confidence=0.75,
            threshold_increment=0.0,
        )
    """

    def score(
        self,
        indicators:          dict[str, IndicatorValues],
        macro:               MacroTrend,
        direction:           str,
        mode:                str,
        regime:              str   = "volatile",
        regime_confidence:   float = 0.5,
        threshold_increment: float = 0.0,
    ) -> ScoreResult:

        tf_main    = _mode_timeframes()[mode]
        tf_confirm = _confirm_timeframes()[mode]
        iv_main    = indicators.get(tf_main)
        iv_confirm = indicators.get(tf_confirm)

        result = ScoreResult(
            direction = direction,
            mode      = mode,
            timeframe = tf_main,
            regime    = regime,
        )

        if iv_main is None:
            result.blocked_reasons.append(f"Sin datos para {tf_main}")
            return result

        # Pesos dinámicos según régimen
        weights = get_scoring_weights(regime, regime_confidence)

        # Condiciones de bloqueo
        if self._check_blocks(iv_main, direction, result):
            return result

        # Puntaje por componente
        scores     = {}
        max_scores = {}

        self._score_rsi(iv_main, direction, scores, max_scores, weights)
        self._score_emas(iv_main, direction, scores, max_scores, weights)
        self._score_macd(iv_main, direction, scores, max_scores, weights)
        self._score_ut_bot(iv_main, direction, scores, max_scores, weights, macro)
        self._score_squeeze(iv_main, direction, scores, max_scores, weights)
        self._score_ichimoku(iv_main, direction, scores, max_scores, weights)
        self._score_vwap(iv_main, direction, scores, max_scores, weights)
        self._score_volume(iv_main, direction, scores, max_scores, weights)
        self._score_bollinger(iv_main, direction, scores, max_scores, weights)
        self._score_macro(macro, direction, scores, max_scores)
        self._score_pivots(iv_main, direction, scores, max_scores, weights)
        self._score_funding(iv_main, direction, scores, max_scores)
        self._score_obi(iv_main, direction, scores, max_scores)
        self._score_candle_patterns(iv_main, direction, mode, scores, max_scores, weights)
        self._score_cci(iv_main, direction, scores, max_scores, weights)
        self._score_stoch(iv_main, direction, scores, max_scores, weights)
        self._score_macd_divergence(iv_main, direction, scores, max_scores, weights)
        self._score_lateralization(iv_main, scores, max_scores, weights)
        self._score_microstructure(iv_main, scores, max_scores)

        if iv_confirm:
            self._score_confirmation_tf(iv_confirm, direction, mode, scores, max_scores)

        # Totales
        total_score = sum(scores.values())
        max_score   = sum(v for v in max_scores.values() if v > 0)
        if max_score == 0:
            max_score = cfg.threshold.max_score

        normalized = max(0.0, total_score / max_score) if max_score > 0 else 0.0

        result.total            = round(total_score, 2)
        result.maximum_possible = round(max_score, 2)
        result.normalized       = round(normalized, 4)
        result.breakdown        = {k: round(v, 2) for k, v in scores.items()}

        # Contexto macro
        result.macro_aligned    = macro.macro_aligned
        result.macro_divergence = macro.daily_vs_weekly_divergence

        if result.macro_divergence:
            result.leverage_multiplier = cfg.leverage.divergence_multiplier
            threshold_increment += _threshold_increments()["macro_divergence"]

        # Determinar nivel de señal
        thresholds   = _thresholds()
        leverage_map = _leverage_by_level()
        tp_sl        = _tp_sl_by_mode()

        result.signal_level = 0
        result.should_trade = False

        t1 = thresholds[1] + threshold_increment
        if normalized >= t1:
            result.signal_level = 1
            result.should_trade = True
        elif normalized >= thresholds[2] + threshold_increment:
            if macro.macro_aligned or macro.daily_aligned:
                result.signal_level = 2
                result.should_trade = True
        elif normalized >= thresholds[3] + threshold_increment:
            if macro.macro_aligned:
                result.signal_level = 3
                result.should_trade = True

        # Con divergencia macro: solo scalps
        if result.macro_divergence and result.should_trade and mode != "scalp":
            result.should_trade = False
            result.blocked_reasons.append("Divergencia macro: solo se permiten scalps")

        # Apalancamiento y TP/SL
        if result.signal_level > 0:
            base_lev = leverage_map.get(result.signal_level, 1)
            result.leverage = max(1, int(base_lev * result.leverage_multiplier))
            mode_params = tp_sl.get(mode, {"tp": 0.006, "sl": 0.004})
            result.approx_tp_pct = (
                cfg.modes.level_3_tp_pct
                if result.signal_level == 3
                else mode_params["tp"]
            )
            result.approx_sl_pct = mode_params["sl"]

        return result

    # ─────────────────────────────────────────
    # CONDICIONES DE BLOQUEO
    # ─────────────────────────────────────────

    def _check_blocks(self, iv: IndicatorValues, direction: str,
                      result: ScoreResult) -> bool:
        blocked = False

        if direction == "long":
            rsi_saturated = iv.rsi > 70
            emas_maxed    = iv.emas_aligned_count == 5 and iv.ema_direction == "up"
        else:
            rsi_saturated = iv.rsi < 30
            emas_maxed    = iv.emas_aligned_count == 5 and iv.ema_direction == "down"

        if rsi_saturated and emas_maxed:
            result.blocked_reasons.append(
                f"RSI saturado ({iv.rsi:.1f}) + 5/5 EMAs alineadas"
            )
            blocked = True

        if iv.lateralization_score >= 0.65:
            result.blocked_reasons.append(
                f"Lateralización alta: {iv.lateralization_score:.2f} ≥ 0.65"
            )
            blocked = True

        return blocked

    # ─────────────────────────────────────────
    # COMPONENTES DE SCORING
    # ─────────────────────────────────────────

    def _score_rsi(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("rsi", 1.0)
        pts = 0.0
        if direction == "long":
            if   iv.rsi < 15: pts = 12
            elif iv.rsi < 20: pts = 10
            elif iv.rsi < 30: pts = 8
            elif iv.rsi < 40: pts = 4
            elif iv.rsi < 50: pts = 1
            if iv.rsi > 70:   pts -= 4
            if iv.rsi_momentum_down: pts += 3
        else:
            if   iv.rsi > 85: pts = 12
            elif iv.rsi > 80: pts = 10
            elif iv.rsi > 70: pts = 8
            elif iv.rsi > 60: pts = 4
            elif iv.rsi > 50: pts = 1
            if iv.rsi < 30:   pts -= 4
            if iv.rsi_momentum_up:   pts += 3
        scores["rsi"]     = pts * w
        max_scores["rsi"] = 15 * w

    def _score_emas(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("emas", 1.0)
        pts = 0.0
        aligned = iv.emas_aligned_count
        dir_match = (
            (direction == "long"  and iv.ema_direction == "up") or
            (direction == "short" and iv.ema_direction == "down")
        )
        if dir_match:
            if   aligned == 5: pts += 10
            elif aligned == 4: pts += 6
            elif aligned == 3: pts += 3
        if iv.ema_separation_growing and dir_match:
            pts += 3
        if iv.ema_compression:
            if iv.squeeze_off or iv.volume_ratio > 1.2:
                pts += 4
        scores["emas"]     = pts * w
        max_scores["emas"] = 17 * w

    def _score_macd(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("macd", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.macd_cross_bullish:        pts += 8
            if iv.macd_histogram > 0:
                pts += 5 if iv.macd_histogram_growing else 2
            if iv.macd_cross_bearish:        pts -= 5
        else:
            if iv.macd_cross_bearish:        pts += 8
            if iv.macd_histogram < 0:
                pts += 5 if iv.macd_histogram_growing else 2
            if iv.macd_cross_bullish:        pts -= 5
        scores["macd"]     = pts * w
        max_scores["macd"] = 13 * w

    def _score_ut_bot(self, iv, direction, scores, max_scores, weights, macro):
        w   = weights.get("ut_bot", 1.0)
        pts = 0.0
        signal_ok = (
            (direction == "long"  and iv.ut_bot_signal == "buy") or
            (direction == "short" and iv.ut_bot_signal == "sell")
        )
        if signal_ok: pts += 8
        if direction == "long"  and iv.price_near_ut_stop_above: pts += 3
        if direction == "short" and iv.price_near_ut_stop_below: pts += 3
        macro_against = (
            (direction == "long"  and macro.weekly_trend == "down") or
            (direction == "short" and macro.weekly_trend == "up")
        )
        if macro_against: pts *= 0.5
        scores["ut_bot"]     = pts * w
        max_scores["ut_bot"] = 11 * w

    def _score_squeeze(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("squeeze", 1.0)
        pts = 0.0
        if iv.squeeze_off:                                        pts += 8
        if iv.squeeze_histogram_positive and iv.squeeze_histogram_growing: pts += 5
        if iv.squeeze_color_change:                               pts += 3
        if iv.squeeze_active:                                     pts -= 3
        # Squeeze es neutral respecto a dirección, pero si histograma es negativo
        # en long o positivo en short, penalizamos
        if direction == "long"  and iv.sqz_histogram < 0:        pts -= 2
        if direction == "short" and iv.sqz_histogram > 0:        pts -= 2
        scores["squeeze"]     = pts * w
        max_scores["squeeze"] = 16 * w

    def _score_ichimoku(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("ichimoku", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.price_above_cloud:  pts += 5
            if iv.tenkan_above_kijun: pts += 4
            if iv.price_in_cloud:     pts -= 2
        else:
            if iv.price_below_cloud:  pts += 5
            if iv.tenkan_below_kijun: pts += 4
            if iv.price_in_cloud:     pts -= 2
        scores["ichimoku"]     = pts * w
        max_scores["ichimoku"] = 9 * w

    def _score_vwap(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("vwap", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.vwap_cross_up_this_candle:   pts = 5
            elif iv.price_above_vwap:          pts = 3
        else:
            if iv.vwap_cross_down_this_candle:  pts = 5
            elif iv.price_below_vwap:           pts = 3
        scores["vwap"]     = pts * w
        max_scores["vwap"] = 5 * w

    def _score_volume(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("volume", 1.0)
        vr  = iv.volume_ratio
        if   vr > 2.0: pts = 8
        elif vr > 1.5: pts = 5
        elif vr > 1.2: pts = 3
        else:          pts = -2
        scores["volume"]     = pts * w
        max_scores["volume"] = 8 * w

    def _score_bollinger(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("bollinger", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.price_at_lower_band:      pts = 6
            elif iv.price_in_lower_third:   pts = 3
            if iv.price_above_upper_band:   pts -= 4
        else:
            if iv.price_at_upper_band:      pts = 6
            elif iv.price_in_upper_third:   pts = 3
            if iv.price_below_lower_band:   pts -= 4
        scores["bollinger"]     = pts * w
        max_scores["bollinger"] = 6 * w

    def _score_macro(self, macro, direction, scores, max_scores):
        """
        Puntaje macro — sin multiplicador de régimen, siempre con peso completo.
        Usa los campos definidos en MacroTrend v1.2.
        """
        pts = 0.0
        up_match   = direction == "long"  and macro.weekly_trend == "up"
        down_match = direction == "short" and macro.weekly_trend == "down"
        trend_ok   = up_match or down_match

        if macro.macro_aligned:
            pts = 15 if trend_ok else -8
        elif macro.daily_aligned:
            # 1D + 4H alineados pero no el 1W
            daily_up   = direction == "long"  and macro.trend_4h == "bullish"
            daily_down = direction == "short" and macro.trend_4h == "bearish"
            pts = 6 if (daily_up or daily_down) else -3
        else:
            # Solo 4H alineado
            h4_up   = direction == "long"  and macro.trend_4h == "bullish"
            h4_down = direction == "short" and macro.trend_4h == "bearish"
            pts = 3 if (h4_up or h4_down) else 0

        scores["macro"]     = pts
        max_scores["macro"] = 15

    def _score_pivots(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("pivots", 1.0)
        pts = 0.0
        pz  = iv.pivot_zone
        if direction == "long":
            if pz == "near_s1":         pts = 5
            elif pz == "near_s2":       pts = 4
            elif pz == "between_s1_p":  pts = 2
            elif pz == "at_pivot":      pts = 1
        else:
            if pz == "near_r1":         pts = 5
            elif pz == "above_r1":      pts = 4
            elif pz == "between_p_r1":  pts = 2
            elif pz == "at_pivot":      pts = 1
        scores["pivots"]     = pts * w
        max_scores["pivots"] = 5 * w

    def _score_funding(self, iv, direction, scores, max_scores):
        pts = 0.0
        fr  = iv.funding_rate
        if direction == "long":
            if   fr < -0.01:  pts = 4
            elif fr < -0.005: pts = 2
            elif fr > 0.01:   pts = -3
        else:
            if   fr > 0.01:   pts = 4
            elif fr > 0.005:  pts = 2
            elif fr < -0.01:  pts = -3
        scores["funding"]     = pts
        max_scores["funding"] = 4

    def _score_obi(self, iv, direction, scores, max_scores):
        """Orderbook Imbalance: +1.0 = solo bids, -1.0 = solo asks."""
        pts = 0.0
        obi = iv.orderbook_imbalance
        if direction == "long":
            if   obi > 0.30:  pts = 4
            elif obi > 0.15:  pts = 2
            elif obi < -0.30: pts = -2
        else:
            if   obi < -0.30: pts = 4
            elif obi < -0.15: pts = 2
            elif obi > 0.30:  pts = -2
        scores["obi"]     = pts
        max_scores["obi"] = 4

    def _score_candle_patterns(self, iv, direction, mode, scores, max_scores, weights):
        w          = weights.get("candle_patterns", 1.0)
        tf_weight  = 1.5 if mode in ("mediano", "swing") else 1.0
        pts        = 0.0
        at_level   = 2.0 if iv.pattern_at_support else 1.0  # patrón en soporte/resistencia

        if direction == "long":
            if iv.candle_pattern == "engulfing_bull":
                pts = 6 * tf_weight * at_level
            elif iv.candle_pattern == "hammer":
                pts = 5 * tf_weight * at_level
            elif iv.candle_pattern == "doji" and iv.pattern_at_support:
                pts = 4 * tf_weight
        else:
            if iv.candle_pattern == "engulfing_bear":
                pts = 6 * tf_weight * at_level
            elif iv.candle_pattern == "shooting_star":
                pts = 5 * tf_weight * at_level
            elif iv.candle_pattern == "doji" and iv.pattern_at_support:
                pts = 4 * tf_weight

        max_base = 12 if mode in ("mediano", "swing") else 8
        scores["candle_patterns"]     = pts * w
        max_scores["candle_patterns"] = max_base * w

    def _score_cci(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("cci", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.cci_cross_up_100:      pts += 3   # salió de sobreventa (< -100)
            elif iv.cci < -150:          pts += 2
            if iv.cci > 100:             pts -= 2
        else:
            if iv.cci_cross_down_100:    pts += 3   # salió de sobrecompra (> +100)
            elif iv.cci > 150:           pts += 2
            if iv.cci < -100:            pts -= 2
        scores["cci"]     = pts * w
        max_scores["cci"] = 4 * w

    def _score_stoch(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("stoch", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.stoch_cross_bullish:  pts = 3   # K cruza D hacia arriba en zona < 20
            elif iv.stoch_oversold:     pts = 2
        else:
            if iv.stoch_cross_bearish:  pts = 3   # K cruza D hacia abajo en zona > 80
            elif iv.stoch_overbought:   pts = 2
        scores["stoch"]     = pts * w
        max_scores["stoch"] = 4 * w

    def _score_macd_divergence(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("macd_divergence", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.macd_divergence_bullish:  pts =  6
            if iv.macd_divergence_bearish:  pts = -6
        else:
            if iv.macd_divergence_bearish:  pts =  6
            if iv.macd_divergence_bullish:  pts = -6
        scores["macd_divergence"]     = pts * w
        max_scores["macd_divergence"] = 6 * w

    def _score_lateralization(self, iv, scores, max_scores, weights):
        w   = weights.get("lateralization", 1.0)
        pts = 0.0
        ls  = iv.lateralization_score
        if   ls >= 0.50: pts = -4
        elif ls >= 0.40: pts = -2
        scores["lateralization"]     = pts * w
        max_scores["lateralization"] = 0  # solo resta, nunca suma

    def _score_microstructure(self, iv, scores, max_scores):
        """
        Market microstructure: spread y latencia del feed.
        Penalizaciones leves — no debe bloquear por sí solo.
        """
        pts = 0.0
        # Spread alto = ejecución cara
        if iv.spread_pct > 0.05:    pts -= 3
        elif iv.spread_pct > 0.03:  pts -= 1
        # Latencia alta del feed = datos potencialmente viejos
        if iv.feed_latency_ms > 500: pts -= 2
        scores["microstructure"]     = pts
        max_scores["microstructure"] = 0  # solo resta

    def _score_confirmation_tf(self, iv_confirm, direction, mode, scores, max_scores):
        """
        Bonus por confirmación en el timeframe secundario.
        Mira la alineación de EMAs y posición respecto a VWAP.
        """
        pts = 0.0
        dir_confirmed = (
            (direction == "long"  and iv_confirm.ema_direction == "up") or
            (direction == "short" and iv_confirm.ema_direction == "down")
        )
        if dir_confirmed:
            pts += 5
            if direction == "long"  and iv_confirm.price_above_vwap: pts += 2
            if direction == "short" and iv_confirm.price_below_vwap: pts += 2

        bonus_max = 8 if mode == "swing" else 6
        scores["confirmation"]     = pts
        max_scores["confirmation"] = bonus_max


# ─────────────────────────────────────────────
# EVALUADOR DE ESTRATEGIA
# ─────────────────────────────────────────────

class StrategyEvaluator:
    """
    Wrapper de alto nivel que evalúa todos los modos y direcciones.
    live_monitor.py y strategy.py usan esta clase.

    Uso:
        evaluator = StrategyEvaluator()
        # Mejor señal operable:
        best = evaluator.evaluate(ivs, macro, active_modes)
        # Todos los resultados:
        all_results = evaluator.evaluate_all(ivs, macro, active_modes)
    """

    def __init__(self):
        self._scorer = Scorer()

    def evaluate_all(
        self,
        indicators:   dict[str, IndicatorValues],
        macro:        MacroTrend,
        active_modes: list[str],
        regime:       str   = "volatile",
        regime_confidence: float = 0.5,
        threshold_increment: float = 0.0,
    ) -> list[ScoreResult]:
        """Devuelve resultados para todas las combinaciones modo × dirección."""
        results = []
        for mode in active_modes:
            for direction in ("long", "short"):
                r = self._scorer.score(
                    indicators          = indicators,
                    macro               = macro,
                    direction           = direction,
                    mode                = mode,
                    regime              = regime,
                    regime_confidence   = regime_confidence,
                    threshold_increment = threshold_increment,
                )
                results.append(r)
        return results

    def evaluate(
        self,
        indicators:   dict[str, IndicatorValues],
        macro:        MacroTrend,
        active_modes: list[str],
        regime:       str   = "volatile",
        regime_confidence: float = 0.5,
        threshold_increment: float = 0.0,
    ) -> Optional[ScoreResult]:
        """
        Devuelve la mejor señal operable (mayor puntaje normalizado entre
        las que tienen should_trade=True), o None si no hay ninguna.
        """
        all_r = self.evaluate_all(
            indicators, macro, active_modes,
            regime, regime_confidence, threshold_increment
        )
        tradeable = [r for r in all_r if r.should_trade]
        if not tradeable:
            return None
        return max(tradeable, key=lambda r: r.normalized)
