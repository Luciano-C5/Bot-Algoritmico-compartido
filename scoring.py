"""
scoring.py  v1.1
================
Sistema de puntaje ponderado continuo.

Cambios respecto a v1.0:
  - THRESHOLDS, LEVERAGE_BY_LEVEL, TP_SL_BY_MODE, THRESHOLD_INCREMENTS,
    MODE_TIMEFRAMES ahora se leen desde config.cfg en vez de estar hardcodeados.
  - Integración con regime_detector: los pesos de cada indicador se ajustan
    dinámicamente según el régimen de mercado detectado.
  - Se agrega parámetro regime_weights opcional en score() para recibir
    los pesos desde el ciclo principal.

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

    direction: str        # "long" o "short"
    mode:      str        # "scalp", "mediano", "swing"
    timeframe: str        # timeframe principal del modo

    total:            float = 0.0
    maximum_possible: float = 0.0
    normalized:       float = 0.0   # total / maximum_possible (0.0 a 1.0)

    signal_level: int  = 0      # 0=no operar, 1=fuerte, 2=moderado, 3=débil
    should_trade: bool = False

    # Desglose por componente (para logs y debugging)
    breakdown: dict[str, float] = field(default_factory=dict)

    # Condiciones de bloqueo activas
    blocked_reasons: list[str] = field(default_factory=list)

    # Contexto macro
    macro_aligned:      bool  = False
    macro_divergence:   bool  = False   # 1D y 1W en direcciones opuestas
    leverage_multiplier: float = 1.0    # se reduce a 0.6 si hay divergencia macro

    # Régimen activo al momento del scoring
    regime: str = "volatile"

    # TP/SL aproximados ajustados al modo y nivel
    approx_tp_pct: float = 0.0   # % sobre capital
    approx_sl_pct: float = 0.0
    leverage:      int   = 1

    def __str__(self) -> str:
        lines = [
            f"[ScoreResult] {self.direction.upper()} | {self.mode} | "
            f"{self.timeframe} | régimen={self.regime}",
            f"  Puntaje: {self.total:.1f} / {self.maximum_possible:.1f} "
            f"({self.normalized*100:.1f}%)",
            f"  Nivel: {self.signal_level} | Operar: {self.should_trade}",
            f"  Leverage: x{self.leverage} (mult: {self.leverage_multiplier})",
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
# Estos valores ya NO están hardcodeados acá.
# Se leen en tiempo de ejecución desde cfg para que cualquier cambio
# hecho desde Flask o terminal se refleje sin reiniciar.

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
        direction:           str,             # "long" o "short"
        mode:                str,             # "scalp", "mediano", "swing"
        regime:              str   = "volatile",
        regime_confidence:   float = 0.5,
        threshold_increment: float = 0.0,     # incrementos por horario/noticias
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

        # Pesos dinámicos según régimen de mercado
        weights = get_scoring_weights(regime, regime_confidence)

        # ── Verificar condiciones de bloqueo ──────────────────
        if self._check_blocks(iv_main, direction, result):
            return result

        # ── Calcular puntaje por componente ───────────────────
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
        self._score_volatility_1m(iv_main, direction, scores, max_scores)
        self._score_cci(iv_main, direction, scores, max_scores, weights)
        self._score_stoch(iv_main, direction, scores, max_scores, weights)
        self._score_macd_divergence(iv_main, direction, scores, max_scores, weights)
        self._score_lateralization(iv_main, scores, max_scores, weights)

        if iv_confirm:
            self._score_confirmation_tf(iv_confirm, direction, mode, scores, max_scores)

        # ── Totales ───────────────────────────────────────────
        total_score = sum(scores.values())
        max_score   = sum(max_scores.values())
        if max_score == 0:
            max_score = cfg.threshold.max_score

        normalized = max(0.0, total_score / max_score) if max_score > 0 else 0.0

        result.total            = round(total_score, 2)
        result.maximum_possible = round(max_score, 2)
        result.normalized       = round(normalized, 4)
        result.breakdown        = {k: round(v, 2) for k, v in scores.items()}

        # ── Contexto macro ────────────────────────────────────
        result.macro_aligned    = macro.macro_aligned
        result.macro_divergence = macro.daily_vs_weekly_divergence

        if result.macro_divergence:
            result.leverage_multiplier = cfg.leverage.divergence_multiplier
            threshold_increment += _threshold_increments()["macro_divergence"]

        # ── Determinar nivel de señal ─────────────────────────
        thresholds     = _thresholds()
        leverage_map   = _leverage_by_level()
        tp_sl          = _tp_sl_by_mode()
        effective_norm = normalized

        result.signal_level = 0
        result.should_trade = False

        # N1: entra siempre si supera el umbral (incluso en pausa de modo)
        t1 = thresholds[1] + threshold_increment
        if effective_norm >= t1:
            result.signal_level = 1
            result.should_trade = True

        # N2: requiere macro alineada
        elif effective_norm >= thresholds[2] + threshold_increment:
            if macro.macro_aligned or macro.daily_aligned:
                result.signal_level = 2
                result.should_trade = True

        # N3: requiere macro muy fuerte (1W + 1D alineados = 15 pts macro)
        elif effective_norm >= thresholds[3] + threshold_increment:
            if macro.macro_aligned:  # los tres TFs alineados
                result.signal_level = 3
                result.should_trade = True

        # Con divergencia macro: solo scalps
        if result.macro_divergence and result.should_trade:
            if mode != "scalp":
                result.should_trade = False
                result.blocked_reasons.append(
                    "Divergencia macro: solo se permiten scalps"
                )

        # ── Apalancamiento y TP/SL ────────────────────────────
        if result.signal_level > 0:
            base_lev = leverage_map.get(result.signal_level, 1)
            result.leverage = max(
                1, int(base_lev * result.leverage_multiplier)
            )
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

    def _check_blocks(
        self,
        iv: IndicatorValues,
        direction: str,
        result: ScoreResult,
    ) -> bool:
        """
        Verifica condiciones que bloquean completamente la entrada.
        Devuelve True si hay bloqueo.

        Bloqueos implementados:
        1. RSI saturado + EMAs completamente alineadas → probable lateralización
        2. Score de lateralización > 0.65
        """
        blocked = False

        # Bloqueo 1: RSI saturado + 5/5 EMAs alineadas
        if direction == "long":
            rsi_saturated = iv.rsi > 70
            emas_aligned  = iv.emas_aligned_count == 5 and iv.ema_direction == "up"
        else:
            rsi_saturated = iv.rsi < 30
            emas_aligned  = iv.emas_aligned_count == 5 and iv.ema_direction == "down"

        if rsi_saturated and emas_aligned:
            result.blocked_reasons.append(
                f"RSI saturado ({iv.rsi:.1f}) + 5/5 EMAs alineadas → "
                "probable lateralización"
            )
            blocked = True

        # Bloqueo 2: Score de lateralización alto
        if iv.lateralization_score >= 0.65:
            result.blocked_reasons.append(
                f"Score lateralización alto: {iv.lateralization_score:.2f} ≥ 0.65"
            )
            blocked = True

        return blocked

    # ─────────────────────────────────────────
    # COMPONENTES DE SCORING
    # Cada método recibe scores y max_scores (dicts mutables)
    # y agrega sus puntos. Los pesos del dict weights
    # multiplican el puntaje base según el régimen.
    # ─────────────────────────────────────────

    def _score_rsi(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("rsi", 1.0)
        pts = 0.0
        if direction == "long":
            if   iv.rsi < 15:  pts = 12
            elif iv.rsi < 20:  pts = 10
            elif iv.rsi < 30:  pts = 8
            elif iv.rsi < 40:  pts = 4
            elif iv.rsi < 50:  pts = 1
            if iv.rsi > 70:    pts -= 4
            if iv.rsi_momentum_down:  pts += 3  # bajando desde >50 en 3 velas
        else:  # short
            if   iv.rsi > 85:  pts = 12
            elif iv.rsi > 80:  pts = 10
            elif iv.rsi > 70:  pts = 8
            elif iv.rsi > 60:  pts = 4
            elif iv.rsi > 50:  pts = 1
            if iv.rsi < 30:    pts -= 4
            if iv.rsi_momentum_up:    pts += 3
        scores["rsi"]     = pts * w
        max_scores["rsi"] = 15 * w   # 12 base + 3 momentum

    def _score_emas(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("emas", 1.0)
        pts = 0.0
        aligned = iv.emas_aligned_count
        direction_match = (
            (direction == "long"  and iv.ema_direction == "up") or
            (direction == "short" and iv.ema_direction == "down")
        )
        if direction_match:
            if   aligned == 5: pts += 10
            elif aligned == 4: pts += 6
            elif aligned == 3: pts += 3
        if iv.ema_separation_growing and direction_match:
            pts += 3
        # Compresión: solo suma si Squeeze está terminando o volumen creciendo
        if iv.ema_compression:
            if iv.squeeze_off or iv.volume_ratio > 1.2:
                pts += 4
        scores["emas"]     = pts * w
        max_scores["emas"] = 17 * w   # 10 + 3 separación + 4 compresión

    def _score_macd(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("macd", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.macd_cross_bullish:         pts += 8
            if iv.macd_histogram > 0:
                if iv.macd_histogram_growing:  pts += 5
                else:                          pts += 2
            if iv.macd_cross_bearish:         pts -= 5
        else:
            if iv.macd_cross_bearish:          pts += 8
            if iv.macd_histogram < 0:
                if iv.macd_histogram_growing:   pts += 5
                else:                           pts += 2
            if iv.macd_cross_bullish:           pts -= 5
        scores["macd"]     = pts * w
        max_scores["macd"] = 13 * w

    def _score_ut_bot(self, iv, direction, scores, max_scores, weights, macro):
        w   = weights.get("ut_bot", 1.0)
        pts = 0.0
        signal_matches = (
            (direction == "long"  and iv.ut_bot_signal == "buy")  or
            (direction == "short" and iv.ut_bot_signal == "sell")
        )
        if signal_matches:
            pts += 8
        if direction == "long"  and iv.price_near_ut_stop_above:  pts += 3
        if direction == "short" and iv.price_near_ut_stop_below:  pts += 3
        # Si va contra la tendencia macro, reducir al 50%
        macro_against = (
            (direction == "long"  and macro.weekly_trend == "down") or
            (direction == "short" and macro.weekly_trend == "up")
        )
        if macro_against:
            pts *= 0.5
        scores["ut_bot"]     = pts * w
        max_scores["ut_bot"] = 11 * w

    def _score_squeeze(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("squeeze", 1.0)
        pts = 0.0
        if iv.squeeze_off:                     pts += 8
        if iv.squeeze_histogram_positive and iv.squeeze_histogram_growing:
            pts += 5
        if iv.squeeze_color_change:            pts += 3
        if iv.squeeze_active:                  pts -= 3
        if direction == "short":
            pts = -pts if pts > 0 else pts
        scores["squeeze"]     = pts * w
        max_scores["squeeze"] = 16 * w

    def _score_ichimoku(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("ichimoku", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.price_above_cloud:    pts += 5
            if iv.tenkan_above_kijun:   pts += 4
            if iv.price_in_cloud:       pts -= 2
        else:
            if iv.price_below_cloud:    pts += 5
            if iv.tenkan_below_kijun:   pts += 4
            if iv.price_in_cloud:       pts -= 2
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
        pts = 0.0
        vr  = iv.volume_ratio   # volumen actual / promedio 20 velas
        if   vr > 2.0:  pts = 8
        elif vr > 1.5:  pts = 5
        elif vr > 1.2:  pts = 3
        else:           pts = -2
        scores["volume"]     = pts * w
        max_scores["volume"] = 8 * w

    def _score_bollinger(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("bollinger", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.price_at_lower_band:         pts = 6
            elif iv.price_in_lower_third:      pts = 3
            if iv.price_above_upper_band:      pts -= 4
        else:
            if iv.price_at_upper_band:         pts = 6
            elif iv.price_in_upper_third:      pts = 3
            if iv.price_below_lower_band:      pts -= 4
        scores["bollinger"]     = pts * w
        max_scores["bollinger"] = 6 * w

    def _score_macro(self, macro, direction, scores, max_scores):
        # Macro no tiene multiplicador de régimen — siempre con peso completo
        pts = 0.0
        aligned_up   = direction == "long"
        aligned_down = direction == "short"
        if macro.macro_aligned:
            trend_match = (
                (aligned_up   and macro.weekly_trend == "up") or
                (aligned_down and macro.weekly_trend == "down")
            )
            pts = 15 if trend_match else -8
        elif macro.weekly_daily_aligned:
            trend_match = (
                (aligned_up   and macro.weekly_trend == "up") or
                (aligned_down and macro.weekly_trend == "down")
            )
            pts = 10 if trend_match else -5
        elif macro.daily_aligned:
            trend_match = (
                (aligned_up   and macro.daily_trend == "up") or
                (aligned_down and macro.daily_trend == "down")
            )
            pts = 4 if trend_match else -3
        scores["macro"]     = pts
        max_scores["macro"] = 15

    def _score_pivots(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("pivots", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.price_at_s1:    pts = 5
            elif iv.price_at_s2:  pts = 4
            elif iv.price_between_pp_s1: pts = 2
        else:
            if iv.price_at_r1:    pts = 5
            elif iv.price_at_r2:  pts = 4
            elif iv.price_between_pp_r1: pts = 2
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
        pts = 0.0
        obi = iv.orderbook_imbalance   # positivo = más bids, negativo = más asks
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
        w  = weights.get("candle_patterns", 1.0)
        tf_weight = 1.5 if mode in ("mediano", "swing") else 1.0
        pts = 0.0
        multiplier = 2.0 if iv.pattern_at_support_resistance else 1.0
        if direction == "long":
            if iv.bullish_engulfing:   pts += 6 * tf_weight * multiplier
            elif iv.hammer:            pts += 5 * tf_weight * multiplier
            elif iv.doji_at_support:   pts += 4 * tf_weight * multiplier
        else:
            if iv.bearish_engulfing:   pts += 6 * tf_weight * multiplier
            elif iv.shooting_star:     pts += 5 * tf_weight * multiplier
            elif iv.doji_at_resistance: pts += 4 * tf_weight * multiplier
        max_base = 12 if mode in ("mediano", "swing") else 8
        scores["candle_patterns"]     = pts * w
        max_scores["candle_patterns"] = max_base * w

    def _score_volatility_1m(self, iv, direction, scores, max_scores):
        pts = min(6.0, iv.bounce_count_1m * 2.0) if iv.bounce_count_1m > 0 else 0.0
        scores["volatility_1m"]     = pts
        max_scores["volatility_1m"] = 6

    def _score_cci(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("cci", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.cci_cross_above_minus100:    pts += 3
            elif iv.cci < -150:                pts += 2
            if iv.cci > 100:                   pts -= 2
        else:
            if iv.cci_cross_below_plus100:     pts += 3
            elif iv.cci > 150:                 pts += 2
            if iv.cci < -100:                  pts -= 2
        scores["cci"]     = pts * w
        max_scores["cci"] = 4 * w

    def _score_stoch(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("stoch", 1.0)
        pts = 0.0
        if direction == "long":
            if iv.stoch_k_cross_up_in_oversold:    pts = 3
            elif iv.stoch_k < 20:                  pts = 2
        else:
            if iv.stoch_k_cross_down_in_overbought: pts = 3
            elif iv.stoch_k > 80:                   pts = 2
        scores["stoch"]     = pts * w
        max_scores["stoch"] = 4 * w

    def _score_macd_divergence(self, iv, direction, scores, max_scores, weights):
        w   = weights.get("macd_divergence", 1.0)
        pts = 0.0
        if direction == "long"  and iv.macd_bullish_divergence:   pts = 6
        if direction == "short" and iv.macd_bearish_divergence:   pts = 6
        if direction == "long"  and iv.macd_bearish_divergence:   pts = -6
        if direction == "short" and iv.macd_bullish_divergence:   pts = -6
        scores["macd_divergence"]     = pts * w
        max_scores["macd_divergence"] = 6 * w

    def _score_lateralization(self, iv, scores, max_scores, weights):
        w   = weights.get("lateralization", 1.0)
        # El score de lateralización ya bloquea si >= 0.65
        # Acá penalizamos levemente si está en zona de advertencia (0.40-0.65)
        pts = 0.0
        ls  = iv.lateralization_score
        if ls >= 0.50:   pts = -4
        elif ls >= 0.40: pts = -2
        scores["lateralization"]     = pts * w
        max_scores["lateralization"] = 0   # no suma, solo puede restar


    def _score_confirmation_tf(self, iv_confirm, direction, mode, scores, max_scores):
        """Bonus por confirmación en el timeframe secundario."""
        pts = 0.0
        matches = (
            (direction == "long"  and iv_confirm.trend_direction == "up")   or
            (direction == "short" and iv_confirm.trend_direction == "down")
        )
        conflicts = (
            (direction == "long"  and iv_confirm.trend_direction == "down") or
            (direction == "short" and iv_confirm.trend_direction == "up")
        )
        if matches:    pts = 8
        if conflicts:  pts = -5
        scores["confirm_tf"]     = pts
        max_scores["confirm_tf"] = 8


# ─────────────────────────────────────────────
# EVALUADOR DE LOS TRES MODOS
# ─────────────────────────────────────────────

class StrategyEvaluator:
    """
    Evalúa los tres modos simultáneamente y devuelve el mejor resultado.

    Uso:
        evaluator = StrategyEvaluator()
        best = evaluator.evaluate(
            indicators=indicators,
            macro=macro,
            regime=regime_result,
            active_modes={"scalp": True, "mediano": True, "swing": True},
            threshold_increment=0.0,
        )
    """

    def __init__(self):
        self._scorer = Scorer()

    def evaluate(
        self,
        indicators:          dict[str, IndicatorValues],
        macro:               MacroTrend,
        regime_regime:       str   = "volatile",
        regime_confidence:   float = 0.5,
        preferred_mode:      Optional[str] = None,
        active_modes:        Optional[dict[str, bool]] = None,
        threshold_increment: float = 0.0,
    ) -> list[ScoreResult]:
        """
        Evalúa todos los modos activos para long y short.

        Devuelve lista de ScoreResult ordenada por normalized descendente.
        Solo incluye resultados con should_trade=True.

        Si preferred_mode está definido (viene del regime_detector),
        ese modo tiene prioridad y los demás se evalúan solo si no hay
        señal en el preferido.
        """
        if active_modes is None:
            active_modes = {"scalp": True, "mediano": True, "swing": True}

        results = []
        modes_to_eval = [m for m, active in active_modes.items() if active]

        # Reordenar: preferred_mode primero
        if preferred_mode and preferred_mode in modes_to_eval:
            modes_to_eval = [preferred_mode] + [
                m for m in modes_to_eval if m != preferred_mode
            ]

        for mode in modes_to_eval:
            for direction in ("long", "short"):
                r = self._scorer.score(
                    indicators          = indicators,
                    macro               = macro,
                    direction           = direction,
                    mode                = mode,
                    regime              = regime_regime,
                    regime_confidence   = regime_confidence,
                    threshold_increment = threshold_increment,
                )
                if r.should_trade:
                    results.append(r)

        results.sort(key=lambda x: x.normalized, reverse=True)
        return results


if __name__ == "__main__":
    print("scoring.py v1.1 — importado correctamente.")
    print(f"  Umbrales desde cfg: N1={cfg.threshold.level_1} N2={cfg.threshold.level_2} N3={cfg.threshold.level_3}")
    print(f"  Leverage desde cfg: N1=x{cfg.leverage.level_1} N2=x{cfg.leverage.level_2} N3=x{cfg.leverage.level_3}")
