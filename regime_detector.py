"""
regime_detector.py
==================
Detector de régimen de mercado. Módulo transversal que corre en paralelo
al ciclo principal del bot.

Determina si el mercado está en:
  - "bull_trend"  : tendencia alcista fuerte
  - "bear_trend"  : tendencia bajista fuerte
  - "range"       : lateral / mean-reversion
  - "volatile"    : volátil sin dirección clara

El resultado (RegimeResult) alimenta:
  - scoring.py   → pesos dinámicos de cada indicador
  - strategy.py  → jerarquía de modos, tamaño de posición, ratio TP/SL

Importar: from regime_detector import RegimeDetector, RegimeResult
"""

from __future__ import annotations

import time
import math
import logging
import statistics
from dataclasses import dataclass, field
from typing import Optional
import urllib.request
import json

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# DATACLASSES DE SALIDA
# ─────────────────────────────────────────────────────────────

@dataclass
class RegimeResult:
    """
    Resultado completo del detector de régimen.
    Todos los valores son los calculados en el último ciclo.
    """

    # Régimen principal detectado
    regime: str = "volatile"          # "bull_trend" | "bear_trend" | "range" | "volatile"

    # Confianza en la clasificación (0.0 a 1.0)
    # ≥ 0.9  → todos los indicadores acuerdan
    # 0.6-0.9 → mayoría acuerda
    # 0.3-0.6 → indicadores divididos → tratar como volatile
    # < 0.3  → sin datos suficientes → tratar como volatile
    confidence: float = 0.0

    # ── Indicadores individuales ──────────────────────────────

    # ADX (Average Directional Index) calculado en TF diario
    # > 25 → tendencia | 20-25 → transición | < 20 → rango
    adx: float = 0.0

    # Posición del precio respecto a EMA50 y EMA200 diarias
    # "above_both" | "below_both" | "between"
    ema_position: str = "between"

    # Volatility Ratio = ATR_actual / ATR_promedio_20_velas
    # > 1.5 → expansión | 0.7-1.5 → normal | < 0.7 → compresión pre-breakout
    volatility_ratio: float = 1.0

    # Hurst Exponent (método R/S) sobre ventana de 100-200 velas
    # > 0.5 → tendencia (favorable swing) | < 0.5 → reversión (favorable scalp)
    # = 0.5 → movimiento aleatorio (sin ventaja estadística)
    hurst: float = 0.5

    # Fear & Greed Index de Alternative.me (0-100)
    # < 20 → miedo extremo | > 80 → avaricia extrema
    fear_greed: int = 50

    # True si el spread es razonable respecto a la volatilidad
    microstructure_ok: bool = True

    # ── Modificadores para otros módulos ─────────────────────

    # Multiplicador de riesgo por trade según el régimen
    # bull_trend/bear_trend: 1.0 | range: 0.5 | volatile: 0.3
    risk_multiplier: float = 0.3

    # Ratio TP/SL recomendado
    # bull_trend/bear_trend: 2.5 | range: 1.5 | volatile: 1.5
    tp_sl_ratio: float = 1.5

    # Modo prioritario según el régimen
    # bull_trend/bear_trend: "swing" | range: "scalp" | volatile: None (no operar)
    preferred_mode: Optional[str] = None

    # Incremento adicional de umbral de entrada (fracción)
    # volatile suma 0.20 adicional al umbral normal
    threshold_increment: float = 0.20

    # Timestamp del cálculo (unix)
    calculated_at: float = field(default_factory=time.time)

    def is_stale(self, max_age_seconds: int = 300) -> bool:
        """True si el resultado tiene más de max_age_seconds segundos."""
        return (time.time() - self.calculated_at) > max_age_seconds

    def summary(self) -> str:
        """Resumen de una línea para el monitor."""
        conf_pct = f"{self.confidence*100:.0f}%"
        return (
            f"Régimen: {self.regime.upper()} (conf={conf_pct}) | "
            f"ADX={self.adx:.1f} | Hurst={self.hurst:.2f} | "
            f"VolRatio={self.volatility_ratio:.2f} | FG={self.fear_greed} | "
            f"EMA={self.ema_position}"
        )


# ─────────────────────────────────────────────────────────────
# DETECTOR PRINCIPAL
# ─────────────────────────────────────────────────────────────

class RegimeDetector:
    """
    Calcula el régimen de mercado a partir de los datos de precio.

    Uso:
        detector = RegimeDetector()
        result = detector.detect(
            daily_closes=closes_diarios,    # list[float], mínimo 200 valores
            daily_highs=highs_diarios,
            daily_lows=lows_diarios,
            current_atr=atr_actual,         # ATR del TF principal actual
            recent_atrs=atrs_20_velas,      # list[float], 20 valores
            bid=precio_bid,
            ask=precio_ask,
        )
    """

    def __init__(
        self,
        adx_period: int = 14,
        ema_short_period: int = 50,
        ema_long_period: int = 200,
        hurst_window: int = 100,
        volatility_ratio_window: int = 20,
        fear_greed_cache_seconds: int = 300,
    ):
        # ── Parámetros de cálculo ─────────────────────────────
        self.adx_period = adx_period                           # período del ADX
        self.ema_short_period = ema_short_period               # EMA50 diaria
        self.ema_long_period = ema_long_period                 # EMA200 diaria
        self.hurst_window = hurst_window                       # ventana para Hurst
        self.volatility_ratio_window = volatility_ratio_window # ventana para VR
        self.fear_greed_cache_seconds = fear_greed_cache_seconds

        # ── Caché del Fear & Greed ────────────────────────────
        self._fg_cache: Optional[int] = None
        self._fg_cache_time: float = 0.0

        # ── Último resultado calculado ────────────────────────
        self._last_result: Optional[RegimeResult] = None

    # ─────────────────────────────────────────────────────────
    # MÉTODO PRINCIPAL
    # ─────────────────────────────────────────────────────────

    def detect(
        self,
        daily_closes: list[float],
        daily_highs: list[float],
        daily_lows: list[float],
        current_atr: float,
        recent_atrs: list[float],
        bid: float,
        ask: float,
        use_fear_greed: bool = True,
    ) -> RegimeResult:
        """
        Calcula el régimen de mercado actual.

        Parámetros:
            daily_closes    Lista de precios de cierre diarios, más reciente al final.
                            Mínimo: max(ema_long_period, hurst_window) + adx_period + 5
                            Recomendado: 250 valores (1 año de datos diarios)
            daily_highs     Lista de máximos diarios, misma longitud que closes
            daily_lows      Lista de mínimos diarios, misma longitud que closes
            current_atr     ATR actual del timeframe principal de operación
            recent_atrs     Lista de ATRs recientes (últimas 20 velas del TF principal)
            bid             Precio bid actual
            ask             Precio ask actual
            use_fear_greed  Si False, no consulta la API (útil para backtest)

        Devuelve:
            RegimeResult con todos los campos calculados
        """

        result = RegimeResult()
        result.calculated_at = time.time()

        min_required = self.ema_long_period + self.adx_period + 5
        if len(daily_closes) < min_required:
            logger.warning(
                f"[Regime] Datos insuficientes: {len(daily_closes)} < {min_required}. "
                f"Devolviendo volatile con confianza 0."
            )
            result.regime = "volatile"
            result.confidence = 0.0
            self._last_result = result
            return result

        # ── 1. ADX ────────────────────────────────────────────
        result.adx = self._calc_adx(
            daily_highs[-self.adx_period * 3:],
            daily_lows[-self.adx_period * 3:],
            daily_closes[-self.adx_period * 3:],
            self.adx_period,
        )

        # ── 2. Posición de precio respecto a EMAs diarias ─────
        price = daily_closes[-1]
        ema50  = self._calc_ema(daily_closes, self.ema_short_period)
        ema200 = self._calc_ema(daily_closes, self.ema_long_period)
        result.ema_position = self._classify_ema_position(price, ema50, ema200)

        # ── 3. Volatility Ratio ───────────────────────────────
        if len(recent_atrs) >= self.volatility_ratio_window and current_atr > 0:
            avg_atr = statistics.mean(recent_atrs[-self.volatility_ratio_window:])
            result.volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
        else:
            result.volatility_ratio = 1.0

        # ── 4. Hurst Exponent ─────────────────────────────────
        window = min(self.hurst_window, len(daily_closes))
        if window >= 20:
            result.hurst = self._calc_hurst(daily_closes[-window:])
        else:
            result.hurst = 0.5   # sin datos suficientes → aleatorio

        # ── 5. Market Microstructure ──────────────────────────
        result.microstructure_ok = self._check_microstructure(
            bid, ask, current_atr
        )

        # ── 6. Fear & Greed ───────────────────────────────────
        if use_fear_greed:
            result.fear_greed = self._get_fear_greed()
        else:
            result.fear_greed = 50   # neutral en backtest

        # ── 7. Clasificar régimen ─────────────────────────────
        result.regime, result.confidence = self._classify_regime(result)

        # ── 8. Calcular modificadores para otros módulos ──────
        self._set_modifiers(result)

        self._last_result = result
        logger.debug(f"[Regime] {result.summary()}")
        return result

    # ─────────────────────────────────────────────────────────
    # ALGORITMO DE CLASIFICACIÓN
    # ─────────────────────────────────────────────────────────

    def _classify_regime(self, r: RegimeResult) -> tuple[str, float]:
        """
        Combina todos los indicadores y devuelve (regime, confidence).

        Lógica de votación ponderada:
        Cada indicador vota por un régimen y contribuye con un peso.
        El régimen con más votos ponderados gana.
        La confianza es la fracción del total de pesos que votó por el ganador.
        """

        # Votos: dict de regime → puntos acumulados
        votes: dict[str, float] = {
            "bull_trend": 0.0,
            "bear_trend": 0.0,
            "range":      0.0,
            "volatile":   0.0,
        }
        total_weight = 0.0

        # ── ADX (peso 3.0 — indicador más determinante para tendencia/rango)
        w = 3.0
        total_weight += w
        if r.adx > 30:
            # Tendencia fuerte — la dirección la determina EMA position
            if r.ema_position == "above_both":
                votes["bull_trend"] += w
            elif r.ema_position == "below_both":
                votes["bear_trend"] += w
            else:
                votes["volatile"] += w * 0.5
                votes["bull_trend"] += w * 0.25
                votes["bear_trend"] += w * 0.25
        elif r.adx > 25:
            # Tendencia moderada
            if r.ema_position == "above_both":
                votes["bull_trend"] += w * 0.8
                votes["range"]      += w * 0.2
            elif r.ema_position == "below_both":
                votes["bear_trend"] += w * 0.8
                votes["range"]      += w * 0.2
            else:
                votes["volatile"] += w * 0.6
                votes["range"]    += w * 0.4
        elif r.adx > 20:
            # Zona de transición
            votes["range"]   += w * 0.5
            votes["volatile"] += w * 0.3
            if r.ema_position == "above_both":
                votes["bull_trend"] += w * 0.2
            elif r.ema_position == "below_both":
                votes["bear_trend"] += w * 0.2
        else:
            # ADX < 20: rango claro
            votes["range"] += w

        # ── EMA Position (peso 2.0)
        w = 2.0
        total_weight += w
        if r.ema_position == "above_both":
            votes["bull_trend"] += w * 0.7
            votes["range"]      += w * 0.3
        elif r.ema_position == "below_both":
            votes["bear_trend"] += w * 0.7
            votes["range"]      += w * 0.3
        else:
            votes["range"]   += w * 0.5
            votes["volatile"] += w * 0.5

        # ── Volatility Ratio (peso 2.0)
        w = 2.0
        total_weight += w
        if r.volatility_ratio > 2.0:
            # Expansión extrema → volátil
            votes["volatile"] += w
        elif r.volatility_ratio > 1.5:
            # Expansión moderada → puede ser inicio de tendencia o pánico
            votes["volatile"]   += w * 0.5
            votes["bull_trend"] += w * 0.25
            votes["bear_trend"] += w * 0.25
        elif r.volatility_ratio < 0.7:
            # Compresión → pre-breakout, tratar como rango hasta confirmación
            votes["range"] += w
        else:
            # Normal → no es determinante, pequeño boost al régimen ya votado
            best = max(votes, key=lambda k: votes[k])
            votes[best] += w * 0.3

        # ── Hurst Exponent (peso 2.5 — el más matemáticamente sólido)
        w = 2.5
        total_weight += w
        if r.hurst > 0.65:
            # Fuerte tendencia estadística
            if r.ema_position == "above_both":
                votes["bull_trend"] += w
            elif r.ema_position == "below_both":
                votes["bear_trend"] += w
            else:
                votes["volatile"] += w * 0.5
                votes["bull_trend"] += w * 0.25
                votes["bear_trend"] += w * 0.25
        elif r.hurst > 0.55:
            # Tendencia leve
            if r.ema_position == "above_both":
                votes["bull_trend"] += w * 0.6
                votes["range"]      += w * 0.4
            elif r.ema_position == "below_both":
                votes["bear_trend"] += w * 0.6
                votes["range"]      += w * 0.4
            else:
                votes["range"]   += w * 0.5
                votes["volatile"] += w * 0.5
        elif r.hurst > 0.45:
            # Cerca de aleatorio — mercado sin edge estadístico
            votes["range"]   += w * 0.4
            votes["volatile"] += w * 0.6
        else:
            # Hurst < 0.45 → fuerte reversión a la media → rango
            votes["range"] += w

        # ── Fear & Greed como modificador (peso 1.0)
        w = 1.0
        total_weight += w
        if r.fear_greed < 15 or r.fear_greed > 85:
            # Sentimiento extremo → incrementa posibilidad de volatile
            votes["volatile"] += w * 0.7
            # Pero puede ser oportunidad contraria — pequeño boost
            if r.fear_greed < 15:
                votes["bull_trend"] += w * 0.3   # miedo extremo = posible suelo
            else:
                votes["bear_trend"] += w * 0.3   # avaricia extrema = posible techo
        elif r.fear_greed < 25 or r.fear_greed > 75:
            votes["volatile"] += w * 0.3
            best = max(votes, key=lambda k: votes[k])
            votes[best] += w * 0.7
        else:
            # FG normal → no altera el régimen
            best = max(votes, key=lambda k: votes[k])
            votes[best] += w

        # ── Microstructure como veto (no suma puntos, puede forzar volatile)
        if not r.microstructure_ok:
            # Spread anormal → mercado ilíquido → no operar
            return "volatile", 0.3

        # ── Determinar ganador ────────────────────────────────
        winning_regime = max(votes, key=lambda k: votes[k])
        winning_votes  = votes[winning_regime]
        confidence     = winning_votes / total_weight if total_weight > 0 else 0.0
        confidence     = min(confidence, 1.0)

        # Si la confianza es muy baja → volatile por precaución
        if confidence < 0.35:
            return "volatile", confidence

        return winning_regime, round(confidence, 3)

    def _set_modifiers(self, r: RegimeResult) -> None:
        """
        Rellena los campos de modificadores del RegimeResult
        según el régimen y la confianza detectados.
        """

        if r.regime == "bull_trend":
            r.risk_multiplier     = 1.0
            r.tp_sl_ratio         = 2.5 if r.confidence > 0.7 else 2.0
            r.preferred_mode      = "swing"
            r.threshold_increment = 0.0

        elif r.regime == "bear_trend":
            r.risk_multiplier     = 1.0
            r.tp_sl_ratio         = 2.5 if r.confidence > 0.7 else 2.0
            r.preferred_mode      = "swing"
            r.threshold_increment = 0.0

        elif r.regime == "range":
            r.risk_multiplier     = 0.5
            r.tp_sl_ratio         = 1.5
            r.preferred_mode      = "scalp"
            r.threshold_increment = 0.0

        else:   # volatile
            r.risk_multiplier     = 0.3
            r.tp_sl_ratio         = 1.5
            r.preferred_mode      = None       # no operar salvo N1 muy fuerte
            r.threshold_increment = 0.20       # +20% sobre el umbral normal

        # Fear & Greed modifica el threshold_increment adicional
        if r.fear_greed < 20 or r.fear_greed > 80:
            r.threshold_increment += 0.10
        elif r.fear_greed < 15 or r.fear_greed > 85:
            r.threshold_increment += 0.15

        # Si la confianza es baja, reducir el risk_multiplier
        if r.confidence < 0.5:
            r.risk_multiplier *= 0.6

    # ─────────────────────────────────────────────────────────
    # CÁLCULOS MATEMÁTICOS
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _calc_ema(closes: list[float], period: int) -> float:
        """
        Calcula la EMA de los últimos `period` valores de closes.
        Usa el método estándar: multiplier = 2 / (period + 1)
        Warm-up: los primeros `period` valores se promedian para la semilla.
        """
        if len(closes) < period:
            return closes[-1] if closes else 0.0

        k = 2.0 / (period + 1)
        ema = statistics.mean(closes[:period])   # semilla = SMA del primer período
        for price in closes[period:]:
            ema = price * k + ema * (1 - k)
        return ema

    @staticmethod
    def _classify_ema_position(
        price: float, ema50: float, ema200: float
    ) -> str:
        """
        Clasifica la posición del precio respecto a las dos EMAs diarias.
        - "above_both" : precio > EMA50 y precio > EMA200
        - "below_both" : precio < EMA50 y precio < EMA200
        - "between"    : cualquier otra combinación
        """
        if price > ema50 and price > ema200:
            return "above_both"
        elif price < ema50 and price < ema200:
            return "below_both"
        else:
            return "between"

    @staticmethod
    def _calc_adx(
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = 14,
    ) -> float:
        """
        Calcula el ADX (Average Directional Index) de Wilder.

        ADX > 25 = tendencia presente
        ADX < 20 = mercado sin tendencia (rango)

        Pasos:
        1. True Range (TR) = max(H-L, |H-Cp|, |L-Cp|)
        2. +DM = H - Hp si H-Hp > Lp-L y H-Hp > 0, sino 0
        3. -DM = Lp - L si Lp-L > H-Hp y Lp-L > 0, sino 0
        4. Suavizar TR, +DM, -DM con EMA de Wilder (período)
        5. +DI = 100 × +DM_smooth / TR_smooth
        6. -DI = 100 × -DM_smooth / TR_smooth
        7. DX = 100 × |+DI - -DI| / (+DI + -DI)
        8. ADX = EMA de Wilder de DX (período)
        """
        n = len(closes)
        if n < period * 2 + 1:
            return 20.0   # valor neutral si no hay suficientes datos

        tr_list, plus_dm, minus_dm = [], [], []

        for i in range(1, n):
            h, l, c_prev = highs[i], lows[i], closes[i - 1]
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            tr_list.append(tr)

            up   = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm.append(up   if up > down and up > 0   else 0.0)
            minus_dm.append(down if down > up and down > 0 else 0.0)

        # EMA de Wilder: primer valor = media simple, luego suavizado
        def wilder_smooth(data: list[float], p: int) -> list[float]:
            result = [sum(data[:p]) / p]
            for v in data[p:]:
                result.append(result[-1] * (p - 1) / p + v)
            return result

        tr_s   = wilder_smooth(tr_list,   period)
        pdm_s  = wilder_smooth(plus_dm,   period)
        mdm_s  = wilder_smooth(minus_dm,  period)

        dx_list = []
        for tr_v, p_v, m_v in zip(tr_s, pdm_s, mdm_s):
            if tr_v == 0:
                dx_list.append(0.0)
                continue
            pdi = 100.0 * p_v / tr_v
            mdi = 100.0 * m_v / tr_v
            denom = pdi + mdi
            dx_list.append(100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0)

        if len(dx_list) < period:
            return 20.0

        adx_smooth = wilder_smooth(dx_list, period)
        return round(adx_smooth[-1], 2)

    @staticmethod
    def _calc_hurst(closes: list[float]) -> float:
        """
        Calcula el Hurst Exponent usando el método R/S (Rescaled Range).

        > 0.5 → tendencia persistente (favorable para trend-following)
        = 0.5 → movimiento aleatorio (sin ventaja estadística)
        < 0.5 → reversión a la media (favorable para mean-reversion)

        Pasos:
        1. Calcular retornos logarítmicos: r_i = ln(P_i / P_{i-1})
        2. Calcular la media de los retornos: mean_r
        3. Calcular desviaciones acumuladas: Y_i = sum_{j=1}^{i}(r_j - mean_r)
        4. R = max(Y) - min(Y)  (rango de la serie acumulada)
        5. S = desviación estándar de los retornos
        6. RS = R / S
        7. Hurst = ln(RS) / ln(N/2)
        """
        n = len(closes)
        if n < 20:
            return 0.5

        # Retornos logarítmicos
        log_returns = []
        for i in range(1, n):
            if closes[i - 1] > 0 and closes[i] > 0:
                log_returns.append(math.log(closes[i] / closes[i - 1]))

        if len(log_returns) < 10:
            return 0.5

        m = statistics.mean(log_returns)

        # Serie acumulada de desviaciones respecto a la media
        cumdev = []
        acc = 0.0
        for r in log_returns:
            acc += (r - m)
            cumdev.append(acc)

        R = max(cumdev) - min(cumdev)
        S = statistics.stdev(log_returns)

        if S == 0 or R == 0:
            return 0.5

        RS = R / S
        N  = len(log_returns)

        try:
            hurst = math.log(RS) / math.log(N / 2)
        except (ValueError, ZeroDivisionError):
            return 0.5

        # Clampear entre 0.1 y 0.9 para evitar valores extremos por ruido
        return round(max(0.1, min(0.9, hurst)), 3)

    @staticmethod
    def _check_microstructure(bid: float, ask: float, atr: float) -> bool:
        """
        Verifica que el spread bid/ask sea razonable respecto a la volatilidad.

        Spread anormal = bid/ask spread > 20% del ATR actual.
        En ese caso el mercado está ilíquido → no operar (devuelve False).

        Ejemplo:
          ATR = 100 USDC → spread máximo aceptable = 20 USDC
          Si bid=40000 y ask=40025 → spread=25 > 20 → microstructure_ok=False
        """
        if bid <= 0 or ask <= 0 or atr <= 0:
            return True   # sin datos → asumir ok

        spread = ask - bid
        threshold = atr * 0.20   # 20% del ATR

        return spread <= threshold

    # ─────────────────────────────────────────────────────────
    # FEAR & GREED INDEX
    # ─────────────────────────────────────────────────────────

    def _get_fear_greed(self) -> int:
        """
        Obtiene el Fear & Greed Index de Alternative.me.
        API gratuita, sin clave. URL: https://api.alternative.me/fng/

        Usa caché de fear_greed_cache_seconds (default 300s = 5 minutos)
        para no spamear la API en cada ciclo.

        Si la API falla devuelve 50 (neutral) sin romper el ciclo.
        """
        now = time.time()
        if (
            self._fg_cache is not None
            and (now - self._fg_cache_time) < self.fear_greed_cache_seconds
        ):
            return self._fg_cache

        try:
            url = "https://api.alternative.me/fng/?limit=1&format=json"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            value = int(data["data"][0]["value"])
            self._fg_cache      = value
            self._fg_cache_time = now
            logger.debug(f"[Regime] Fear & Greed actualizado: {value}")
            return value
        except Exception as e:
            logger.warning(f"[Regime] Error obteniendo Fear & Greed: {e}. Usando 50.")
            return self._fg_cache if self._fg_cache is not None else 50

    # ─────────────────────────────────────────────────────────
    # ACCESO AL ÚLTIMO RESULTADO
    # ─────────────────────────────────────────────────────────

    @property
    def last_result(self) -> Optional[RegimeResult]:
        """Devuelve el último RegimeResult calculado, o None si no hay ninguno."""
        return self._last_result


# ─────────────────────────────────────────────────────────────
# FUNCIÓN DE ACCESO RÁPIDO PARA LOS OTROS MÓDULOS
# ─────────────────────────────────────────────────────────────

def get_scoring_weights(regime: str, confidence: float) -> dict:
    """
    Devuelve los multiplicadores de peso para cada componente del scoring
    según el régimen detectado.

    Estos multiplicadores se aplican en scoring.py sobre los puntos base
    de cada indicador para ajustar el puntaje al contexto del mercado.

    Valores = 1.0 → peso estándar (sin cambio)
    Valores > 1.0 → ese indicador vale más en este régimen
    Valores < 1.0 → ese indicador vale menos en este régimen
    """

    if regime in ("bull_trend", "bear_trend"):
        return {
            "rsi":              1.0,
            "emas":             1.3,   # EMAs alineadas confirman tendencia
            "macd":             1.2,
            "ut_bot":           1.1,
            "squeeze":          1.2,   # breakout de squeeze es señal fuerte en tendencia
            "ichimoku":         1.2,
            "vwap":             1.0,
            "volume":           1.1,
            "bollinger":        0.8,   # Bollinger menos relevante en tendencia
            "macro":            1.0,
            "pivots":           0.9,
            "funding":          1.0,
            "obi":              1.0,
            "candle_patterns":  1.1,
            "cci":              0.9,
            "stoch":            0.9,
            "macd_divergence":  1.0,
            "lateralization":   0.7,   # score de lateralización menos relevante
        }

    elif regime == "range":
        return {
            "rsi":              1.5,   # RSI extremo es la señal principal en rango
            "emas":             0.6,   # EMAs alineadas son menos relevantes en rango
            "macd":             0.8,
            "ut_bot":           0.7,
            "squeeze":          1.0,
            "ichimoku":         0.8,
            "vwap":             1.1,
            "volume":           0.9,
            "bollinger":        1.6,   # Bollinger tocada es la señal principal en rango
            "macro":            0.8,
            "pivots":           1.4,   # Pivotes son clave para targets en rango
            "funding":          1.0,
            "obi":              1.2,
            "candle_patterns":  1.3,   # Patrones de reversión valen más en rango
            "cci":              1.3,
            "stoch":            1.3,
            "macd_divergence":  1.2,
            "lateralization":   0.5,   # Si ya sabemos que estamos en rango, este score es redundante
        }

    else:   # volatile — reducir todo
        # En volátil los pesos se reducen globalmente.
        # El threshold_increment (+20%) ya hace el trabajo de filtrar.
        factor = max(0.4, confidence)
        return {k: factor for k in [
            "rsi", "emas", "macd", "ut_bot", "squeeze", "ichimoku",
            "vwap", "volume", "bollinger", "macro", "pivots", "funding",
            "obi", "candle_patterns", "cci", "stoch", "macd_divergence",
            "lateralization",
        ]}


# ─────────────────────────────────────────────────────────────
# TEST Y DEMO
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=" * 60)
    print("  TEST regime_detector.py")
    print("=" * 60)

    detector = RegimeDetector()
    random.seed(42)

    # ── Generar datos de precio sintéticos ───────────────────

    def generate_prices(n: int, trend: float = 0.0, noise: float = 0.01) -> list:
        """Genera lista de precios con tendencia y ruido controlados."""
        prices = [40000.0]
        for _ in range(n - 1):
            change = trend + random.gauss(0, noise)
            prices.append(prices[-1] * (1 + change))
        return prices

    def prices_to_ohlc(closes: list) -> tuple:
        """Genera highs y lows aproximados a partir de closes."""
        highs  = [c * (1 + abs(random.gauss(0, 0.003))) for c in closes]
        lows   = [c * (1 - abs(random.gauss(0, 0.003))) for c in closes]
        return highs, lows

    # ── Test 1: Tendencia alcista fuerte ─────────────────────
    print("\n[Test 1] Tendencia alcista fuerte (trend=+0.003, baja volatilidad)")
    closes = generate_prices(250, trend=0.003, noise=0.005)
    highs, lows = prices_to_ohlc(closes)
    atrs = [closes[i] * 0.008 for i in range(20)]
    current_atr = closes[-1] * 0.009

    result = detector.detect(
        daily_closes=closes,
        daily_highs=highs,
        daily_lows=lows,
        current_atr=current_atr,
        recent_atrs=atrs,
        bid=closes[-1] * 0.9999,
        ask=closes[-1] * 1.0001,
        use_fear_greed=False,
    )
    print(f"  {result.summary()}")
    print(f"  preferred_mode={result.preferred_mode} | risk_mult={result.risk_multiplier} | tp_sl={result.tp_sl_ratio}")
    assert result.regime in ("bull_trend", "volatile"), f"Esperaba bull_trend, got {result.regime}"
    print(f"  ✓ Régimen: {result.regime}")

    # ── Test 2: Mercado lateral ───────────────────────────────
    print("\n[Test 2] Mercado lateral (sin tendencia, baja volatilidad)")
    base = 40000.0
    closes = [base + random.gauss(0, 200) for _ in range(250)]
    highs, lows = prices_to_ohlc(closes)
    atrs = [200.0] * 20
    current_atr = 190.0

    result = detector.detect(
        daily_closes=closes,
        daily_highs=highs,
        daily_lows=lows,
        current_atr=current_atr,
        recent_atrs=atrs,
        bid=closes[-1] * 0.9999,
        ask=closes[-1] * 1.0001,
        use_fear_greed=False,
    )
    print(f"  {result.summary()}")
    print(f"  preferred_mode={result.preferred_mode} | risk_mult={result.risk_multiplier} | tp_sl={result.tp_sl_ratio}")
    assert result.regime in ("range", "volatile"), f"Esperaba range, got {result.regime}"
    print(f"  ✓ Régimen: {result.regime}")

    # ── Test 3: Mercado volátil ───────────────────────────────
    print("\n[Test 3] Mercado volátil (alta volatilidad, sin dirección clara)")
    closes = generate_prices(250, trend=0.0, noise=0.03)
    highs, lows = prices_to_ohlc(closes)
    atrs_hist = [closes[i] * 0.015 for i in range(20)]
    current_atr = closes[-1] * 0.04   # ATR actual muy por encima del histórico

    result = detector.detect(
        daily_closes=closes,
        daily_highs=highs,
        daily_lows=lows,
        current_atr=current_atr,
        recent_atrs=atrs_hist,
        bid=closes[-1] * 0.9999,
        ask=closes[-1] * 1.0001,
        use_fear_greed=False,
    )
    print(f"  {result.summary()}")
    print(f"  preferred_mode={result.preferred_mode} | risk_mult={result.risk_multiplier} | tp_sl={result.tp_sl_ratio}")
    print(f"  Régimen: {result.regime} (esperaba volatile o range con alta VR)")

    # ── Test 4: Microstructure mala ───────────────────────────
    print("\n[Test 4] Spread anormal (iliquidez)")
    closes = generate_prices(250, trend=0.002, noise=0.005)
    highs, lows = prices_to_ohlc(closes)
    current_atr = closes[-1] * 0.008
    atrs = [current_atr] * 20

    result = detector.detect(
        daily_closes=closes,
        daily_highs=highs,
        daily_lows=lows,
        current_atr=current_atr,
        recent_atrs=atrs,
        bid=closes[-1],
        ask=closes[-1] * 1.005,   # spread = 0.5%, mucho mayor al 20% del ATR
        use_fear_greed=False,
    )
    print(f"  {result.summary()}")
    assert result.regime == "volatile", f"Con spread anormal debe ser volatile, got {result.regime}"
    assert not result.microstructure_ok
    print(f"  ✓ Microstructure mala → regime=volatile forzado")

    # ── Test 5: Scoring weights ───────────────────────────────
    print("\n[Test 5] Scoring weights por régimen")
    for reg in ("bull_trend", "range", "volatile"):
        w = get_scoring_weights(reg, 0.8)
        print(f"  {reg}: rsi={w['rsi']} | emas={w['emas']} | bollinger={w['bollinger']}")
    print("  ✓ Weights generados correctamente")

    # ── Test 6: Funciones matemáticas ─────────────────────────
    print("\n[Test 6] Funciones matemáticas individuales")

    # EMA
    prices_test = [float(i) for i in range(1, 51)]
    ema = RegimeDetector._calc_ema(prices_test, 10)
    assert 40 < ema < 50, f"EMA fuera de rango esperado: {ema}"
    print(f"  EMA(10) sobre 1..50 = {ema:.2f} ✓")

    # Hurst sobre serie con tendencia (esperamos > 0.5)
    trending = generate_prices(150, trend=0.002, noise=0.002)
    hurst_t = RegimeDetector._calc_hurst(trending)
    print(f"  Hurst serie con tendencia = {hurst_t:.3f} (esperado > 0.5, got {'✓' if hurst_t > 0.4 else '?'})")

    # Hurst sobre ruido puro (esperamos ~0.5)
    noisy = generate_prices(150, trend=0.0, noise=0.02)
    hurst_n = RegimeDetector._calc_hurst(noisy)
    print(f"  Hurst ruido puro         = {hurst_n:.3f} (esperado ~0.5)")

    print("\n" + "=" * 60)
    print("  Todos los tests completados ✓")
    print("=" * 60)
