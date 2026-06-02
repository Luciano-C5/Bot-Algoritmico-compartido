"""
strategy.py
===========
Capa de decisión. Recibe un ScoreResult y el estado actual
del sistema y decide exactamente qué hacer:
- Tamaño de posición
- Precio de entrada
- Stop Loss
- Tres TPs escalonados
- Trailing stop
- Cierre parcial en EMAs intermedias
- Gestión de riesgo (pausas, contadores, modo revisión)

No ejecuta órdenes. Devuelve Decision objects que orders.py ejecuta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum

from indicators import IndicatorValues, analyze_macro_trend
from scoring import ScoreResult, StrategyEvaluator, THRESHOLDS


# ─────────────────────────────────────────────
# ENUMS Y ESTRUCTURAS
# ─────────────────────────────────────────────

class Action(Enum):
    OPEN_LONG     = "open_long"
    OPEN_SHORT    = "open_short"
    CLOSE_LONG    = "close_long"
    CLOSE_SHORT   = "close_short"
    PARTIAL_CLOSE = "partial_close"
    MOVE_SL       = "move_sl"
    HOLD          = "hold"
    PAPER_ONLY    = "paper_only"   # modo revisión activo


@dataclass
class TakeProfit:
    price:      float
    size_pct:   float   # % del total de la posición a cerrar en este TP
    order_id:   str = ""


@dataclass
class Decision:
    """
    Todo lo que orders.py necesita para ejecutar (o no) una operación.
    """
    action:     Action
    direction:  str = ""        # "long" o "short"
    mode:       str = ""        # "scalp", "mediano", "swing"

    # Entrada
    entry_price:    float = 0.0
    size_usdc:      float = 0.0     # capital a usar en USDC
    leverage:       int   = 1

    # SL y TPs
    sl_price:       float = 0.0
    tp_levels:      list[TakeProfit] = field(default_factory=list)

    # Trailing stop
    trailing_active:    bool  = False
    trailing_distance:  float = 0.0     # % desde el precio actual
    trailing_trigger:   float = 0.0     # % de avance para activar

    # Contexto
    score:          Optional[ScoreResult] = None
    reason:         str = ""
    timestamp:      datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Para cierre parcial
    close_pct:      float = 0.0     # % de la posición a cerrar

    def __str__(self) -> str:
        if self.action == Action.HOLD:
            return f"[Decision] HOLD — {self.reason}"
        if self.action == Action.PAPER_ONLY:
            return f"[Decision] PAPER ONLY (modo revisión) — {self.reason}"

        lines = [
            f"[Decision] {self.action.value.upper()} | {self.mode} | "
            f"x{self.leverage}",
            f"  Entrada:  ${self.entry_price:,.2f}",
            f"  Tamaño:   ${self.size_usdc:.2f} USDC",
            f"  SL:       ${self.sl_price:,.2f}",
        ]
        for i, tp in enumerate(self.tp_levels, 1):
            lines.append(f"  TP{i}:      ${tp.price:,.2f} ({tp.size_pct:.0f}%)")
        if self.trailing_active:
            lines.append(f"  Trailing: {self.trailing_distance:.2f}% "
                         f"(activa desde +{self.trailing_trigger:.2f}%)")
        return "\n".join(lines)


@dataclass
class Position:
    """Estado de la posición abierta actual."""
    open:           bool  = False
    direction:      str   = ""
    mode:           str   = ""
    entry_price:    float = 0.0
    size_usdc:      float = 0.0
    leverage:       int   = 1
    sl_price:       float = 0.0
    tp_levels:      list[TakeProfit] = field(default_factory=list)
    remaining_pct:  float = 1.0     # fracción de la posición original que queda

    trailing_active:   bool  = False
    trailing_distance: float = 0.0
    trailing_stop:     float = 0.0  # precio actual del trailing stop
    highest_price:     float = 0.0  # para long: máximo alcanzado
    lowest_price:      float = 0.0  # para short: mínimo alcanzado

    breakeven_set:  bool  = False
    opened_at:      Optional[datetime] = None
    score_at_open:  Optional[ScoreResult] = None

    @property
    def unrealized_pnl_pct(self) -> float:
        """PNL no realizado como % del capital (no de la posición)."""
        if not self.open or self.entry_price <= 0:
            return 0.0
        # Se calcula después con el precio actual


# ─────────────────────────────────────────────
# GESTIÓN DE RIESGO
# ─────────────────────────────────────────────

@dataclass
class RiskState:
    """
    Estado del sistema de gestión de riesgo.
    Persiste entre ciclos (se guarda/carga desde JSON).
    """
    # Contadores por modo
    consecutive_losses: dict[str, int] = field(
        default_factory=lambda: {"scalp": 0, "mediano": 0, "swing": 0}
    )
    daily_losses: int = 0
    total_trades_today: int = 0

    # Pausas por modo (timestamp hasta cuando está pausado)
    mode_paused_until: dict[str, Optional[datetime]] = field(
        default_factory=lambda: {"scalp": None, "mediano": None, "swing": None}
    )
    global_paused_until: Optional[datetime] = None

    # Modo revisión (paper trading)
    review_mode: bool = False
    review_mode_until: Optional[datetime] = None
    review_mode_reason: str = ""

    # Historial reciente para calcular winrate
    recent_results: list[str] = field(default_factory=list)  # "win" o "loss"
    MAX_RECENT = 20

    # Reset diario
    last_reset_date: str = ""

    def reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_reset_date != today:
            self.daily_losses      = 0
            self.total_trades_today = 0
            self.consecutive_losses = {"scalp": 0, "mediano": 0, "swing": 0}
            self.mode_paused_until  = {"scalp": None, "mediano": None, "swing": None}
            self.global_paused_until = None
            self.last_reset_date    = today

    def register_result(self, mode: str, won: bool) -> list[str]:
        """
        Registra el resultado de una operación y aplica pausas si corresponde.
        Devuelve lista de eventos ocurridos (para logging/telegram).
        """
        events = []
        self.total_trades_today += 1
        result_str = "win" if won else "loss"

        self.recent_results.append(result_str)
        if len(self.recent_results) > self.MAX_RECENT:
            self.recent_results.pop(0)

        if not won:
            self.daily_losses += 1
            self.consecutive_losses[mode] = self.consecutive_losses.get(mode, 0) + 1
        else:
            self.consecutive_losses[mode] = 0

        # 3 pérdidas consecutivas en el mismo modo → pausa ese modo 20 velas
        velas_pausa = {"scalp": 20*15, "mediano": 20*60, "swing": 20*240}  # minutos
        consec = self.consecutive_losses.get(mode, 0)
        if consec >= 3:
            mins = velas_pausa.get(mode, 300)
            until = datetime.now(timezone.utc) + timedelta(minutes=mins)
            self.mode_paused_until[mode] = until
            events.append(
                f"PAUSA {mode.upper()}: 3 pérdidas consecutivas. "
                f"Pausa hasta {until.strftime('%H:%M UTC')}"
            )

        # 5 pérdidas totales en el día → pausa global hasta medianoche UTC
        if self.daily_losses >= 5:
            midnight = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            self.global_paused_until = midnight
            events.append(
                f"PAUSA GLOBAL: 5 pérdidas en el día. "
                f"Reanuda a medianoche UTC"
            )

        # Winrate bajo → modo revisión
        if len(self.recent_results) >= self.MAX_RECENT:
            wins = self.recent_results.count("win")
            winrate = wins / len(self.recent_results)
            if winrate < 0.38 and not self.review_mode:
                self.review_mode       = True
                self.review_mode_until = (datetime.now(timezone.utc)
                                          + timedelta(days=30))
                self.review_mode_reason = (
                    f"Winrate {winrate*100:.1f}% en últimas "
                    f"{self.MAX_RECENT} operaciones"
                )
                events.append(
                    f"⚠ MODO REVISIÓN ACTIVADO: {self.review_mode_reason}"
                )

        return events

    def is_mode_paused(self, mode: str) -> bool:
        now = datetime.now(timezone.utc)
        if self.global_paused_until and now < self.global_paused_until:
            return True
        paused_until = self.mode_paused_until.get(mode)
        if paused_until and now < paused_until:
            return True
        return False

    def is_review_mode(self) -> bool:
        if not self.review_mode:
            return False
        if self.review_mode_until and datetime.now(timezone.utc) > self.review_mode_until:
            self.review_mode = False
            return False
        return True

    @property
    def winrate_recent(self) -> float:
        if not self.recent_results:
            return 0.5
        return self.recent_results.count("win") / len(self.recent_results)

    def save(self, path: str = "risk_state.json") -> None:
        data = {
            "consecutive_losses":  self.consecutive_losses,
            "daily_losses":        self.daily_losses,
            "total_trades_today":  self.total_trades_today,
            "mode_paused_until":   {
                k: v.isoformat() if v else None
                for k, v in self.mode_paused_until.items()
            },
            "global_paused_until": (self.global_paused_until.isoformat()
                                    if self.global_paused_until else None),
            "review_mode":         self.review_mode,
            "review_mode_until":   (self.review_mode_until.isoformat()
                                    if self.review_mode_until else None),
            "review_mode_reason":  self.review_mode_reason,
            "recent_results":      self.recent_results,
            "last_reset_date":     self.last_reset_date,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = "risk_state.json") -> "RiskState":
        try:
            with open(path) as f:
                data = json.load(f)
            rs = cls()
            rs.consecutive_losses  = data.get("consecutive_losses",
                                               {"scalp":0,"mediano":0,"swing":0})
            rs.daily_losses        = data.get("daily_losses", 0)
            rs.total_trades_today  = data.get("total_trades_today", 0)
            rs.recent_results      = data.get("recent_results", [])
            rs.last_reset_date     = data.get("last_reset_date", "")
            rs.review_mode         = data.get("review_mode", False)
            rs.review_mode_reason  = data.get("review_mode_reason", "")

            def _parse_dt(s):
                return datetime.fromisoformat(s) if s else None

            rs.mode_paused_until = {
                k: _parse_dt(v)
                for k, v in data.get("mode_paused_until",
                                     {"scalp":None,"mediano":None,"swing":None}).items()
            }
            rs.global_paused_until = _parse_dt(data.get("global_paused_until"))
            rs.review_mode_until   = _parse_dt(data.get("review_mode_until"))
            return rs
        except (FileNotFoundError, json.JSONDecodeError):
            return cls()


# ─────────────────────────────────────────────
# CALCULADOR DE PARÁMETROS DE OPERACIÓN
# ─────────────────────────────────────────────

class OperationBuilder:
    """
    Dado un ScoreResult y los indicadores actuales, calcula
    los parámetros exactos de la operación: entry, SL, TPs, trailing.
    """

    # Distancias de trailing por modo (% desde precio)
    TRAILING_DISTANCE = {
        "scalp":   0.0027,   # 0.27%
        "mediano": 0.0045,   # 0.45%
        "swing":   0.0150,   # 1.5%
    }
    # Avance necesario para activar el trailing (% desde entrada)
    TRAILING_TRIGGER = {
        "scalp":   0.0030,   # 0.30%
        "mediano": 0.0050,   # 0.50%
        "swing":   0.0100,   # 1.00%
    }

    # SL base por modo como % del capital (no de la posición)
    SL_BASE_PCT = {
        "scalp":   0.004,    # 0.4%
        "mediano": 0.005,    # 0.5%
        "swing":   0.008,    # 0.8%
    }

    # TP base por modo como % del capital
    TP_BASE_PCT = {
        "scalp":   0.006,    # 0.6%
        "mediano": 0.010,    # 1.0%
        "swing":   0.025,    # 2.5%
    }

    def build(
        self,
        score:      ScoreResult,
        iv_main:    IndicatorValues,
        capital:    float,          # USDC disponible
        price:      float,
    ) -> Decision:
        mode      = score.mode
        direction = score.direction
        leverage  = score.leverage

        # ── Tamaño de posición ────────────────────────────────────────
        # Usamos el 100% del capital disponible (una posición a la vez)
        size_usdc = capital

        # ── SL: ajustado por ATR y niveles cercanos ───────────────────
        sl_price = self._calc_sl(price, direction, mode, iv_main, leverage)

        # ── TPs escalonados ───────────────────────────────────────────
        tp_levels = self._calc_tps(price, direction, mode, iv_main,
                                    leverage, score.signal_level)

        # ── Trailing stop ─────────────────────────────────────────────
        trailing_distance = self.TRAILING_DISTANCE[mode]
        trailing_trigger  = self.TRAILING_TRIGGER[mode]

        action = Action.OPEN_LONG if direction == "long" else Action.OPEN_SHORT

        return Decision(
            action           = action,
            direction        = direction,
            mode             = mode,
            entry_price      = price,
            size_usdc        = size_usdc,
            leverage         = leverage,
            sl_price         = sl_price,
            tp_levels        = tp_levels,
            trailing_active  = True,
            trailing_distance = trailing_distance * 100,   # en %
            trailing_trigger  = trailing_trigger  * 100,
            score            = score,
            reason           = f"N{score.signal_level} {score.normalized*100:.1f}%",
        )

    def _calc_sl(
        self, price: float, direction: str, mode: str,
        iv: IndicatorValues, leverage: int
    ) -> float:
        """
        SL inteligente: base por modo ajustado por ATR y EMAs cercanas.
        Siempre expresado como % del capital, convertido a precio.
        """
        base_pct = self.SL_BASE_PCT[mode]

        # Ajuste por ATR: si la volatilidad es alta, SL un poco más lejos
        if iv.atr_pct > 0:
            atr_factor = min(iv.atr_pct / 0.3, 1.5)   # normalizado, max 1.5x
            base_pct   = base_pct * atr_factor

        # Convertir % de capital a movimiento de precio
        # PNL% = (precio_salida - precio_entrada) / precio_entrada * leverage
        # base_pct = movimiento_precio * leverage  →  movimiento = base_pct / leverage
        price_move = (base_pct / leverage) * price

        if direction == "long":
            sl = price - price_move
            # No poner SL por encima de la EMA200 si está cerca (sería en terreno malo)
            if iv.ema200 > 0 and sl < iv.ema200 * 0.995:
                sl = max(sl, iv.ema200 * 0.995)
        else:
            sl = price + price_move
            if iv.ema200 > 0 and sl > iv.ema200 * 1.005:
                sl = min(sl, iv.ema200 * 1.005)

        return round(sl, 2)

    def _calc_tps(
        self, price: float, direction: str, mode: str,
        iv: IndicatorValues, leverage: int, signal_level: int
    ) -> list[TakeProfit]:
        """
        3 TPs escalonados. El primero más conservador, el tercero más ambicioso.
        Se ajustan según EMAs próximas en el camino y pivotes.
        """
        base_pct = self.TP_BASE_PCT[mode]
        if signal_level == 3:
            base_pct = 0.0015   # nivel 3: TP reducido

        # Tres niveles: 40%, 70%, 100% del TP base
        # Tamaños: 40% de la posición en TP1, 35% en TP2, 25% en TP3
        tp_distances = [base_pct * 0.4, base_pct * 0.7, base_pct * 1.0]
        tp_sizes     = [40.0, 35.0, 25.0]

        tps = []
        for i, (dist_pct, size) in enumerate(zip(tp_distances, tp_sizes)):
            price_move = (dist_pct / leverage) * price
            if direction == "long":
                tp_price = price + price_move
                # Si hay una EMA importante en el camino, ajustar al nivel de la EMA
                tp_price = self._adjust_tp_to_ema(tp_price, price, direction, iv)
            else:
                tp_price = price - price_move
                tp_price = self._adjust_tp_to_ema(tp_price, price, direction, iv)

            tps.append(TakeProfit(
                price    = round(tp_price, 2),
                size_pct = size,
            ))

        return tps

    def _adjust_tp_to_ema(
        self, tp_price: float, entry: float,
        direction: str, iv: IndicatorValues
    ) -> float:
        """
        Si hay una EMA importante entre la entrada y el TP,
        ajusta el TP a justo antes de esa EMA (deja margen del 0.1%).
        """
        emas = {
            "ema7":   iv.ema7,
            "ema25":  iv.ema25,
            "ema50":  iv.ema50,
            "ema99":  iv.ema99,
            "ema200": iv.ema200,
        }

        if direction == "long":
            # Buscar EMAs entre la entrada y el TP
            emas_in_path = [v for v in emas.values()
                            if v > 0 and entry < v < tp_price]
            if emas_in_path:
                nearest = min(emas_in_path)
                # Colocar TP justo antes de la EMA (margen 0.1%)
                tp_price = nearest * 0.999
        else:
            emas_in_path = [v for v in emas.values()
                            if v > 0 and tp_price < v < entry]
            if emas_in_path:
                nearest = max(emas_in_path)
                tp_price = nearest * 1.001

        return tp_price


# ─────────────────────────────────────────────
# GESTOR DE POSICIÓN ABIERTA
# ─────────────────────────────────────────────

class PositionManager:
    """
    Dado el precio actual y la posición abierta, decide
    si hay que hacer algo: mover SL, activar trailing,
    cerrar parcialmente, cerrar todo.
    """

    def evaluate(
        self,
        position: Position,
        price:    float,
        iv:       IndicatorValues,
        score:    Optional[ScoreResult] = None,   # score actual del mercado
    ) -> list[Decision]:
        """
        Evalúa la posición abierta y devuelve lista de decisiones.
        Puede devolver múltiples decisiones (ej: mover SL + cierre parcial).
        """
        if not position.open:
            return []

        decisions = []

        # Actualizar máximo/mínimo alcanzado
        if position.direction == "long":
            position.highest_price = max(position.highest_price, price)
        else:
            position.lowest_price = min(
                position.lowest_price if position.lowest_price > 0 else price,
                price
            )

        # ── 1. Trailing stop ──────────────────────────────────────────
        trail_decision = self._evaluate_trailing(position, price, iv)
        if trail_decision:
            decisions.append(trail_decision)
            if trail_decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
                return decisions   # posición cerrada, no evaluar más

        # ── 2. SL inteligente ─────────────────────────────────────────
        sl_decision = self._evaluate_sl(position, price, iv, score)
        if sl_decision:
            decisions.append(sl_decision)
            if sl_decision.action in (Action.CLOSE_LONG, Action.CLOSE_SHORT):
                return decisions

        # ── 3. Cierre parcial en EMAs intermedias ─────────────────────
        partial_decisions = self._evaluate_partial_close(position, price, iv)
        decisions.extend(partial_decisions)

        # ── 4. Señal contraria fuerte ─────────────────────────────────
        if score:
            opposite = "short" if position.direction == "long" else "long"
            if (score.direction == opposite and
                    score.signal_level >= 2 and score.should_trade):
                close_action = (Action.CLOSE_LONG
                                if position.direction == "long"
                                else Action.CLOSE_SHORT)
                decisions.append(Decision(
                    action = close_action,
                    reason = f"Señal contraria fuerte: {score.direction} "
                             f"N{score.signal_level} {score.normalized*100:.1f}%",
                ))

        return decisions

    def _evaluate_trailing(
        self, position: Position, price: float, iv: IndicatorValues
    ) -> Optional[Decision]:
        mode = position.mode

        # Calcular avance desde entrada como % del capital
        if position.entry_price <= 0:
            return None

        if position.direction == "long":
            advance_pct = (price - position.entry_price) / position.entry_price
            advance_pct *= position.leverage
        else:
            advance_pct = (position.entry_price - price) / position.entry_price
            advance_pct *= position.leverage

        # Trigger para activar trailing
        triggers = {"scalp": 0.003, "mediano": 0.005, "swing": 0.010}
        trigger  = triggers.get(mode, 0.005)

        distances = {"scalp": 0.0027, "mediano": 0.0045, "swing": 0.015}
        distance  = distances.get(mode, 0.005)

        if advance_pct >= trigger:
            position.trailing_active = True

        if not position.trailing_active:
            return None

        # Calcular nuevo nivel del trailing stop
        if position.direction == "long":
            new_trail = price * (1 - distance)
            if new_trail > position.trailing_stop:
                position.trailing_stop = new_trail
                # Mover también el SL si el trailing lo supera
                if position.trailing_stop > position.sl_price:
                    position.sl_price = position.trailing_stop
                    return Decision(
                        action   = Action.MOVE_SL,
                        sl_price = position.trailing_stop,
                        reason   = f"Trailing stop subido a "
                                   f"${position.trailing_stop:,.2f}",
                    )
            # ¿El precio tocó el trailing stop?
            if price <= position.trailing_stop:
                return Decision(
                    action = Action.CLOSE_LONG,
                    reason = f"Trailing stop alcanzado: ${position.trailing_stop:,.2f}",
                )
        else:
            new_trail = price * (1 + distance)
            if position.trailing_stop == 0 or new_trail < position.trailing_stop:
                position.trailing_stop = new_trail
                if position.trailing_stop < position.sl_price or position.sl_price == 0:
                    position.sl_price = position.trailing_stop
                    return Decision(
                        action   = Action.MOVE_SL,
                        sl_price = position.trailing_stop,
                        reason   = f"Trailing stop bajado a "
                                   f"${position.trailing_stop:,.2f}",
                    )
            if price >= position.trailing_stop:
                return Decision(
                    action = Action.CLOSE_SHORT,
                    reason = f"Trailing stop alcanzado: ${position.trailing_stop:,.2f}",
                )

        return None

    def _evaluate_sl(
        self,
        position: Position,
        price:    float,
        iv:       IndicatorValues,
        score:    Optional[ScoreResult],
    ) -> Optional[Decision]:
        """
        SL inteligente: evalúa si cerrar, mantener o mover a breakeven.
        """
        if position.sl_price <= 0:
            return None

        sl_hit = (position.direction == "long"  and price <= position.sl_price or
                  position.direction == "short" and price >= position.sl_price)

        if not sl_hit:
            # Mover SL a breakeven si hubo cierre parcial
            if (not position.breakeven_set and
                    position.remaining_pct < 1.0 and
                    position.entry_price > 0):
                # Breakeven = entrada + comisiones (0.1% ambos lados)
                fee_factor = 1.001 if position.direction == "long" else 0.999
                be_price   = position.entry_price * fee_factor
                if ((position.direction == "long"  and position.sl_price < be_price) or
                    (position.direction == "short" and position.sl_price > be_price)):
                    position.sl_price    = be_price
                    position.breakeven_set = True
                    return Decision(
                        action   = Action.MOVE_SL,
                        sl_price = be_price,
                        reason   = "SL movido a breakeven tras cierre parcial",
                    )
            return None

        # SL alcanzado — evaluar si cerrar o mantener
        # Solo mantener si: señal original muy fuerte Y tendencia no cambió
        if (score and
                position.score_at_open and
                position.score_at_open.signal_level == 1 and
                score.direction == position.direction and
                score.normalized >= THRESHOLDS[2]):
            # Tendencia sigue válida, no cerrar todavía
            # PERO: si EMAs cruzaron en contra, cerrar igual
            if position.direction == "long":
                emas_crossed_against = iv.emas_aligned_bearish >= 4
            else:
                emas_crossed_against = iv.emas_aligned_bullish >= 4

            if not emas_crossed_against:
                return None   # mantener, esperar recuperación

        close_action = (Action.CLOSE_LONG
                        if position.direction == "long"
                        else Action.CLOSE_SHORT)
        return Decision(
            action = close_action,
            reason = f"SL alcanzado: ${position.sl_price:,.2f}",
        )

    def _evaluate_partial_close(
        self, position: Position, price: float, iv: IndicatorValues
    ) -> list[Decision]:
        """
        Cierre parcial cuando el precio llega a una EMA importante en el camino.
        Lógica: fuerza de ruptura × fuerza del soporte = % a cerrar.
        """
        decisions = []

        emas = {
            "ema7":   iv.ema7,
            "ema25":  iv.ema25,
            "ema50":  iv.ema50,
            "ema99":  iv.ema99,
            "ema200": iv.ema200,
        }

        for ema_name, ema_val in emas.items():
            if ema_val <= 0:
                continue

            # ¿El precio llegó a esta EMA en el camino hacia el TP?
            tolerance = ema_val * 0.002  # 0.2% de tolerancia

            reached = False
            if (position.direction == "long" and
                    position.entry_price < ema_val and
                    abs(price - ema_val) <= tolerance):
                reached = True
            elif (position.direction == "short" and
                    position.entry_price > ema_val and
                    abs(price - ema_val) <= tolerance):
                reached = True

            if not reached:
                continue

            # Calcular fuerza de la ruptura (qué tan fuerte llegó aquí)
            if iv.atr > 0:
                momentum = abs(price - position.entry_price) / iv.atr
                ruptura_fuerte = momentum > 0.8
                ruptura_moderada = 0.4 < momentum <= 0.8
            else:
                ruptura_fuerte   = False
                ruptura_moderada = True

            # Peso del soporte (simplificado: EMA200 > EMA99 > EMA50 > EMA25 > EMA7)
            soporte_weights = {
                "ema200": 4, "ema99": 3, "ema50": 3, "ema25": 2, "ema7": 1
            }
            soporte = soporte_weights.get(ema_name, 2)
            soporte_fuerte   = soporte >= 4
            soporte_moderado = soporte == 3

            # Determinar % a cerrar según la tabla del spec
            close_pct = 0.0
            if ruptura_fuerte and soporte_fuerte:
                close_pct = 30.0
            elif ruptura_fuerte and not soporte_fuerte:
                close_pct = 15.0
            elif ruptura_moderada and soporte_fuerte:
                close_pct = 50.0
            elif ruptura_moderada and soporte_moderado:
                close_pct = 35.0
            elif ruptura_moderada:
                close_pct = 20.0
            elif soporte_fuerte:
                close_pct = 60.0

            if close_pct > 0 and position.remaining_pct > 0.1:
                actual_close = min(close_pct, position.remaining_pct * 100 - 10)
                if actual_close > 0:
                    decisions.append(Decision(
                        action    = Action.PARTIAL_CLOSE,
                        close_pct = actual_close,
                        reason    = f"Cierre parcial {actual_close:.0f}% en "
                                    f"{ema_name} (${ema_val:,.2f}): "
                                    f"ruptura={'fuerte' if ruptura_fuerte else 'moderada'} "
                                    f"soporte={soporte}",
                    ))

        return decisions


# ─────────────────────────────────────────────
# ESTRATEGIA PRINCIPAL
# ─────────────────────────────────────────────

class Strategy:
    """
    Punto de entrada principal. El bot llama a evaluate() en cada ciclo.

    Uso:
        strategy = Strategy(capital=100.0)
        decision = strategy.evaluate(snapshot, indicators, macro)
        if decision.action != Action.HOLD:
            orders.execute(decision)
    """

    def __init__(self, capital: float = 100.0, testnet: bool = True):
        self.capital          = capital
        self.testnet          = testnet
        self.position         = Position()
        self.risk             = RiskState.load()
        self.evaluator        = StrategyEvaluator()
        self.op_builder       = OperationBuilder()
        self.pos_manager      = PositionManager()
        self._active_modes    = ["scalp", "mediano", "swing"]

    # ── Modos activos ─────────────────────────────────────────────────

    def set_mode(self, mode: str, active: bool) -> None:
        if active and mode not in self._active_modes:
            self._active_modes.append(mode)
        elif not active and mode in self._active_modes:
            self._active_modes.remove(mode)

    @property
    def active_modes(self) -> list[str]:
        """Modos activos y no pausados."""
        return [m for m in self._active_modes
                if not self.risk.is_mode_paused(m)]

    # ── Ciclo principal ───────────────────────────────────────────────

    def evaluate(
        self,
        indicators: dict,
        macro,
        threshold_increment: float = 0.0,
    ) -> Decision:
        """
        Evalúa el mercado y devuelve la decisión para este ciclo.
        """
        self.risk.reset_daily_if_needed()

        price = 0.0
        iv_15m = indicators.get("15m")
        if iv_15m:
            price = iv_15m.current_price

        if price <= 0:
            return Decision(action=Action.HOLD, reason="Sin precio disponible")

        # ── Modo revisión activo ──────────────────────────────────────
        if self.risk.is_review_mode():
            # Calcular igual pero no operar
            best = self.evaluator.evaluate(
                indicators, macro, self.active_modes, threshold_increment
            )
            signal_info = (f"{best.direction} {best.mode} N{best.signal_level}"
                           if best else "ninguna")
            return Decision(
                action = Action.PAPER_ONLY,
                reason = f"Modo revisión hasta "
                         f"{self.risk.review_mode_until.strftime('%Y-%m-%d') if self.risk.review_mode_until else '?'}"
                         f" | Señal detectada: {signal_info}",
            )

        # ── Posición abierta: gestionar ───────────────────────────────
        if self.position.open:
            iv_main = indicators.get(
                {"scalp":"15m","mediano":"1h","swing":"4h"}.get(
                    self.position.mode, "15m"
                )
            )
            if iv_main is None:
                return Decision(action=Action.HOLD, reason="Sin datos para posición abierta")

            # Score actual para detectar señal contraria
            current_score = self.evaluator.evaluate(
                indicators, macro, self.active_modes, threshold_increment
            )

            mgmt_decisions = self.pos_manager.evaluate(
                self.position, price, iv_main, current_score
            )

            if mgmt_decisions:
                return mgmt_decisions[0]   # ejecutar de a una por ciclo

            return Decision(action=Action.HOLD, reason="Posición abierta, sin acción")

        # ── Sin posición: buscar entrada ──────────────────────────────
        if not self.active_modes:
            return Decision(action=Action.HOLD,
                            reason="Todos los modos pausados")

        best = self.evaluator.evaluate(
            indicators, macro, self.active_modes, threshold_increment
        )

        if best is None or not best.should_trade:
            return Decision(action=Action.HOLD, reason="Sin señales operables")

        # Señal encontrada — construir la operación
        iv_main = indicators.get(best.timeframe)
        if iv_main is None:
            return Decision(action=Action.HOLD,
                            reason=f"Sin datos para {best.timeframe}")

        decision = self.op_builder.build(best, iv_main, self.capital, price)
        return decision

    # ── Registro de resultados ────────────────────────────────────────

    def register_trade_result(self, mode: str, won: bool) -> list[str]:
        """
        Llamar cuando se cierra una operación.
        Devuelve lista de eventos (pausas, alertas) para telegram/log.
        """
        events = self.risk.register_result(mode, won)
        self.risk.save()
        return events

    def open_position(self, decision: Decision) -> None:
        """Actualizar el estado interno cuando se abre una posición."""
        self.position = Position(
            open         = True,
            direction    = decision.direction,
            mode         = decision.mode,
            entry_price  = decision.entry_price,
            size_usdc    = decision.size_usdc,
            leverage     = decision.leverage,
            sl_price     = decision.sl_price,
            tp_levels    = decision.tp_levels,
            remaining_pct = 1.0,
            highest_price = decision.entry_price,
            lowest_price  = decision.entry_price,
            opened_at     = decision.timestamp,
            score_at_open = decision.score,
        )

    def close_position(self) -> None:
        """Limpiar el estado de posición."""
        self.position = Position()


# ─────────────────────────────────────────────
# TEST RÁPIDO
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import time
    from market_feed import create_feed
    from indicators import IndicatorCalculator

    print("Probando strategy.py contra testnet...\n")
    feed = create_feed('live', symbol='BTCUSDC', testnet=True)
    feed.start()
    time.sleep(2)

    snap  = feed.get_snapshot()
    calc  = IndicatorCalculator()
    ivs   = calc.calculate(snap)
    macro = analyze_macro_trend(ivs)

    strategy = Strategy(capital=100.0, testnet=True)
    decision = strategy.evaluate(ivs, macro)

    print(decision)
    print()

    # Simular apertura de posición si hay señal
    if decision.action in (Action.OPEN_LONG, Action.OPEN_SHORT):
        print("Simulando apertura...")
        strategy.open_position(decision)
        print(f"Posición abierta: {strategy.position.direction} "
              f"@ ${strategy.position.entry_price:,.2f}")
        print(f"SL: ${strategy.position.sl_price:,.2f}")
        for i, tp in enumerate(strategy.position.tp_levels, 1):
            print(f"TP{i}: ${tp.price:,.2f} ({tp.size_pct:.0f}%)")

        # Simular un ciclo de gestión
        print("\nSimulando gestión de posición...")
        snap2 = feed.get_snapshot()
        ivs2  = calc.calculate(snap2)
        price2 = snap2.current_close
        iv_main = ivs2.get("15m")
        if iv_main:
            mgmt = strategy.pos_manager.evaluate(
                strategy.position, price2, iv_main
            )
            if mgmt:
                for d in mgmt:
                    print(f"  → {d}")
            else:
                print("  → Sin acción en este ciclo")

    # Mostrar estado del riesgo
    print(f"\nEstado de riesgo:")
    print(f"  Pérdidas consecutivas: {strategy.risk.consecutive_losses}")
    print(f"  Pérdidas hoy: {strategy.risk.daily_losses}")
    print(f"  Winrate reciente: {strategy.risk.winrate_recent*100:.1f}%")
    print(f"  Modo revisión: {strategy.risk.is_review_mode()}")

    feed.stop()
