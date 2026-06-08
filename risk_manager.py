"""
risk_manager.py
===============
Gestión de riesgo centralizada. Capa transversal que corre en paralelo
al ciclo principal y puede bloquear cualquier modo de operación.

Responsabilidades:
  - Contadores de pérdidas consecutivas por modo
  - Pausa por modo (3 pérdidas → 20 velas del TF del modo)
  - Pausa global diaria (5 pérdidas → hasta medianoche UTC)
  - Modo revisión (paper trading) por winrate bajo
  - Gestión de riesgo dinámica: tamaño de posición según régimen y ATR
  - Persistencia del estado en risk_state.json entre reinicios

Uso:
    from risk_manager import RiskManager
    rm = RiskManager()

    # Al cerrar una operación:
    rm.record_trade(mode="scalp", won=False)

    # Antes de abrir una operación:
    if rm.is_blocked(mode="scalp"):
        return  # no operar

    # Para calcular el tamaño de posición:
    size = rm.position_size(capital=100.0, sl_pct=0.004, regime="range")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from config import cfg

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# ESTADO PERSISTENTE
# ─────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    """
    Estado completo del gestor de riesgo.
    Se serializa a JSON y se carga al arrancar el bot.
    """

    # Contadores de pérdidas consecutivas por modo
    consecutive_losses: dict = field(default_factory=lambda: {
        "scalp": 0, "mediano": 0, "swing": 0
    })

    # Timestamp de fin de pausa por modo (unix). 0 = sin pausa.
    mode_pause_until: dict = field(default_factory=lambda: {
        "scalp": 0.0, "mediano": 0.0, "swing": 0.0
    })

    # Pérdidas totales del día actual
    daily_losses: int = 0

    # Timestamp de pausa global (0 = sin pausa global)
    global_pause_until: float = 0.0

    # Fecha del último reset diario (ISO 8601 UTC)
    last_reset_date: str = ""

    # Últimas N operaciones para calcular winrate
    # Cada elemento: {"won": bool, "ts": float}
    recent_trades: list = field(default_factory=list)

    # True si el bot está en modo revisión (paper trading)
    review_mode: bool = False

    # Timestamp de inicio del modo revisión (0 = no activo)
    review_mode_started: float = 0.0

    # Intentos fallidos de Nivel 3 hoy
    n3_failures_today: int = 0

    # True si N3 está desactivado por límite de intentos
    n3_disabled: bool = False


# ─────────────────────────────────────────────────────────────
# GESTOR DE RIESGO
# ─────────────────────────────────────────────────────────────

class RiskManager:
    """
    Gestiona todas las reglas de riesgo del bot.

    Los parámetros se leen desde config.cfg (risk y modes).
    El estado persiste en risk_state.json entre reinicios.
    """

    # Duración de pausa por modo en segundos según el TF principal
    # scalp=15m, mediano=1h, swing=4h; 20 velas cada uno
    _PAUSE_SECONDS = {
        "scalp":   20 * 15 * 60,    #  5 horas
        "mediano": 20 * 60 * 60,    # 20 horas
        "swing":   20 * 4 * 60 * 60, # 80 horas
    }

    def __init__(self, state_file: Optional[str] = None):
        self._state_file = state_file or cfg.risk.state_file
        self._state = RiskState()
        self._load()
        self._maybe_daily_reset()

    # ─────────────────────────────────────────────────────────
    # API PÚBLICA
    # ─────────────────────────────────────────────────────────

    def record_trade(
        self,
        mode: str,
        won: bool,
        score_pct: float = 0.0,
        level: int = 1,
    ) -> None:
        """
        Registra el resultado de una operación cerrada.

        Parámetros:
            mode        "scalp" | "mediano" | "swing"
            won         True si la operación fue ganadora
            score_pct   Puntaje de la señal como fracción del máximo (0.0-1.0)
            level       Nivel de señal (1, 2 o 3)
        """
        self._maybe_daily_reset()

        # Registrar en historial reciente
        self._state.recent_trades.append({
            "won": won, "ts": time.time(), "mode": mode
        })
        # Mantener solo las últimas N operaciones
        max_window = cfg.risk.winrate_window
        if len(self._state.recent_trades) > max_window:
            self._state.recent_trades = self._state.recent_trades[-max_window:]

        if not won:
            # Incrementar pérdida consecutiva del modo
            self._state.consecutive_losses[mode] = (
                self._state.consecutive_losses.get(mode, 0) + 1
            )
            self._state.daily_losses += 1

            # Registrar fallo de N3
            if level == 3:
                self._state.n3_failures_today += 1
                if self._state.n3_failures_today >= 2:
                    self._state.n3_disabled = True
                    logger.warning("[Risk] N3 desactivado: 2 intentos fallidos hoy.")

            # Activar pausa por modo si corresponde
            max_consec = cfg.risk.max_consecutive_losses_per_mode
            if self._state.consecutive_losses[mode] >= max_consec:
                pause_secs = self._PAUSE_SECONDS.get(mode, 3600)
                self._state.mode_pause_until[mode] = time.time() + pause_secs
                logger.warning(
                    f"[Risk] Modo '{mode}' pausado por {pause_secs/3600:.1f}h "
                    f"({max_consec} pérdidas consecutivas)."
                )

            # Activar pausa global si corresponde
            if self._state.daily_losses >= cfg.risk.max_daily_losses:
                midnight = self._next_midnight_utc()
                self._state.global_pause_until = midnight
                logger.warning(
                    f"[Risk] Pausa global activada hasta medianoche UTC "
                    f"({cfg.risk.max_daily_losses} pérdidas en el día)."
                )

        else:
            # Pérdida ganada: resetear contador consecutivo del modo
            self._state.consecutive_losses[mode] = 0

        # Verificar si hay que activar modo revisión
        self._check_review_mode()
        self._save()

    def is_blocked(
        self,
        mode: str,
        score_pct: float = 0.0,
        level: int = 1,
    ) -> bool:
        """
        Devuelve True si el modo está bloqueado y NO debe operar.

        Una señal N1 con score_pct >= 0.75 puede romper la pausa de modo
        (pero no la pausa global ni el modo revisión).

        Parámetros:
            mode        "scalp" | "mediano" | "swing"
            score_pct   Puntaje como fracción del máximo (0.0-1.0)
            level       Nivel de la señal actual
        """
        self._maybe_daily_reset()
        now = time.time()

        # Modo revisión bloquea todo
        if self._state.review_mode:
            if self._review_expired():
                self._exit_review_mode()
            else:
                return True

        # Pausa global bloquea todo
        if now < self._state.global_pause_until:
            # Señal N1 muy fuerte puede romper la pausa de modo pero NO la global
            return True

        # N3 desactivado
        if level == 3 and self._state.n3_disabled:
            return True

        # Pausa por modo
        pause_until = self._state.mode_pause_until.get(mode, 0.0)
        if now < pause_until:
            # Señal N1 override: score >= 75% del máximo puede romper la pausa
            override_level  = cfg.risk.override_pause_level
            override_score  = 0.75
            if level <= override_level and score_pct >= override_score:
                logger.info(
                    f"[Risk] Señal N{level} ({score_pct*100:.0f}%) rompe pausa "
                    f"del modo '{mode}'. Contador de pérdidas NO se resetea."
                )
                return False
            return True

        return False

    def is_review_mode(self) -> bool:
        """True si el bot está en modo revisión (paper trading)."""
        if self._state.review_mode and self._review_expired():
            self._exit_review_mode()
        return self._state.review_mode

    def position_size(
        self,
        capital: float,
        sl_pct: float,
        regime: str = "volatile",
        volatility_ratio: float = 1.0,
        confidence: float = 0.5,
    ) -> float:
        """
        Calcula el tamaño de posición en USDC según el régimen y la volatilidad.

        Fórmula:
            tamaño = (riesgo_por_trade / sl_pct) × capital

        El riesgo_por_trade varía según el régimen:
            bull_trend / bear_trend : 1.0% del capital
            range                   : 0.5% del capital
            volatile                : 0.3% del capital

        Si la volatilidad_ratio > 1.5 (expansión) → reduce tamaño un 20%
        Si la confianza del régimen < 0.5 → reduce tamaño un 20% adicional

        Parámetros:
            capital           Capital disponible en USDC
            sl_pct            Distancia al Stop Loss como fracción del precio
                              (ej: 0.004 = 0.4%)
            regime            Régimen actual de mercado
            volatility_ratio  ATR_actual / ATR_promedio. > 1.5 = alta volatilidad
            confidence        Confianza del detector de régimen (0.0-1.0)

        Devuelve:
            Tamaño de la posición en USDC (nunca > capital)
        """
        # Riesgo base por régimen
        risk_map = {
            "bull_trend":  0.010,
            "bear_trend":  0.010,
            "range":       0.005,
            "volatile":    0.003,
        }
        risk_pct = risk_map.get(regime, 0.003)

        # Ajuste por volatilidad alta
        if volatility_ratio > 1.5:
            risk_pct *= 0.8

        # Ajuste por confianza baja del régimen
        if confidence < 0.5:
            risk_pct *= 0.8

        # Calcular tamaño
        if sl_pct <= 0:
            return 0.0

        size = (risk_pct / sl_pct) * capital

        # Limitar al capital disponible
        return min(size, capital)

    def get_status(self) -> dict:
        """
        Devuelve el estado actual del risk manager.
        Usado por live_monitor y Flask para mostrar el estado.
        """
        self._maybe_daily_reset()
        now = time.time()

        winrate = self._calc_winrate()

        pauses = {}
        for mode in ("scalp", "mediano", "swing"):
            until = self._state.mode_pause_until.get(mode, 0.0)
            if until > now:
                remaining_min = int((until - now) / 60)
                pauses[mode] = f"pausado {remaining_min}min"
            else:
                pauses[mode] = "activo"

        global_paused = now < self._state.global_pause_until
        global_remaining = (
            int((self._state.global_pause_until - now) / 60)
            if global_paused else 0
        )

        return {
            "review_mode":        self._state.review_mode,
            "global_paused":      global_paused,
            "global_remaining_min": global_remaining,
            "mode_status":        pauses,
            "daily_losses":       self._state.daily_losses,
            "winrate":            winrate,
            "winrate_window":     len(self._state.recent_trades),
            "n3_disabled":        self._state.n3_disabled,
            "consecutive_losses": dict(self._state.consecutive_losses),
        }

    def reset_mode_pause(self, mode: str) -> None:
        """Levanta manualmente la pausa de un modo (comando manual)."""
        self._state.mode_pause_until[mode] = 0.0
        self._save()
        logger.info(f"[Risk] Pausa del modo '{mode}' levantada manualmente.")

    def reset_global_pause(self) -> None:
        """Levanta manualmente la pausa global (comando manual)."""
        self._state.global_pause_until = 0.0
        self._save()
        logger.info("[Risk] Pausa global levantada manualmente.")

    # ─────────────────────────────────────────────────────────
    # LÓGICA INTERNA
    # ─────────────────────────────────────────────────────────

    def _maybe_daily_reset(self) -> None:
        """
        Resetea todos los contadores diarios a medianoche UTC.
        Se llama al inicio de cada operación pública.
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._state.last_reset_date == today:
            return

        logger.info(f"[Risk] Reset diario. Fecha anterior: {self._state.last_reset_date}")
        self._state.daily_losses        = 0
        self._state.global_pause_until  = 0.0
        self._state.n3_failures_today   = 0
        self._state.n3_disabled         = False
        self._state.last_reset_date     = today

        # Resetear contadores consecutivos
        for mode in self._state.consecutive_losses:
            self._state.consecutive_losses[mode] = 0

        # Levantar pausas de modo que hayan expirado
        now = time.time()
        for mode in self._state.mode_pause_until:
            if self._state.mode_pause_until[mode] <= now:
                self._state.mode_pause_until[mode] = 0.0

        self._save()

    def _check_review_mode(self) -> None:
        """
        Activa el modo revisión si el winrate de las últimas N operaciones
        cae por debajo del umbral configurado.
        """
        if self._state.review_mode:
            return

        winrate = self._calc_winrate()
        window  = len(self._state.recent_trades)
        required_window = cfg.risk.winrate_window

        if window < required_window:
            return  # no hay suficientes datos aún

        threshold = cfg.risk.review_mode_winrate_threshold
        if winrate < threshold:
            self._state.review_mode         = True
            self._state.review_mode_started = time.time()
            logger.warning(
                f"[Risk] MODO REVISIÓN ACTIVADO. "
                f"Winrate={winrate*100:.1f}% < {threshold*100:.0f}% "
                f"en las últimas {window} operaciones. "
                f"Duración: {cfg.risk.review_mode_days} días."
            )
            # Aquí se enviará alerta por Telegram cuando esté implementado

    def _review_expired(self) -> bool:
        """True si el período de modo revisión ya terminó."""
        if not self._state.review_mode_started:
            return True
        days_elapsed = (
            time.time() - self._state.review_mode_started
        ) / 86400
        return days_elapsed >= cfg.risk.review_mode_days

    def _exit_review_mode(self) -> None:
        """Sale del modo revisión al terminar el período."""
        self._state.review_mode         = False
        self._state.review_mode_started = 0.0
        logger.info("[Risk] Período de revisión terminado. Bot vuelve a operar.")
        self._save()

    def _calc_winrate(self) -> float:
        """Calcula el winrate de las operaciones recientes."""
        trades = self._state.recent_trades
        if not trades:
            return 1.0
        wins = sum(1 for t in trades if t.get("won", False))
        return wins / len(trades)

    @staticmethod
    def _next_midnight_utc() -> float:
        """Timestamp unix de la próxima medianoche UTC."""
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # Si ya pasó la medianoche de hoy, devolver la de mañana
        from datetime import timedelta
        next_midnight = midnight + timedelta(days=1)
        return next_midnight.timestamp()

    # ─────────────────────────────────────────────────────────
    # PERSISTENCIA
    # ─────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Guarda el estado en risk_state.json."""
        try:
            with open(self._state_file, "w") as f:
                json.dump(asdict(self._state), f, indent=2)
        except Exception as e:
            logger.error(f"[Risk] Error guardando estado: {e}")

    def _load(self) -> None:
        """Carga el estado desde risk_state.json si existe."""
        if not os.path.exists(self._state_file):
            logger.info("[Risk] Sin estado previo. Iniciando fresco.")
            return
        try:
            with open(self._state_file) as f:
                data = json.load(f)
            # Actualizar campos del estado con los datos cargados
            for key, value in data.items():
                if hasattr(self._state, key):
                    setattr(self._state, key, value)
            logger.info(f"[Risk] Estado cargado desde {self._state_file}")
        except Exception as e:
            logger.error(f"[Risk] Error cargando estado: {e}. Iniciando fresco.")


# ─────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile

    logging.basicConfig(level=logging.INFO)
    print("=" * 55)
    print("  TEST risk_manager.py")
    print("=" * 55)

    # Usar archivo temporal para no contaminar el estado real
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp_file = f.name

    rm = RiskManager(state_file=tmp_file)

    # ── Test 1: Sin pérdidas → no bloqueado ──────────────────
    print("\n[Test 1] Sin pérdidas → not blocked")
    assert not rm.is_blocked("scalp")
    print("  ✓ scalp no bloqueado sin pérdidas")

    # ── Test 2: 3 pérdidas consecutivas → pausa de modo ──────
    print("\n[Test 2] 3 pérdidas consecutivas → pausa scalp")
    for _ in range(3):
        rm.record_trade("scalp", won=False)
    assert rm.is_blocked("scalp"), "scalp debe estar bloqueado"
    assert not rm.is_blocked("mediano"), "mediano no debe estar bloqueado"
    assert not rm.is_blocked("swing"), "swing no debe estar bloqueado"
    print("  ✓ scalp pausado, mediano y swing siguen activos")

    # ── Test 3: N1 con score alto rompe la pausa ──────────────
    print("\n[Test 3] N1 score=80% rompe pausa de modo")
    assert not rm.is_blocked("scalp", score_pct=0.80, level=1)
    print("  ✓ N1 80% rompe pausa correctamente")

    # ── Test 4: N2 no rompe la pausa ─────────────────────────
    print("\n[Test 4] N2 no rompe pausa")
    assert rm.is_blocked("scalp", score_pct=0.80, level=2)
    print("  ✓ N2 no rompe pausa (solo N1 puede)")

    # ── Test 5: Reset manual de pausa ────────────────────────
    print("\n[Test 5] Reset manual de pausa")
    rm.reset_mode_pause("scalp")
    assert not rm.is_blocked("scalp")
    print("  ✓ Pausa levantada manualmente")

    # ── Test 6: 5 pérdidas → pausa global ────────────────────
    print("\n[Test 6] 5 pérdidas en el día → pausa global")
    rm2 = RiskManager(state_file=tmp_file + "2")
    for _ in range(5):
        rm2.record_trade("scalp", won=False, level=1)
    assert rm2.is_blocked("scalp"), "global pause debe bloquear todo"
    assert rm2.is_blocked("swing"), "global pause debe bloquear swing también"
    print("  ✓ Pausa global activa tras 5 pérdidas")

    # ── Test 7: N3 se desactiva tras 2 fallos ─────────────────
    print("\n[Test 7] N3 se desactiva tras 2 fallos")
    rm3 = RiskManager(state_file=tmp_file + "3")
    rm3.record_trade("scalp", won=False, level=3)
    rm3.record_trade("scalp", won=False, level=3)
    assert rm3.is_blocked("scalp", level=3)
    print("  ✓ N3 desactivado tras 2 fallos")

    # ── Test 8: position_size por régimen ─────────────────────
    print("\n[Test 8] position_size por régimen")
    for regime, expected_risk in [
        ("bull_trend", 0.010),
        ("range",      0.005),
        ("volatile",   0.003),
    ]:
        size = rm.position_size(
            capital=100.0, sl_pct=0.004, regime=regime, volatility_ratio=1.0
        )
        expected = min((expected_risk / 0.004) * 100.0, 100.0)
        print(f"  {regime}: size={size:.2f} USDC (esperado ~{expected:.2f})")
        assert abs(size - expected) < 0.01, f"Diff: {size} vs {expected}"
    print("  ✓ Todos los tamaños correctos")

    # ── Test 9: Modo revisión por winrate bajo ─────────────────
    print("\n[Test 9] Modo revisión por winrate bajo")
    rm4 = RiskManager(state_file=tmp_file + "4")
    # Simular 20 operaciones con winrate de 30% (< 38%)
    for i in range(20):
        rm4.record_trade("scalp", won=(i % 10 == 0))  # 2 de 20 = 10%
    status = rm4.get_status()
    print(f"  Winrate: {status['winrate']*100:.0f}%")
    print(f"  Review mode: {status['review_mode']}")
    assert status["review_mode"], "Debe activarse modo revisión"
    assert rm4.is_blocked("scalp"), "Revisión debe bloquear"
    print("  ✓ Modo revisión activado correctamente")

    # ── Test 10: get_status ────────────────────────────────────
    print("\n[Test 10] get_status")
    status = rm.get_status()
    assert "mode_status" in status
    assert "daily_losses" in status
    assert "winrate" in status
    print(f"  Status: {status}")
    print("  ✓ get_status devuelve todos los campos")

    # Limpiar archivos temporales
    for f in [tmp_file, tmp_file+"2", tmp_file+"3", tmp_file+"4"]:
        try:
            os.remove(f)
        except Exception:
            pass

    print("\n" + "=" * 55)
    print("  Todos los tests pasaron ✓")
    print("=" * 55)
