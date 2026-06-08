"""
live_monitor.py
===============
Loop en tiempo real. Muestra lo que el bot estaría "pensando"
cada ciclo sin ejecutar ninguna orden.

Útil para:
- Entender el comportamiento del sistema antes de operar en vivo
- Debuggear señales inesperadas
- Ver cómo reacciona el sistema a movimientos del mercado

Correr con:
    py -3.12 live_monitor.py
"""

import os
import sys
import time
import threading
from datetime import datetime, timezone

from market_feed import create_feed
from indicators import IndicatorCalculator, analyze_macro_trend
from scoring import StrategyEvaluator, THRESHOLDS

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────

SYMBOL       = "BTCUSDC"
TESTNET      = True
CYCLE_SEC    = 60       # cada cuántos segundos evaluar
ACTIVE_MODES = ["scalp", "mediano", "swing"]

# Colores ANSI para terminal
class C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    WHITE  = "\033[97m"

def _clr(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


# ─────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────

def clear_screen():
    pass  # desactivado para mantener historial visible

def print_header(price: float, cycle: int):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    env = "TESTNET" if TESTNET else "PRODUCCIÓN"
    print(_clr("═" * 65, C.BLUE))
    print(_clr(f"  BOT MONITOR  |  {SYMBOL}  |  {env}  |  Ciclo #{cycle}", C.BOLD))
    print(_clr(f"  Precio: ", C.WHITE) +
          _clr(f"${price:,.2f}", C.CYAN) +
          _clr(f"  |  {now}", C.GRAY))
    print(_clr("═" * 65, C.BLUE))

def print_macro(macro):
    def trend_str(t):
        if t == "bullish":
            return _clr("▲ ALCISTA", C.GREEN)
        elif t == "bearish":
            return _clr("▼ BAJISTA", C.RED)
        return _clr("─ NEUTRAL", C.GRAY)

    print(f"\n  {_clr('MACRO', C.BOLD)}  "
          f"1W: {trend_str(macro.trend_1w)}  "
          f"1D: {trend_str(macro.trend_1d)}  "
          f"4H: {trend_str(macro.trend_4h)}")

    if macro.divergence_daily_weekly:
        print(_clr("  ⚠ DIVERGENCIA 1D/1W — solo scalps, leverage reducido", C.YELLOW))
    print()

def print_indicators_summary(ivs: dict):
    """Muestra un resumen compacto de los indicadores clave."""
    rows = [
        ("TF",    "RSI",   "EMAs↑", "EMAs↓", "MACD H",  "Squeeze",    "Vol x", "Lat"),
    ]
    for tf in ["5m", "15m", "1h", "4h"]:
        iv = ivs.get(tf)
        if iv is None:
            continue
        sqz = "OFF✓" if iv.sqz_off else ("ON·" if iv.sqz_on else "---")
        rows.append((
            tf,
            f"{iv.rsi:.0f}",
            f"{iv.emas_aligned_bullish}/5",
            f"{iv.emas_aligned_bearish}/5",
            f"{iv.macd_histogram:+.2f}",
            sqz,
            f"{iv.volume_ratio:.2f}",
            f"{iv.lateralization_score:.2f}",
        ))

    # Header
    print(_clr("  " + "─" * 61, C.GRAY))
    h = rows[0]
    print(_clr(f"  {h[0]:<5} {h[1]:<6} {h[2]:<7} {h[3]:<7} "
               f"{h[4]:<10} {h[5]:<8} {h[6]:<7} {h[7]}", C.GRAY))
    print(_clr("  " + "─" * 61, C.GRAY))

    for r in rows[1:]:
        tf, rsi, bull, bear, macd_h, sqz, vol, lat = r
        # Color del RSI
        rsi_val = float(rsi)
        if rsi_val < 30:
            rsi_c = C.GREEN
        elif rsi_val > 70:
            rsi_c = C.RED
        else:
            rsi_c = C.WHITE

        print(f"  {_clr(tf, C.CYAN):<5} "
              f"{_clr(rsi, rsi_c):<15} "
              f"{_clr(bull, C.GREEN):<16} "
              f"{_clr(bear, C.RED):<15} "
              f"{macd_h:<10} "
              f"{_clr(sqz, C.YELLOW):<8} "
              f"{vol:<7} "
              f"{lat}")
    print()

def print_scores(all_results: list):
    """Muestra la tabla de puntajes de todos los modos."""
    print(_clr("  PUNTAJES POR MODO:", C.BOLD))
    print(_clr("  " + "─" * 61, C.GRAY))

    for r in all_results:
        pct = r.normalized * 100
        bar_len = int(pct / 3)
        bar = "█" * bar_len + "░" * (33 - bar_len)

        if r.should_trade:
            color = C.GREEN
            status = f"✓ NIVEL {r.signal_level}  x{r.leverage}"
        elif r.blocked_reasons:
            color = C.GRAY
            status = f"⊘ {r.blocked_reasons[0][:30]}"
        else:
            color = C.GRAY
            status = "✗"

        direction_str = _clr("SHORT", C.RED) if r.direction == "short" else _clr("LONG ", C.GREEN)
        mode_str      = f"{r.mode:<8}"
        pct_str       = _clr(f"{pct:>5.1f}%", color)

        print(f"  {direction_str} {_clr(mode_str, C.CYAN)} "
              f"{pct_str} {_clr(bar, color)} {_clr(status, color)}")

    print()

def print_best(best):
    """Muestra la mejor oportunidad con desglose."""
    if best is None:
        print(_clr("  Sin señales operables en este ciclo.", C.GRAY))
        # Mostrar cuánto falta para el umbral mínimo
        print(_clr(f"  (Umbral mínimo nivel 3: {THRESHOLDS[3]*100:.0f}% del puntaje máximo)", C.GRAY))
        print()
        return

    dir_color = C.GREEN if best.direction == "long" else C.RED
    dir_str   = best.direction.upper()

    print(_clr(f"  ★ SEÑAL OPERABLE: {dir_str} {best.mode.upper()} "
               f"— Nivel {best.signal_level} — x{best.leverage}", dir_color))
    print(f"  Puntaje: {_clr(f'{best.normalized*100:.1f}%', dir_color)} "
          f"({best.total:.0f}/{best.maximum_possible:.0f} puntos)")
    print(f"  TP aprox: {_clr(f'+{best.approx_tp_pct:.2f}%', C.GREEN)}  "
          f"SL aprox: {_clr(f'-{best.approx_sl_pct:.2f}%', C.RED)}")

    # Top 5 contribuyentes
    top = sorted(best.breakdown.items(), key=lambda x: -x[1])[:5]
    contributors = "  Top señales: " + " | ".join(
        _clr(f"{k}={v:+.0f}", C.GREEN if v > 0 else C.RED)
        for k, v in top if v != 0
    )
    print(contributors)
    print()

def print_footer(next_in: int):
    print(_clr("─" * 65, C.GRAY))
    print(_clr(f"  Próxima evaluación en {next_in}s  |  "
               f"Ctrl+C para salir", C.GRAY))


# ─────────────────────────────────────────────
# HISTORIAL DE SEÑALES
# ─────────────────────────────────────────────

signal_history = []   # lista de (timestamp, direction, mode, pct, level)
MAX_HISTORY = 8

def update_history(best):
    if best and best.should_trade:
        signal_history.append((
            datetime.now(timezone.utc).strftime("%H:%M"),
            best.direction,
            best.mode,
            best.normalized * 100,
            best.signal_level,
        ))
        if len(signal_history) > MAX_HISTORY:
            signal_history.pop(0)

def print_history():
    if not signal_history:
        return
    print(_clr("  HISTORIAL DE SEÑALES (últimas sesión):", C.BOLD))
    for ts, direction, mode, pct, level in reversed(signal_history):
        color = C.GREEN if direction == "long" else C.RED
        print(f"  {_clr(ts, C.GRAY)}  "
              f"{_clr(direction.upper(), color):<14} "
              f"{_clr(mode, C.CYAN):<10} "
              f"{_clr(f'{pct:.1f}%', color)}  "
              f"N{level}")
    print()


# ─────────────────────────────────────────────
# LOOP PRINCIPAL
# ─────────────────────────────────────────────

def run():
    feed      = create_feed('live', symbol=SYMBOL, testnet=TESTNET)
    calc      = IndicatorCalculator()
    evaluator = StrategyEvaluator()

    feed.start()
    print(_clr("\nConectado. Primera evaluación en curso...\n", C.CYAN))
    time.sleep(2)

    cycle = 0

    try:
        while True:
            cycle += 1
            t0 = time.monotonic()

            # ── Calcular ──────────────────────────────────────────────
            snap  = feed.get_snapshot()
            ivs   = calc.calculate(snap)
            macro = analyze_macro_trend(ivs)

            all_results = evaluator.evaluate_all(ivs, macro, ACTIVE_MODES)
            best        = evaluator.evaluate(ivs, macro, ACTIVE_MODES)

            update_history(best)

            elapsed = (time.monotonic() - t0) * 1000

            # ── Display ───────────────────────────────────────────────
            clear_screen()
            print_header(snap.current_close, cycle)
            print_macro(macro)
            print_indicators_summary(ivs)
            print_scores(all_results)
            print_best(best)
            print_history()

            print(_clr(f"  Cálculo: {elapsed:.0f}ms  |  "
                       f"Latencia feed: {snap.feed_latency_ms:.1f}ms", C.GRAY))

            # ── Esperar hasta el próximo ciclo ────────────────────────
            wait = CYCLE_SEC
            while wait > 0:
                sys.stdout.write(f"\r  Próxima evaluación en {wait}s  |  Ctrl+C para salir   ")
                sys.stdout.flush()
                time.sleep(1)
                wait -= 1
            print()  # nueva línea antes del próximo ciclo

    except KeyboardInterrupt:
        print(_clr("\n\nMonitor detenido por el usuario.", C.YELLOW))
    finally:
        feed.stop()


if __name__ == '__main__':
    # Habilitar colores ANSI en Windows
    if sys.platform == 'win32':
        os.system('color')
        # Para terminales modernas en Windows
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    run()
