# ============================================================
# CALCULADORA DE GESTIÓN DE RIESGO
# Uso: Antes de CUALQUIER operación, correr este script.
# Si los números no cierran, NO se entra al trade.
# ============================================================

def calcular_trade(
    capital_total: float,
    precio_entrada: float,
    precio_stop_loss: float,
    precio_take_profit: float,
    porcentaje_riesgo: float = 1.0  # Por defecto, arriesgamos 1% del capital
):
    """
    Calcula todos los parámetros de un trade antes de ejecutarlo.
    
    Parámetros:
    - capital_total: Tu capital en USDT (ej: 1000)
    - precio_entrada: Precio al que vas a comprar/vender (ej: 100000)
    - precio_stop_loss: Precio donde cortás la pérdida (ej: 98000)
    - precio_take_profit: Precio objetivo de ganancia (ej: 104000)
    - porcentaje_riesgo: Qué % del capital arriesgás. Nunca subir de 2.
    """
    
    print("\n" + "="*50)
    print("      ANÁLISIS PRE-TRADE")
    print("="*50)
    
    # --- CÁLCULO 1: Dinero en riesgo ---
    riesgo_en_usdt = capital_total * (porcentaje_riesgo / 100)
    print(f"\n💰 Capital total:          {capital_total:.2f} USDT")
    print(f"⚠️  Riesgo aceptado ({porcentaje_riesgo}%):  {riesgo_en_usdt:.2f} USDT")
    
    # --- CÁLCULO 2: Distancia porcentual al Stop Loss ---
    distancia_sl = abs(precio_entrada - precio_stop_loss) / precio_entrada
    print(f"\n📍 Precio de entrada:      {precio_entrada:.2f}")
    print(f"🛑 Stop Loss:              {precio_stop_loss:.2f} ({distancia_sl*100:.2f}% de distancia)")
    
    # --- CÁLCULO 3: Tamaño de posición correcto ---
    # Esta es la fórmula clave: cuánto dinero total necesitamos poner
    # para que si el precio llega al SL, perdamos EXACTAMENTE nuestro riesgo máximo
    tamano_posicion = riesgo_en_usdt / distancia_sl
    print(f"\n📊 Tamaño de posición:     {tamano_posicion:.2f} USDT")
    
    # --- CÁLCULO 4: Apalancamiento necesario ---
    apalancamiento = tamano_posicion / capital_total
    print(f"⚡ Apalancamiento:         {apalancamiento:.2f}x")
    
    # --- CÁLCULO 5: Ratio Riesgo/Beneficio ---
    ganancia_potencial = abs(precio_take_profit - precio_entrada) / precio_entrada
    rr_ratio = ganancia_potencial / distancia_sl
    ganancia_en_usdt = riesgo_en_usdt * rr_ratio
    
    print(f"\n🎯 Take Profit:            {precio_take_profit:.2f}")
    print(f"💵 Ganancia potencial:     {ganancia_en_usdt:.2f} USDT")
    print(f"📐 Ratio Riesgo/Beneficio: 1:{rr_ratio:.2f}")
    
    # --- VEREDICTO FINAL ---
    print("\n" + "-"*50)
    print("VEREDICTO:")
    
    errores = []
    
    if apalancamiento > 10:
        errores.append(f"❌ Apalancamiento {apalancamiento:.1f}x es demasiado alto. Ajustá el Stop Loss.")
    
    if rr_ratio < 2:
        errores.append(f"❌ Ratio 1:{rr_ratio:.2f} es insuficiente. Necesitás mínimo 1:2.")
    
    if distancia_sl < 0.005:
        errores.append(f"❌ Stop Loss demasiado ajustado ({distancia_sl*100:.2f}%). Riesgo de barrido.")
    
    if not errores:
        print("✅ TRADE VÁLIDO - Los números cierran matemáticamente.")
        print(f"   Entrá con {tamano_posicion:.2f} USDT a {apalancamiento:.1f}x de apalancamiento.")
    else:
        print("🚫 TRADE RECHAZADO - Motivos:")
        for error in errores:
            print(f"   {error}")
    
    print("="*50 + "\n")
    
    return {
        "valido": len(errores) == 0,
        "riesgo_usdt": riesgo_en_usdt,
        "tamano_posicion": tamano_posicion,
        "apalancamiento": apalancamiento,
        "rr_ratio": rr_ratio,
        "ganancia_potencial_usdt": ganancia_en_usdt
    }


# ============================================================
# ZONA DE PRUEBA - Modificá estos números con tu trade real
# ============================================================

if __name__ == "__main__":
    
    # Ejemplo 1: Trade conservador en BTC
    calcular_trade(
        capital_total=1000,
        precio_entrada=100000,
        precio_stop_loss=97000,    # SL al 3% abajo
        precio_take_profit=106000  # TP al 6% arriba → Ratio 1:2
    )
    
    # Ejemplo 2: Trade agresivo (para que veas cómo el sistema lo rechaza)
    calcular_trade(
        capital_total=1000,
        precio_entrada=100000,
        precio_stop_loss=99500,    # SL muy ajustado, 0.5%
        precio_take_profit=100800, # TP insuficiente
        porcentaje_riesgo=2.0
    )
