"""
Simulación Computacional de Matemática Cardiovascular
Modelo de Parámetros Concentrados (0D) - Efecto Windkessel

Mentor: PhD. Andrés J. C. Vásquez
Grupo: Semillero de Computación Científica - AppliScience

Referencias Académicas:
[1] Formaggia, L., Quarteroni, A., & Veneziani, A. (Eds.). (2009).
    "Cardiovascular Mathematics: Modeling and simulation of the circulatory system".
    Springer Science & Business Media.
"""
import matplotlib
matplotlib.use('TkAgg')
import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

class SegmentoArterial0D:
    """Clase que modela un vaso sanguíneo usando parámetros concentrados (0D)."""

    def __init__(self, R=1.0, L=0.01, C=1.70, P_ext=5.0, Q_amp=20.0, t_shift=0.0):
        # Parámetros físicos del vaso
        self.R = R          # Resistencia viscosa
        self.L = L          # Inercia del fluido
        self.C = C          # Distensibilidad de la pared (Capacitancia)
        self.P_ext = P_ext  # Presión de salida o lecho capilar
        self.Q_amp = Q_amp  # Amplitud del flujo de entrada (controla la amplitud de la curva)
        self.t_shift = t_shift # Desfase temporal para alinear con la señal real

    def flujo_entrada(self, t):
        """Simula un perfil de flujo pulsátil (sístole y diástole)."""
        t_mod = (t - self.t_shift) % 1.0
        if t_mod < 0.3:  # Fase sistólica (aprox 30% del ciclo cardíaco)
            # El factor / 0.6 asegura que la onda senoidal empiece y termine en 0 de forma suave durante los 0.3s
            return self.Q_amp * np.sin(2 * np.pi * t_mod / 0.6)**2
        return 0.0         # Fase diastólica

    def sistema_dae(self, t, y):
        """Define el sistema de ecuaciones diferencial-algebraicas (DAE)."""
        P1, Qout = y  # Extraemos los valores actuales de Presión y Flujo
        Qin = self.flujo_entrada(t) # Calculamos cuánto flujo entra en este instante 't'

        # =====================================================================
        # AQUÍ ESTÁN LAS ECUACIONES DIFERENCIALES EXPLÍCITAS (DAE):
        # =====================================================================

        # 1. Ecuación de Conservación de Masa (Cambio de Presión en el tiempo)
        # Físicamente: dP/dt depende de la diferencia entre el flujo de entrada y salida,
        # amortiguada por la elasticidad de la arteria (Capacitancia C).
        dP1_dt = (Qin - Qout) / self.C

        # 2. Ecuación de Balance de Cantidad de Movimiento (Cambio de Flujo en el tiempo)
        # Físicamente: dQ/dt es impulsada por el gradiente de presión y frenada
        # por la resistencia viscosa (R), dependiendo de la inercia de la sangre (Inductancia L).
        dQout_dt = (P1 - self.R * Qout - self.P_ext) / self.L

        # =====================================================================

        # Retornamos las derivadas al integrador numérico (solve_ivp)
        return [dP1_dt, dQout_dt]

    def simular(self, t_span, y0, t_eval):
        """Resuelve el sistema utilizando el integrador de SciPy (Runge-Kutta 45)."""
        return solve_ivp(self.sistema_dae, t_span, y0, t_eval=t_eval, method='RK45')

# --- Bloque de Ejecución Principal para Google Colab ---
if __name__ == "__main__":
    # PASO 1: CONFIGURAR EL MODELO CON PARÁMETROS OPTIMIZADOS
    # Instanciamos un segmento (ej. aorta).
    # Ajustamos todos los parámetros físicos (R, L, C, P_ext, Q_amp) y el desfase temporal (t_shift)
    # para alinear perfectamente la curva generada con la señal real.
    aorta = SegmentoArterial0D(
        R=4.989, 
        L=0.0826, 
        C=0.0878, 
        P_ext=74.08, 
        Q_amp=24.46, 
        t_shift=-0.1476
    )

    # PASO 2: PREPARAR EL TIEMPO Y CONDICIONES INICIALES
    # Configuración del tiempo de simulación y condiciones iniciales
    tiempo_total = 3.0 # segundos (simularemos 3 ciclos cardíacos aprox)
    t_eval = np.linspace(0, tiempo_total, 1000)
    # Condiciones iniciales optimizadas para coincidir con la curva real desde el t=0
    condiciones_iniciales = [106.54, 31.06] # [Presión inicial (P1), Flujo de salida inicial (Qout)]

    # PASO 3: RESOLVER EL SISTEMA DE ECUACIONES (SIMULACIÓN)
    # Ejecución del solver numérico usando los parámetros establecidos
    solucion = aorta.simular([0, tiempo_total], condiciones_iniciales, t_eval)

    # =====================================================================
    # PASO 4: COMPARACIÓN CON UNA IMAGEN REAL
    # =====================================================================
    # Señal del modelo obtenida de la simulación
    t_modelo = solucion.t
    P_modelo = solucion.y[0]

    # ---------------------------------------------
    # PASO 4.1: CARGAR SEÑAL REAL
    # ---------------------------------------------

    df = pd.read_csv(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "presion_real.csv"),
        sep=';',
        decimal=',',
        header=None,
        names=["tiempo", "presion"]
    )
    df = df.sort_values("tiempo")
    df = df.drop_duplicates()

    # Extraer arrays de la base de datos real
    t_real = df["tiempo"].values
    P_real = df["presion"].values

    # ---------------------------------------------
    # PASO 4.2: INTERPOLAR LA SEÑAL REAL
    # ---------------------------------------------

    # Esto hace que ambas señales tengan
    # exactamente los mismos tiempos.

    P_real_interp = np.interp(
        t_modelo,
        t_real,
        P_real
    )

    # ---------------------------------------------
    # PASO 5: CÁLCULO DE MÉTRICAS MATEMÁTICAS (COMPARACIÓN)
    # ---------------------------------------------

    # Error cuadrático medio
    mse = mean_squared_error(
        P_real_interp,
        P_modelo
    )

    # Correlación
    corr, _ = pearsonr(
        P_real_interp,
        P_modelo
    )

    # ---------------------------------------------
    # PASO 6: MOSTRAR RESULTADOS NUMÉRICOS
    # ---------------------------------------------

    print("\n===== RESULTADOS =====")

    print(f"MSE = {mse:.4f}")

    print(f"Correlación = {corr:.4f}")


    # ---------------------------------------------
    # PASO 7: VISUALIZACIÓN GRÁFICA
    # ---------------------------------------------

    plt.figure(figsize=(12,6))
    # Señal del modelo
    plt.plot(
        t_modelo,
        P_modelo,
        label='Modelo Windkessel 0D',
        linewidth=2,
        color='darkred'
    )

    # Señal real
    plt.plot(
        t_modelo,
        P_real_interp,
        '--',
        label='Registro fisiológico real',
        linewidth=2,
        color='navy'
    )

    plt.title(
        'Comparación entre Modelo Matemático y Señal Real',
        fontsize=14
    )

    plt.xlabel('Tiempo (s)', fontsize=12)

    plt.ylabel('Presión arterial (mmHg)', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.legend()

    # Mostrar gráfica
    plt.show()