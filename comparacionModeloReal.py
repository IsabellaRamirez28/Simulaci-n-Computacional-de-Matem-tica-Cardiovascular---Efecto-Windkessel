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
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

class SegmentoArterial0D:
    """Clase que modela un vaso sanguíneo usando parámetros concentrados (0D)."""

    def __init__(self, R=1.0, L=0.01, C=1.70, P_ext=5.0):
        # Parámetros físicos del vaso
        self.R = R          # Resistencia viscosa
        self.L = L          # Inercia del fluido
        self.C = C          # Distensibilidad de la pared (Capacitancia)
        self.P_ext = P_ext  # Presión de salida o lecho capilar

    def flujo_entrada(self, t):
        """Simula un perfil de flujo pulsátil (sístole y diástole)."""
        if (t % 1) < 0.3:  # Fase sistólica (aprox 30% del ciclo cardíaco)
            return 20 * np.sin(2 * np.pi * t)**2
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
    # Instanciamos un segmento (ej. aorta) con valores representativos
    aorta = SegmentoArterial0D(R=0.8, L=0.015, C=0.008, P_ext=80.0)

    # Configuración del tiempo de simulación y condiciones iniciales
    tiempo_total = 3.0 # segundos (simularemos 3 ciclos cardíacos aprox)
    t_eval = np.linspace(0, tiempo_total, 1000)
    condiciones_iniciales = [90.0, 0.0] # [Presión inicial (P1), Flujo de salida inicial (Qout)]

    # Ejecución del solver numérico
    solucion = aorta.simular([0, tiempo_total], condiciones_iniciales, t_eval)

    # =====================================================================
    # AQUÍ ESTÁ LA PARTE DE COMPARACIÓN CON UNA IMAGEN REAL:
    # =====================================================================
    # Señal del modelo
    t_modelo = solucion.t
    P_modelo = solucion.y[0]

    # ---------------------------------------------
    # CARGAR SEÑAL REAL
    # ---------------------------------------------

    df = pd.read_csv(
        "presion_real.csv",
        sep=';',
        decimal=',',
        header=None,
        names=["tiempo", "presion"]
    )
    df = df.sort_values("tiempo")
    df = df.drop_duplicates()

    # Extraer arrays
    t_real = df["tiempo"].values
    P_real = df["presion"].values

    # ---------------------------------------------
    # INTERPOLAR LA SEÑAL REAL
    # ---------------------------------------------

    # Esto hace que ambas señales tengan
    # exactamente los mismos tiempos.

    P_real_interp = np.interp(
        t_modelo,
        t_real,
        P_real
    )

    # ---------------------------------------------
    # COMPARACIÓN MATEMÁTICA
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
    # RESULTADOS NUMÉRICOS
    # ---------------------------------------------

    print("\n===== RESULTADOS =====")

    print(f"MSE = {mse:.4f}")

    print(f"Correlación = {corr:.4f}")


    # ---------------------------------------------
    # VISUALIZACIÓN
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
    plt.tight_layout()
    plt.show()