import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

R = 1.0
L = 0.01
C = 0.005
P_ext = 80.0

def flujo_entrada(t):

    if (t % 1) < 0.3:
        return 20 * np.sin(2 * np.pi * t)**2

    return 0.0

def sistema(t, y):

    P, Q = y

    Qin = flujo_entrada(t)

    dP_dt = (Qin - Q) / C

    dQ_dt = (P - R*Q - P_ext) / L

    return [dP_dt, dQ_dt]

t_eval = np.linspace(0, 3, 1000)

sol = solve_ivp(
    sistema,
    [0, 3],
    [90, 0],
    t_eval=t_eval
)

t_modelo = sol.t
P_modelo = sol.y[0]

df = pd.read_csv(
    "presion_real.csv",
    sep=';',
    decimal=',',
    header=None,
    names=["tiempo", "presion"]
)

df = df.sort_values("tiempo")

t_real = df["tiempo"].values
P_real = df["presion"].values

P_real_interp = np.interp(
    t_modelo,
    t_real,
    P_real
)

mse = mean_squared_error(
    P_real_interp,
    P_modelo
)

corr, _ = pearsonr(
    P_real_interp,
    P_modelo
)

print("MSE =", mse)
print("Correlación =", corr)

plt.figure(figsize=(10,5))

plt.plot(
    t_modelo,
    P_modelo,
    label="Modelo Windkessel",
    linewidth=2,
    color='#b30000'
)

plt.plot(
    t_modelo,
    P_real_interp,
    '--',
    label="Registro real",
)

plt.xlabel("Tiempo (s)")
plt.ylabel("Presión (mmHg)")

plt.title("Comparación Modelo vs Registro Real")

plt.legend()
plt.grid(True)

plt.show()