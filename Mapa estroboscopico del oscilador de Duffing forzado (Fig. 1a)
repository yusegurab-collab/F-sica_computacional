"""
Ejercicio 2 - Mapa estroboscopico del oscilador de Duffing forzado (Fig. 1a)

"""
import numpy as np
import os
import matplotlib.pyplot as plt

# --- Parametros fisicos -------------------------------------------------
DELTA, ALPHA, BETA, GAMMA, OMEGA = 0.02, -1.0, 5.0, 8.0, 0.5

# --- Parametros numericos -----------------------------------------------
N_IC = 400        # numero de condiciones iniciales
N_TRANS = 300     # periodos descartados (transitorio)
N_PLOT = 300      # periodos graficados
SPP = 200         # pasos RK4 por periodo (h = T/SPP)
SEMILLA = 1

T = 2 * np.pi / OMEGA
h = T / SPP


def f(t, x, v):
    """Lado derecho del sistema de primer orden: x' = v, v' = ..."""
    dx = v
    dv = -DELTA * v + ALPHA * x - BETA * x**3 + GAMMA * np.cos(OMEGA * t)
    return dx, dv


def rk4_paso(t, x, v):
    k1x, k1v = f(t, x, v)
    k2x, k2v = f(t + h / 2, x + h / 2 * k1x, v + h / 2 * k1v)
    k3x, k3v = f(t + h / 2, x + h / 2 * k2x, v + h / 2 * k2v)
    k4x, k4v = f(t + h, x + h * k3x, v + h * k3v)
    x = x + h / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
    v = v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return x, v


def main():
    print("Calculando... (puede tardar unos segundos)")
    rng = np.random.default_rng(SEMILLA)
    x = rng.uniform(-3, 3, N_IC)
    v = rng.uniform(-3, 3, N_IC)

    t = 0.0
    X, V = [], []
    for n in range(N_TRANS + N_PLOT):
        for _ in range(SPP):          # un periodo completo del forzamiento
            x, v = rk4_paso(t, x, v)
            t += h
        if n >= N_TRANS:              # mapa estroboscopico: t = 2*pi*n/omega
            X.append(x.copy())
            V.append(v.copy())
    X, V = np.array(X), np.array(V)
    print(f"x en [{X.min():.2f}, {X.max():.2f}],  xdot en [{V.min():.2f}, {V.max():.2f}]")

    fig, ax = plt.subplots(figsize=(5, 5))
    colores = plt.cm.rainbow(np.linspace(0, 1, N_IC))
    for j in range(N_IC):
        ax.plot(X[:, j], V[:, j], ".", ms=1.2, color=colores[j])
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\dot{x}$")
    ax.set_title(r"Ej. 2: $\delta=0.02$ (ecuación literal, 400 CI)", fontsize=10)
    ax.set_xlim(1.05, 1.75)
    fig.tight_layout()
    fig.savefig("fig_ej2.png", dpi=170)
    print("Figura guardada en:", os.path.abspath("fig_ej2.png"))
    plt.show()   # abre la ventana con la grafica


if __name__ == "__main__":
    main()
