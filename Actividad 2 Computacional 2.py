import numpy as np
from numpy import sin, cos
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------------------------------
# System parameters
# -----------------------------------------------------
g = 9.81
l = 1.5       
m = 1.0       # Masa del péndulo
M = 2.0       # Masa del bloque
k = 15.0

# Relación de masas auxiliar para simplificar las ecuaciones
mu = M / m

# -----------------------------------------------------
# Initial conditions
# -----------------------------------------------------
# Estado inicial: [x, v, theta, omega]
x0 = 0.2                       
v0 = 0.0                       
th0 = np.radians(90)
ome0 = 0.0                     

# -----------------------------------------------------
# method parameters
# -----------------------------------------------------
tmax = 20
dt = 0.01
STRIDE = 2

# -----------------------------------------------------
# Dinámica: péndulo acoplado a un bloque en movimiento 
# -----------------------------------------------------
def dyn(t, y):
    x, v, th, w = y
    
    sin_t = sin(th)
    cos_t = cos(th)
    den = mu + sin_t**2
    
    # Ecuaciones de movimiento (1) del documento de la Actividad 2
    a1 = ((g * cos_t + l * w**2) * sin_t - (k / m) * x) / den
    a2 = -(1 / l) * (g * (1 + mu) * sin_t + cos_t * (l * w**2 * sin_t - (k / m) * x)) / den
    
    return np.array([v, a1, w, a2])

# -----------------------------------------------------
# Fourth-order Runge-Kutta method
# -----------------------------------------------------
def rk4(f, t, y, h):
    k1 = h * f(t, y)
    k2 = h * f(t + h/2, y + k1/2)
    k3 = h * f(t + h/2, y + k2/2)
    k4 = h * f(t + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

# -----------------------------------------------------
# Integration using RK4
# -----------------------------------------------------
n = int(tmax / dt)
t = np.linspace(0, n * dt, n + 1)
y = np.empty((n + 1, 4))
y[0] = np.array([x0, v0, th0, ome0])

for i in range(n):
    y[i + 1] = rk4(dyn, t[i], y[i], dt)

# -----------------------------------------------------
# Separate variables after integration
# -----------------------------------------------------
x_bloque = y[:, 0]
v_bloque = y[:, 1]
th = y[:, 2]
ome = y[:, 3]

# -----------------------------------------------------
# Kinematics (Posiciones cartesianas para animación)
# -----------------------------------------------------
# El bloque se mueve solo en el eje X (Y = 0)
x_M = x_bloque
y_M = np.zeros_like(x_bloque)

# La masa m se mide de forma relativa desde la posición del bloque
x_m = x_M + l * sin(th)
y_m = y_M - l * cos(th)

# -----------------------------------------------------
# Figure & Visual Elements Setup
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
l_max = max(x_M) + l + 0.5
l_min = min(x_M) - l - 0.5
ax.set(xlim=(l_min - 1.0, l_max), 
       ylim=(-l - 0.5, 1.0), 
       aspect="equal", 
       title="Péndulo acoplado a un bloque (RK4)")
ax.title.set_fontsize(16)
ax.grid(alpha=0.3)

# --- NUEVO: Dimensiones del bloque rectangular ---
b_width = 0.4   # Ancho del bloque
b_height = 0.3  # Alto del bloque

# Crear el rectángulo del bloque (centrado horizontalmente en x_M, verticalmente en y=0)
patch_bloque = plt.Rectangle((x_M[0] - b_width/2, -b_height/2), b_width, b_height, 
                             fc='gray', ec='black', lw=1.5, zorder=3)
ax.add_patch(patch_bloque)

# --- NUEVO: Pared fija izquierda (origen del resorte) ---
pared_x = l_min - 0.8
ax.axvline(x=pared_x, color='black', lw=3)

# --- NUEVO: Línea del resorte ---
line_resorte, = ax.plot([], [], "-", lw=1.5, color="darkorange", zorder=2)

# Elementos del péndulo originales
line_pendulo, = ax.plot([], [], "o-", lw=2, color="blue", markersize=8, zorder=4)
trace, = ax.plot([], [], "-", lw=1.2, alpha=0.5, color="red", zorder=1)
clock = ax.text(0.05, 0.90, "", transform=ax.transAxes, fontsize=12)

# -----------------------------------------------------
# Create Animation
# -----------------------------------------------------
def animate(i):
    # 1. Actualizar posición del bloque (esquina inferior izquierda del rectángulo)
    patch_bloque.set_xy((x_M[i] - b_width/2, -b_height/2))
    
    # 2. GENERAR EL EFECTO DE ZIGZAG DEL RESORTE
    # Conectamos desde la pared fija hasta la cara izquierda del bloque
    inicio_resorte = pared_x
    fin_resorte = x_M[i] - b_width/2
    num_vueltas = 15
    
    # Crear puntos en X espaciados y alternar en Y para hacer el zigzag
    x_res = np.linspace(inicio_resorte, fin_resorte, num_vueltas * 2 + 1)
    y_res = np.zeros_like(x_res)
    y_res[1:-1:2] = 0.08   # Picos hacia arriba
    y_res[2:-1:2] = -0.08  # Picos hacia abajo
    line_resorte.set_data(x_res, y_res)
    
    # 3. Actualizar la varilla y masa m del péndulo
    line_pendulo.set_data([x_M[i], x_m[i]], [y_M[i], y_m[i]])
    trace.set_data(x_m[:i+1], y_m[:i+1])
    
    # 4. Actualizar tiempo
    clock.set_text(f"t = {t[i]:.1f} s")
    
    return patch_bloque, line_resorte, line_pendulo, trace, clock

# -----------------------------------------------------
# Animation Execution
# -----------------------------------------------------
ani = FuncAnimation(fig, animate, frames=range(0, n + 1, STRIDE),
                    interval=STRIDE * dt * 1000, blit=True)
plt.tight_layout()
plt.show()
