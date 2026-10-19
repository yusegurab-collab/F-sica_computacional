import numpy as np
import matplotlib.pyplot as plt

# Parámetros del sistema 
mu = 3.0       # Relación de masas M/m (Caso integrable clásico del paper)
E = 5.0        # Energía total constante del sistema para definir la escala

# Arreglo angular (de 0 a 2pi) para trazar toda la trayectoria límite
theta = np.linspace(0, 2 * np.pi, 1000)

# Ecuación de la curva de velocidad cero en coordenadas polares (Energía cinética T = 0)
r = E / (mu - np.cos(theta))

# Conversión a coordenadas cartesianas espaciales
x = r * np.sin(theta)
y = -r * np.cos(theta)

# Configuración del gráfico
fig, ax = plt.subplots(figsize=(6, 6))

# Dibujo de la frontera límite
ax.plot(x, y, color='#1f77b4', lw=2.5, label='Curva de Velocidad Cero (Frontera)')

# Sombreado verde de la Región Admisible 
ax.fill(x, y, color='#2ca02c', alpha=0.2, label='Región Admisible de Movimiento ($T \\geq 0$)')

# Marcador para la polea de pivote 
ax.plot(0, 0, 'ko', markersize=8, label='Polea / Pivote Fijo (Origen)')

# Detalles
ax.set_title("Región Admisible de la Máquina de Atwood Oscilante (SAM)", fontsize=11, fontweight='bold', pad=15)
ax.set_xlabel("Posición Horizontal $x = r \\sin(\\theta)$", fontsize=10)
ax.set_ylabel("Posición Vertical $y = -r \\cos(\\theta)$", fontsize=10)

# Configuración simétrica de los ejes y cuadrícula de fondo
max_val = np.max(np.abs([x, y])) * 1.2
ax.set_xlim(-max_val, max_val)
ax.set_ylim(-max_val, 0.5 * max_val)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.axvline(0, color='black', lw=0.8, ls='--')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=9)
ax.set_aspect('equal')

# Guardar la gráfica 
plt.tight_layout()
plt.savefig('region_admisible_SAM.png', dpi=300)
plt.show()
