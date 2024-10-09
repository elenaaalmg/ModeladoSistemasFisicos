# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 00:43:56 2024

@author: elena
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import plots

def MRA(y, t, m, k, b, g):
    x1, x2 = y
    
    dxdt = [x2,
            g - (b/m)*x2 - (k/m)*x1]
    
    return dxdt

def spicySolution(y0, t, m, k, b, g):

    sol = odeint(MRA, y0, t, args = (m, k, b, g))
    
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    
    return x1, x2

if __name__ == '__main__':
    
    # parametros del sistema
    m = 1
    k = 2.5
    b = 0.3
    g = 9.81
    
    # condiciones iniciales
    y0 = [15, 0.0]
    
    # parametros de simulación
    t = np.linspace(0, 50, 1000)
    
    # solución
    x1, x2 = spicySolution(y0, t, m, k, b, g)
    
    # graficamos la solución en el timpo
    plots.plotMultiple(t, [x1, x2], labels = [r'$x_1(t)$', r'$x_2(t)$'], xlabel = "time")
    
    # retrato fase
    plots.plot(x1, x2, xlabel = 'x1(t)', ylabel = 'x2(t)')
    
    plt.figure()
    fig, ax = plt.subplots(figsize=(10,8))
    ax.scatter(x1, x2)
    q = ax.quiver(x1[:-1], x2[:-1], x1[1:] - x1[:-1], x2[1:] - x2[:-1], scale_units = 'xy', angles = 'xy', scale = 1, color = 'red')
    ax.quiverkey(q, X = np.max(x1), Y = np.max(x2), U=1, label='Quiver key, length = 10', labelpos='E', color = 'black')
    plt.xlabel('x1(t)')
    plt.ylabel('x2(t)')
    plt.grid()
    plt.show()
    
    # lyapunov
    z1 = np.linspace(-np.max(x1), np.max(x1), 50)
    z2 = np.linspace(-np.max(x2), np.max(x2), 50)
    
    X1, X2 = np.meshgrid(z1, z2)
    
    V = lambda x1, x2: (1/2)*m*(x2**2) + (1/2)*k*(x1**2) - m*g*x1  #función candidata de lyapunov
    u = k*X1 - m*g
    v = m*X2
    
    # graficamos la función 3d con sus campos vectoriales
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,8))
    surf = ax.plot_surface(X1, X2, V(X1,X2), cmap = 'viridis', linewidth=0, antialiased=False)
    ax.quiver(X1, X2, V(X1, X2), u, v, 0, length=0.01, color = 'r')
    ax.contour(X1, X2, V(X1,X2), cmap = 'viridis', offset = -1)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('V(x1,x2)')
    plt.title('Función definida positiva')
    plt.show()
    
    # campos vectoriales
    fig, ax = plt.subplots(figsize=(10,8))
    q = ax.quiver(X1, X2, u, v)
    ax.quiverkey(q, X = np.max(X1), Y = np.max(X2), U=1,
                 label='Quiver key, length = 10', labelpos='E')
    plt.show()
    
    # curvas de nivel
    plt.figure(figsize=(10,8))
    contours = plt.contour(X1, X2, V(X1,X2), 10)
    plt.clabel(contours, inline=True, fontsize=8)
    plt.colorbar()
    q = plt.quiver(x1[:-1], x2[:-1], x1[1:] - x1[:-1], x2[1:] - x2[:-1], scale_units = 'xy', angles = 'xy', scale = 1)
    plt.quiverkey(q, X = np.max(x1), Y = np.max(x2), U=1,
                 label='Quiver key, length = 10', labelpos='E')
    plt.title('Curvas de nivel')
    plt.show()
    
    # función con retrato fase
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,8))
    surf = ax.plot_surface(X1, X2, V(X1,X2), cmap = 'viridis', linewidth=0, antialiased=True)
    ax.contour(X1, X2, V(X1,X2), cmap = 'viridis', offset = -1)
    ax.plot3D(x1, x2, V(x1, x2), 'o--k')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('V(x1,x2)')
    plt.title('Función definida positiva')
    plt.show()
    