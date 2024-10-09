# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 13:03:29 2024

@author: elena
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plots
# % matplotlib ipympl

def system(y ,t):
    x1, x2 = y  #descomponemos el vector en posición (x1) y velocidad (x2)
    
    dxdt = [x2 - x1*(x1**2 + x2**2),
            -x1 - x2*(x1**2 + x2**2)]
    
    return dxdt

def spicySolution(y0, t):

    sol = odeint(system, y0, t)
    
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    
    return x1, x2

if __name__ == '__main__':
    
    #simulación usando spicy
    x0 = [1e-2, 1e-2]
    # ts = np.linspace(0, 5, 1e-3)
    ts = np.arange(0, 25, 1e-3)
    x1, x2 = spicySolution(x0, ts)
    
    y = [x1, x2]
    label = [ r'$x_1(t)$', r'$x_2(t)$']
    plots.plotMultiple(ts, y)
    plots.plot(x1, x2)


    # # Generar dominio D
    # z1 = np.linspace(-0.02, 0.02, 50)
    # z2 = np.linspace(-0.02, 0.02, 50)
    z1 = np.linspace(-np.max(x1), np.max(x1))
    z2 = np.linspace(-np.max(x2), np.max(x2))
    X1, X2 = np.meshgrid(z1, z2)
    
    # # Definir la función potencial de lyapunov
    V = lambda x1, x2 : x1**2 + x2**2
    # V = lambda x1, x2 : -2*x1*x2 - 4*(x1**2)*(x2**2) - 2*(x1**3)*x2 - 2*x1*(x2**3) - x1**4 + x2**2 - x1**2 - 3*(x2**4)
    
    
    # Generamos la grafica de superficie 3D del potencial V
    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, figsize = (10,8))
    
    # Graficamos la superficie del potencial V
    surf = ax.plot_surface(X1, X2, V(X1, X2), cmap = 'viridis', linewidth = 0, antialiased = False)
    
    # Añadimos contornos en la parte inferior
    ax.contour(X1, X2, V(X1, X2), cmap = 'viridis', offset = -1)
    ax.plot3D(x1, x2, V(x1, x2), 'o--k')
    
    # Definir el campo vectorial u y v (derivadas)
    u = 2*X1
    v = 2*X2
    # u = 2*X1 + 2*X2
    # v = 2*X1 + 6*X1
    
    # Usamos quiver para graficar un campo vectorial en 2D.
    plt.figure()
    plt.quiver(X1, X2, V(X1, X2), X1, X2)
    plt.show()
    
    plt.figure()
    contours = plt.contour(X1, X2, V(X1, X2), 10)
    plt.clabel(contours, inline = True, fontsize = 8)
    # plt.plot(x1, x2)
    plt.plot(x1, x2, 'o--k')
    plt.show
 
    

