# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 20:35:52 2024

@author: elena
"""

import numpy as np
from scipy.integrate import odeint
import plots
import control as ctrl
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

def magneticLevitator(y, t, m, g, R, c, L, u):
    x1, x2, x3 = y
    
    # u = pulseTrain(t, Uhigh, Ulow, highDuration, lowDuration) #generamos el tren de pulsos como entrada
    
    dxdt = [
            x2,
            g - ((c/m)*((x3**2)/(x1))),
            (-R/L)*x3 + (u/L)
        ]
    
    return dxdt

def spicySolution(y0, t, m, g, R, c, L, u):
    sol = odeint(magneticLevitator, y0, t, args = (m, g, R, c, L, u))
    
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    x3 = sol[:, 2]
    
    return x1, x2, x3


if __name__ == '__main__':
    
    # parámetros del sistema
    m = 0.05  # masa
    c = 0.0049 # constante
    R  = 10 # resietancia
    L = 0.050 # inductancia
    g = 9.81  # gravedad
    u = 10 # entrada del sistema (voltaje)
    
    # parámetros de simulación
    y0 = [0.5, 0.0, 7.07]
    ts = np.linspace(0, 3, 1000)
    x1, x2, x3 = spicySolution(y0, ts, m, g, R, c, L, u)
    
    eq1 = (c/(m*g))*(u**2/R**2)
    eq2 = 0
    eq3 = u/R
    
    A = np.array([[0, 1, 0],
                  [(c/m)*(eq3**2/eq1**2), 0 , -(c/m)*(eq3/eq1)],
                  [0, 0, (-R/L)]])
    
    B = np.array([[0],
               [0],
               [1/L]])
    
    C = [1, 0, 0]
    

    # obteniendo los eigenvalores del sistema
    eigenvalues = np.linalg.eigvals(A)
    print(eigenvalues)
    
    # pasamos nuestro sistema de espacio de estados a función de transferencia
    levitTF = ctrl.ss2tf(A, B, C, 0)
    print(levitTF)
    
    # graficamos ante una entrada de tipo escalon
    t, ysr = ctrl.step_response(levitTF)
    
    # obtemos el mapa de polos y ceros 
    ctrl.pzmap(levitTF)
    
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection='3d')
    ax.plot3D(x1, x2, x3)
    ax.set_xlabel('$x_{1}(t)$')
    ax.set_ylabel('$x_{2}(t)$')
    ax.set_zlabel('$x_{3}(t)$')
    ax.set_title('Espacio fase')
    plt.show()
    
    plots.plot(t, ysr)
    
    y = [x1, x2, x3]
    label = [ r'$x_1(t)$', r'$x_2(t)$', r'$x_3(t)$']
    plots.plotMultiple(ts, y, "Resultados de simulación", label)
    plots.plot(y[0], y[1], "Espacio Fase 1")
    plots.plot(y[0], y[2], "Espacio Fase 2")
    plots.plot(y[1], y[2], "Espacio Fase 3")
    