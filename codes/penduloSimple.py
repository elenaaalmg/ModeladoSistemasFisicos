# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:59:20 2024

@author: Elena Almanza García

En este script se proponen tres maneras de solución a un sistema de ecuaciones 
de segundo orden con dos incógnitas y una parte no lineal. 
Para este caso en particular, del sistema del pendulo simple.
"""

import numpy as np
from scipy.integrate import odeint
import criteriosEstadisticos as bm
import plots

def scalarSolution(m, l, kf, g, h, tfin, y1_0, y2_0):
    """
   Resuelve el sistema de ecuaciones que modela el comportamiento del péndulo 
   simple, desde un enfoque escalar, utilizando el método de Euler.

   Parameters
   ----------
   m : float
       Masa del péndulo en kilogramos.
   l : float
       Longitud de la cuerda del péndulo en metros.
   kf : float
       Coeficiente de fricción.
   g : float
       Gravedad.
   h : float
       Paso de integración.
   tfin : float
       Tiempo final de la simulación.
   y1_0 : float
       Posición angular inicial (en radianes).
   y2_0 : float
       Velocidad angular inicial (en rad/s).

   Returns
   -------
   tuple[np.ndarray, np.ndarray, np.ndarray]
       t: np.ndarray
           Vector de tiempo de la simulación.
       x1: np.ndarray
           Solución numérica para la posición angular a lo largo del tiempo.
       x2: np.ndarray
           Solución numérica para la velocidad angular a lo largo del tiempo.
   """
   
    N = int((tfin - h)/h)
    
    t = np.zeros(N)
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    
    # condiciones iniciales
    x1[0] = y1_0 
    x2[0] = y2_0 
    
    # integración por euler
    for k in range(N - 1):
        t[k + 1] = t[k] + h
        x1[k + 1] = x1[k] + h*(x2[k])
        x2[k + 1] = x2[k] + h*((-g/l)*np.sin(x1[k]) - (kf/m)*x2[k])
        
    return t, x1, x2

def matrixSolution(A, l, g, h, tfin, y1_0, y2_0):
    """
   Resuelve el sistema de ecuaciones que modela el comportamiento del péndulo 
   simple, desde un enfoque escalar, utilizando el método de Euler.

   Parameters
   ----------
   A : np.ndarray
       Matriz de coeficientes.
   l : float
       Longitud de la cuerda del péndulo en metros.
   g : float
       Gravedad.
   h : float
       Paso de integración.
   tfin : float
       Tiempo final de la simulación.
   y1_0 : float
       Posición angular inicial (en radianes).
   y2_0 : float
       Velocidad angular inicial (en rad/s).

   Returns
   -------
   tuple[np.ndarray, np.ndarray, np.ndarray]
       t: np.ndarray
           Vector de tiempo de la simulación.
       Y: np.ndarray
           Matriz donde cada fila es la solución de la posición y aceleración 
           angular en el tiempo, respectivamente
   """
    N = int((tfin - h) / h)
    
    t = np.zeros(N)
    Y = np.zeros((2, N))  # matriz para almacenar x1 (posición) y x2 (velocidad)
    
    #condiciones iniciales
    Y[0, 0] = y1_0  # posición inicial
    Y[1, 0] = y2_0  # velocidad inicial
    
    for k in range(N - 1):
        t[k + 1] = t[k] + h
        
        Y[:, k + 1] = Y[:, k] + h*np.dot(A, Y[:, k]) #parte lienal de la ecuación
        Y[1, k + 1] = Y[1, k + 1] + h*(-g/l)*np.sin(Y[0, k])  # parte no lineal de la ecuación
    
    return t, Y

def singlePendulum(y ,t, m, l, kf, g):
    """
    Función que representa el sistema del péndulo simple en espacio de estados.

    Parmeters
    ----------
    y : np.ndarray
        Vector de variables [posición, velocidad].
    t : float
        Tiempo actual.
    m : float
        Masa del péndulo en kilogramos.
    l : float
        Longitud de la cuerda del péndulo en metros.
    kf : float
        Coeficiente de fricción.
    g : float
        Gravedad.

    Returns
    -------
    np.ndarray
       Matriz donde cada fila es la derivada numérica de la posición y aceleración 
       en el tiempo actual.
    """
    x1, x2 = y  #descomponemos el vector en posición (x1) y velocidad (x2)
    
    dxdt = [x2,
            -(g/l)*np.sin(x1) - (kf/m)*x2]
    
    return dxdt

def spicySolution(y0, t, m, l, kf, g):
    """
    Función que da solución numérica al sistema en espacio de estados.

    Parameters
    ----------
    y0 : np.ndarray
        Vector que contiene las condiciones iniciales del sistema.
    t : list or np.ndarray
        Vector de tiempo para la simulación.
    m, l, kf, g : float
        Parámetros del sistema.

   Returns
   -------
   np.ndarray
      Matriz donde cada fila es la solución de la posición y aceleración 
      angular en el tiempo, respectivamente
    """
    sol = odeint(singlePendulum, y0, t, args = (m, l, kf, g))
    
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    
    return sol

def derivation(m, l, kf, g, h, tfin, y1_0, y2_0):
    """
    Función que deriva numéricamente de distintas maneras la solución para
    hacer una comparación de los resultados.

    Parameters
    ----------
    m, l, kf, g : float
        Parámetros del sistema.
    tfin : float
        Tiempo final de la simulación.
    y1_0 : float
        Posición angular inicial (en radianes).
    y2_0 : float
        Velocidad angular inicial (en rad/s).

   Returns
   -------
   tuple[np.ndarray, np.ndarray, np.ndarray]
       xpp_1: np.ndarray
           Segunda derivada numérica de x1.
       xp_2: np.ndarray
           Derivada numérica de x2
       xpp_num: np.ndarray
           DErivada numérica usando numpy
    """
    
    t, x1, x2 = scalarSolution(m, l, kf, g, h, tfin, y1_0, y2_0)
    
    N = len(x1)
    xpp_1 = np.zeros(N - 1)
    xp_2 = np.zeros(N - 1)
    
    # derivación nuumerica
    # usando la segunda derivada en x1
    for k in range(1, N - 1):
        xpp_1[k] = (x1[k + 1] - 2*x1[k] + x1[k - 1]) / (h**2)
    
    # derivando x2 para obtener directamente la acelaración
    for k in range(N - 1):
        xp_2[k] = (x2[k + 1] - x2[k])/h # DHD
        
    #usando numpy
    h = t[1] - t[0]
    xpp_num = np.diff(np.diff(x1)/h)/h
        
    return xpp_1, xp_2, xpp_num


if __name__ == '__main__':
    
    # parámetros del sistema
    m = 0.5  # masa
    l = 0.30  # longitud
    kf = 0.1  # coeficiente de fricción
    g = 9.81  # gravedad
    
    # parámetros de simulación (integración)
    h = 1e-3  # paso de integración
    tfin = 20  # tiempo de simulación
    y1_0 = np.pi/4  # posición inicial
    y2_0 = 0  # velocidad inicial
    
    # simulación escalar
    t, x1, x2 = scalarSolution(m, l, kf, g, h, tfin, y1_0, y2_0)
    plots.plot(t, x1, 'Movimiento del Péndulo (Solución Escalar)')

    
    # simulación matricial
    # matriz de coeficientes A
    A = np.array([[0, 1],
                  [0, -(kf/m)]])
    tM, Y = matrixSolution(A, l, g, h, tfin, y1_0, y2_0)
    plots.plot(tM, Y[0], 'Movimiento del Péndulo (Solución Matricial)')
    
    #simulación usando spicy
    y0 = [np.pi/4, 0.0]
    ts = np.linspace(0, 15, 1000)
    sol = spicySolution(y0, ts, m, l, kf, g)
    
    y = [sol[:, 0], sol[:, 1]]
    label = [ r'$x_1(t)$', r'$x_2(t)$']
    plots.plotMultiple(ts, y, 'Posición y aceleración angular a lo largo del tiempo', label)
    
    
    # resultado de aceleracion por derivación númerica
    xp_1, xpp_2, xpp_num = derivation(m, l, kf, g, h, tfin, y1_0, y2_0)
    
    # aceleración obtenida analíticamente
    xpp = (-g/l)*np.sin(x1) - (kf/m)*x2

    
    #datos para graficar la derivada
    y_data = [xpp[:-1], xp_1, xpp_2]
    labels = ['Aceleración analítica', 'Aceleración numérica con x1', 'Aceleración numérica con x2']
    line_styles = ['-', '-.', '--']  # Continua, punteada, y guiones
    plots.plotMultiple(t[:-1], y_data, 'Aceleración angular', labels, line_styles)
    
    
    # derivación hacia adelante manual y usando numpy
    y_ = [xpp[:-2], xpp_num]
    line_styles_ = ['-', '-.']
    plots.plotMultiple(t[:-2], y_, lineStyles = line_styles_)
    
    resultados = bm.benchmark(xpp[:-2], xpp_num)
    print(resultados)
    
    
        
   
    