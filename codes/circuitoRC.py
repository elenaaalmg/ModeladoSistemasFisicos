# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 23:14:36 2024

@author: Elena Almanza Garcia

Este script contiene el código para el modelado de un circuito RC
"""

import numpy as np
from scipy.integrate import odeint
import criteriosEstadisticos as bm
import plots 

def circuitRC(y, t, Vin, R, C):
    """
    Función que representa el sistema del circuito RC en espacio de estados.

    Parmeters
    ----------
    y : float
        Valor de la variable de estado (por lo general, voltaje en el capacitor).
    t : float
        Tiempo actual.
    Vin : float
        Voltaje de entrada del sistema.
    R : float
        Resistencia del circuito en ohmios.
    C : float
        Capacitancia del circuito en faradios.

    Returns
    -------
    float
        Derivada numérica del sistema en el tiempo actual.
    """
    
    Vc = y
    
    dxdt = (Vin/(R*C)) - (Vc/(R*C))
    
    return dxdt

def spicySolution(y0, t, Vin, R, C):
    """
    Función que da solución numérica al sistema en espacio de estados.

    Parameters
    ----------
    y0 : float
        Condición inicial del sistema (voltaje inicial en el capacitor).
    t : list or np.ndarray
        Vector de tiempo para la simulación.
    Vin, R, C : float
        Parámetros del sistema.

    Returns
    -------
    np.ndarray
        Solución numérica del voltaje en el capacitor a lo largo del tiempo.
    """
    
    Vc = odeint(circuitRC, y0, t, args = (Vin, R, C))
    
    return Vc


if __name__ == '__main__':
    
    # parámetros del sistema
    Vin = 5 #voltaje de entrada
    R = 1000 #resistencia
    C = 1000e-6 #capacitancia
    
    # parámetros de simulación (integración)
    t = np.linspace(0, 8, 1000)
    y0 = 0 #condición inicial
    
    # Vc = (1/R*C)*np.exp(-(1/R*C)*t)*Vin
    Vc = R*C*Vin*(1 - np.exp(-t/(R*C))) #solución obtenida analíticamente 
    Vc_num = spicySolution(y0, t, Vin, R, C) #solución numérica
    
    #datos para graficar y comparar las soluciones
    yData = [Vc, Vc_num]
    labels = ['Analítica', 'Numerica']
    lineStyles = ['-', '--']
    plots.plotMultiple(t, yData, labels = labels, lineStyles = lineStyles)
    
    results = bm.benchmark(Vc[1:], Vc_num[1:])
    print(results)

