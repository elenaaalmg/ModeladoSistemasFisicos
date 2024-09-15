# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:17:39 2024

@author: Elena Almanza García

En este script se resuelve el sistema que modela un levitador magnético.
"""

import numpy as np
from scipy.integrate import odeint
import plots

def pulseTrain(t, Uhigh, Ulow, highDuration, lowDuration):
    """
    Genera un tren de pulsos basado en duraciones de nivel alto y bajo.
    
    Parameters
    ----------
    t : np.ndarray
        Vector de tiempo en el que se evalúa el tren de pulsos.
    Uhigh : float
        Valor del pulso cuando está en nivel alto.
    Ulow : float
        Valor del pulso cuando está en nivel bajo.
    highDuration : float
        Duración del pulso en nivel alto.
    lowDuration : float
        Duración del pulso en nivel bajo.
    
    Returns
    -------
    np.ndarray
        Vector con los valores del tren de pulsos en cada instante de tiempo.
    """
    period = highDuration + lowDuration #periodo total del tren del pulsos
    
    #evaluamos en cada instante del tiempo si estamos dentro de la fase en alto del ciclo. 
    #si estamos dentro de los primeros highduration del ciclo, devolvemos uHigh, de lo contrario uLow
    #(t%period) obtiene el tiempo dentro del ciclo actual
    
    return np.where((t % period) < highDuration, Uhigh, Ulow)

def magneticLevitator(y, t, m, g, R, c, L, u):
# def magneticLevitator(y, t, m, g, R, c, L, Uhigh, Ulow, highDuration, lowDuration):
    """
    Función que representa el sistema del levitador magnético en espacio de estados.

    Parameters
    ----------
    y : np.ndarray
        Vector de variables de estado [posición, velocidad, corriente].
    t : float
        Tiempo actual.
    m : float
        Masa de la bola en kilogramos.
    g : float
        Gravedad.
    R : float
        Resistencia del circuito en ohms.
    c : float
        Constante relacionada con la fuerza magnética.
    L : float
        Inductancia del circuito.
    u : float
        Entrada del sistema (voltaje del circuito)

    Returns
    -------
    np.ndarray
       Matriz donde cada elemento es la derivada numérica de [posición, velocidad, corriente]
       en el tiempo actual.
    """
    x1, x2, x3 = y
    
    # u = pulseTrain(t, Uhigh, Ulow, highDuration, lowDuration) #generamos el tren de pulsos como entrada
    
    dxdt = [
            x2,
            g - ((c/m)*((x3**2)/(x1))),
            (-R/L)*x3 + (u/L)
        ]
    
    return dxdt

def spicySolution(y0, t, m, g, R, c, L, u):
# def spicySolution(y0, t, m, g, R, c, L, Uhigh, Ulow, highDuration, lowDuration):
    """
    Función que da solución numérica al sistema en espacio de estados.

    Parameters
    ----------
    y0 : np.ndarray
        Vector que contiene las condiciones iniciales del sistema.
    t : list or np.ndarray
        Vector de tiempo para la simulación.
    m, g, R, c, L, u : float
        Parámetros del sistema.

   Returns
   -------
   np.ndarray
      Matriz donde cada fila es la solución de la posición y aceleración 
      angular en el tiempo, respectivamente
    """
    
    # sol = odeint(magneticLevitator, y0, t, args = (m, g, R, c, L, Uhigh, Ulow, highDuration, lowDuration))
    sol = odeint(magneticLevitator, y0, t, args = (m, g, R, c, L, u))
    
    return sol


if __name__ == '__main__':
    
    # parámetros del sistema
    m = 0.05  # masa
    c = 0.0049 # constante
    R  = 10 # resietancia
    L = 0.060 # inductancia
    g = 9.81  # gravedad
    u = 10 # entrada del sistema (voltaje)
    
    # parámetros del tren de pulsos
    # Uhigh = 10  # valor en alto
    # Ulow = 0   # valor en bajo
    # highDuration = 0.5  # duración del pulso en alto
    # lowDuration = 0.5  # duración del pulso en bajo 
    
    # parámetros de simulación
    y0 = [0.5, 0.0, 7.07]
    t = np.linspace(0, 3, 1000)
    sol = spicySolution(y0, t, m, g, R, c, L, u)
    # sol = spicySolution(y0, t, m, g, R, c, L, Uhigh, Ulow, highDuration, lowDuration)
    
    y = [sol[:, 0], sol[:, 1], sol[:, 2]]
    label = [ r'$x_1(t)$', r'$x_2(t)$', r'$x_3(t)$']
    plots.plotMultiple(t, y, "Resultados de simulación", label)
    
    plots.plot(y[0], y[1], "Espacio Fase: posición vs velocidad")