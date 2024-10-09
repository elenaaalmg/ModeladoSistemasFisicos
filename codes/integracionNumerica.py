# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:46:49 2024

@author: Elena Almanza García

Este script contiene el método de Euler generalizado para integrar numéricamente 
desde un enfoque escalar y matricial.
"""

import numpy as np
import plots

def eulerScalarSolution(f1, f2, h, tfin, y1_0, y2_0):
    """
   Esta función resuelve un sistema de 2 ecuaciones desde un enfoque escalar, 
   utilizando el método de Euler.

   Parameters
   ----------
   f1 : callable
       Primera función del sistema de ecuaciones.
    f1 : callable
        Segunda función del sistema de ecuaciones.
   h : float
       Paso de integración.
   tfin : float
       Tiempo final de la simulación.
   y1_0 : float
       Primera condición inicial del sistema.
   y2_0 : float
       Segunda condición inicial del sistema.

   Returns
   -------
   tuple[np.ndarray, np.ndarray, np.ndarray]
       t: np.ndarray
           Vector de tiempo de la simulación.
       x1: np.ndarray
           Solución numérica para x1.
       x2: np.ndarray
           Solución numérica para x2.
   """
    N = int((tfin - h)/h)
    
    t = np.zeros(N)
    x1 = np.zeros(N)
    x2 = np.zeros(N)
    
    # condiciones iniciales
    x1[0] = y1_0 
    x2[0] = y2_0
    
    for k in range(N - 1):
        t[k + 1] = t[k] + h
        x1[k + 1] = x1[k] + h*f1(x1[k], x2[k])
        x2[k + 1] = x2[k] + h*f2(x1[k], x2[k])
        
    return t, x1, x2

def genericEulerSolution(f, y0, h, tfin):
    """
    Resuelve un sistema de ecuaciones diferenciales de forma genérica usando el 
    método de Euler.
    
    Parameters
    ----------
    f : np.ndarray
        Función (matriz) que representa el sistema de ecuaciones diferenciales.
    y0 : np.ndarray
        Vector de condiciones iniciales.
    h : float
        paso de tiempo.
    tfin : float
        tiempo final de la simulación.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
    t : np.ndarray
        Vector de tiempo.
    Y: np.ndarray
        Matriz donde cada fila es el valor de cada variable en el tiempo.
    """
    
    N = int((tfin - h)/h)  # número de pasos
    numEqs = len(y0)    # número de ecuaciones del sistema
    
    t = np.zeros(N)
    Y = np.zeros((numEqs, N))  # matriz para almacenar todas las variables
    
    # Condiciones iniciales
    Y[:, 0] = y0
    
    for k in range(N - 1):
        t[k + 1] = t[k] + h
        Y[:, k + 1] = Y[:, k] + h * f(Y[:, k])
    
    return t, Y

def genericMatrixEuler(A, F, y0, h, tfin):
    """
    Resuelve un sistema de ecuaciones diferenciales separando la parte
    lineal y no lineal, usando el método de Euler.
    
    Parámetros:
    A : np.ndarray
        Matriz de coeficientes para el sistema lineal.
    F : np.ndarray
        Matriz con la parte no lineal del sistema
    y0: np.ndarray
        Vector de condiciones iniciales.
    h : float
        paso de tiempo.
    tfin : float
        tiempo final de la simulación.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
    t : np.ndarray
        Vector de tiempo.
    Y: np.ndarray
        Matriz donde cada fila es el valor de cada variable en el tiempo.
    """
    
    N = int((tfin - h)/h)  # número de pasos
    numEqs = len(y0)    # número de ecuaciones del sistema
    
    t = np.zeros(N)
    Y = np.zeros((numEqs, N))  # matriz para almacenar todas las variables
    
    # Condiciones iniciales
    Y[:, 0] = y0
    
    for k in range(N - 1):
        t[k + 1] = t[k] + h
        
        linearPart = np.dot(A, Y[:, k])  # parte lineal
        nonlinearPart = F(Y[:, k]) if F else 0  # parte no lineal (opcional)
        # nonlinearPart = F(Y[:, k]) # parte no lineal
        
        Y[:, k + 1] = Y[:, k] + h * (linearPart + nonlinearPart)
    
    return t, Y

if __name__ == '__main__':
    
    # parámetros del sistema
    m = 0.5  # masa
    l = 0.30  # longitud
    kf = 0.1  # coeficiente de fricción
    g = 9.81  # gravedad
    
    # parámetros de simulación (integración)
    h = 1e-3  # paso de integración
    tfin = 20  # tiempo de simulación
    
    # Probando solución escalar para un sistema de dos ecuaciones
    # y1_0 = np.pi/4  # posición inicial
    # y2_0 = 0  # velocidad inicial
    # f1 = lambda x1, x2: x2
    # f2 = lambda x1, x2: (-g/l)*np.sin(x1) - (kf/m)*x2
    # t, x1, x2 = eulerScalarSolution(f1, f2, h, tfin, y1_0, y2_0)
    # plots.plot(t, x1, 'Movimiento del Péndulo (Solución Escalar)')
    
    # Probando solución genérica
    # def singlePendulum(y, m, l, kf, g):
    #     x1, x2 = y
    #     dxdt = np.array([x2,
    #                      (-g/l)*np.sin(x1) - (kf/m)*x2])
    #     return dxdt
    # y0 = [np.pi/4, 0.0]  # condiciones iniciales
    # f = lambda y: singlePendulum(y, m, l, kf, g)
    # t, Y = genericEulerSolution(f, y0, h, tfin)
    # plots.plot(t, Y[0,:], 'Movimiento del Péndulo (Solución Genérica Escalar)')
    
    # Probando solución matricial genérica
    def singlePendulum_nonlinear(y, m, l, g):
        x1, x2 = y
        return np.array([0, (-g/l) * np.sin(x1)])
    A = np.array([[0, 1],
              [0, -(kf/m)]]) #matriz de coeficientes 
    F = lambda y: singlePendulum_nonlinear(y, m, l, g) #parte no lineal
    y0 = [np.pi/4, 0.0]
    t, Y = genericMatrixEuler(A, None, y0, h, tfin)
    plots.plot(t, Y[0,:], 'Movimiento del Péndulo (Solución Matricial Genérica)')

    
