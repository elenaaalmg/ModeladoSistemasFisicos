# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 11:24:27 2024

@author: Elena Almanza García

En este script se realizan los siguientes puntos:
    1. Obtener puntos de equilibrio
    2. Clasificar de puntos de equilibrio
    3. Bosquejar de retrajo fase.
    4. Linealizar el sistema
    
de acuerdo con el sistema que modela un circuito de diodo tunel.
"""

import sympy as sp
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import plots

def h(x1):
    """
    Función que contiene el polinomio que simula la característica del diodo 
    tunel.
    
    Parameters
    ----------
    x1 : np.ndarray
        Vector de datos que se evaluan en la función en cada instante de tiempo.

    Returns
    -------
    np.ndarray
        Vector con el resultado de la evaluación en cada instante de tiempo.
    """
    
    return 17.76*x1 - 103.79*x1**2 + 229.62*x1**3 - 226.31*x1**4 + 83.72*x1**5

def tunnelDiode(y, t, E, R, C, L):
    """
    Función que representa el sistema del circuito del diodo tunel en espacio
    de estados.

    Parameters
    ----------
    y : np.ndarray
        Vector de variables de estado.
    t : float
        Tiempo actual.
    E : float
        Entrada del sistema (voltaje).
    R : float
        Resistencia del circuito en ohms.
    C : float
        Capacitancia.
    L : float
        Inductancia del circuito.

    Returns
    -------
    np.ndarray
       Matriz donde cada elemento es la derivada numérica de las variables en el 
       tiempo actual.
    """
    x1, x2 = y
    
    dxdt = [
            (1/C)*(-h(x1) + x2),
            (1/L)*(E - R*x2 - x1)
        ]
    
    return dxdt

def tunnelDiodeSympy(E, R, C, L):
    """
    Representa el sistema de ecuaciones diferenciales del circuito del diodo tunel
    utilizando la librería SymPy.
    
    Parameters
    ----------
    E, R, C, L : float
        Parámetros del sistema.
    
    Returns
    -------
    tuple[sympy expression, sympy expression]
        x1p : sympy expression
            Expresión simbólica de la derivada de la primera variable de estado (x1).
        x2p : sympy expression
            Expresión simbólica de la derivada de la segunda variable de estado (x2).
    """
    
    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    
    x1p = (1/C)*(-h(x1) + x2)
    x2p = (1/L)*(E - R*x2 - x1)

    return x1p, x2p

def spicySolution(y0, t, E, R, C, L):
    """
    Función que da solución numérica al sistema en espacio de estados.

    Parameters
    ----------
    y0 : np.ndarray
        Vector que contiene las condiciones iniciales del sistema.
    t : list or np.ndarray
        Vector de tiempo para la simulación.
    E, R, C, L : float
        Parámetros del sistema.

   Returns
   -------
   np.ndarray
      Matriz donde cada fila es la solución en el tiempo de las variables.
    """
    sol = odeint(tunnelDiode, y0, t, args = (E, R, C, L))
    
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    
    return sol

def roots(R, E):
    """
    Calcula las raíces de dos polinomios asociados al sistema del diodo tunel
    usando numpy.

    Parameters
    ----------
    R, E : float
        Parámetros del sistema.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        roots_coef : np.ndarray
            Raíces del primer polinomio.
        roots_coef2 : np.ndarray
            Raíces del segundo polinomio.
    """
    
    coef = [83.72, -226.31, 229.62, -103.79, 17.76, 0]
    coef2 = [(-R)*coef[0], (-R)*coef[1], (-R)*coef[2], (-R)*coef[3], (-R)*coef[4] - 1, E]
    # coef2 = [(-R)*83.72, (-R)*(-226.31), (-R)*229.62, (-R)*(-103.79), (-R)*17.76 - 1, E]
 
    return np.roots(coef), np.roots(coef2)

def jacobianMatrix(F, X):
    """
    Calcula la matriz Jacobiana de un sistema de ecuaciones.

    Parameters
    ----------
    F : sympy.Matrix
        Matriz que contiene las funciones o ecuaciones del sistema.
    X : sympy.Matrix
        Matriz que contiene las variables del sistema.

    Returns
    -------
    sympy.Matrix
        La matriz Jacobiana, que es la matriz de las derivadas parciales
        de cada función con respecto a cada variable.
    """
    return F.jacobian(X)

def equilibrio(X):
    x1, x2 = X
    eq1 = -h(x1) + x2
    eq2 = E - R*x2 - x1
    return [eq1, eq2]

if __name__ == '__main__':
    
    #parametros del sistema
    E = 1.2
    R = 1500
    C = 2e-12
    L = 5e-6
    
    # graficamos la característica del diodo tunel
    x = np.linspace(0, 1, 1000)
    hx = h(x)
    plots.plot(x, hx,'Característica voltaje - corriente del diodo tunel') 
    
    # encontrando los puntos d equilibrio
    x1p, x2p = tunnelDiodeSympy(E, R, C, L)
    eq1 = sp.solve(x1p)
    eq2 = sp.solve(x2p)
    print([eq1, eq2]) # de forma simbolica
    
    breakPoints = roots(R, E) # valores numéricos
    print(breakPoints) 
    
    # usamos fsolve para encontrar los puntos de equilibrio
    # puntosEquilibrio = fsolve(equilibrio, [0, 0])
    # print(f'Puntos de equilibrio: {puntosEquilibrio}')
    
    # simulación usando spicy
    # damos varias condiciones iniciales
    Y0 = [
        [-0.4, 0.4], [1.0, 1.0], [1.5, 1.5], 
        [0.5, 1.0], [1.0, 1.5], [1.5, 0.5], 
        [1.0, 0.5], [1.5, 1.0], [0.8, 0.8], 
        [1.4, 1.4], [-0.2, 1.0], [-0.4, 2],
        [1.6, 2.0], [1.6, 0.4], [1.6, -0.2]
    ]
    t = np.linspace(0, 10e-8, 1000) # simulación más larga para capturar trayectorias
    
    # hacemos un bosquejo del retrato fase (trayectorias) para multiples condiciones iniclaes
    x1Data = []
    x2Data = []
    for y0 in Y0:
        sol = spicySolution(y0, t, E, R, C, L)
        x1 = sol[:, 0]
        x2 = sol[:, 1]
        x1Data.append(x1)
        x2Data.append(x2)
    
    plots.plotMultiple(x1Data, x2Data, 'Retrato fase de un diodo túnel para diferentes condiciones iniciales',
                       xlabel = 'x1', ylabel = 'x2')
    
    
    #linealzar el sistema encontrando la matriz jacobiana
    F = sp.Matrix([x1p, x2p])
    x1 = sp.Symbol('x1')
    x2 = sp.Symbol('x2')
    X = sp.Matrix([x1, x2])
    J = jacobianMatrix(F, X)
    
    
    # evaluamos la matriz jacobiana en los puntos de equilibrio 
    for eq in breakPoints:
        Jeq = J.subs({x1: eq[0], x2: eq[1]})
        print(f"Jacobiana en el punto de equilibrio {eq}:")
        sp.pprint(Jeq)
    
        # Calculamos los eigenvalores
        eigenvalues = Jeq.eigenvals()
        print(f"Autovalores: {eigenvalues}")
    
    
    #clasificación puntos de equilibrio
    
    

