# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:38:31 2024

@author: Elena Almanza García

En este script analizamos el sistema de un motor DC
"""
import numpy as np
import control as ctrl
import plots
import matplotlib.pyplot as plt

def motorDC(J, b, K, R, L):

    """
    Función que contiene las matrices que representan el sistema de un motor DC
    en espacio de estados.

    Parameters
    ----------
    J : float
       Momento de inercia del rotor.
    b : float
        Constante de fricción de viscosidad del motor.
    K : float
        Constante de torque
    R : float
        Resistencia eléctrica.
    L : float
        Inductancia eléctrica.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A : np.ndarray
            Matriz de coeficientes del sistema.
        B : np.ndarray
            Matriz con la entrada del sistema.
        C : np.ndarray
            Matriz que contiene la salida del sistema (por defecto es la primera)
    """
# =============================================================================
#   Tenemos el siguiente sistema 
# 
#                   x1p = x2
#                   x2p = (1/J)*(K*x3 - b*x2)
#                   x3p = (1/L)*(u - K*x2 - R*x3)
#   
# El cual podemos representar de forma matricial como una combinación lienal
# de la siguiente manera
# =============================================================================
    A = np.array([[0, 1, 0],
                  [0, (-b/J), (K/J)],
                  [0, (-K/L), (-R/L)]])
    
    B = np.array([[0],
                   [0],
                   [1/L]])
    
    # si queremos conocer las multiples salidas del sistema, tenemos que separarlas
    # en vectores porque control solo soporta sistemas de tipo SISO
    C1 = np.array([1, 0, 0])
    C2 = np.array([0, 1, 0])
    C3 = np.array([0, 0, 1])
    
    return A, B, C1

if __name__ == '__main__':
    
    # parámetros del sistema
    J = 3.22284e-6 # momento de inercia de un rotor
    b = 3.5077e-6 # constante de fricción de viscosidad del motor
    K = 0.0274 # constante de torque
    R = 4 # resistencia eléctrica
    L = 2.75e-6 # inductacia eléctrica
    
    A, B, C = motorDC(J, b, K, R, L)
    
    # Pasamos nuestro sistema a espacio de estados
    motorSS = ctrl.ss(A, B, C, 0)
    print(motorSS)
    
    # Con la matriz A podemos encontrar los eigenvalores para determinar estabilidad
    eigenvalues = np.linalg.eigvals(A)
    print(eigenvalues)
    
    # Pasamos de un sistema en espacio de estados a una función de transferencia
    motorTF = ctrl.ss2tf(motorSS)
    motorTF
    
    # Obtenemos los polos y el mapa de polos y ceros del sistema
    print(ctrl.poles(motorTF))
    ctrl.pzmap(motorTF)
    
    # Obtenemos la respuesta del sistema ante una entrada de tipo escalón
    y, t = ctrl.step_response(motorTF, T_num = 0.6) # T reduce la escala (eje y)
    plots.plot(t, y, 'Step Response')