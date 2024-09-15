# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 12:51:19 2024

@author: Elena Almanza García

En este script se utiliza la librería de control para obtener una función de 
transferencia de la forma G(s) = Y(s)/X(s), que está dada por la relación entre 
la salida (Y(s)) y la entrada (X(s)) de un sistema; la respuesta del sistema
ante una entrada de tipo escalón y  el mapa de polos y ceros, de tres sistemas:
    Circuito RC
    Masa Resorte
    Masa Resorte Amortiguador
"""

import control as ctrl
import numpy as np
import matplotlib.pyplot as plt
import plots

def polesZeros(system):
    """
    Calcula y muestra los polos y ceros de un sistema en forma de función de transferencia.

    Parameters
    ----------
    systemen : ctrl.TransferFunction
        Sistema de control representado como una función de transferencia.

   Returns
    -------
    tuple[np.ndarray, np.ndarray]
        poles: Un arreglo de los polos del sistema.
        zeros: Un arreglo de los ceros del sistema.
    """
    
    poles = ctrl.poles(system)
    zeros = ctrl.zeros(system)
    
    # Mapa de polos y ceros del sistema
    plt.figure()
    ctrl.pzmap(system)
    plt.show()
    
    return poles, zeros

def stepResponse(system):
    
    """
    Calcula y grafica la respuesta al escalón de un sistema lineal en forma de función de transferencia.
    
    Parameters
    ----------
    system : ctrl.TransferFunction
        Sistema dinámico representado en forma de función de transferencia.
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        t: Vector de tiempo en el que se evalúa la respuesta.
        y: Respuesta del sistema ante una entrada escalón.
    """

    t, y = ctrl.step_response(system)
    plots.plot(t, y, 'Step response', xlabel = 'Time')
    
    return t, y
    

if __name__ == '__main__':
    
# =============================================================================
# Circuito RC.
# A partir de la ecuación diferencial que representa el comportamiento 
# del el sistema del circuito RC:
# 
#                       dV_c/dt = V_in/RC - V/RC
# 
# Es posible obtener una función de transferencia usando Laplace, obteniendo:
# 
#                       G(s) = (1/RC)/(s + (1/RC))
# =============================================================================
   
    #parámetros del sistema
    R = 1000 #resistencia
    C = 1000e-6 #capacitancia
    
    #solución analítica de la ecuación diferencial del circuito RC
    Vc = lambda Vin, t, R, C : Vin*(1 - np.exp(-t/(R*C)))
    
    # Usando la librería de control, tenemos dos maneras diferentes para definir una función de transferencia
    #primera forma para definir una función de tramsferencia
    num = 1/(R*C)
    den = [1, 1/(R*C)]
    sys1 = ctrl.tf(num, den)
    sys1
    
    #segunda forma para definir una función de transferencia 
    s = ctrl.tf('s')
    sys2 = (1/(R*C))/(s + (1/(R*C)))
    sys2
    
    #obtenemos y mostramos el mapa de polos y ceros del sistema
    polesRC, zerosRC = polesZeros(sys2)
    print(polesRC, zerosRC)
    
    #calculamos la respuesta ante una entrada de tipo escalon
    t, yOut = stepResponse(sys2)
    
    #comparamos la respuesta de la función de transferencia con la solución analítica
    yData = [yOut, Vc(1, t, R, C)]
    lineStyles = ['-', '--']
    plots.plotMultiple(t, yData, 'Step response', lineStyles = lineStyles)
    


# =============================================================================
# Masa Resorte.
# De la ecuación diferencial que describe el comportamiento del sistema M-R
# 
#                       m*xpp + k*x = F_x
# 
# Es posible obtener una función de transferencia usando Laplace, obteniendo:
# 
#                       G(s) = (1/m)(1/(s^2 + (k/m)))
# =============================================================================

    #parámetros del sistema
    m = 0.5 #masa
    k = 0.03 #coeficiente de fricción
    
    #definimos la función de transferencia
    s = ctrl.tf('s')
    Gs = (1/m)/(s**2 + (k/m))
    
    #obtenemos los polos y ceros
    print(polesZeros(Gs))
    
    #respuesta ante entrada de tipo escalon
    stepResponse(Gs)
    
    
    
# =============================================================================
# Masa Resorte Amortiguador.
# De la ecuación diferencial que describe el comportamiento del sistema M-R-A
# 
#                       m*xpp + bxp + k*x = F_x
# 
# Es posible obtener una función de transferencia usando Laplace, obteniendo:
# 
#                       G(s) = (1/m)/(s^2 + (b/m)*s + (k/m))
# =============================================================================
    
    #parámetros del sistema
    ma = 5 #masa
    ka = 0.01 #coeficiente de fricción
    b = 0.09 #coeficiente de amortiguamiento
    
    #definimos la función de transferencia
    s = ctrl.tf('s')
    Gs2 = (1/ma)/(s**2 + (b/ma)*s + (ka/ma))
    Gs2
    
    #obtenemos los polos y ceros
    print(polesZeros(Gs2))
    
    #respuesta ante entrada de tipo escalon
    stepResponse(Gs2)
    
    
    
    
    
    
 
