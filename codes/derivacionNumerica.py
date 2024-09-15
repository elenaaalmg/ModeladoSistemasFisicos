# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:39:41 2024

@author: Elena Almanza García

El contenido de este script sirve para obtener la derivada numérica de una función
"""

import numpy as np
import plots

def numericalDerivation(x, h, fx):
    """
    Calcula la derivada numérica de una función cuando no se conoce el cuerpo de ésta
    usando la diferencia hacia adelante.

    Parameters
    ----------
    x : list or np.ndarray
        Vector de puntos donde se evalúa la función.
    h : float
        Incremento diferencial.
    fx : list or np.ndarray
        "función" a derivar, conjunto de datos.

    Returns
    -------
    xd : np.ndarray
        Derivada numérica usando la diferencia hacia adelante (DHD).
    """
    
    N = len(x)
    xd = np.zeros(N - 1) #vector de longitud N - 1. por la derivación numerica perdemos un dato
    
    for k in range(N - 1):
        xd[k] = (fx[k + 1] - fx[k])/h #si no conozco el cuerpo de la función
    
    return xd

def numericalDerivationBody(x, h, f):
    """
    Calcula la derivada numérica de una función cuando se conoce el cuerpo de ésta
    usando tres métodos: diferencia hacia adelante, hacia atrás y central.

    Parameters
    ----------
    x : list or np.ndarray
        Vector de puntos donde se evalúa la función.
    h : float
        Incremento diferencial.
    f : callable
        Función a derivar.

    Returns
    -------
    xdDHD : np.ndarray
        Derivada numérica usando la diferencia hacia adelante (DHD).
    xdDHA : np.ndarray
        Derivada numérica usando la diferencia hacia atrás (DHA).
    xdDC : np.ndarray
        Derivada numérica usando la diferencia central (DC).
    """
    
    N = len(x)
    xdDHD = np.zeros(N - 1)
    xdDHA = np.zeros(N - 1)
    xdDC = np.zeros(N - 1)
    
    for k in range(N - 1):
        
        #Diferencia hacia delante
        xdDHD[k] = (f(x[k] + h) - f(x[k]))/h 
        
        #Diferencia hacia atrás
        xdDHA[k] =(f(x[k]) - f(x[k] - h))/h
        
        #Diferencia Central
        xdDC[k] = (f(x[k] + h) - f(x[k] - h))/(2*h)  #ojo con los parentésis

    return xdDHD, xdDHA, xdDC

if __name__ == '__main__':
    
    x = np.linspace(-2, 2, 1000) #vector x. inicio, fin, cantidad de datos
    h = x[1] - x[0] #delta_x
    # f = lambda x: 2*x #def f(x): return 2*x
    # fp = lambda x: 2*np.ones(len(x)) #vectorizamos la constante para evitar error de dimensión
    f = lambda x: 3*x**2 + 3*x
    fp = lambda x: 6*x + 3*np.ones(len(x))
    fx = f(x) #en la practica esto es lo que vamos a tener (conjunto de datos)
    
    xd = numericalDerivation(x, h, fx)
    xdDHD, xdDHA, xdDC = numericalDerivationBody(x, h, f)    
    
    e = fp(x[:-1]) - xd #error
    
    #comparando la derivada analítca con la numérica
    yData = [fp(x[:-1]), xd] #recuerda que evaluamos fp con un dato menos porque al derivar numéricamente perdemos un dato
    labels2 = ['Derivada analítica', 'Derivada numérica']
    lineStyle2 = ['-', '--']
    plots.plotMultiple(x[:-1], yData, 'Derivación analítica vs numérica', labels2, lineStyle2)
    
    #datos para graficar. comparamos las distintas formas de diferenciar una función
    yData2 = [xdDHD, xdDHA, xdDC]
    labels = ['Diferencia hacia adelante', 'Diferencia hacia atrás', 'Diferencia central']
    lineStyle = ['-', '--', '-.']
    plots.plotMultiple(x[:-1], yData2, 'Diferentes formas de diferenciar', labels, lineStyle)
    
    #verificamos con cuál diferenciación obtenemos un menor error
    mseDHD = np.mean(fp(x[:-1]) - xdDHD)
    mseDHA = np.mean(fp(x[:-1]) - xdDHA)
    mseDC = np.mean(fp(x[:-1]) - xdDC)
    
    print(mseDHD, mseDHA, mseDC)
    
    plots.plot(x[:-1], e, 'Error')
    






