# -*- coding: utf-8 -*-
"""
@author: Elena Almanza García

Este script contiene funciones para generar gráficas.
"""

import matplotlib.pyplot as plt

def plot(x, y, title = None, label = None, xlabel = None, ylabel = None):
    
    """
    Genera un gráfico del comportamiento de una soala función.

    Parameters
    ----------
    x : list or np.ndarray
        Datos para el eje x.
    y : list or np.ndarray
        Datos para el eje y.
    title : str, opcional
        Título de la gráfica (por defecto None).
    label : str, opcional
        Etiqueta para la función (por defecto None).
    xlabel : str, opcional
        Nombre para el eje x (por defecto None).
    ylabel : str, opcional
        Nombre para el eje y (por defecto None).
    """
    
    plt.figure()
    plt.plot(x, y, label = label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if label:
        plt.legend()
    plt.show()

def plotMultiple(xData, yData, title = None, labels = None, lineStyles = None, xlabel = None , ylabel = None):
    
    """
    Genera un gráfico del comportamiento de múltiples funciones sobre los mismos valores de x.

    Parameters
    ----------
    xData : list or np.ndarray
        Vector que contiene los valores para los cuales se evalúan las funciones.
    yData : list or np.ndarray
        Vector que contiene las funciones a graficar.
    title : str, opcional
        Título de la gráfica (por defecto None).
    labels : list, opcional
        Vector de etiquetas para diferenciar las funciones (por defecto None).
    lineStyles : list, opcional
        Vector de diferentes estilos de línea para diferenciar mejor las funciones (por defecto None).
    xlabel : str, opcional
        Nombre para el eje x (por defecto None).
    ylabel : str, opcional
        Nombre para el eje y (por defecto None).
    """
    
    plt.figure()
    
    #si no se especifican estilos, usa una línea continua por defecto para todas
    if lineStyles is None:
        lineStyles = ['-'] * len(yData)
        
    # Si xData es un solo vector, duplicarlo para que coincida con yData
    if isinstance(xData[0], (int, float)):
        xData = [xData] * len(yData)    
    
    #graficar cada conjunto de datos con su correspondiente estilo de línea
    if labels is None:
        for x, y, style in zip(xData, yData, lineStyles):
            plt.plot(x, y, style)
    else:
        for x, y, label, style in zip(xData, yData, labels, lineStyles):
            plt.plot(x, y, style, label = label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    plt.grid(True)
    
    #solo mostrar la leyenda si se proporcionaron etiquetas
    if labels:
        plt.legend()
    
    plt.show()