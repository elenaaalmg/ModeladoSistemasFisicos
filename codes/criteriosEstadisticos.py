# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 10:46:19 2024

@author: Elena Almanza García

Este scriot se utiliza para calcular criterios estadísticos y evaluar el desempeño
de los datos dados.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

def benchmark(yg, y):
    """
    Calcula varios criterios estadísticos y el desempeño de los datos.

    Parameters
    ----------
    yg : np.ndarray
        Datos predichos.
    y : np.ndarray
        Datos reales.

    Returns
    -------
    dict
        Diccionario con todos los datos estadísticos calculados: MSE, RMSE, MAE, MAPE y FIT.
    """
    
    def mse(yg, y):
        return np.mean((y - yg)**2)
    
    def rmse(yg, y):
        return np.sqrt(mse(yg, y))
        #return np.sqrt(np.mean((y - yg)**2))
    
    def mae(yg, y):
        return np.mean(np.abs(y - yg))
    
    def mape(yg, y):
        e = y - yg
    
        # N = len(y)
        # return (100/N)*np.sum(e/y)
        return 100*np.mean(np.abs(e/y))
        
    def fit(yg, y):
        return 100*(1 - (np.linalg.norm(y - yg)/np.linalg.norm(y - np.mean(y))))
        
    results = {
        'MSE': mse(y, yg),
        'RMSE' :rmse(y, yg),
        'MAE': mae(y, yg),
        'MAPE': mape(y, yg),
        'FIT': fit(y, yg)
        }
        
    return results
