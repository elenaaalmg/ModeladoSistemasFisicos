# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:14:42 2024

@author: Elena Almanza García

Este script es util para reciclar código. Se determina la estabilidad de un sistema
y si éste es controlable.
"""
import numpy as np

def stability(eigenvalues):
    if np.all(np.real(eigenvalues) < 0):
        print('\nsistema estable')
    else:
        print('\nsistema inestable')

def controllability(U):
    if np.linalg.det(U) == 0:
        print('\nsistema no controlable')
    else:
        print('\nsistema controlable')
        
