# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:42:13 2024

@author: Elena Almanza García

En este script realizamos el procedimiento para la asignación de eigenvalores
"""

import numpy as np
import control as ctrl
import plots
import get_kr
import estabilidadControlabilidad as ec

def singlePendulumControl():
    # Tenemos las matrices que representan el sistema del pendulo simple 
    # con los valores numéricos sustituidos.
    A = np.array([
                    [0, 1],
                    [-16.53, -0.2]
                  ])
    
    
    B = np.array([
                    [0],
                    [22.2]
                    ])
    
    C = [1, 0]

    
    # Realizando la asiganación de eigenvalores
    # 1. Encontramos los eigenvalores reales del sistema:
    # ab1 = 0.2v, ab1 = 10.5     v = lamnda
    
    # 2. Proponemos dos valores de lamnda deseados para enocntrar dos eigenvalores más
    # v1 = 10, v2 = 11 
    # a1 = -21v, a2 = 110
    eig_d = [-10, -11]
    
    # 3. Obtenemos la ganancia de retroalimentación Kb = [ab1 - a1, ab2 - a2]
    
    Kb = [-20.8, -93.49]
    
    # 4. Obtenemos la transformación equivalente S := P^-1 = [B, A*B]
    
    # 5. Obtenemos la ganancia K = Kb*P = Kb*S^-1
    
    #inversa de matriz S, que enocntramos manualmente en el paso 4
    S_ = np.array([
                    [-4.44, -22.2],
                    [-22.2, 0]
                  ]) 
    
    K = -(1/(22.2)**2)*(Kb@S_)
    
    # El procedimiento anterior fue para obtener el valor de K manualmente, 
    # pero también podemos obtenerlo usando la librería de control de dos maneras
    
    K_ak = ctrl.acker(A, B, eig_d)
    K_p = ctrl.place(A, B, eig_d)
    
    # habiendo encontrado K, representamos el sistema en lazo cerrado
    r = 1 #referencia, a donde quiero llegar
    
    Alc_k = A - K_ak*B*C
    Blc_k = r*B
    
    # sistema en lazo cerrado en espacio de estados
    singlePLc_k = ctrl.ss(Alc_k, Blc_k, C, 0)
    
    # respuesta del sistema en lazo cerrado ante entrada tipo escalon
    t, ysr, xsr = ctrl.step_response(singlePLc_k, return_x = True)
    plots.plot(t, ysr, 'respuesta del sistema ante entrada tipo escalón')
    plots.plot(xsr[0, :], xsr[1, :], 'retrato fase')
    
    ctrl.step_info(singlePLc_k)

if __name__ == '__main__':
    
    singlePendulumControl()
    
    A = np.array([
                    [0, 1],
                    [0, -1]
                  ])

    
    B = np.array([
                    [0],
                    [10]
                    ])
    
    C = [1, 0]
    
    r = 1 #referencia, a donde quiero llegar 
    
    sys = ctrl.ss(A, B, C, 0)
    print('\nSistema en espacio de estados: \n', sys)
    
    # calculando los eigenvalores
    eigenvalues = np.linalg.eigvals(A)
    print('\neigenvalores: ', eigenvalues)
    
    # verificamos estabilidad
    ec.stability(eigenvalues)
    
    # repuesta del sistema en lazo abierto ante una entrada de tipo escalón
    t, y, x = ctrl.step_response(sys, return_x = True)
    plots.plot(t, y, 'respuesta del sistema en lazo abierto ante entrada tipo escalón')
    plots.plot(x[0, :], x[1, :], 'retrato fase en lazo abierto')
    
    # determinamos si el sistema es controlable
    U = ctrl.ctrb(A, B)
    print('matriz de controlabilidad: \n', U)
    ec.controllability(U)
    
    # sistema en lazo cerrado 
    Alc = A - B*C
    Blc = r*B
    
    syslc = ctrl.ss(Alc, Blc, C, 0)
    
    # calculando los eigenvalores
    eigenvalues = np.linalg.eigvals(A)
    print('\neigenvalores: ', eigenvalues)
    
    # verificamos estabilidad
    ec.stability(eigenvalues)
    
    t, y, x = ctrl.step_response(syslc, return_x = True)
    plots.plot(t, y, 'respuesta del sistema en lazo cerrado ante entrada tipo escalón')
    plots.plot(x[0, :], x[1, :], 'retrato fase en lazo cerrado')
    
    # determinamos si el sistema es controlable
    U = ctrl.ctrb(A, B)
    print('matriz de controlabilidad: \n', U)
    ec.controllability(U)
    
    # definimos eigenvalores deseados
    # eig_d = np.array([-2 + 2j, -2 -2j])
    eig_d = np.array([-3 + 3j, -3 -3j])
    
    # encontramos el valor de K
    K = ctrl.place(A, B, eig_d)
    print("\nK: ", K)
    
    kr = get_kr.get_kr(sys, K)
    print("\nKr: ", kr)
    
    #sistema en lazo cerrado con retroalimentacion K
    Alc_k = A - K*B
    Blc_k = kr*B
    
    sysLc_k = ctrl.ss(Alc_k, Blc_k, C, 0)
    
    # calculando los eigenvalores en lazo cerrado con retro K
    eigenvalueslc_k = np.linalg.eigvals(Alc_k)
    print('\neigenvalores en lazo cerrado: ', eigenvalueslc_k)
    
    # verificamos estabilidad
    ec.stability(eigenvalueslc_k)
        
    # respuesta ante entrada tipo escalon
    t, y, x = ctrl.step_response(sysLc_k, return_x = True)
    plots.plot(t, y, 'respuesta del sistema en lazo cerrado con retro K ante entrada tipo escalón')
    plots.plot(x[0, :], x[1, :], 'retrato fase en lazo cerrado con retro K')
    
    info = ctrl.step_info(sysLc_k)
    print(info)
    
