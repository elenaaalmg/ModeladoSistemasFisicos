"""
Created on Tue Sep 24 18:08:26 2024

@author: Elena Almanza García

En este script determinamos si un sistema es controlable
"""
import numpy as np
import sympy as sp
import control as ctrl
from scipy.integrate import odeint
import plots 
import integracionNumerica
import estabilidadControlabilidad as ec

def singlePendulum(y ,t, m, l, kf, g, u):
    x1, x2 = y
    
    dxdt = [x2,
            -(g/l)*np.sin(x1) - (kf/m)*x2 + u/(m*l**2)]
    
    return dxdt

def spicySolution(y0, t, m, l, kf, g, u):
    sol = odeint(singlePendulum, y0, t, args = (m, l, kf, g, u))
    
    x1 = sol[:, 0]
    x2 = sol[:, 1]
    
    return x1, x2

if __name__ == '__main__':
    
    # parámetros del sistema
    m = 0.5  # masa
    l = 0.30  # longitud
    kf = 0.1  # coeficiente de fricción
    g = 9.81  # gravedad
    u = np.sin(np.pi/4)*m*g*l # tau deseado (entrada del sistema modificado)
    r = 1 #referencia
    
    # parametros de simulación
    y0 = [0, 0]
    ts = np.linspace(0, 15, 1000)
    
    #solución del sistema
    x1, x2 = spicySolution(y0, ts, m, l, kf, g, u)
    plots.plotMultiple(ts, [x1, x2], 'solución del sistema')
    
    # puntos de equilibrio
    # eq1 = np.arcsin(u/(g*l*m))
    eq1 = u
    eq2 = 0
    
    # Linealizando el sistema
    x1s = sp.Symbol('x1s')
    x2s = sp.Symbol('x2s')
    F = sp.Matrix([x2s, -(g/l)*sp.sin(x1s) - (kf/m)*x2s + u/(m*l**2)])
    X = sp.Matrix([x1s, x2s])
    J = F.jacobian(X)
    print('\nmatriz Jacobiana: \n', J)
    
    # matrices del sistema linealizado en torno al punto de equilibrio
    A = np.array([
                    [0, 1],
                    [-(g/l)*np.cos(eq1), -(kf/m)]
                  ])

    
    B = np.array([
                    [0],
                    [1/(m*l**2)]
                    ])
    
    C = [1, 0]
    
    # representación del sistema en espacio de estados
    singleP = ctrl.ss(A, B, C, 0)
    print('\nSistema en espacio de estados: \n', singleP)
    
    # # resolviendo el sistema linealizado usando euler
    # te, Y = integracionNumerica.genericMatrixEuler(A, None, y0, 1e-3, 15)
    # plots.plotMultiple(te, Y, 'solución del sistema linealizado')
    # plots.plot(Y[0,:], Y[1,:])
    
    # calculando los eigenvalores
    eigenvalues = np.linalg.eigvals(A)
    print('\neigenvalores: ', eigenvalues)
    
    #verificando estabilidad del sistema 
    ec.stability(eigenvalues)
    
    # respuesta ante entrada tipo escalon 
    t, ysr, xsr = ctrl.step_response(singleP, return_x = True)
    plots.plot(t, ysr, 'respuesta del sistema ante entrada tipo escalón')
    plots.plot(xsr[0, :], xsr[1, :], 'retrato fase')
    
    # determinamos si el sistema es controlable
    U = ctrl.ctrb(A, B)
    print('matriz de controlabilidad: \n', U)
    ec.controllability(U)
        
    # =============================================================================
    #   Tenemos la siguiente función que representan nuestro sistema en lazo
    #   cerrado en espacio de estados
    # 
    #                       dXdt = (A - BC)X + rB                         
    # =============================================================================
          
    Alc = A - B*C #matriz A en lazo cerrado
    Blc = r*B #matriz B en lazo cerrado
    
    singlePLc = ctrl.ss(Alc, Blc, C, 0)
    
    # # resolvemos el sistema usando integración por euler
    # te, Y = integracionNumerica.genericMatrixEuler(Alc, None, y0, 1e-3, 15)
    # plots.plotMultiple(te, Y, 'solución al sistema en lazo cerrado')
    
    # calculando los eigenvalores
    eigenvaluesLc = np.linalg.eigvals(Alc)
    print('\neigenvalores: ', eigenvaluesLc)
    
    #estabilidad del sistema 
    ec.stability(eigenvaluesLc)
        
    # respuesta ante entrada tipo escalon
    t, ysr, xsr = ctrl.step_response(singlePLc, return_x = True)
    plots.plot(t, ysr, 'respuesta del sistema en lazo cerrado ante entrada tipo escalón')
    plots.plot(xsr[0, :], xsr[1, :], 'retrato fase')
    
    # controlabilidad 
    ULc = ctrl.ctrb(Alc, Blc)
    print('matriz de controlabilidad: \n', ULc)
    ec.controllability(ULc)
    
    
    