# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 13:01:52 2024

@author: Elena Almanza García

En este script se realiza un control de tipo proporcional (P). Consiste en una
interfaz gráfica que permite mover un valor de Kp
"""
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import control  as ctrl
import get_kr

# Función para actualizar la gráfica
def update_plot(event=None):
    
    # parámetros del sistema
    m = 1
    b = 10
    k = 20
    
    A = np.array([
                    [0,1],
                    [-k/m,-b/m]
                ])
    
    B = np.array([
                    [0],
                    [1/m]
                  ])
    C = [1,0]
    D = 0
    
    # definimos el sistema en lazo abierto
    sys1 = ctrl.ss(A, B, C, D)
    
    # Obtener los valores del slider
    Kp = Kp_slider.get()
    
    try:
        # Limpiar el gráfico anterior
        ax.cla()
        
        # representación del sistema en función de transferencia
        systf = ctrl.ss2tf(sys1)
        
        # cerramos el lazo
        syslc_k = ctrl.feedback(systf*Kp) 
        
        info = ctrl.step_info(syslc_k)
        info_str = "\n".join([f"{key}: {value}" for key, value in info.items()])
        info_label.config(text=f"Step info:\n{info_str}")
        
        # respuesta del sistema ante una entrada de tipo escalon
        t, y= ctrl.step_response(syslc_k)
        
        # dibujamos la grafica
        ax.plot(t, y, label = 'Esfuerzo de control')
        ax.grid(True)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 2])
        ax.legend()
        
        # Redibujar el canvas
        canvas.draw()
    
    except ValueError as e:
        print(f"Error: {e}")


# Crear la ventana principal
root = tk.Tk()
root.title("Gráfica de Esfuerzo de control")
root.resizable(True,True)

# Crear el gráfico usando matplotlib
fig, ax = plt.subplots(figsize=(6, 6))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, rowspan=4, columnspan=4)

# Crear slider para mover kp
Kp_slider = tk.Scale(root, from_=0, to=300, orient='horizontal', label='Kp', resolution=0.01, command=update_plot)
Kp_slider.set(1)
Kp_slider.grid(row=0, column=5, padx=10, pady=10, columnspan=3)

info_label = tk.Label(root, text="Step info: ")
info_label.grid(row=2, column=5, padx=10, pady=10)

# Dibujar la primera gráfica
update_plot()

# Iniciar el loop principal
root.mainloop()