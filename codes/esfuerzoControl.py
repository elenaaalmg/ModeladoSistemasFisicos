"""
Created on Tue Oct  8 10:44:05 2024

@author: Roberto Montoya Delgado 
         Elena Almanza García 
         
En este script se realiza una interfaz gráfica que permite mover los eigenvalores
del sistema para visualizar como se comporta el sistema con cada una de ellos 
"""
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import control  as ctrl
import get_kr



# Función para actualizar la gráfica
def update_plot(event=None):
    
    A = np.array([
                    [0,1],
                    [0,-1]
                ])
    
    B = np.array([
                    [0],
                    [10]
                  ])
    C = [1,0]
    D = 0
    
    # definimos el sistema en lazo abierto
    sys1 = ctrl.ss(A, B, C, D)
    
    # Obtener los valores de los sliders
    LR = LambR_slider.get()
    LI = LambI_slider.get()
    
    # definimos los eigenvalores deseados del sistema
    eig_d = np.array([LR + 1j*LI, LR - 1j*LI])
    
    try:
        # Limpiar el gráfico anterior
        ax.cla()
        
        # encontramos el valor de K y Kr
        k = ctrl.place(A, B, eig_d)
        kr = get_kr.get_kr(sys1, k)
        
        # definimos label para que los valores de k y kr aparezcan dentro de la interfaz
        k_label.config(text=f"K: {k}")
        kr_label.config(text=f"Kr: {kr}")
        
        # definimos el sistema en lazo cerrado
        Alc_k = A - k*B
        Blc_k = kr*B
        
        syslc_k = ctrl.ss(Alc_k, Blc_k, C, D)
        
        # mostramos en la interfaz la información del sistema
        info = ctrl.step_info(syslc_k)
        info_str = "\n".join([f"{key}: {value}" for key, value in info.items()])
        info_label.config(text=f"Step info:\n{info_str}")
        
        # calculando los eigenvalores en lazo cerrado
        eigenvalues = np.linalg.eigvals(Alc_k)
        eigenvalues_label.config(text = f"Eigenvalores: {eigenvalues}")
        
        # verificamos la estabilidad del sistema
        if np.all(np.real(eigenvalues) < 0):
            stability_label.config(text="Sistema estable", fg="green")
        else:
            stability_label.config(text="Sistema inestable", fg="red")
        
        # respuesta del sistema ante una entrada de tipo escalon
        t, y = ctrl.step_response(syslc_k)
        
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

# Crear los sliders para valor real e imaginario 
LambR_slider = tk.Scale(root, from_=-10, to=10, orient='horizontal', label='Parte Real', resolution=0.01, command=update_plot)
LambR_slider.set(1) # valor inicial del slider
LambR_slider.grid(row=0, column=5, padx=10, pady=10, columnspan=3)

LambI_slider = tk.Scale(root, from_=0, to=10, orient='horizontal', label='Parte imaginaria', resolution=0.01, command=update_plot)
LambI_slider.set(1)
LambI_slider.grid(row=1, column=5, padx=10, pady=10, columnspan=3)

# Crear etiquetas para mostrar valores
k_label = tk.Label(root, text="K: ")
k_label.grid(row=2, column=5, padx=10, pady=10)

kr_label = tk.Label(root, text="Kr: ")
kr_label.grid(row=2, column=6, padx=10, pady=10)

eigenvalues_label = tk.Label(root, text = "Eigenvalores: ")
eigenvalues_label.grid(row = 3, column=5, padx = 10, pady = 10)

stability_label =tk.Label(root, text = "Estabilidad: ")
stability_label.grid(row = 4, column=5, padx = 10, pady = 10)

info_label = tk.Label(root, text="Step info: ")
info_label.grid(row=5, column=5, padx=10, pady=10)


# Dibujar la primera gráfica
update_plot()

# Iniciar el loop principal
root.mainloop()
