# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 04:54:07 2024

@author: elena
"""

import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sympy as sym
import control as ctrl
from scipy.optimize import fsolve
from scipy.integrate import odeint


# Configurar tema oscuro para matplotlib
plt.style.use("dark_background")

# Configurar tema oscuro para customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# ------------------------------- Sistema --------------------------------- #
# variables globales
setpoint = 1
Kp = 1
Ki = 1
Kd = 1
step = 100
lengthGraph = 20

ks = 2
k1 = 2
K1 = 0.1
dx = 1
n = 2
k2 = 2
K2 = 1
dy = 1
u = 1


Y0 = [0.2, 0.1]
t = np.linspace(0, lengthGraph, step)

A = np.array([[-dx - 1.15319697623924*k1/(K1 + 0.288014802072835) + 0.332137798862537*k1/(K1 + 0.288014802072835)**2, -0.288014802072835*k1/(K1 + 0.288014802072835)],
                      [-3.47204377276108*0.288014802072835**(2*n)*k2*n/(0.288014802072835**n + K2**n)**2 + 3.47204377276108*0.288014802072835**n*k2*n/(0.288014802072835**n + K2**n), -dy]])

B = np.array([[0],
              [1]])
C = [0, 1]
D = 0

sys = ctrl.ss(A, B, C, D)


def sysAutonomous(Y, t, ks, k1, K1, dx, n, k2, K2, dy):
  x, y = Y

  dxdt = [ks - k1*y*(x/(K1 + x)) - dx*x,
          k2*(x**n/(K2**n + x**n)) - dy*y]
  return dxdt

def solutionAutonomous(Y0, t, ks, k1, K1, dx, n, k2, K2, dy):
    sol = odeint(sysAutonomous, Y0, t, args = (ks, k1, K1, dx, n, k2, K2, dy))

    x = sol[:, 0]
    y = sol[:, 1]

    return sol
 
def controlSignal(t, entrada):
    # Parámetros de la señal de control
    A = 1  # Amplitud de la entrada
    omega = 1  # Frecuencia de la señal sinusoidal
    t1, t2 = 5, 10  # Intervalo de tiempo para la señal escalón
    
    if entrada == "Escalón":
        if t1 <= t <= t2:
            return A
        else:
            return 0
        
    elif entrada == "Senoidal":
        return A*np.sin(omega*t)
    
    elif entrada == "Cuadrada":
        return np.sign(A*np.sin(omega*t))

def sysNoAutonomous(Y, t, ks, k1, K1, dx, n, k2, K2, dy, entrada):
    x, y = Y
    u = controlSignal(t, entrada)  # Evaluar la señal de control en el tiempo actual
    dxdt = [ks - k1*y*(x/(K1 + x)) - dx*x,
            k2*((x**n)/(K2**n + x**n)) - dy*y + u]
    return dxdt

def solutionNoAutonomous(Y0, t, ks, k1, K1, dx, n, k2, K2, dy, entrada):
    sol = odeint(sysNoAutonomous, Y0, t, args=(ks, k1, K1, dx, n, k2, K2, dy, entrada))
    return sol

def plot_autonomousSys():
    sol = solutionAutonomous(Y0, t, ks, k1, K1, dx, n, k2, K2, dy)  
    ax_open_loop1.clear()
    ax_open_loop1.plot(t, sol[:, 0], color = 'cyan', label = "x (p53)")
    ax_open_loop1.plot(t, sol[:, 1], color = 'orange', label = "y (Mdm2)")
    ax_open_loop1.set_xlim(0, 15)
    ax_open_loop1.set_title("Solución del sistema autónomo")
    ax_open_loop1.grid(color='gray', linestyle='--', linewidth=0.5)
    ax_open_loop1.legend()
    canvas_open_loop1.draw()

def plot_noAutonomousSys(entrada):
    """Actualiza la gráfica del lazo abierto según la entrada seleccionada."""
    sol = solutionNoAutonomous(Y0, t, ks, k1, K1, dx, n, k2, K2, dy, entrada)  
    ax_open_loop.clear()
    ax_open_loop.plot(t, sol[:, 0], color = 'cyan', label = "x (p53)")
    ax_open_loop.plot(t, sol[:, 1], color = 'orange', label = "y (Mdm2)")
    ax_open_loop.set_xlim(0, 15)
    ax_open_loop.set_title("Solución del sistema no autónomo")
    ax_open_loop.grid(color='gray', linestyle='--', linewidth=0.5)
    ax_open_loop.legend()
    canvas_open_loop.draw()
    
def update_lines_visibility():
    """Actualiza la visibilidad de las líneas según los checkboxes."""
    for name, var in control_scheme_vars.items():
        lines[name].set_visible(var.get())
    canvas_closed_loop.draw()

            
def update(frame):   
    for name in data:
        # Simular datos dinámicos
        if name == "Lazo Cerrado":
            y = lazoCerrado(frame)
            data[name] = y
        elif name == "Retroalimentación de Estados":
            y = CREControl(frame)
            data[name] = y
        elif name == "Con Observador":
            y = CREOControl(frame)
            data[name] = y
        elif name == "PID":
            y = PIDControl()
            data[name][frame] = y[frame]
        
        ax_closed_loop.set_ylim(np.min(data[name]) - 0.1, np.max(data[name]) + 0.1)

        lines[name].set_ydata(data[name])
    
    ax_closed_loop.legend()
    canvas_closed_loop.draw()
    return lines.values()

def update_setpoint(*args):
    global setpoint
    setpoint = float(setpoint_entry.get())
    setpoint_line.set_ydata(np.ones_like(time)*setpoint)
    PIDControl()

    
def PIDControl(*args):
    global setpoint
    
    Kp = float(Kp_slider.get())
    Ki = float(Ki_slider.get())
    Kd = float(Kd_slider.get())
    
    systf = ctrl.ss2tf(sys)
    
    # Crear controlador PID
    s = ctrl.tf('s')
    pid = Kp + Ki/s + Kd*s

    # Sistema en lazo cerrado
    system = ctrl.feedback(systf*pid)

    # Crear señal de entrada
    reference = np.ones_like(t)*setpoint
    input_signal = reference 

    _, output = ctrl.forced_response(system, T=t, U=input_signal)
    # _, output = ctrl.step_response(system, T=t)
    
    return output

def lazoCerrado(frame):
    """
    //TODO
    """
    output = np.sin(time + frame / 10)
    
    return output

def CREControl(frame):
   """
   //TODO
   """
   output =  2 * ((time + frame) % 20) - 1
   
   return output

def CREOControl(frame):
    """
    //TODO
    """
    output = np.sign(np.sin(time + frame / 10))
    
    return output
    
# ------------------------------- App frame --------------------------------- #
root = ctk.CTk()
root.title("Control de Sistema p53-Mdm2")
root.geometry("1200x800")

# ------------------------------- Pestañas --------------------------------- #
# Crear las pestañas
tabview = ctk.CTkTabview(root, width = 1180, height = 780)
tabview.grid(row = 0, column = 0, sticky = "nsew")

# Configurar expansión dinámica de las pestañas
root.grid_rowconfigure(0, weight = 1)
root.grid_columnconfigure(0, weight = 1)

# Agregar pestañas
tab_open_loop = tabview.add("Lazo Abierto")
tab_closed_loop = tabview.add("Lazo Cerrado")

# ------------------------------- Pestaña lazo abierto --------------------------------- #
ctk.CTkLabel(tab_open_loop, text="Sistema en Lazo Abierto", font=("Arial", 24, "bold")).grid(row=0, column=0, columnspan=2, pady = (30, 0))

# frames para organizar las gráficas
frame_left = ctk.CTkFrame(tab_open_loop, corner_radius = 10, fg_color = "gray20")
frame_left.grid(row = 1, column = 0, padx = (50, 10), pady = (80, 10), sticky = "nsew")

frame_right = ctk.CTkFrame(tab_open_loop, corner_radius=10, fg_color = "gray20")
frame_right.grid(row = 1, column = 1, padx = (50, 10), pady = (80, 10), sticky = "nsew")

# gráfica sistema autónomo
fig_open_loop1, ax_open_loop1 = plt.subplots(figsize=(8, 4))
ax_open_loop1.set_title("Respuesta del Sistema Autónomo")
ax_open_loop1.grid(color='gray', linestyle='--', linewidth=0.5)
canvas_open_loop1 = FigureCanvasTkAgg(fig_open_loop1, master=frame_left)
canvas_open_loop1.get_tk_widget().grid(row = 0, column = 0, padx = 10, pady = (50, 0), sticky = "nsew")

# Dropdown para seleccionar una entrada para el sistema no autónomo 
ctk.CTkLabel(frame_right, text = "Seleccione Entrada", font=("Arial", 12, "bold")).grid(row = 0, column = 0, pady = 10)
input_dropdown = ctk.CTkComboBox(frame_right, values=["Escalón", "Senoidal", "Cuadrada"], command = plot_noAutonomousSys)
input_dropdown.grid(row = 1, column = 0, padx = 10, pady = 0, sticky = "nsew")

# gráfica sistema no autónomo
fig_open_loop, ax_open_loop = plt.subplots(figsize=(8, 4))
ax_open_loop.set_title("Respuesta del Sistema No Autónomo")
ax_open_loop.grid(color='gray', linestyle='--', linewidth=0.5)
canvas_open_loop = FigureCanvasTkAgg(fig_open_loop, master=frame_right)
canvas_open_loop.get_tk_widget().grid(row = 3, column = 0, padx = 10, pady = 10, sticky = "nsew")

# Graficar inicial
plot_autonomousSys()
plot_noAutonomousSys("Escalón")  # Graficar con la entrada inicial

# ------------------------------- Pestaña lazo cerrado --------------------------------- #
ctk.CTkLabel(tab_closed_loop, text="Sistema en Lazo Cerrado", font=("Arial", 24, "bold")).grid(row=0, column=0, columnspan=2, pady = (30, 0), sticky = "snew")

# frames para organizar las gráficas
frame_set = ctk.CTkFrame(tab_closed_loop, corner_radius = 10, fg_color = "gray20")
frame_set.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = "nsew")

frame_check = ctk.CTkFrame(frame_set, corner_radius=10, fg_color = "gray20")
frame_check.grid(row = 7, column = 0, columnspan = 2, padx = 10, pady = (120, 10), sticky = "nsew")

frame_plot = ctk.CTkFrame(tab_closed_loop, corner_radius=10, fg_color = "gray20")
frame_plot.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = "nsew")

# Variables de control
time = np.linspace(0, lengthGraph, step)
data = {
    "Lazo Cerrado": np.sin(time),
    "Retroalimentación de Estados": np.zeros(step),
    "Con Observador": np.zeros(step),
    "PID": np.zeros(step),
}

# Variables para los checkboxes
control_scheme_vars = {
    "Lazo Cerrado": ctk.BooleanVar(value = False),
    "Retroalimentación de Estados": ctk.BooleanVar(value = False),
    "Con Observador": ctk.BooleanVar(value = False),
    "PID": ctk.BooleanVar(value = False),
    }

# setpoint
ctk.CTkLabel(frame_set, text = "Setpoint", font = ("Arial", 12, "bold")).grid(row=2, column=0, padx=30, sticky="nsew")
setpoint_entry = ctk.CTkEntry(frame_set, placeholder_text = "Ingrese referencia")
setpoint_entry.grid(row=3, column=0, padx=30, sticky="nsew")
setpoint_button = ctk.CTkButton(frame_set, text = "Actualizar Setpoint", command=update_setpoint)
setpoint_button.grid(row=4, column=0, padx = 30, pady = 5, sticky="nsew")

# Controladores PID
ctk.CTkLabel(frame_set, text = "Ganancias de controlador PID", font=("Arial", 12, "bold")).grid(row=0, column=1, padx = (60, 10),  pady = 10, sticky = "nsew")

kp_label = ctk.CTkLabel(frame_set, text="Kp: 0.0")
kp_label.grid(row=1, column=1, padx = (60, 10), pady = (5, 0), sticky="nsew")
Kp_slider = ctk.CTkSlider(frame_set, from_ = 0, to = 10, number_of_steps = 100,
    command = lambda value: kp_label.configure(text = f"Kp: {value:.1f}")
)
Kp_slider.set(0)
Kp_slider.grid(row = 2, column=1, padx = (60, 10), pady=(8, 8), sticky="nsew")

ki_label = ctk.CTkLabel(frame_set, text = "Ki: 0.0")
ki_label.grid(row = 3, column = 1, padx = (60, 10), pady = (5, 0), sticky="nsew")
Ki_slider = ctk.CTkSlider(frame_set, from_ = 0, to = 10, number_of_steps = 100,
    command = lambda value: ki_label.configure(text = f"Ki: {value:.1f}")
)
Ki_slider.set(0)
Ki_slider.grid(row = 4, column=1, padx = (60, 10), pady=(12, 12), sticky="nsew")

kd_label = ctk.CTkLabel(frame_set, text="Kd: 0.0")
kd_label.grid(row = 5, column=1, padx = (60, 10), pady = (5, 0), sticky="nsew")
Kd_slider = ctk.CTkSlider(frame_set, from_ = 0, to = 10, number_of_steps = 100,
    command = lambda value: kd_label.configure(text = f"Kd: {value:.1f}")
)
Kd_slider.set(0)
Kd_slider.grid(row = 6, column=1, padx = (60, 10), pady=(0, 5), sticky="nsew")

ctk.CTkLabel(frame_check, text="Esquemas de control", font=("Arial", 14, "bold")).grid(row=0, column=0, padx = 50,  pady = 10, sticky = "nsew")
for i, (name, var) in enumerate(control_scheme_vars.items(), start = 1):
    ctk.CTkCheckBox(
        frame_check, text = name, variable=var, command=update_lines_visibility
    ).grid(row = i, column = 0, sticky = "snew", padx = 10, pady = 5)

# Crear gráfica
fig_closed_loop, ax_closed_loop = plt.subplots(figsize=(10, 8))
ax_closed_loop.set_title("Esquemas de control para el sistema p53-Mdm2")
ax_closed_loop.set_xlim(0, lengthGraph)
ax_closed_loop.grid(color='gray', linestyle='--', linewidth=0.5)
setpoint_line, = ax_closed_loop.plot(time, np.ones_like(time)*setpoint, 'r--', label="Setpoint")

lines = {name: ax_closed_loop.plot(time, data[name], label = name, visible = False)[0] for name in data}

canvas_closed_loop = FigureCanvasTkAgg(fig_closed_loop, master = frame_plot)
canvas_closed_loop.draw()
canvas_closed_loop.get_tk_widget().grid(row=0, column=0, padx = 10, pady = 10, sticky="nsew")


# Animación
ani = FuncAnimation(fig_closed_loop, update, frames=step, interval = 10, blit = True) 
root.mainloop()