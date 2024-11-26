# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 04:54:07 2024

@author: elena
"""

import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import sympy as sym
import control as ctrl
from scipy.optimize import fsolve
from scipy.integrate import odeint
import get_kr

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
r = 1


Y0 = [0.2, 0.1]
t = np.linspace(0, lengthGraph, step)

A = np.array([[-dx - 1.15319697623924*k1/(K1 + 0.288014802072835) + 0.332137798862537*k1/(K1 + 0.288014802072835)**2, -0.288014802072835*k1/(K1 + 0.288014802072835)],
                      [-3.47204377276108*0.288014802072835**(2*n)*k2*n/(0.288014802072835**n + K2**n)**2 + 3.47204377276108*0.288014802072835**n*k2*n/(0.288014802072835**n + K2**n), -dy]])

B = np.array([[0],
              [1]])
C = [0, 1]
D = 0

# definimos el sistema en lazo abierto
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
        if name == "CREO":
            lines[name][0].set_visible(var.get())  
            lines[name][1].set_visible(var.get())  
        else:
            lines[name][0].set_visible(var.get())

    canvas_closed_loop.draw()

            
def update(frame):  
    try:
        for name in data:
            # Simular datos dinámicos
            if name == "LC":
                y = lazoCerrado(frame)
                data[name][0][frame] = y[frame]
                lines[name][0].set_ydata(data[name][0])
            elif name == "CRE":
                y = CREControl(frame)
                data[name][0][frame] = y[frame]
                lines[name][0].set_ydata(data[name][0])
            elif name == "CREO":
                y1, y2 = CREOControl(frame)
                data[name][0][frame] = y1[frame]
                data[name][1][frame] = y2[frame]
                
                # Actualizar ambas líneas
                lines[name][0].set_ydata(data[name][0])  
                lines[name][1].set_ydata(data[name][1])  
            elif name == "PID":
                y = PIDControl()
                data[name][0][frame] = y[frame]
                lines[name][0].set_ydata(data[name][0])
            
            ax_closed_loop.set_ylim(np.min(data[name]) - 0.1, np.max(data[name]) + 0.1)
        
        ax_closed_loop.legend()
        canvas_closed_loop.draw()

    except ValueError as e:
        print(f"Error: {e}")
        
    return  [line for line_group in lines.values() for line in line_group]

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

    _, sysPID = ctrl.forced_response(system, T=t, U=input_signal)
    
    # mostramos en la interfaz la información del sistema
    info = ctrl.step_info(system)
    info_str = "\n".join([f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}" 
                           for key, value in info.items()])
    
    infoPID_textbox.insert("0.0", text = f"{info_str}")
    
    return sysPID

def lazoCerrado(frame):
    
    # Definimos las matrices del sistema cerrado
    A_cl = A - B*C
    B_cl = B*r
    
    # Sistema de lazo cerrado propuesto:
    sys_lc = ctrl.ss(A_cl, B_cl, C, D)
    
    #Respuesta de tipo escalon del sistema de lazo cerrado
    _, yout_lc = ctrl.step_response(sys_lc)
    
    # mostramos en la interfaz la información del sistema
    info = ctrl.step_info(sys_lc)
    info_str = "\n".join([f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}" 
                           for key, value in info.items()])
    
    infoLC_textbox.insert("0.0", text = f"{info_str}")
    
    return yout_lc

def CREControl(frame):
   
    # Obtener los valores de los sliders
    LR = float(LambR_slider.get())
    LI = float(LambI_slider.get())
   
    # definimos los eigenvalores deseados del sistema
    eig_d = np.array([LR + 1j*LI, LR - 1j*LI])
    
    sys = ctrl.ss(A, B, C, D)
    
    
    # encontramos el valor de K y Kr
    K = ctrl.place(A, B, eig_d)
    kr = get_kr.get_kr(sys, K)
    
    # definimos el sistema en lazo cerrado con retroalimentación k
    A_cl_k = A - B@K
    B_cl_k = B*kr
    
    sys_K = ctrl.ss(A_cl_k, B_cl_k, C, D)
    
    
    # respuesta a una entrada tipo escalón
    _, yout = ctrl.step_response(sys_K)

    
    # mostramos en la interfaz la información del sistema
    info = ctrl.step_info(sys_K)
    info_str = "\n".join([f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}" 
                           for key, value in info.items()])
    
    infoCRE_textbox.insert("0.0", text = f"{info_str}")
   
    return yout

def CREOControl(frame):
    def factorial(n):
        if n == 0:
            return 1
        else:
            return n*factorial(n-1)
    
    def nchoosek(n, k_f):
        return int(factorial(n)/(factorial(k_f)*factorial(n-k_f)))
    
    def lyapunov(lamb,N):
        S = np.zeros((N,N))
    
        for i in range(1,N+1):
            for l in range(1,N+1):
                c = nchoosek(l+i-2,i-1)
                S[l-1][i-1] = ((-1)**(l+i)*c)/lamb**(l+i-1)
    
        return S

    LR = float(LambR_slider.get())
    LI = float(LambI_slider.get())
    
    eig_d_K = np.array([LR + 1j*LI, LR - 1j*LI])
    K_pf = ctrl.place(A, B, eig_d_K)
    
    # 5. Calculamos la constante kr
    sys = ctrl.ss(A, B, C, D)
    kr = get_kr.get_kr(sys, K_pf)
    
    # 6. Simulamos el control por retroalimentacion de estados
    Alc_pf = A - B*K_pf
    Blc_pf = B*kr
    
    sys_lc_K_pf = ctrl.ss(Alc_pf, Blc_pf, C, D)
    
    _, y_pf = ctrl.step_response(sys_lc_K_pf)
    
    lamda = 30

    L = lyapunov(lamda, 2)
    
    h = 1e-2
    tfin = 10
    N = int(np.ceil((tfin -h)/h))
    t = h + np.arange(0, N)*h
    
    # Referencia
    ref = np.ones(N)
    u = np.zeros(N)
    
    x0 = [0.2, 0.1]
    x1 = np.hstack((x0[0], np.zeros(N-1)))
    x2 = np.hstack((x0[1], np.zeros(N-1)))
    
    x0_g = [0.5, 0.5]
    x1_g = np.hstack((x0_g[0], np.zeros(N-1)))
    x2_g = np.hstack((x0_g[1], np.zeros(N-1)))
    
    
    for i in range(N-1):
    
        # Entrada de control
        u[i] = -(K_pf[0][0]*x1_g[i] + K_pf[0][1]*x2_g[i]) + kr*ref[i]
    
        # Planta
        x1[i+1] = x1[i] + h*((-dx - 1.15319697623924*k1/(K1 + 0.288014802072835) + 0.332137798862537*k1/(K1 + 0.288014802072835)**2)*x1[i] +  (-0.288014802072835*k1/(K1 + 0.288014802072835)) * x2[i])
        x2[i+1] = x2[i] + h*((-3.47204377276108*0.288014802072835**(2*n)*k2*n/(0.288014802072835**n + K2**n)**2 + 3.47204377276108*0.288014802072835**n*k2*n/(0.288014802072835**n + K2**n))* x1[i] + (-dy)* x2[i] + u[i])
    
        # Observador
        x1_g[i+1] = x1_g[i] + h*((-dx - 1.15319697623924*k1/(K1 + 0.288014802072835) + 0.332137798862537*k1/(K1 + 0.288014802072835)**2)*x1_g[i] +  (-0.288014802072835*k1/(K1 + 0.288014802072835)) * x2_g[i] + L[0,0]*(x2[i] - x2_g[i]))
        x2_g[i+1] = x2_g[i] + h*((-3.47204377276108*0.288014802072835**(2*n)*k2*n/(0.288014802072835**n + K2**n)**2 + 3.47204377276108*0.288014802072835**n*k2*n/(0.288014802072835**n + K2**n))* x1_g[i] + (-dy)* x2_g[i] + u[i] + L[1,1]*(x2[i] - x2_g[i]) )
    
    # mostramos en la interfaz la información del sistema
    info = ctrl.step_info(sys_lc_K_pf)
    info_str = "\n".join([f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}" 
                           for key, value in info.items()])
    
    infoCREO_textbox.insert("0.0", text = f"{info_str}")
    
    return x2, x2_g
    
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
frame_plot = ctk.CTkFrame(tab_closed_loop, corner_radius=10, fg_color = "gray20")
frame_plot.grid(row = 1, column = 0, padx = 10, pady = 10, sticky = "nsew")

frame_set = ctk.CTkFrame(tab_closed_loop, corner_radius = 10, fg_color = "gray20")
frame_set.grid(row = 1, column = 1, padx = 10, pady = 10, sticky = "nsew")

frame_check = ctk.CTkFrame(frame_set, corner_radius=10, fg_color = "gray20")
frame_check.grid(row = 7, column = 0, columnspan = 2, padx = 10, pady = (120, 10), sticky = "nsew")

frame_info = ctk.CTkFrame(tab_closed_loop, corner_radius=10, fg_color = "gray20")
frame_info.grid(row = 1, column = 2, padx = 10, pady = 10, sticky = "nsew")

# setpoint
ctk.CTkLabel(frame_set, text = "Setpoint", font = ("Arial", 12, "bold")).grid(row=2, column=0, padx=30, sticky="nsew")
setpoint_entry = ctk.CTkEntry(frame_set, placeholder_text = "Ingrese referencia")
setpoint_entry.grid(row=3, column=0, padx=30, sticky="nsew")
setpoint_button = ctk.CTkButton(frame_set, text = "Actualizar Setpoint", command=update_setpoint)
setpoint_button.grid(row=4, column=0, padx = 30, pady = 5, sticky="nsew")

# Controladores PID
ctk.CTkLabel(frame_set, text = "Ganancias de controlador PID", font=("Arial", 12, "bold")).grid(row=0, column=1, padx = (60, 10),  pady = 10, sticky = "nsew")

kp_label = ctk.CTkLabel(frame_set, text="Kp: 1.5")
kp_label.grid(row=1, column=1, padx = (60, 10), pady = (5, 0), sticky="nsew")
Kp_slider = ctk.CTkSlider(frame_set, from_ = 0, to = 10, number_of_steps = 100,
    command = lambda value: kp_label.configure(text = f"Kp: {value:.1f}")
)
Kp_slider.set(1.5)
Kp_slider.grid(row = 2, column=1, padx = (60, 10), pady=(8, 8), sticky="nsew")

ki_label = ctk.CTkLabel(frame_set, text = "Ki: 1.5")
ki_label.grid(row = 3, column = 1, padx = (60, 10), pady = (5, 0), sticky="nsew")
Ki_slider = ctk.CTkSlider(frame_set, from_ = 0, to = 10, number_of_steps = 100,
    command = lambda value: ki_label.configure(text = f"Ki: {value:.1f}")
)
Ki_slider.set(1.5)
Ki_slider.grid(row = 4, column=1, padx = (60, 10), pady=(12, 12), sticky="nsew")

kd_label = ctk.CTkLabel(frame_set, text="Kd: 0.5")
kd_label.grid(row = 5, column=1, padx = (60, 10), pady = (5, 0), sticky="nsew")
Kd_slider = ctk.CTkSlider(frame_set, from_ = 0, to = 10, number_of_steps = 100,
    command = lambda value: kd_label.configure(text = f"Kd: {value:.1f}")
)
Kd_slider.set(0.5)
Kd_slider.grid(row = 6, column=1, padx = (60, 10), pady=(0, 5), sticky="nsew")

# slider para modificar los eigenvalores del sistema
LambR_label = ctk.CTkLabel(frame_check, text = "Parte Real: -2.4")
LambR_label.grid(row = 1, column = 1, padx = (60, 10), pady = (5, 0), sticky="nsew")
LambR_slider = ctk.CTkSlider(frame_check, from_ = -10, to = 10, number_of_steps = 100,
    command = lambda value: LambR_label.configure(text = f"Parte Real: {value:.1f}")
)
LambR_slider.set(-2.4)
LambR_slider.grid(row = 2, column = 1, padx = (60, 10), pady=(12, 12), sticky="nsew")

LambI_label = ctk.CTkLabel(frame_check, text = "Parte Imaginaria: -0.4")
LambI_label.grid(row = 3, column = 1, padx = (60, 10), pady = (5, 0), sticky="nsew")
LambI_slider = ctk.CTkSlider(frame_check, from_ = -10, to = 10, number_of_steps = 100,
    command = lambda value: LambI_label.configure(text = f"Parte Imaginaria: {value:.1f}")
)
LambI_slider.set(-0.4)
LambI_slider.grid(row = 4, column=1, padx = (60, 10), pady=(12, 12), sticky="nsew")

# Variables de control
time = np.linspace(0, lengthGraph, step)
# data = {
#     "LC": np.sin(time),
#     "CRE": np.zeros(step),
#     "CREO": np.zeros(step),
#     "PID": np.zeros(step),
# }

data = {
    "LC": [np.zeros(step), np.zeros(step)],
    "CRE": [np.zeros(step), np.zeros(step)],
    "CREO": [np.zeros(step), np.zeros(step)],
    "PID": [np.zeros(step), np.zeros(step)]
}

# Variables para los checkboxes
control_scheme_vars = {
    "LC": ctk.BooleanVar(value = False),
    "CRE": ctk.BooleanVar(value = False),
    "CREO": ctk.BooleanVar(value = False),
    "PID": ctk.BooleanVar(value = False),
    }

ctk.CTkLabel(frame_check, text="Esquemas de control", font=("Arial", 14, "bold")).grid(row=0, column=0, padx = 50,  pady = 10, sticky = "nsew")
for i, (name, var) in enumerate(control_scheme_vars.items(), start = 1):
    ctk.CTkCheckBox(
        frame_check, text = name, variable=var, command=update_lines_visibility
    ).grid(row = i, column = 0, sticky = "snew", padx = 10, pady = 5)
    
# variables para la info del sistema
ctk.CTkLabel(frame_info, text = "Step info control LC", font = ("Arial", 12, "bold")).grid(row=0, column=0, sticky="nsew")
infoLC_textbox = ctk.CTkTextbox(frame_info, width=150, height=100, font=("Arial", 12))
infoLC_textbox.grid(row=1, column=0)

ctk.CTkLabel(frame_info, text = "Step info control CRE", font = ("Arial", 12, "bold")).grid(row=2, column=0, sticky="nsew")
infoCRE_textbox = ctk.CTkTextbox(frame_info, width=150, height=100, font=("Arial", 12))
infoCRE_textbox.grid(row=3, column=0)

ctk.CTkLabel(frame_info, text = "Step info control CREO", font = ("Arial", 12, "bold")).grid(row=4, column=0, sticky="nsew")
infoCREO_textbox = ctk.CTkTextbox(frame_info, width=150, height=100, font=("Arial", 12))
infoCREO_textbox.grid(row=5, column=0)

ctk.CTkLabel(frame_info, text = "Step info control PID", font = ("Arial", 12, "bold")).grid(row=6, column=0, sticky="nsew")
infoPID_textbox = ctk.CTkTextbox(frame_info, width=150, height=100, font=("Arial", 12))
infoPID_textbox.grid(row=7, column=0)

# Gráfica
fig_closed_loop, ax_closed_loop = plt.subplots(figsize=(8, 6))
ax_closed_loop.set_title("Esquemas de control para el sistema p53-Mdm2")
ax_closed_loop.set_xlim(0, lengthGraph)
ax_closed_loop.grid(color='gray', linestyle='--', linewidth=0.5)
setpoint_line, = ax_closed_loop.plot(time, np.ones_like(time)*setpoint, 'r--', label="Setpoint")

# lines = {name: ax_closed_loop.plot(time, data[name], label = name, visible = False)[0] for name in data}
lines = {}
for name in data:
    if name == "CREO":
        # Dos líneas para las dos salidas de "CREO"
        lines[name] = [
            ax_closed_loop.plot(time, data[name][0], label=f"{name} - Planta", visible=False)[0],
            ax_closed_loop.plot(time, data[name][1], '--', label=f"{name} - Observador", visible=False)[0]
        ]
    else:
        # Una línea para las demás opciones
        lines[name] = [
            ax_closed_loop.plot(time, data[name][0], label=name, visible=False)[0]
        ]

canvas_closed_loop = FigureCanvasTkAgg(fig_closed_loop, master = frame_plot)
canvas_closed_loop.draw()
canvas_closed_loop.get_tk_widget().grid(row=0, column=0, padx = 10, pady = (50, 10), sticky="nsew")

# Animación
ani = FuncAnimation(fig_closed_loop, update, frames=step, interval = 10, blit = True) 
root.mainloop()