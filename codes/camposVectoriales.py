# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:04:40 2024

@author: elena
"""

import numpy as np
import matplotlib.pyplot as plt

# Definimos el dominio
x1 = np.linspace(-3,3,25)
x2 = np.linspace(-2,2,25)
a = 1/4

# #Generar dominio D
X1, X2 = np.meshgrid(x1, x2)

# Definir la función potencial
# V = lambda x1, x2 : x1**2 + x2**2
V = lambda x1, x2 : (1/2)*x1**2 + (1/4)*a*x2**4

# Generamos la grafica de superficie 3D del potencial V
fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, figsize = (10,8))

# Graficamos la superficie del potencial V
surf = ax.plot_surface(X1, X2, V(X1, X2), cmap = 'viridis', linewidth = 0, antialiased = False)

# Añadimos contornos en la parte inferior
ax.contour(X1, X2, V(X1, X2), cmap = 'viridis', offset = -1)

# Definir el campo vectorial u y v
# # u = -2*X1
# # v = -2*X2

u = X1
v = a*X2**3

# #Usamos quiver para graficar un campo vectorial en 2D.
# plt.figure()
# plt.quiver(X1, X2, V(X1, X2), X1, X2)

# Generamos la tercer componente del campo, en este caso 0
w = np.zeros_like(X1)

# Graficamos el campo vectorial en 3D
ax.quiver(X1, X2, V(X1, X2), u, v, 0, length=0.2, color='r')


plt.show()








