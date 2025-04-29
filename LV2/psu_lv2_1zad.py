import numpy as np
import matplotlib.pyplot as plt

# Definiranje koordinata točaka
tocke_x = np.array([1, 2, 3, 3, 1])
tocke_y = np.array([1, 2, 2, 1, 1])

# Crtanje grafa
plt.plot(tocke_x, tocke_y, marker='o', linestyle='-', color='blue', markersize=8, linewidth=2)

# Postavke osi i naslova
plt.xlabel("x os")
plt.ylabel("y os")
plt.title("Primjer")
plt.xlim(0, 4)
plt.ylim(0, 4)

# Prikaz grafa
plt.show()
