import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Učitavanje slike
image = mpimg.imread("example.png")
w, h, d = image.shape
print("Original image:", image.dtype, image.min(), image.max())

# Priprema podataka za KMeans
X = image.reshape((-1, 3))
X_for_kmeans = X

# Ako su vrijednosti float u 0–1, pretvori ih u 0–255 za KMeans
scale_back = False
if image.dtype == np.float32 and image.max() <= 1.0:
    X_for_kmeans = (X * 255).astype(np.uint8)
    scale_back = True

# KMeans
n_colors = 10
kmeans = KMeans(n_clusters=n_colors, n_init=10)
kmeans.fit(X_for_kmeans)
labels = kmeans.predict(X_for_kmeans)
values = kmeans.cluster_centers_

# Ako treba, skaliraj natrag u 0–1
if scale_back:
    values = values / 255.0

# Rekonstrukcija
compressed_img = values[labels].reshape((w, h, 3))
compressed_img = compressed_img.astype(image.dtype)

# Prikaz
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Originalna slika")
plt.imshow(image)
plt.axis('off')

plt.subplot(1,2,2)
plt.title(f"Kvantizirana slika ({n_colors} boja)")
plt.imshow(compressed_img)
plt.axis('off')

plt.tight_layout()
plt.show()
