import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt


# Učitavanje slike
image = mpimg.imread('example_grayscale.png')
X = image.reshape((-1, 1))

# Kvantizacija s 10 klastera
n_colors = 10
kmeans = KMeans(n_clusters=n_colors, n_init=10)
kmeans.fit(X)
labels = kmeans.predict(X)
values = kmeans.cluster_centers_.squeeze()

# Rekonstrukcija slike
image_compressed = np.choose(labels, values)
image_compressed = image_compressed.reshape(image.shape)

# Prikaz
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Originalna slika')
plt.imshow(image, cmap='gray')

plt.subplot(1,2,2)
plt.title(f'Kvantizirana slika ({n_colors} klastera)')
plt.imshow(image_compressed, cmap='gray')
plt.show()

# Kompresija
original_bits = 8  # grayscale: 256 vrijednosti -> 8 bitova
compressed_bits = int(np.ceil(np.log2(n_colors)))  # log2(10) ~= 4
compression_ratio = original_bits / compressed_bits
print(f"Kompresija: {compression_ratio:.2f}x")
