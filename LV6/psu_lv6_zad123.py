import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from funkcija_6_1 import generate_data

##########1. ZADATAK

X = generate_data(500, flagc=5)

fig1 = plt.figure("Zadatak 1 – KMeans", figsize=(6, 5))
ax1 = fig1.add_subplot(1, 1, 1)

kmeans = KMeans(n_clusters=3, n_init=10)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

ax1.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30)
ax1.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centri')
ax1.set_title('KMeans klasteriranje')
ax1.legend()

##########2. ZADATAK
fig2 = plt.figure("Zadatak 2 – Elbow metoda", figsize=(6, 5))
ax2 = fig2.add_subplot(1, 1, 1)

inertias = []
cluster_range = range(1, 21)
for k in cluster_range:
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

ax2.plot(cluster_range, inertias, marker='o')
ax2.set_title('Inercija vs. broj klastera')
ax2.set_xlabel('Broj klastera')
ax2.set_ylabel('Inercija')
ax2.grid(True)



########## 3. ZADATAK

fig3 = plt.figure("Zadatak 3 – Dendrogrami", figsize=(12, 8))
methods = ['single', 'complete', 'average', 'ward']

for i, method in enumerate(methods):
    ax = fig3.add_subplot(2, 2, i + 1)
    Z = linkage(X, method=method)
    dendrogram(Z,
               truncate_mode='lastp',
               p=30,
               leaf_rotation=90.,
               leaf_font_size=8,
               ax=ax)
    ax.set_title(f'Dendrogram – {method}')
    ax.set_xlabel("Uzorci")
    ax.set_ylabel("Udaljenost")
    ax.grid(True)

plt.tight_layout()
plt.show()
