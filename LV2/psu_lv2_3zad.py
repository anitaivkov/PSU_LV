import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# a) Učitavanje slike
slika = Image.open("tiger.png")
matrica_slike = np.array(slika)

# a) Posvjetljivanje slike (povećavanje brightness)
svjetlija_slika = np.clip(matrica_slike * 1.5, 0, 255).astype(np.uint8)

# b) Rotacija slike za 90° u smjeru kazaljke na satu
rotirana_slika = np.rot90(matrica_slike, k=-1)

# c) Zrcaljenje slike (horizontalno)
zrcaljena_slika = np.flip(matrica_slike, axis=1)

# d) Smanjenje rezolucije slike (npr. 10 puta)
visina, širina, kanali = matrica_slike.shape
nova_visina = visina // 10
nova_širina = širina // 10
smanjena_slika = np.array(slika.resize((nova_širina, nova_visina)))

# e) Prikaz samo 1/4 slike (po širini)
maskirana_slika = np.zeros_like(matrica_slike)
maskirana_slika[:, :širina//4, :] = matrica_slike[:, :širina//4, :]

# Prikaz svih rezultata
fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].imshow(matrica_slike)
axs[0, 0].set_title("Originalna slika")

axs[0, 1].imshow(svjetlija_slika)
axs[0, 1].set_title("Posvjetljena slika")

axs[0, 2].imshow(rotirana_slika)
axs[0, 2].set_title("Rotirana slika (90°)")

axs[1, 0].imshow(zrcaljena_slika)
axs[1, 0].set_title("Zrcaljena slika")

axs[1, 1].imshow(smanjena_slika)
axs[1, 1].set_title("Smanjena rezolucija")

axs[1, 2].imshow(maskirana_slika)
axs[1, 2].set_title("Prikaz 1/4 slike")

for ax in axs.flat:
    ax.axis('off')

plt.show()
