import numpy as np
import matplotlib.pyplot as plt

def stvori_sahovsko_polje(velicina_kvadrata, broj_kvadrata_visina, broj_kvadrata_sirina):
    # Kreiranje jednog crnog i jednog bijelog kvadrata
    crni_kvadrat = np.zeros((velicina_kvadrata, velicina_kvadrata), dtype=np.uint8)
    bijeli_kvadrat = np.ones((velicina_kvadrata, velicina_kvadrata), dtype=np.uint8) * 255

    # Stvaranje jednog reda ploče s naizmjeničnim crno-bijelim kvadratima
    red_neparni = np.hstack([crni_kvadrat, bijeli_kvadrat] * (broj_kvadrata_sirina // 2))
    red_parni = np.hstack([bijeli_kvadrat, crni_kvadrat] * (broj_kvadrata_sirina // 2))

    # Slaganje redova kako bismo dobili cijelu ploču
    sahovsko_polje = np.vstack([red_neparni, red_parni] * (broj_kvadrata_visina // 2))

    return sahovsko_polje

# Parametri: veličina kvadrata, broj kvadrata po visini i širini
velicina_kvadrata = 25
broj_kvadrata_visina = 8
broj_kvadrata_sirina = 8

# Generiranje šahovske ploče
slika = stvori_sahovsko_polje(velicina_kvadrata, broj_kvadrata_visina, broj_kvadrata_sirina)

# Prikaz slike
plt.imshow(slika, cmap='gray', vmin=0, vmax=255)
plt.axis("on")
plt.show()
