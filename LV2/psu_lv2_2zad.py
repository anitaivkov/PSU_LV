import numpy as np
import matplotlib.pyplot as plt

# a) Učitavanje podataka iz mtcars.csv
podaci = np.loadtxt(open("mtcars.csv", "rb"), usecols=(1,2,3,4,5,6), delimiter=",", skiprows=1)

# Imenovanje varijabli (stupaca u skupu podataka)
potrosnja_mpg = podaci[:, 0]  # milja po galonu
snaga_hp = podaci[:, 1]       # konjske snage
težina_wt = podaci[:, 4]      # težina automobila
cilindri_cyl = podaci[:, 5]   # broj cilindara

# b) Scatter graf: Potrošnja vs. Konjske snage
plt.scatter(snaga_hp, potrosnja_mpg, color='b', alpha=0.6)
plt.xlabel("Konjske snage (hp)")
plt.ylabel("Potrošnja (mpg)")
plt.title("Ovisnost potrošnje goriva o konjskim snagama")
plt.show()

# c) Prikazivanje težine vozila kroz veličinu točkica
plt.scatter(snaga_hp, potrosnja_mpg, s=težina_wt*50, color='g', alpha=0.6)
plt.xlabel("Konjske snage (hp)")
plt.ylabel("Potrošnja (mpg)")
plt.title("Ovisnost potrošnje goriva o konjskim snagama (veličina označava težinu vozila)")
plt.show()

# d) Izračun minimalne, maksimalne i srednje potrošnje
minimalna_potrosnja = np.min(potrosnja_mpg)
maksimalna_potrosnja = np.max(potrosnja_mpg)
srednja_potrosnja = np.mean(potrosnja_mpg)

print(f"Minimalna potrošnja: {minimalna_potrosnja:.2f} mpg")
print(f"Maksimalna potrošnja: {maksimalna_potrosnja:.2f} mpg")
print(f"Srednja potrošnja: {srednja_potrosnja:.2f} mpg")

# e) Filtriranje automobila s 6 cilindara i izračun statistike potrošnje
potrosnja_6_cilindara = potrosnja_mpg[cilindri_cyl == 6]

min_6 = np.min(potrosnja_6_cilindara)
max_6 = np.max(potrosnja_6_cilindara)
srednja_6 = np.mean(potrosnja_6_cilindara)

print(f"Za automobile sa 6 cilindara:")
print(f"Minimalna potrošnja: {min_6:.2f} mpg")
print(f"Maksimalna potrošnja: {max_6:.2f} mpg")
print(f"Srednja potrošnja: {srednja_6:.2f} mpg")

