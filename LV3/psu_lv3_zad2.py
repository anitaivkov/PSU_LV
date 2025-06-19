import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

mtcars = pd.read_csv("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/LV/LV3/mtcars.csv")

####2. ZADATAK

# 1. Barplot
avg_mpg_by_cyl = mtcars.groupby('cyl')['mpg'].mean().reset_index()

plt.figure(figsize=(8, 6))
sns.barplot(x='cyl', y='mpg', data=avg_mpg_by_cyl, palette='viridis')
plt.title('Srednja potrošnja (mpg) prema broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Srednja potrošnja (mpg)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('barplot_mpg_by_cyl.png')
print("Grafikon barplot_mpg_by_cyl.png je generiran.")

# 2. Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x='cyl', y='wt', data=mtcars, palette='plasma')
plt.title('Distribucija težine (wt) prema broju cilindara')
plt.xlabel('Broj cilindara')
plt.ylabel('Težina (x1000 lbs)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('boxplot_weight_by_cyl.png')
print("Grafikon boxplot_weight_by_cyl.png je generiran.")

# 3. 
# 'am' stupac: 0 = automatic, 1 = manual
# Boxplot je korišten
# Iz boxplota za potrošnju i vrstu mjenjača se može vidjeti da automobili s 
# ručnim mjenjačem (1) imaju veću potrošnju u usporedbi s automobilima s automatskim mjenjačem (0)
plt.figure(figsize=(8, 6))
sns.boxplot(x='am', y='mpg', data=mtcars, palette='cividis')
plt.title('Potrošnja (mpg) prema vrsti mjenjača')
plt.xlabel('Vrsta mjenjača')
plt.ylabel('Potrošnja (mpg)')
plt.xticks([0, 1], ['Automatski (0)', 'Ručni (1)'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('boxplot_mpg_by_transmission.png')
print("Grafikon boxplot_mpg_by_transmission.png je generiran.")

# 4. 
plt.figure(figsize=(10, 7))
sns.scatterplot(x='hp', y='qsec', hue='am', data=mtcars, palette='dark', s=100, alpha=0.8)
plt.title('Odnos ubrzanja (qsec) i snage (hp) prema vrsti mjenjača')
plt.xlabel('Snaga (hp)')
plt.ylabel('Ubrzanje (qsec - vrijeme za 1/4 milje)')
plt.legend(title='Mjenjač', labels=['Automatski (0)', 'Ručni (1)'])
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('scatterplot_hp_qsec_by_transmission.png')
print("Grafikon scatterplot_hp_qsec_by_transmission.png je generiran.")
plt.show()
