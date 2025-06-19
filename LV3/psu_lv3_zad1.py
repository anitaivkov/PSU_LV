import pandas as pd

mtcars = pd.read_csv("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/LV/LV3/mtcars.csv")

####1. ZADATAK

# 1.
najveca_potrosnja = mtcars.sort_values(by='mpg', ascending=False).head(5) # 
print("\n1. 5 automobila s najvećom potrošnjom:")
print(najveca_potrosnja[['car', 'mpg']])

# 2.
osam_cilindara_najmanja_potrosnja = mtcars[mtcars['cyl'] == 8].sort_values(by='mpg', ascending=True).head(3)
print("\n2. Tri automobila s 8 cilindara i najmanjom potrošnjom:")
print(osam_cilindara_najmanja_potrosnja[['car', 'mpg', 'cyl']])

# 3. 
srednja_potrosnja_6_cilindara = mtcars[mtcars['cyl'] == 6]['mpg'].mean() # 
print(f"\n3. Srednja potrošnja automobila sa 6 cilindara: {srednja_potrosnja_6_cilindara:.2f} mpg")

# 4.
srednja_potrosnja_4_cilindra_tezina = mtcars[(mtcars['cyl'] == 4) & (mtcars['wt'] >= 2) & (mtcars['wt'] <= 2.2)]['mpg'].mean() # 
print(f"\n4. Srednja potrošnja automobila s 4 cilindra mase između 2000 i 2200 lbs: {srednja_potrosnja_4_cilindra_tezina:.2f} mpg")

# 5.
broj_mjenjaca = mtcars['am'].value_counts()
print("\n5. Broj automobila prema vrsti mjenjača (0=automatski, 1=ručni):")
print(broj_mjenjaca)

# 6.
automatski_mjenjac_preko_100hp = mtcars[(mtcars['am'] == 0) & (mtcars['hp'] > 100)].shape[0] # 
print(f"\n6. Broj automobila s automatskim mjenjačem i snagom preko 100 konjskih snaga: {automatski_mjenjac_preko_100hp}")

# 7.
mtcars['masa_kg'] = mtcars['wt'] * 453.592
print("\n7. Masa svakog automobila u kilogramima:")
print(mtcars[['car', 'masa_kg']])
