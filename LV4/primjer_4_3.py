import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###4. zadatak

# ucitavanje ociscenih podataka
df = pd.read_csv("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/LV/LV4/cars_processed.csv")
print(df.info())

# razliciti prikazi
sns.pairplot(df, hue='fuel')

sns.relplot(data=df, x='km_driven', y='selling_price', hue='fuel')

# Pitanje 3
print(f"Najskuplji: {df.loc[df['selling_price'].idxmax(), 'name']} ({np.exp(df['selling_price'].max()):.2f} INR)")
print(f"Najjeftiniji: {df.loc[df['selling_price'].idxmin(), 'name']} ({np.exp(df['selling_price'].min()):.2f} INR)")

# Pitanje 4
print(f"Automobila 2012.g.: {df[df['year'] == 2012].shape[0]}")

# Pitanje 5
print(f"Najviše km: {df.loc[df['km_driven'].idxmax(), 'name']} ({df['km_driven'].max()} km)")
print(f"Najmanje km: {df.loc[df['km_driven'].idxmin(), 'name']} ({df['km_driven'].min()} km)")

# Pitanje 6
print(f"Najčešći broj sjedala: {df['seats'].mode()[0]}")

# Pitanje 7
print(f"Prosjek km Dizel: {df[df['fuel'] == 'Diesel']['km_driven'].mean():.2f} km")
print(f"Prosjek km Benzin: {df[df['fuel'] == 'Petrol']['km_driven'].mean():.2f} km")

df = df.drop(['name','mileage'], axis=1)

obj_cols = df.select_dtypes(object).columns.values.tolist()
num_cols = df.select_dtypes(np.number).columns.values.tolist()

fig = plt.figure(figsize=[15,8])
for col in range(len(obj_cols)):
    plt.subplot(2,2,col+1)
    sns.countplot(x=obj_cols[col], data=df)

df.boxplot(by ='fuel', column =['selling_price'], grid = False)

df.hist(['selling_price'], grid = False)

tabcorr = df.corr(numeric_only=True)
sns.heatmap(df.corr(numeric_only=True), annot=True, linewidths=2, cmap= 'coolwarm') 

plt.show()

# 1. 6699 mjerenja (automobila)
# 2. Tipovi stupaca: name: name- object, year- int64, selling_price- float64, km_driven- int64,
#                   fuel- object, seller_type- object, transmission- object, owner- object,
#                   mileage- float64, engine- int64, max_power- float64, seats- int64
# 3. Najskuplji: BMW X7 xDrive 30d DPE (7200000.00 INR); Najjeftiniji: Maruti 800 AC (29999.00 INR)
# 4. Automobila 2012.g.: 575
# 5. Najviše km: Maruti Wagon R LXI Minor (577414 km); Najmanje km: Maruti Eeco 5 STR With AC Plus HTR CNG (1 km)
# 6. Najčešći broj sjedala: 5
# 7. Prosjek km Dizel: 100000.00 km; Prosjek km Benzin: 100000.00 km
