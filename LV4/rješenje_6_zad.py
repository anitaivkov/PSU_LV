import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error


# ucitavanje podataka
df = pd.read_csv('cars_processed.csv')

# Prethodno čišćenje NaN vrijednosti na cijelom DataFrameu prije odabira X i y
# Identificirajemo sve relevantne numeričke i kategoričke stupce
relevant_cols = ['km_driven', 'year', 'engine', 'max_power', 'seats',
                 'fuel', 'seller_type', 'transmission', 'owner', 'selling_price']
df_cleaned_all = df[relevant_cols].dropna()

# 1. Izbacite veličine koje nisu potrebne (name, mileage - već su izbačene ili zanemarene)
# Odabir numeričkih i kategoričkih varijabli
numeric_features = ['km_driven', 'year', 'engine', 'max_power', 'seats']
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']
target = 'selling_price'

X_numeric = df_cleaned_all[numeric_features]
X_categorical = df_cleaned_all[categorical_features]
y = df_cleaned_all[target]

# Primjena one-hot kodiranja na kategoričke varijable
X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True) # drop_first=True izbjegava multikolinearnost

# Spajanje numeričkih i kodiranih kategoričkih varijabli
X_combined = pd.concat([X_numeric, X_categorical_encoded], axis=1)


# 2. Podijelite skup na train i test u omjeru 80% – 20%.
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=300)

# 3. Skalirajte ulazne podatke
Scaler = StandardScaler()
X_train_s = Scaler.fit_transform(X_train)
X_test_s = Scaler.transform(X_test)

# 4. Odredite parametre linearnog regresijskog modela.
linear_model = LinearRegression()
linear_model.fit(X_train_s, y_train)

# 5. Evaluirajte izgrađeni model na trening i testnom skupu
y_pred_train = linear_model.predict(X_train_s)
y_pred_test = linear_model.predict(X_test_s)

print("\n--- Evaluacija modela (numeričke + kategoričke veličine) ---")
print("R2 test:", r2_score(y_test, y_pred_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Max error test:", max_error(y_test, y_pred_test))
print("MAE test:", mean_absolute_error(y_test, y_pred_test))

print("\nR2 train:", r2_score(y_train, y_pred_train))
print("RMSE train:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Max error train:", max_error(y_train, y_pred_train))
print("MAE train:", mean_absolute_error(y_train, y_pred_train))

# Prikaz predikcija vs stvarne vrijednosti za testni skup
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') # Idealna linija
plt.xlabel('Stvarna cijena (log)')
plt.ylabel('Predviđena cijena (log)')
plt.title('Stvarna vs Predviđena cijena na testnom skupu (numeričke + kategoričke)')
plt.grid(True)
plt.show()

# Značajno su se poboljšali rezultati, dodavanjem varijable poput tipa goriva,
# tipa mjenjača i broja vlasnika imaju veliki utjecaj na cijenu automobila.
