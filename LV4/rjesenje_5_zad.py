import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, max_error


# ucitavanje podataka
df = pd.read_csv("C:/Users/anita/Desktop/Faks_NOVO/2. GODINA/IV. semestar/Primjenjeno strojno učenje/LV/LV4/cars_processed.csv")
print(df.info())

# 1. Izbacite veličine koje nisu potrebne
X = df[['km_driven', 'year', 'engine', 'max_power', 'seats']] 
y = df['selling_price']

X = X.dropna()

df_cleaned = df[['km_driven', 'year', 'engine', 'max_power', 'seats', 'selling_price']].dropna()
X = df_cleaned[['km_driven', 'year', 'engine', 'max_power', 'seats']]
y = df_cleaned['selling_price']

# 2. Podijelite skup na train i test u omjeru 80% – 20%.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=300)

# 3. Skalirajte ulazne podatke
Scaler = StandardScaler() 
X_train_s = Scaler.fit_transform(X_train)
X_test_s = Scaler.transform(X_test)

# 4. Odredite parametre linearnog regresijskog modela
linear_model = LinearRegression()
linear_model.fit(X_train_s, y_train)

# 5. Evaluirajte izgrađeni model na trening i testnom skupu
y_pred_train = linear_model.predict(X_train_s)
y_pred_test = linear_model.predict(X_test_s)

print("\n--- Evaluacija modela (samo numeričke veličine) ---")
print("R2 test:", r2_score(y_test, y_pred_test))
print("RMSE test:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Max error test:", max_error(y_test, y_pred_test))
print("MAE test:", mean_absolute_error(y_test, y_pred_test))

print("\nR2 train:", r2_score(y_train, y_pred_train))
print("RMSE train:", np.sqrt(mean_squared_error(y_train, y_pred_train)))
print("Max error train:", max_error(y_train, y_pred_train))
print("MAE train:", mean_absolute_error(y_train, y_pred_train))

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--') 
plt.xlabel('Stvarna cijena (log)')
plt.ylabel('Predviđena cijena (log)')
plt.title('Stvarna vs Predviđena cijena na testnom skupu (numeričke veličine)')
plt.grid(True)
plt.show()

print("\n--- Analiza promjene pogreške s brojem ulaznih veličina ---")
features_to_test = [
    ['km_driven'],
    ['km_driven', 'year'],
    ['km_driven', 'year', 'max_power'],
    ['km_driven', 'year', 'max_power', 'engine'],
    ['km_driven', 'year', 'max_power', 'engine', 'seats']
]

for features in features_to_test:
    print(f"\nModel s veličinama: {features}")
    X_current = df_cleaned[features]
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_current, y, test_size=0.2, random_state=300)
    
    scaler_c = StandardScaler()
    X_train_s_c = scaler_c.fit_transform(X_train_c)
    X_test_s_c = scaler_c.transform(X_test_c)
    
    model_c = LinearRegression()
    model_c.fit(X_train_s_c, y_train_c)
    
    y_pred_test_c = model_c.predict(X_test_s_c)
    r2_test_c = r2_score(y_test_c, y_pred_test_c)
    rmse_test_c = np.sqrt(mean_squared_error(y_test_c, y_pred_test_c))
    
    print(f"R2 test: {r2_test_c:.4f}")
    print(f"RMSE test: {rmse_test_c:.4f}")

# Općeniti odgovor na pitanje 6:
# Povećanjem broja relevantnih ulaznih veličina (značajki), pogreška na testnom skupu pada
# jer model dobiva više informacija o ciljnoj varijabli i može donositi točnije predikcije.
# Dodavanje irelevantnih ili redundantnih značajki može dovesti do preprilagođavanja (overfittinga),
# posebno ako je skup podataka mali, što može rezultirati pogoršanjem performansi na testnom skupu.
# Postoji točka optimalnog broja značajki gdje se postiže najbolja generalizacija.