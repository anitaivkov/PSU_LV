import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# Učitavanje podataka
df = pd.read_csv('occupancy_processed.csv')
feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

# a) Podjela podataka
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# b) Skaliranje
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Izgradnja modela
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# Evaluacija
y_pred = logreg.predict(X_test_scaled)

# Matrica zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Slobodna', 'Zauzeta'])
disp.plot(cmap=plt.get_cmap('Blues'))
plt.title('Matrica zabune - Logistička regresija')
plt.show()

# Metrike
print(f"Točnost: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))
