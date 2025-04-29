import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

# Učitavanje podataka
df = pd.read_csv('occupancy_processed.csv')
feature_names = ['S3_Temp', 'S5_CO2']
target_name = 'Room_Occupancy_Count'

X = df[feature_names].to_numpy()
y = df[target_name].to_numpy()

# a) Podjela podataka na skup za učenje i testiranje (80%-20%) uz stratifikaciju
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# b) Skaliranje ulaznih veličina pomoću StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# c) Izgradnja KNN klasifikatora s početnim brojem susjeda K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# d) Evaluacija klasifikatora na testnom skupu podataka
y_pred = knn.predict(X_test_scaled)

# d.a) Prikaz matrice zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Slobodna', 'Zauzeta'])
disp.plot(cmap=plt.cm.get_cmap('Blues'))
plt.title('Matrica zabune')
plt.show()

# d.b) Izračun točnosti klasifikacije
accuracy = accuracy_score(y_test, y_pred)
print(f"Točnost klasifikacije: {accuracy:.4f}")

# d.c) Izračun preciznosti i odziva po klasama
print("Izvještaj o klasifikaciji:")
print(classification_report(y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))

# e) Utjecaj broja susjeda na rezultate (testiranje s različitim K)
for k in [1, 3, 5, 10]:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    y_pred_k = knn_k.predict(X_test_scaled)
    acc_k = accuracy_score(y_test, y_pred_k)
    print(f"K={k}: Točnost={acc_k:.4f}")

# f) Utjecaj skaliranja ulaznih veličina (bez skaliranja)
knn_no_scaling = KNeighborsClassifier(n_neighbors=5)
knn_no_scaling.fit(X_train, y_train)
y_pred_no_scaling = knn_no_scaling.predict(X_test)
accuracy_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
print(f"Točnost bez skaliranja: {accuracy_no_scaling:.4f}")
