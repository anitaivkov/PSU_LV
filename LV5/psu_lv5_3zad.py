import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
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

# c) Izgradnja stabla odlučivanja
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_scaled, y_train)

# d) Evaluacija stabla odlučivanja
y_pred = decision_tree.predict(X_test_scaled)

# d.a) Prikaz matrice zabune
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Slobodna', 'Zauzeta'])
disp.plot(cmap='Blues')
plt.title('Matrica zabune')
plt.show()

# d.b) Izračun točnosti klasifikacije
accuracy = accuracy_score(y_test, y_pred)
print(f'Točnost klasifikacije: {accuracy:.4f}')

# d.c) Izračun preciznosti i odziva po klasama
print('Izvještaj o klasifikaciji:')
print(classification_report(y_test, y_pred, target_names=['Slobodna', 'Zauzeta']))

# a) Vizualizacija dobivenog stabla odlučivanja
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, filled=True, feature_names=feature_names, class_names=['Slobodna', 'Zauzeta'])
plt.title('Vizualizacija stabla odlučivanja')
plt.show()

# b) Utjecaj parametra max-depth na rezultate
print("Utjecaj max-depth na točnost:")
for max_depth in [1, 3, 5, 10, None]:
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train_scaled, y_train)
    y_pred_depth = dt.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_depth)
    print(f'max_depth={max_depth}: Točnost={acc:.4f}')

# c) Utjecaj nedostatka skaliranja ulaznih veličina
dt_no_scaling = DecisionTreeClassifier(random_state=42)
dt_no_scaling.fit(X_train, y_train)
y_pred_no_scaling = dt_no_scaling.predict(X_test)

cm_no_scaling = confusion_matrix(y_test, y_pred_no_scaling)
disp_no_scaling = ConfusionMatrixDisplay(confusion_matrix=cm_no_scaling, display_labels=['Slobodna', 'Zauzeta'])
disp_no_scaling.plot(cmap='Blues')
plt.title('Matrica zabune bez skaliranja')
plt.show()

accuracy_no_scaling = accuracy_score(y_test, y_pred_no_scaling)
print(f'Točnost bez skaliranja: {accuracy_no_scaling:.4f}')
print('Izvještaj o klasifikaciji bez skaliranja:')
print(classification_report(y_test, y_pred_no_scaling, target_names=['Slobodna', 'Zauzeta']))
