import numpy as np
from tensorflow import keras
from keras import layers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from keras import ops
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# MNIST podatkovni skup
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# TODO: prikazi nekoliko slika iz train skupa
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"Labela: {y_train[i]}")
plt.axis()
plt.show()


# Skaliranje vrijednosti piksela na raspon [0,1]
x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

# Slike 28x28 piksela se predstavljaju vektorom od 784 elementa
x_train_s = x_train_s.reshape(60000, 784)
x_test_s = x_test_s.reshape(10000, 784)

# Kodiraj labele (0, 1, ... 9) one hot encoding-om
y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

# TODO: kreiraj mrezu pomocu keras.Sequential(); prikazi njenu strukturu pomocu .summary()
model = Sequential()
model.add(Input(shape = (784, )))  
model.add(Dense(2, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

# TODO: definiraj karakteristike procesa ucenja pomocu .compile()
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

# TODO: provedi treniranje mreze pomocu .fit()
model.fit(x_train_s, y_train_s, epochs=25, batch_size=32, verbose=2)

# TODO: Izracunajte tocnost mreze na skupu podataka za ucenje i skupu podataka za testiranje
train_loss, train_acc = model.evaluate(x_train_s, y_train_s)
test_loss, test_acc = model.evaluate(x_test_s, y_test_s)
print(f"\nTočnost na skupu za učenje: {train_acc:.4f}")
print(f"Točnost na skupu za testiranje: {test_acc:.4f}")

# TODO: Prikazite matricu zabune na skupu podataka za testiranje
y_test_pred = np.argmax(model.predict(x_test_s), axis=1)
cm = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Matrica zabune - Test skup")
plt.xlabel("Predviđeno")
plt.ylabel("Stvarno")
plt.show()

# TODO: Prikazi nekoliko primjera iz testnog skupa podataka koje je izgrađena mreza pogresno klasificirala
misclassified_idx = np.where(y_test != y_test_pred)[0]
plt.figure(figsize=(10, 10))
for i, idx in enumerate(misclassified_idx[:9]):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    plt.title(f"Stvarno: {y_test[idx]}, Predviđeno: {y_test_pred[idx]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
