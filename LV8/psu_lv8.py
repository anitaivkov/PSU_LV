import tensorflow as tf
from keras import layers, models
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

#učitavanje mnist: 60 000 slika za treniranje i 10 000 za testiranje
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

#expand_dims dodaje kanal zbog CNN-a koji prihvaća slike samo s kanalima
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

#definicija modela; 
model = models.Sequential([
        #1. sloj: uči 32 filtera (jezgre) veličine 3x3
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        #smanjuje dimenzije slike
    layers.MaxPooling2D((2, 2)),
        #dodatno ekstrahiranje opet pomoću cov2d i maxpooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
        #pretvara 2d matricu u 1d vektor
    layers.Flatten(),
        #128 neurona
    layers.Dense(128, activation='relu'),
        #10 neurona
    layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

#tensorboard: omogućava vizualizaciju trening procesa
tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)
#sprema model kada je val_accuracy najbolja
checkpoint_callback = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1, callbacks=[tensorboard_callback, checkpoint_callback])

#evaluacija najboljeg modela
model = tf.keras.models.load_model('best_model.h5')
train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f'Točnost na skupu za učenje: {train_acc * 100:.2f}%')
print(f'Točnost na skupu za testiranje: {test_acc * 100:.2f}%')
y_train_pred = np.argmax(model.predict(x_train), axis=1)
cm_train = confusion_matrix(y_train, y_train_pred)

y_test_pred = np.argmax(model.predict(x_test), axis=1)
cm_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Matrica zabune - Skup za učenje')
plt.xlabel('Predviđene klase')
plt.ylabel('Stvarne klase')
plt.show()
plt.figure(figsize=(8, 6))
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.title('Matrica zabune - Skup za testiranje')
plt.xlabel('Predviđene klase')
plt.ylabel('Stvarne klase')
plt.show()

