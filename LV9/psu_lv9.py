import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Postavke
image_size = (48, 48)
batch_size = 32
epochs = 15
num_classes = 43  # broj prometnih znakova

# Putanje
train_path = "gtsrb/Train"
test_path = "gtsrb/Test"

# Image generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_gen = datagen.flow_from_directory(
    train_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

# Model po slici 9.2
model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(48, 48, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=2),
    
    Conv2D(64, (3,3), padding='same', activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=2),

    Conv2D(128, (3,3), padding='same', activation='relu'),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2), strides=2),

    Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbackovi
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_accuracy')
tensorboard = TensorBoard(log_dir="logs")

# Treniranje
"""
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint, tensorboard]
)
"""
# Učitavanje najboljeg modela
model.load_weights("best_model.h5")

# Priprema testnog skupa
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_path,
    target_size=image_size,
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Predikcija i matrica zabune
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# Ispis matrice zabune
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matrica zabune")
plt.xlabel("Predviđene oznake")
plt.ylabel("Stvarne oznake")
plt.show()

# Ispis točnosti
print("Točnost na testnom skupu:", np.mean(y_pred == y_true))
print(classification_report(y_true, y_pred))
